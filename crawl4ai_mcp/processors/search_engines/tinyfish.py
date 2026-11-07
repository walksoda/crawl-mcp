"""
TinyFish Search API client.

Docs: https://docs.tinyfish.ai/search-api/reference
The query string accepts operators such as site: and filetype:.

One request returns one page of about 10 results for one domain_type (web,
news or research_paper), so larger num_results values page with page=0..10
and each result type is a separate series of requests. Search is free up to
12,000 requests per day; the limit is 30 requests per minute per account.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from .common import failure, failure_kind_for_status, main_result_type, make_result, normalize_region

# Result types this engine returns; each one is a domain_type value.
TINYFISH_RESULT_TYPES = ('web', 'news', 'research_paper')
# recency_minutes accepts 1 to 5,256,000 minutes (3650 days).
MAX_RECENCY_MINUTES = 5_256_000
# Error codes of HTTP 402: the daily free allowance is used up.
_QUOTA_CODES = frozenset({'INSUFFICIENT_CREDITS', 'FREE_ALLOWANCE_EXHAUSTED'})


class TinyFishSearchClient:
    """TinyFish Search API client."""

    name = 'tinyfish'
    base_url = 'https://api.search.tinyfish.ai/'
    max_page = 10

    # The limit is per minute, so only short Retry-After waits are worth taking.
    max_rate_limit_retries = 1
    max_retry_wait = 5.0

    def __init__(self):
        self.api_key = os.getenv('TINYFISH_API_KEY')
        self.timeout = aiohttp.ClientTimeout(total=30)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_params(
        self,
        query: str,
        result_type: str = 'web',
        page: int = 0,
        language: Optional[str] = None,
        region: Optional[str] = None,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query parameters for one page of one result type.

        research_paper rejects date and recency filters, so recent_days becomes
        pub_year_min, the year recent_days days ago (a whole-year range).
        """
        params: Dict[str, Any] = {'query': query, 'page': page}
        if result_type != 'web':
            params['domain_type'] = result_type
        if language:
            params['language'] = str(language).strip().lower()
        country = normalize_region(region)
        if country:
            params['location'] = country
        if recent_days and recent_days > 0:
            if result_type == 'research_paper':
                start = datetime.now(timezone.utc) - timedelta(days=recent_days)
                params['pub_year_min'] = start.year
            else:
                params['recency_minutes'] = min(recent_days * 1440, MAX_RECENCY_MINUTES)
        return params

    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: Optional[str] = 'en',
        region: Optional[str] = 'us',
        safe_search: bool = True,
        recent_days: Optional[int] = None,
        result_types: Sequence[str] = ('web',),
    ) -> Dict[str, Any]:
        """
        Search each requested result type, paging until num_results.

        safe_search is accepted for a common signature; the API has no such
        filter. The main type (web when requested, else the first) goes to
        results, the others to extra_results. A failure of another type is
        reported in extra_errors and keeps the main results.
        """
        if not self.is_configured():
            return failure('not_configured', 'TinyFish Search API not configured',
                           suggestion='Set TINYFISH_API_KEY')

        result_types = tuple(result_types) or ('web',)
        main = main_result_type(result_types)
        ordered = (main,) + tuple(t for t in result_types if t != main)
        headers = {'Accept': 'application/json', 'X-API-Key': self.api_key}
        collected: Dict[str, List[Dict[str, Any]]] = {}
        extra_failures: Dict[str, Dict[str, Any]] = {}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                for rtype in ordered:
                    bucket: List[Dict[str, Any]] = []
                    error = await self._collect_guarded(
                        session, headers, bucket, rtype, query, num_results, language, region, recent_days,
                    )
                    if error:
                        if rtype == main:
                            return error
                        extra_failures[rtype] = error
                    collected[rtype] = bucket
        except Exception as e:
            return _exception_failure(e)

        if not any(collected.values()):
            if extra_failures:
                # Nothing to return and a request failed: keep its failure_kind so auto moves on.
                return dict(next(iter(extra_failures.values())))
            return failure('no_results', f'No {main} results found' if main != 'web' else 'No search results found',
                           query=query)

        out = {
            'success': True,
            'query': query,
            'result_type': main,
            'total_results': len(collected[main]),
            'results': collected[main],
            'processing_method': 'tinyfish_search_api',
        }
        extras = {t: collected.get(t, []) for t in result_types if t != main}
        if extras:
            out['extra_results'] = extras
        if extra_failures:
            out['extra_errors'] = {t: f['error'] for t, f in extra_failures.items()}
        return out

    async def _collect_guarded(self, *args) -> Optional[Dict[str, Any]]:
        """_collect with exceptions turned into failures, so one type's timeout keeps the others."""
        try:
            return await self._collect(*args)
        except Exception as e:
            return _exception_failure(e)

    async def _get_page(self, session, params: Dict[str, Any], headers: Dict[str, str]):
        """GET one page, waiting out a short Retry-After on HTTP 429. Returns (status, data)."""
        for attempt in range(self.max_rate_limit_retries + 1):
            async with session.get(self.base_url, params=params, headers=headers) as response:
                data = await _json_or_none(response)
                status = response.status
                wait = _retry_after(response.headers) if status == 429 else None
            if status != 429 or wait is None or wait > self.max_retry_wait or attempt == self.max_rate_limit_retries:
                return status, data
            await asyncio.sleep(max(wait, 0.2))
        return status, data

    async def _collect(
        self, session, headers, bucket, rtype, query, num_results, language, region, recent_days,
    ) -> Optional[Dict[str, Any]]:
        """Fill bucket with results of one type. Returns a failure dict or None."""
        seen: set = set()
        for page in range(self.max_page + 1):
            params = self.build_params(query, rtype, page, language, region, recent_days)
            status, data = await self._get_page(session, params, headers)
            if status != 200:
                return _tinyfish_error(status, data)
            if not isinstance(data, dict) or not isinstance(data.get('results'), list):
                return failure('error', 'TinyFish Search API returned an unexpected response shape')

            items = data.get('results') or []
            added = _append_new(bucket, seen, rtype, items, num_results)
            # An empty page, or one holding only repeats, means the results ran out.
            if len(bucket) >= num_results or not added:
                return None
        return None


def _append_new(bucket: list, seen: set, rtype: str, items, limit: int) -> int:
    """Add new results of one type to bucket, stopping at limit. Returns how many were added."""
    added = 0
    for item in items:
        if len(bucket) >= limit:
            break
        url = item.get('url') if isinstance(item, dict) else None
        if not url or url in seen:
            continue
        seen.add(url)
        bucket.append(_make_typed_result(rtype, len(bucket) + 1, url, item))
        added += 1
    return added


def _make_typed_result(rtype: str, rank: int, url: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """One result entry with the fields that come with its result type."""
    # date is free text: "4 days ago", "Nov 4, 2023" or an RFC 1123 date for news.
    extra: Dict[str, Any] = {'published_date': item.get('date') or None}
    if rtype == 'news':
        extra['source'] = item.get('publisher') or item.get('site_name')
    elif rtype == 'research_paper':
        authors = item.get('authors')
        extra.update(
            authors=[a for a in authors if isinstance(a, str)] if isinstance(authors, list) else None,
            venue=item.get('venue'),
            year=item.get('year'),
            cited_by_count=item.get('cited_by_count'),
            pdf_url=item.get('pdf_url'),
        )
    return make_result(rank, url, item.get('title'), item.get('snippet'), **extra)


def _retry_after(headers) -> Optional[float]:
    """Seconds from Retry-After, else from X-RateLimit-Reset (an epoch time in seconds)."""
    headers = headers or {}
    raw = headers.get('Retry-After')
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    reset = headers.get('X-RateLimit-Reset')
    if reset:
        try:
            return max(0.0, float(reset) - datetime.now(timezone.utc).timestamp())
        except ValueError:
            pass
    return None


def _exception_failure(e: Exception) -> Dict[str, Any]:
    if isinstance(e, asyncio.TimeoutError):
        return failure('network', 'TinyFish Search API timed out')
    if isinstance(e, aiohttp.ClientError):
        return failure('network', f'Network error: {e}')
    return failure('error', f'TinyFish Search API error: {e}')


async def _json_or_none(response) -> Optional[Any]:
    try:
        return await response.json(content_type=None)
    except Exception:
        return None


def _tinyfish_error(status: int, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    err = (data or {}).get('error') if isinstance(data, dict) else None
    err = err if isinstance(err, dict) else {}
    code = err.get('code')
    message = err.get('message') or f'HTTP {status}'
    kind = 'quota' if code in _QUOTA_CODES else failure_kind_for_status(status)
    suggestion = None
    if code == 'INSUFFICIENT_CREDITS':
        suggestion = 'The daily free allowance (12,000 searches, resets 00:00 UTC) is used up; add funds to the TinyFish wallet'
    elif code == 'FREE_ALLOWANCE_EXHAUSTED':
        suggestion = 'The daily free allowance is used up; it resets at 00:00 UTC'
    elif status == 429:
        suggestion = 'TinyFish allows 30 requests per minute per account; retry after a minute'
    return failure(kind, f'TinyFish Search API error: {message}', status_code=status,
                   api_error_code=code, suggestion=suggestion)
