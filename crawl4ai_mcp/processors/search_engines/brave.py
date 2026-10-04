"""
Brave Search API client (Web Search endpoint).

Docs: https://api-dashboard.search.brave.com/app/documentation/web-search/query
The query string accepts operators such as site:, filetype: and lang:.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

import aiohttp

from .common import (
    failure, failure_kind_for_status, main_result_type, make_result, normalize_region, strip_html,
)

# Values the country parameter accepts; anything else is sent as ALL.
BRAVE_COUNTRIES = frozenset(
    'AR AU AT BE BR CA CL DK FI FR DE GR HK IN ID IT JP KR MY MX NL NZ NO CN '
    'PL PT PH RU SA ZA ES SE CH TW TR GB US ALL'.split()
)
# search_lang codes that differ from the plain language code.
_LANG_ALIASES = {'zh': 'zh-hans', 'pt': 'pt-pt'}
# recent_days values that match Brave's rolling freshness windows (24 hours,
# 7 days, 31 days, 365 days). Other values become a date range.
_FRESHNESS_WINDOWS = {1: 'pd', 7: 'pw', 31: 'pm', 365: 'py'}
# Forum post and FAQ answer text longer than this is cut, so a dozen entries stay readable.
FORUM_TEXT_LIMIT = 1000
# Result types read from sections of the web search response. "news" comes
# from the News Search endpoint instead: the web response's news section is
# empty for most queries that name a specific event or a concept.
_WEB_SECTIONS = ('web', 'discussions', 'faq', 'videos')


class BraveSearchClient:
    """Brave Search API client."""

    name = 'brave'
    base_url = 'https://api.search.brave.com/res/v1/web/search'
    news_url = 'https://api.search.brave.com/res/v1/news/search'
    max_per_page = 20
    max_news_per_page = 50
    max_page_offset = 9

    # Per-second limit hits (X-RateLimit-Reset of a few seconds) are retried;
    # a longer reset means the monthly quota is used up.
    max_rate_limit_retries = 2
    max_retry_wait = 2.0

    def __init__(self):
        self.api_key = os.getenv('BRAVE_SEARCH_API_KEY')
        self.timeout = aiohttp.ClientTimeout(total=20)
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop = None

    def _request_lock(self) -> asyncio.Lock:
        """One request at a time per client, so concurrent batch queries do not trip the per-second limit."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def _get_page(self, session, params: Dict[str, Any], headers: Dict[str, str], url: Optional[str] = None):
        """GET one page, waiting out short per-second rate limits. Returns (status, data)."""
        async with self._request_lock():
            for attempt in range(self.max_rate_limit_retries + 1):
                async with session.get(url or self.base_url, params=params, headers=headers) as response:
                    data = await _json_or_none(response)
                    status = response.status
                    wait = _reset_seconds(response.headers) if status == 429 else None
                if status != 429 or wait is None or wait > self.max_retry_wait or attempt == self.max_rate_limit_retries:
                    return status, data
                await asyncio.sleep(max(wait, 0.2))
        return status, data

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_params(
        self,
        query: str,
        count: int,
        page_offset: int = 0,
        language: Optional[str] = None,
        region: Optional[str] = None,
        safe_search: bool = True,
        recent_days: Optional[int] = None,
        result_types: Sequence[str] = ('web',),
    ) -> Dict[str, Any]:
        """Query parameters for one page of results; result_types becomes result_filter."""
        params: Dict[str, Any] = {
            'q': query,
            'count': count,
            'offset': page_offset,
            'safesearch': 'moderate' if safe_search else 'off',
            'text_decorations': 'false',
        }
        if result_types:
            params['result_filter'] = ','.join(result_types)
        if language:
            lang = str(language).strip().lower()
            params['search_lang'] = _LANG_ALIASES.get(lang, lang)
        country = normalize_region(region)
        if country:
            params['country'] = country if country in BRAVE_COUNTRIES else 'ALL'
        if recent_days and recent_days > 0:
            params['freshness'] = brave_freshness(recent_days)
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
        Search, paging with offset when num_results exceeds one page.

        result_types selects web, discussions, faq, videos and news. The first
        four come from one web search request (count and offset apply to web
        only, so paging adds web results and the other sections come from the
        first page). news comes from a separate News Search request, paged
        50 at a time. The main type (web when requested, else the first) goes
        to results, the others to extra_results.
        """
        if not self.is_configured():
            return failure('not_configured', 'Brave Search API not configured',
                           suggestion='Set BRAVE_SEARCH_API_KEY')

        result_types = tuple(result_types) or ('web',)
        main = main_result_type(result_types)
        web_types = tuple(t for t in result_types if t in _WEB_SECTIONS)
        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': self.api_key,
        }
        collected: Dict[str, list] = {t: [] for t in result_types}
        extra_errors: Dict[str, str] = {}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                if web_types:
                    error = await self._collect_web(
                        session, headers, collected, web_types, query, num_results,
                        language, region, safe_search, recent_days,
                    )
                    if error:
                        return error
                if 'news' in result_types:
                    error = await self._collect_news(
                        session, headers, collected['news'], query, num_results,
                        language, region, safe_search, recent_days,
                    )
                    if error:
                        if main == 'news':
                            return error
                        # The web request already succeeded; keep its results.
                        extra_errors['news'] = error['error']
        except asyncio.TimeoutError:
            return failure('network', 'Brave Search API timed out')
        except aiohttp.ClientError as e:
            return failure('network', f'Network error: {e}')
        except Exception as e:
            return failure('error', f'Brave Search API error: {e}')

        if not any(collected.values()):
            if main != 'web':
                return failure(
                    'no_results', f'No {main} results found', query=query,
                    suggestion=f'Brave returns {main} only for some queries; '
                               'try another wording or result_type web',
                )
            return failure('no_results', 'No search results found', query=query)

        out = {
            'success': True,
            'query': query,
            'result_type': main,
            'total_results': len(collected[main]),
            'results': collected[main],
            'processing_method': 'brave_search_api',
        }
        extras = {t: collected[t] for t in result_types if t != main}
        if extras:
            out['extra_results'] = extras
        if extra_errors:
            out['extra_errors'] = extra_errors
        return out

    async def _collect_web(
        self, session, headers, collected, web_types, query, num_results,
        language, region, safe_search, recent_days,
    ) -> Optional[Dict[str, Any]]:
        """Fill collected from the web search endpoint. Returns a failure dict or None."""
        seen: set = set()
        # offset counts pages of `count` results, so every page uses the same count.
        count = min(self.max_per_page, num_results)
        page_offset = 0
        while page_offset <= self.max_page_offset:
            # Later pages only add web results.
            page_types = web_types if page_offset == 0 else ('web',)
            params = self.build_params(
                query, count, page_offset, language, region, safe_search, recent_days, page_types
            )
            status, data = await self._get_page(session, params, headers)
            if status != 200:
                return _brave_error(status, data)
            if not isinstance(data, dict):
                return failure('error', 'Brave Search API returned a response that is not a JSON object')

            for rtype in page_types:
                items = (data.get(rtype) or {}).get('results') or []
                _append_new(collected[rtype], seen, rtype, items, num_results)

            if 'web' not in web_types or len(collected['web']) >= num_results:
                return None
            # A page can hold fewer than `count` results while more remain.
            more = (data.get('query') or {}).get('more_results_available')
            if not more or not (data.get('web') or {}).get('results'):
                return None
            page_offset += 1
            # Plans have a per-second limit; spread page requests out.
            await asyncio.sleep(1.0)
        return None

    async def _collect_news(
        self, session, headers, bucket, query, num_results,
        language, region, safe_search, recent_days,
    ) -> Optional[Dict[str, Any]]:
        """Fill bucket from the News Search endpoint (up to 50 per page). Returns a failure dict or None."""
        seen: set = set()
        count = min(self.max_news_per_page, num_results)
        page_offset = 0
        while page_offset <= self.max_page_offset:
            params = self.build_params(
                query, count, page_offset, language, region, safe_search, recent_days, result_types=()
            )
            status, data = await self._get_page(session, params, headers, url=self.news_url)
            if status != 200:
                return _brave_error(status, data)
            if not isinstance(data, dict):
                return failure('error', 'Brave News Search API returned a response that is not a JSON object')

            items = data.get('results') or []
            _append_new(bucket, seen, 'news', items, num_results)
            if len(bucket) >= num_results or len(items) < count:
                return None
            page_offset += 1
            await asyncio.sleep(1.0)
        return None


def brave_freshness(recent_days: int) -> str:
    """
    Brave freshness for the past recent_days days.

    1, 7, 31 and 365 use the rolling windows pd, pw, pm and py. Other values
    become a UTC date range from recent_days days ago through today, which
    can reach back up to one day further because whole days are counted.
    """
    window = _FRESHNESS_WINDOWS.get(recent_days)
    if window:
        return window
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=recent_days)
    return f"{start.isoformat()}to{today.isoformat()}"


def _append_new(bucket: list, seen: set, rtype: str, items, limit: int) -> None:
    """Add items of one result type to bucket, skipping repeats and stopping at limit."""
    for item in items:
        if len(bucket) >= limit:
            return
        url = item.get('url') if isinstance(item, dict) else None
        # One page can hold several FAQ entries, so the question is part of the key.
        key = (rtype, url, item.get('question') if rtype == 'faq' else None)
        if not url or key in seen:
            continue
        seen.add(key)
        bucket.append(_make_typed_result(rtype, len(bucket) + 1, url, item))


def _make_typed_result(rtype: str, rank: int, url: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """One result entry with the fields specific to its Brave section."""
    if rtype == 'faq':
        answer = strip_html(item.get('answer'))
        return make_result(
            rank, url, strip_html(item.get('title')), _truncate(answer) if answer else None,
            question=strip_html(item.get('question')) or None,
        )
    extra: Dict[str, Any] = {'age': item.get('age'), 'published_date': item.get('page_age')}
    if rtype == 'discussions':
        extra.update(_forum_fields(item))
    elif rtype == 'videos':
        video = item.get('video') if isinstance(item.get('video'), dict) else {}
        extra.update(duration=video.get('duration'), creator=video.get('creator'),
                     publisher=video.get('publisher'))
    elif rtype == 'news':
        meta = item.get('meta_url') if isinstance(item.get('meta_url'), dict) else {}
        profile = item.get('profile') if isinstance(item.get('profile'), dict) else {}
        # News Search results carry no breaking flag, so none is added.
        extra.update(source=profile.get('name') or meta.get('hostname'))
    return make_result(rank, url, strip_html(item.get('title')), strip_html(item.get('description')), **extra)


def _truncate(text: str, limit: int = FORUM_TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + '...'


def _forum_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """forum_name, num_answers, score, question, top_comment from a discussion result's data."""
    data = item.get('data') if isinstance(item.get('data'), dict) else {}
    question = strip_html(data.get('question'))
    top_comment = strip_html(data.get('top_comment'))
    return {
        'forum_name': data.get('forum_name'),
        'num_answers': data.get('num_answers'),
        'score': data.get('score'),
        'question': _truncate(question) if question else None,
        'top_comment': _truncate(top_comment) if top_comment else None,
    }


def _reset_seconds(headers) -> Optional[float]:
    """First value of X-RateLimit-Reset ("1, 1234567": per-second window, then monthly)."""
    raw = (headers or {}).get('X-RateLimit-Reset')
    if not raw:
        return None
    try:
        return float(str(raw).split(',')[0].strip())
    except ValueError:
        return None


async def _json_or_none(response) -> Optional[Dict[str, Any]]:
    try:
        return await response.json(content_type=None)
    except Exception:
        return None


def _brave_error(status: int, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    err = (data or {}).get('error') or {}
    code = err.get('code')
    detail = err.get('detail') or f'HTTP {status}'
    # An invalid key is reported as 422 SUBSCRIPTION_TOKEN_INVALID.
    kind = 'auth' if code == 'SUBSCRIPTION_TOKEN_INVALID' else failure_kind_for_status(status)
    return failure(kind, f'Brave Search API error: {detail}', status_code=status, api_error_code=code)
