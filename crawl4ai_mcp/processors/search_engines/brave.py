"""
Brave Search API client (Web Search endpoint).

Docs: https://api-dashboard.search.brave.com/app/documentation/web-search/query
The query string accepts operators such as site:, filetype: and lang:.
"""

import asyncio
import os
from datetime import date, timedelta
from typing import Any, Dict, Optional

import aiohttp

from .common import failure, failure_kind_for_status, make_result, normalize_region, strip_html

# Values the country parameter accepts; anything else is sent as ALL.
BRAVE_COUNTRIES = frozenset(
    'AR AU AT BE BR CA CL DK FI FR DE GR HK IN ID IT JP KR MY MX NL NZ NO CN '
    'PL PT PH RU SA ZA ES SE CH TW TR GB US ALL'.split()
)
# search_lang codes that differ from the plain language code.
_LANG_ALIASES = {'zh': 'zh-hans', 'pt': 'pt-pt'}
# Forum post text longer than this is cut, so a dozen posts stay readable.
FORUM_TEXT_LIMIT = 1000


class BraveSearchClient:
    """Brave Search API client."""

    name = 'brave'
    base_url = 'https://api.search.brave.com/res/v1/web/search'
    max_per_page = 20
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

    async def _get_page(self, session, params: Dict[str, Any], headers: Dict[str, str]):
        """GET one page, waiting out short per-second rate limits. Returns (status, data)."""
        async with self._request_lock():
            for attempt in range(self.max_rate_limit_retries + 1):
                async with session.get(self.base_url, params=params, headers=headers) as response:
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
        result_type: str = 'web',
    ) -> Dict[str, Any]:
        """Query parameters for one page of results (result_type: 'web' or 'discussions')."""
        params: Dict[str, Any] = {
            'q': query,
            'count': count,
            'offset': page_offset,
            'safesearch': 'moderate' if safe_search else 'off',
            'result_filter': result_type,
            'text_decorations': 'false',
        }
        if language:
            lang = str(language).strip().lower()
            params['search_lang'] = _LANG_ALIASES.get(lang, lang)
        country = normalize_region(region)
        if country:
            params['country'] = country if country in BRAVE_COUNTRIES else 'ALL'
        if recent_days and recent_days > 0:
            today = date.today()
            start = today - timedelta(days=recent_days)
            params['freshness'] = f"{start.isoformat()}to{today.isoformat()}"
        return params

    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: Optional[str] = 'en',
        region: Optional[str] = 'us',
        safe_search: bool = True,
        recent_days: Optional[int] = None,
        result_type: str = 'web',
    ) -> Dict[str, Any]:
        """
        Search, paging with offset when num_results exceeds one page.

        result_type='discussions' returns Brave's forum cluster instead of web
        results. count and offset do not apply to it, so one request is sent
        and the cluster is cut to num_results.
        """
        if not self.is_configured():
            return failure('not_configured', 'Brave Search API not configured',
                           suggestion='Set BRAVE_SEARCH_API_KEY')

        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': self.api_key,
        }
        results = []
        seen = set()
        # offset counts pages of `count` results, so every page uses the same count.
        count = min(self.max_per_page, num_results)
        page_offset = 0
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                while len(results) < num_results and page_offset <= self.max_page_offset:
                    params = self.build_params(
                        query, count, page_offset, language, region, safe_search, recent_days, result_type
                    )
                    status, data = await self._get_page(session, params, headers)
                    if status != 200:
                        return _brave_error(status, data)
                    if not isinstance(data, dict):
                        return failure('error', 'Brave Search API returned a response that is not a JSON object')

                    items = (data.get(result_type) or {}).get('results') or []
                    for item in items:
                        url = item.get('url') if isinstance(item, dict) else None
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        results.append(make_result(
                            len(results) + 1,
                            url,
                            strip_html(item.get('title')),
                            strip_html(item.get('description')),
                            age=item.get('age'),
                            published_date=item.get('page_age'),
                            **(_forum_fields(item) if result_type == 'discussions' else {}),
                        ))
                        if len(results) >= num_results:
                            break

                    if result_type == 'discussions':
                        break
                    # A page can hold fewer than `count` results while more remain.
                    more = (data.get('query') or {}).get('more_results_available')
                    if not more or not items:
                        break
                    page_offset += 1
                    # Plans have a per-second limit; spread page requests out.
                    await asyncio.sleep(1.0)
        except asyncio.TimeoutError:
            return failure('network', 'Brave Search API timed out')
        except aiohttp.ClientError as e:
            return failure('network', f'Network error: {e}')
        except Exception as e:
            return failure('error', f'Brave Search API error: {e}')

        if not results:
            if result_type == 'discussions':
                return failure(
                    'no_results', 'No discussion results found', query=query,
                    suggestion='Brave returns discussions only for some queries; '
                               'try another wording or result_type web',
                )
            return failure('no_results', 'No search results found', query=query)

        return {
            'success': True,
            'query': query,
            'result_type': result_type,
            'total_results': len(results),
            'results': results,
            'processing_method': 'brave_search_api',
        }


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
