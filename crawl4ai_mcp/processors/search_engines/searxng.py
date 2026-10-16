"""
SearXNG metasearch client (self-hosted, no API key).

Docs: https://docs.searxng.org/dev/search_api.html
SearXNG passes the query to its own engines (Google, Bing, DuckDuckGo, ...),
so site:/filetype: operators reach the engines that understand them. The
instance must allow the json format (search.formats in settings.yml);
otherwise it answers HTTP 403.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

from .common import failure, failure_kind_for_status, make_result, normalize_region

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 120.0
# SearXNG has no day-count filter, only these ranges. recent_days is rounded up.
TIME_RANGES = ((1, 'day'), (7, 'week'), (31, 'month'), (365, 'year'))
# Result pages fetched at most to reach num_results (about 10 results per page).
MAX_PAGES = 5


def time_range_for(recent_days: Optional[int]) -> Optional[str]:
    """Smallest SearXNG time_range that covers recent_days; None for no filter."""
    if not recent_days or recent_days <= 0:
        return None
    for days, name in TIME_RANGES:
        if recent_days <= days:
            return name
    return None


def _read_base_url() -> Optional[str]:
    """SEARXNG_URL when it is an http(s) URL with a host, else None."""
    raw = os.getenv('SEARXNG_URL', '').strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        logging.warning(f"Ignoring SEARXNG_URL '{raw}': expected an http:// or https:// URL")
        return None
    return raw.rstrip('/')


def _read_timeout() -> float:
    """SEARXNG_TIMEOUT in seconds (0 < value <= MAX_TIMEOUT), else DEFAULT_TIMEOUT."""
    raw = os.getenv('SEARXNG_TIMEOUT', '').strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0 < value <= MAX_TIMEOUT:
        logging.warning(f"Invalid SEARXNG_TIMEOUT '{raw}'. Using {DEFAULT_TIMEOUT:g}")
        return DEFAULT_TIMEOUT
    return value


def _read_engines() -> Optional[str]:
    """SEARXNG_ENGINES as a clean comma list, or None for the instance's defaults."""
    names = [n.strip() for n in os.getenv('SEARXNG_ENGINES', '').split(',') if n.strip()]
    return ','.join(names) or None


class SearXNGSearchClient:
    """SearXNG /search client (format=json)."""

    name = 'searxng'
    max_results = 100

    def __init__(self):
        self.base_url = _read_base_url()
        self.engines = _read_engines()
        self.timeout = aiohttp.ClientTimeout(total=_read_timeout())

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def build_params(
        self,
        query: str,
        page: int = 1,
        language: Optional[str] = None,
        region: Optional[str] = None,
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query parameters for one result page."""
        params: Dict[str, Any] = {
            'q': query,
            'format': 'json',
            'categories': 'general',
            'safesearch': 1 if safe_search else 0,
            'pageno': page,
        }
        if self.engines:
            params['engines'] = self.engines
        if language:
            lang = str(language).strip().lower()
            country = normalize_region(region)
            params['language'] = f'{lang}-{country}' if country else lang
        time_range = time_range_for(recent_days)
        if time_range:
            params['time_range'] = time_range
        return params

    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: Optional[str] = 'en',
        region: Optional[str] = 'us',
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search, reading result pages until num_results unique URLs (at most MAX_PAGES pages)."""
        if not self.is_configured():
            return failure('not_configured', 'SearXNG not configured', suggestion='Set SEARXNG_URL')

        num_results = max(1, min(self.max_results, num_results))
        results: List[Dict[str, Any]] = []
        seen = set()
        unresponsive: List[str] = []
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                for page in range(1, MAX_PAGES + 1):
                    params = self.build_params(query, page, language, region, safe_search, recent_days)
                    async with session.get(f'{self.base_url}/search', params=params) as response:
                        if response.status != 200:
                            if results:
                                break  # keep the results of the earlier pages
                            return _searxng_error(response.status)
                        try:
                            data = await response.json(content_type=None)
                        except Exception:
                            data = None

                    if not isinstance(data, dict) or not isinstance(data.get('results', []), list):
                        return failure('error', 'SearXNG returned an unexpected response shape')
                    for entry in data.get('unresponsive_engines') or []:
                        label = ': '.join(str(x) for x in entry) if isinstance(entry, (list, tuple)) else str(entry)
                        if label not in unresponsive:
                            unresponsive.append(label)

                    added = 0
                    for item in data.get('results') or []:
                        url = item.get('url') if isinstance(item, dict) else None
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        added += 1
                        results.append(make_result(
                            len(results) + 1,
                            url,
                            item.get('title'),
                            ' '.join((item.get('content') or '').split()),
                            published_date=item.get('publishedDate'),
                            source_engines=item.get('engines') or None,
                            score=item.get('score'),
                        ))
                        if len(results) >= num_results:
                            break
                    if len(results) >= num_results or added == 0:
                        break
        except asyncio.TimeoutError:
            return failure('network', 'SearXNG timed out')
        except aiohttp.ClientError as e:
            return failure('network', f'Network error: {e}')
        except Exception as e:
            return failure('error', f'SearXNG error: {e}')

        if not results:
            if unresponsive:
                # Every engine behind SearXNG failed, so auto mode should try the next engine.
                return failure('server', 'All SearXNG engines failed: ' + '; '.join(unresponsive),
                               query=query, unresponsive_engines=unresponsive)
            return failure('no_results', 'No search results found', query=query)

        out: Dict[str, Any] = {
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results,
            'processing_method': 'searxng',
        }
        time_range = time_range_for(recent_days)
        if time_range:
            out['time_range'] = time_range
        if unresponsive:
            out['unresponsive_engines'] = unresponsive
        return out


def _searxng_error(status: int) -> Dict[str, Any]:
    hint = None
    if status == 403:
        hint = 'Enable the json format in the SearXNG settings.yml (search.formats)'
    return failure(failure_kind_for_status(status), f'SearXNG error: HTTP {status}',
                   status_code=status, suggestion=hint)
