"""
Keenable search API client.

Docs: https://docs.keenable.ai/api-reference/search
Keenable reads the query as natural language. site: and -site: terms in the
query become the site and exclude_domains parameters (site takes one domain,
so a query with several site: terms keeps them in the query text). There is no
file type or language filter.

Opt-in only: the engine is configured by KEENABLE_API_KEY, or by
KEENABLE_KEYLESS=true for the keyless endpoint, and it is not part of the
default auto order. Keyless requests are rate limited per IP
(https://docs.keenable.ai/rate-limits). Keyed and keyless requests are both
logged under Keenable's default data policy
(https://docs.keenable.ai/data-privacy).
"""

import asyncio
import os
from typing import Any, Dict, Optional

import aiohttp

from .common import failure, failure_kind_for_status, make_result
from .exa import split_site_operators

BASE_URL = 'https://api.keenable.ai'
# Sent on keyless requests, which need an app name.
KEYLESS_TITLE = 'crawl-mcp'
_TRUE = ('1', 'true', 'yes', 'on')


class KeenableSearchClient:
    """Keenable POST /v1/search client (keyed) or /v1/search/public (keyless, opt-in)."""

    name = 'keenable'
    max_results = 50

    def __init__(self):
        self.api_key = os.getenv('KEENABLE_API_KEY', '').strip() or None
        self.keyless = os.getenv('KEENABLE_KEYLESS', '').strip().lower() in _TRUE
        self.timeout = aiohttp.ClientTimeout(total=30)

    def is_configured(self) -> bool:
        return bool(self.api_key) or self.keyless

    @property
    def endpoint(self) -> str:
        return f'{BASE_URL}/v1/search' if self.api_key else f'{BASE_URL}/v1/search/public'

    def headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        else:
            headers['X-Keenable-Title'] = KEYLESS_TITLE
        return headers

    def build_body(
        self,
        query: str,
        num_results: int,
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """JSON body for POST /v1/search."""
        text, include, exclude = split_site_operators(query)
        body: Dict[str, Any] = {
            'max_results': max(1, min(self.max_results, num_results)),
        }
        if len(include) == 1:
            body['site'] = include[0]
        elif include:
            # Only one site filter exists; keep the original terms in the text.
            text = ' '.join(query.split())
            exclude = []
        if exclude:
            body['exclude_domains'] = exclude
        # A query made only of site: terms still needs some text.
        body['query'] = text or query
        if safe_search:
            body['moderation'] = 'moderate'
        if recent_days and recent_days > 0:
            body['published_after'] = f'{int(recent_days)}d'
        return body

    async def search(
        self,
        query: str,
        num_results: int = 10,
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_configured():
            return failure('not_configured', 'Keenable not configured',
                           suggestion='Set KEENABLE_API_KEY (or KEENABLE_KEYLESS=true)')

        body = self.build_body(query, num_results, safe_search, recent_days)
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.endpoint, json=body, headers=self.headers()) as response:
                    try:
                        data = await response.json(content_type=None)
                    except Exception:
                        data = None
                    if response.status != 200:
                        return _keenable_error(response.status, data)
        except asyncio.TimeoutError:
            return failure('network', 'Keenable API timed out')
        except aiohttp.ClientError as e:
            return failure('network', f'Network error: {e}')
        except Exception as e:
            return failure('error', f'Keenable API error: {e}')

        if not isinstance(data, dict) or not isinstance(data.get('results', []), list):
            return failure('error', 'Keenable API returned an unexpected response shape')

        results = []
        seen = set()
        for item in data.get('results') or []:
            url = item.get('url') if isinstance(item, dict) else None
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(make_result(
                len(results) + 1,
                url,
                item.get('title'),
                ' '.join((item.get('snippet') or item.get('description') or '').split()),
                published_date=item.get('published_at'),
            ))

        if not results:
            return failure('no_results', 'No search results found', query=query)

        return {
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results,
            'processing_method': 'keenable_search_api' if self.api_key else 'keenable_search_public',
        }


def _keenable_error(status: int, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    message = data.get('message') or data.get('error') or f'HTTP {status}'
    return failure(failure_kind_for_status(status), f'Keenable API error: {message}',
                   status_code=status, api_error_code=data.get('code'), suggestion=data.get('fix'))
