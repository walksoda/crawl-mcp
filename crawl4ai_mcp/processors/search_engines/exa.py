"""
Exa search API client.

Docs: https://exa.ai/docs/reference/search
Exa reads the query as natural language and has no search operators.
site:/-site: terms in the query are moved to includeDomains/excludeDomains.
There is no file type filter.
"""

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .common import failure, failure_kind_for_status, make_result, normalize_region

_SITE_RE = re.compile(r'(?<!\S)(-?)site:(\S+)', re.IGNORECASE)


def split_site_operators(query: str) -> Tuple[str, List[str], List[str]]:
    """Return (query without site: terms, include domains, exclude domains)."""
    include: List[str] = []
    exclude: List[str] = []

    def take(match: 're.Match[str]') -> str:
        (exclude if match.group(1) else include).append(match.group(2))
        return ''

    stripped = ' '.join(_SITE_RE.sub(take, query).split())
    return stripped, include, exclude


class ExaSearchClient:
    """Exa /search client."""

    name = 'exa'
    base_url = 'https://api.exa.ai/search'
    max_results = 100

    def __init__(self):
        self.api_key = os.getenv('EXA_API_KEY')
        self.timeout = aiohttp.ClientTimeout(total=30)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_body(
        self,
        query: str,
        num_results: int,
        region: Optional[str] = None,
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """JSON body for POST /search."""
        text, include, exclude = split_site_operators(query)
        body: Dict[str, Any] = {
            # A query made only of site: terms still needs some text.
            'query': text or query,
            'type': 'auto',
            'numResults': max(1, min(self.max_results, num_results)),
            'contents': {'highlights': {'maxCharacters': 400}},
        }
        if include:
            body['includeDomains'] = include
        if exclude:
            body['excludeDomains'] = exclude
        country = normalize_region(region)
        if country:
            body['userLocation'] = country
        if safe_search:
            body['moderation'] = True
        if recent_days and recent_days > 0:
            start = datetime.now(timezone.utc) - timedelta(days=recent_days)
            body['startPublishedDate'] = start.strftime('%Y-%m-%dT00:00:00.000Z')
        return body

    async def search(
        self,
        query: str,
        num_results: int = 10,
        region: Optional[str] = 'us',
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_configured():
            return failure('not_configured', 'Exa API not configured', suggestion='Set EXA_API_KEY')

        headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
        body = self.build_body(query, num_results, region, safe_search, recent_days)
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.base_url, json=body, headers=headers) as response:
                    try:
                        data = await response.json(content_type=None)
                    except Exception:
                        data = None
                    if response.status != 200:
                        return _exa_error(response.status, data)
        except asyncio.TimeoutError:
            return failure('network', 'Exa API timed out')
        except aiohttp.ClientError as e:
            return failure('network', f'Network error: {e}')
        except Exception as e:
            return failure('error', f'Exa API error: {e}')

        if not isinstance(data, dict) or not isinstance(data.get('results', []), list):
            return failure('error', 'Exa API returned an unexpected response shape')

        results = []
        for item in data.get('results') or []:
            url = item.get('url') if isinstance(item, dict) else None
            if not url:
                continue
            highlights = [h for h in item.get('highlights') or [] if isinstance(h, str) and h]
            results.append(make_result(
                len(results) + 1,
                url,
                item.get('title'),
                ' ... '.join(' '.join(h.split()) for h in highlights),
                published_date=item.get('publishedDate'),
                author=item.get('author'),
            ))

        if not results:
            return failure('no_results', 'No search results found', query=query)

        out = {
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results,
            'processing_method': 'exa_search_api',
        }
        cost = (data.get('costDollars') or {}).get('total')
        if cost is not None:
            out['cost_dollars'] = cost
        return out


def _exa_error(status: int, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    message = data.get('error') or f'HTTP {status}'
    tag = data.get('tag')
    return failure(failure_kind_for_status(status), f'Exa API error: {message}',
                   status_code=status, api_error_code=tag)
