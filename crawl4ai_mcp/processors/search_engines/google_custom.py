"""
Google Custom Search JSON API client.

The API is closed to new customers and will be discontinued on 2027-01-01
(https://developers.google.com/custom-search/v1/overview).
"""

import asyncio
import os
from typing import Any, Dict, Optional

import aiohttp

from .common import failure, failure_kind_for_status, make_result


class CustomSearchAPIClient:
    """Google Custom Search API client."""

    name = 'google_custom'
    base_url = 'https://www.googleapis.com/customsearch/v1'

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.timeout = aiohttp.ClientTimeout(total=20)

    def is_configured(self) -> bool:
        """Check if Custom Search API is properly configured"""
        return bool(self.api_key and self.search_engine_id)

    def build_params(
        self,
        query: str,
        num: int,
        start: int = 1,
        language: Optional[str] = None,
        region: Optional[str] = None,
        safe_search: bool = True,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query parameters for one page (at most 10 results)."""
        params: Dict[str, Any] = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': num,
            'start': start,
            'safe': 'active' if safe_search else 'off',
        }
        if language:
            params['hl'] = language
        if region:
            params['gl'] = region
        if recent_days and recent_days > 0:
            params['dateRestrict'] = f'd{recent_days}'
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
        """Search, paging 10 results at a time (the API stops at start=91)."""
        if not self.is_configured():
            return failure('not_configured', 'Custom Search API not configured',
                           suggestion='Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID')

        results = []
        start = 1
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                while len(results) < num_results and start <= 91:
                    num = min(10, num_results - len(results))
                    params = self.build_params(query, num, start, language, region, safe_search, recent_days)
                    async with session.get(self.base_url, params=params) as response:
                        try:
                            data = await response.json(content_type=None)
                        except Exception:
                            data = None
                        if response.status != 200 or (isinstance(data, dict) and 'error' in data):
                            err = (data or {}).get('error') or {}
                            message = err.get('message') if isinstance(err, dict) else None
                            return failure(
                                failure_kind_for_status(response.status if response.status != 200 else 400),
                                f'Custom Search API error: {message or "HTTP " + str(response.status)}',
                                status_code=response.status,
                            )
                    if not isinstance(data, dict):
                        return failure('error', 'Custom Search API returned a response that is not a JSON object')

                    items = data.get('items') or []
                    for item in items:
                        url = item.get('link') if isinstance(item, dict) else None
                        if not url:
                            continue
                        results.append(make_result(
                            len(results) + 1, url, item.get('title'), item.get('snippet'),
                            metadata=item.get('pagemap'),
                        ))

                    if len(items) < num:
                        break
                    start += 10
                    await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            return failure('network', 'Custom Search API timed out')
        except aiohttp.ClientError as e:
            return failure('network', f'Network error: {e}')
        except Exception as e:
            return failure('error', f'Custom Search API error: {e}')

        if not results:
            return failure('no_results', 'No search results found', query=query)

        return {
            'success': True,
            'query': query,
            'total_results': len(results),
            'results': results,
            'processing_method': 'google_custom_search_api',
        }
