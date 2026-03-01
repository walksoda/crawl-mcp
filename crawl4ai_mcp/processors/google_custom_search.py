"""
Google Custom Search API Client
Handles Google Custom Search API requests with pagination and error handling.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import aiohttp


class CustomSearchAPIClient:
    """Google Custom Search API client"""

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def is_configured(self) -> bool:
        """Check if Custom Search API is properly configured"""
        return bool(self.api_key and self.search_engine_id)

    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: str = 'en',
        region: str = 'us'
    ) -> Dict[str, Any]:
        """Perform search using Google Custom Search API"""
        if not self.is_configured():
            return {
                'success': False,
                'error': 'Custom Search API not configured. Missing API key or search engine ID.',
                'suggestion': 'Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in environment'
            }

        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(10, num_results),
                'gl': region,
                'hl': language,
                'safe': 'active'
            }

            results = []
            start_index = 1
            results_collected = 0

            async with aiohttp.ClientSession() as session:
                while results_collected < num_results and start_index <= 91:
                    current_params = params.copy()
                    current_params['start'] = start_index
                    current_params['num'] = min(10, num_results - results_collected)

                    async with session.get(self.base_url, params=current_params) as response:
                        if response.status == 429:
                            return {
                                'success': False,
                                'error': 'Custom Search API rate limit exceeded',
                                'status_code': 429,
                                'suggestion': 'Reduce search frequency or upgrade API quota'
                            }
                        elif response.status != 200:
                            return {
                                'success': False,
                                'error': f'Custom Search API error: HTTP {response.status}',
                                'status_code': response.status
                            }

                        data = await response.json()

                        if 'error' in data:
                            return {
                                'success': False,
                                'error': f"Custom Search API error: {data['error'].get('message', 'Unknown error')}",
                                'api_error': data['error']
                            }

                        if 'items' in data:
                            for i, item in enumerate(data['items']):
                                try:
                                    parsed_url = urlparse(item['link'])
                                    domain = parsed_url.netloc

                                    result = {
                                        'rank': results_collected + i + 1,
                                        'url': item['link'],
                                        'domain': domain,
                                        'title': item.get('title', 'No title'),
                                        'snippet': item.get('snippet', 'No description available'),
                                        'type': self._classify_url(item['link'])
                                    }

                                    if 'pagemap' in item:
                                        result['metadata'] = item['pagemap']

                                    results.append(result)
                                    results_collected += 1

                                except Exception:
                                    continue

                        search_info = data.get('searchInformation', {})
                        total_results = int(search_info.get('totalResults', '0'))

                        if results_collected >= num_results or len(data.get('items', [])) < 10:
                            break

                        start_index += 10

                        await asyncio.sleep(0.1)

            if not results:
                return {
                    'success': False,
                    'error': 'No search results found',
                    'query': query,
                    'suggestion': 'Try a different search query'
                }

            return {
                'success': True,
                'query': query,
                'total_results': len(results),
                'results': results,
                'processing_method': 'google_custom_search_api'
            }

        except aiohttp.ClientError as e:
            return {
                'success': False,
                'error': f'Network error: {str(e)}',
                'suggestion': 'Check internet connection and try again'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Custom Search API error: {str(e)}'
            }

    def _classify_url(self, url: str) -> str:
        """Classify URL by type based on domain and path (shared with GoogleSearchProcessor)"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            path = parsed.path

            if any(social in domain for social in ['youtube.com', 'youtu.be']):
                return 'video'
            elif any(social in domain for social in ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com']):
                return 'social_media'
            elif any(social in domain for social in ['reddit.com', 'quora.com', 'stackoverflow.com']):
                return 'forum'
            elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com']):
                return 'news'
            elif any(edu in domain for edu in ['.edu', '.ac.', 'scholar.google', 'arxiv.org']):
                return 'academic'
            elif any(gov in domain for gov in ['.gov', '.mil', '.org']):
                return 'official'
            elif any(shop in domain for shop in ['amazon.com', 'ebay.com', 'etsy.com']):
                return 'ecommerce'
            elif any(doc in domain for doc in ['github.com', 'docs.', 'wiki']):
                return 'documentation'
            elif any(filetype in path for filetype in ['.pdf', '.doc', '.ppt']):
                return 'document'
            else:
                return 'general'

        except Exception:
            return 'unknown'
