"""
Google Search Processing Module
Handles Google search queries and result processing with Custom Search API fallback
"""

import asyncio
import re
import os
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from googlesearch import search

from .google_search_helpers import RateLimiter
from .google_custom_search import CustomSearchAPIClient
from .google_search_analysis import GoogleSearchAnalysisMixin


class GoogleSearchProcessor(GoogleSearchAnalysisMixin):
    """Process Google search queries and return structured results with Custom Search API fallback"""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.custom_search_client = CustomSearchAPIClient()

        self.search_mode = os.getenv('GOOGLE_SEARCH_MODE', 'hybrid').lower()
        self.max_retries = int(os.getenv('GOOGLE_SEARCH_MAX_RETRIES', '3'))
        self.fallback_delay = float(os.getenv('GOOGLE_SEARCH_FALLBACK_DELAY', '2.0'))

        valid_modes = ['googlesearch_only', 'custom_search_only', 'hybrid']
        if self.search_mode not in valid_modes:
            logging.warning(f"Invalid GOOGLE_SEARCH_MODE '{self.search_mode}'. Using 'hybrid'")
            self.search_mode = 'hybrid'

        self.search_patterns = [
            r'site:([^\s]+)',
            r'filetype:([^\s]+)',
            r'"([^"]+)"',
            r'after:(\d{4})',
            r'before:(\d{4})'
        ]

    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate and analyze search query"""
        try:
            if not query or not query.strip():
                return {
                    'valid': False,
                    'error': 'Search query cannot be empty'
                }

            query = query.strip()
            if len(query) > 500:
                return {
                    'valid': False,
                    'error': 'Search query too long (max 500 characters)'
                }

            patterns_found = {}
            for pattern in self.search_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    pattern_name = pattern.split('(')[0].replace(':', '').replace('[', '').replace('^', '')
                    patterns_found[pattern_name] = matches

            return {
                'valid': True,
                'query': query,
                'length': len(query),
                'patterns': patterns_found,
                'is_advanced': len(patterns_found) > 0
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Query validation error: {str(e)}'
            }

    async def search_google(
        self,
        query: str,
        num_results: int = 10,
        language: str = 'en',
        region: str = 'us',
        safe_search: bool = True,
        search_genre: Optional[str] = None,
        include_snippets: bool = True
    ) -> Dict[str, Any]:
        """Perform Google search with flexible API selection and fallback."""
        try:
            validation = self.validate_query(query)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'query': query
                }

            enhanced_query = self._enhance_query_with_genre(query, search_genre)
            num_results = max(1, min(100, num_results))

            if self.search_mode == 'custom_search_only':
                return await self._search_with_custom_api(
                    enhanced_query, num_results, language, region, search_genre, validation
                )
            elif self.search_mode == 'googlesearch_only':
                return await self._search_with_googlesearch(
                    enhanced_query, num_results, language, region, search_genre, validation
                )
            else:  # hybrid mode
                return await self._search_with_fallback(
                    enhanced_query, num_results, language, region, search_genre, validation
                )

        except Exception as e:
            return {
                'success': False,
                'error': f'Search processing error: {str(e)}',
                'query': query
            }

    async def _search_with_fallback(
        self,
        query: str,
        num_results: int,
        language: str,
        region: str,
        search_genre: Optional[str],
        validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hybrid search with automatic fallback on 429 errors"""
        attempts = []

        if self.rate_limiter.can_make_request('googlesearch'):
            googlesearch_result = await self._search_with_googlesearch(
                query, num_results, language, region, search_genre, validation,
                record_rate_limit=True
            )
            attempts.append(('googlesearch-python', googlesearch_result))

            if googlesearch_result['success'] or not self._is_rate_limit_error(googlesearch_result):
                return googlesearch_result

            logging.warning(f"429 error detected with googlesearch-python, falling back to Custom Search API")
            await asyncio.sleep(self.fallback_delay)
        else:
            wait_time = self.rate_limiter.get_wait_time('googlesearch')
            attempts.append(('googlesearch-python', {
                'success': False,
                'error': f'Rate limit reached. Wait {wait_time:.1f} seconds',
                'rate_limited': True
            }))

        if self.custom_search_client.is_configured() and self.rate_limiter.can_make_request('custom_search'):
            custom_search_result = await self._search_with_custom_api(
                query, num_results, language, region, search_genre, validation,
                record_rate_limit=True
            )
            attempts.append(('google_custom_search_api', custom_search_result))

            if custom_search_result['success']:
                custom_search_result['fallback_info'] = {
                    'primary_method': 'googlesearch-python',
                    'fallback_method': 'google_custom_search_api',
                    'fallback_reason': 'Rate limit or 429 error',
                    'attempts': attempts
                }
                return custom_search_result
        else:
            if not self.custom_search_client.is_configured():
                attempts.append(('google_custom_search_api', {
                    'success': False,
                    'error': 'Custom Search API not configured',
                    'suggestion': 'Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID'
                }))
            else:
                wait_time = self.rate_limiter.get_wait_time('custom_search')
                attempts.append(('google_custom_search_api', {
                    'success': False,
                    'error': f'Rate limit reached. Wait {wait_time:.1f} seconds',
                    'rate_limited': True
                }))

        return {
            'success': False,
            'error': 'All search methods failed or rate limited',
            'query': query,
            'fallback_info': {
                'search_mode': 'hybrid',
                'attempts': attempts,
                'suggestion': 'Wait for rate limits to reset or configure missing API credentials'
            }
        }

    async def _search_with_googlesearch(
        self,
        query: str,
        num_results: int,
        language: str,
        region: str,
        search_genre: Optional[str],
        validation: Dict[str, Any],
        record_rate_limit: bool = False
    ) -> Dict[str, Any]:
        """Search using googlesearch-python library with 429 error detection"""
        try:
            search_results = []

            loop = asyncio.get_event_loop()

            def do_search():
                return list(search(
                    query,
                    num_results=num_results,
                    lang=language,
                    sleep_interval=1.0,
                    region=region,
                    safe='active'
                ))

            urls = await loop.run_in_executor(None, do_search)

            if record_rate_limit:
                self.rate_limiter.record_request('googlesearch', 'success')

            for i, url in enumerate(urls):
                if not url:
                    continue

                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc

                    title, snippet = await self._extract_title_and_snippet(url)

                    result = {
                        'rank': i + 1,
                        'url': url,
                        'domain': domain,
                        'title': title,
                        'snippet': snippet,
                        'type': self._classify_url(url)
                    }

                    search_results.append(result)

                except Exception:
                    continue

            if not search_results:
                suggestions = self._generate_simplified_query_suggestions(query)
                suggestion_text = 'Try a broader or different search query'

                if suggestions:
                    quoted_suggestions = [f"'{s}'" for s in suggestions]
                    suggestion_text = f"Try these simpler searches: {', '.join(quoted_suggestions)}"

                return {
                    'success': False,
                    'error': 'No search results found',
                    'query': query,
                    'suggestion': suggestion_text,
                    'alternative_queries': suggestions
                }

            domains = [result['domain'] for result in search_results]
            unique_domains = list(set(domains))
            domain_counts = {domain: domains.count(domain) for domain in unique_domains}

            type_counts = {}
            for result in search_results:
                result_type = result['type']
                type_counts[result_type] = type_counts.get(result_type, 0) + 1

            return {
                'success': True,
                'query': query,
                'enhanced_query': query,
                'total_results': len(search_results),
                'results': search_results,
                'search_metadata': {
                    'query_info': validation,
                    'search_params': {
                        'num_results_requested': num_results,
                        'language': language,
                        'region': region,
                        'safe_search': True,
                        'search_genre': search_genre,
                        'enhanced_query': query
                    },
                    'result_stats': {
                        'total_results': len(search_results),
                        'unique_domains': len(unique_domains),
                        'domain_distribution': domain_counts,
                        'result_types': type_counts
                    }
                },
                'processing_method': 'googlesearch-python'
            }

        except Exception as e:
            error_msg = str(e).lower()

            if record_rate_limit:
                status = '429' if '429' in error_msg or 'too many requests' in error_msg else 'error'
                self.rate_limiter.record_request('googlesearch', status)

            if '429' in error_msg or 'too many requests' in error_msg or 'rate limit' in error_msg:
                return {
                    'success': False,
                    'error': f'Rate limit exceeded: {str(e)}',
                    'status_code': 429,
                    'query': query,
                    'suggestion': 'Wait before retrying or use Custom Search API'
                }

            return {
                'success': False,
                'error': f'googlesearch-python error: {str(e)}',
                'query': query,
                'suggestion': 'Try a different search query or check your internet connection'
            }

    async def _search_with_custom_api(
        self,
        query: str,
        num_results: int,
        language: str,
        region: str,
        search_genre: Optional[str],
        validation: Dict[str, Any],
        record_rate_limit: bool = False
    ) -> Dict[str, Any]:
        """Search using Google Custom Search API"""
        result = await self.custom_search_client.search(
            query, num_results, language, region
        )

        if record_rate_limit:
            status = '429' if result.get('status_code') == 429 else ('success' if result['success'] else 'error')
            self.rate_limiter.record_request('custom_search', status)

        if result['success']:
            results = result['results']
            domains = [r['domain'] for r in results]
            unique_domains = list(set(domains))
            domain_counts = {domain: domains.count(domain) for domain in unique_domains}

            type_counts = {}
            for r in results:
                result_type = r['type']
                type_counts[result_type] = type_counts.get(result_type, 0) + 1

            result['search_metadata'] = {
                'query_info': validation,
                'search_params': {
                    'num_results_requested': num_results,
                    'language': language,
                    'region': region,
                    'safe_search': True,
                    'search_genre': search_genre,
                    'enhanced_query': query
                },
                'result_stats': {
                    'total_results': len(results),
                    'unique_domains': len(unique_domains),
                    'domain_distribution': domain_counts,
                    'result_types': type_counts
                }
            }
        else:
            if ('No search results found' in result.get('error', '') or
                result.get('total_results', 0) == 0):
                suggestions = self._generate_simplified_query_suggestions(query)
                if suggestions:
                    quoted_suggestions = [f"'{s}'" for s in suggestions]
                    result['suggestion'] = f"Try these simpler searches: {', '.join(quoted_suggestions)}"
                    result['alternative_queries'] = suggestions

        return result

    def _is_rate_limit_error(self, result: Dict[str, Any]) -> bool:
        """Check if result indicates a rate limiting error"""
        if not result.get('success', True):
            error_msg = result.get('error', '').lower()
            return (
                result.get('status_code') == 429 or
                '429' in error_msg or
                'rate limit' in error_msg or
                'too many requests' in error_msg
            )
        return False

    def _enhance_query_with_genre(self, query: str, genre: Optional[str]) -> str:
        """Enhance search query based on specified genre"""
        if not genre:
            return query

        genre_enhancements = {
            'pdf': 'filetype:pdf',
            'documents': 'filetype:pdf OR filetype:doc OR filetype:docx',
            'presentations': 'filetype:ppt OR filetype:pptx',
            'spreadsheets': 'filetype:xls OR filetype:xlsx',
            'japanese': 'site:jp OR lang:ja',
            'english': 'lang:en'
        }

        enhancement = genre_enhancements.get(genre.lower())
        if enhancement:
            enhanced_query = f"{query} ({enhancement})"
            return enhanced_query
        else:
            return query

    def get_available_genres(self) -> Dict[str, str]:
        """Get list of available search genres with descriptions"""
        return {
            'pdf': 'PDF documents only',
            'documents': 'Document files (PDF, Word, etc.)',
            'presentations': 'Presentation files (PowerPoint, etc.)',
            'spreadsheets': 'Spreadsheet files (Excel, etc.)',
            'japanese': 'Japanese language content and .jp domains',
            'english': 'English language content'
        }

    # _extract_title_and_snippet and _classify_url are inherited from GoogleSearchAnalysisMixin
