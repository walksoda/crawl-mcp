"""
Google Search Processing Module
Handles Google search queries and result processing with Custom Search API fallback
"""

import asyncio
import re
import os
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse, urljoin
from googlesearch import search
import aiohttp
import logging
from bs4 import BeautifulSoup
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta


@dataclass
class SearchRequest:
    """Represents a search request with rate limiting info"""
    timestamp: datetime
    method: str  # 'googlesearch' or 'custom_search'
    status: str  # 'success', 'error', '429'


class RateLimiter:
    """Rate limiting manager for search APIs"""
    
    def __init__(self):
        self.requests = defaultdict(list)  # method -> list of SearchRequest
        self.rpm_limits = {
            'googlesearch': int(os.getenv('GOOGLESEARCH_PYTHON_RPM', '60')),
            'custom_search': int(os.getenv('CUSTOM_SEARCH_API_RPM', '100'))
        }
    
    def can_make_request(self, method: str) -> bool:
        """Check if a request can be made without exceeding rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[method] = [
            req for req in self.requests[method] 
            if req.timestamp > cutoff
        ]
        
        return len(self.requests[method]) < self.rpm_limits[method]
    
    def record_request(self, method: str, status: str):
        """Record a completed request"""
        self.requests[method].append(SearchRequest(
            timestamp=datetime.now(),
            method=method,
            status=status
        ))
    
    def get_wait_time(self, method: str) -> float:
        """Get suggested wait time in seconds before next request"""
        if self.can_make_request(method):
            return 0.0
        
        # Find oldest request in current window
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        oldest_request = min(
            (req for req in self.requests[method] if req.timestamp > cutoff),
            key=lambda r: r.timestamp,
            default=None
        )
        
        if oldest_request:
            wait_until = oldest_request.timestamp + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            return max(0.0, wait_seconds + 1.0)  # Add 1 second buffer
        
        return 0.0


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
            # Build request parameters
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(10, num_results),  # Custom Search API max is 10 per request
                'gl': region,
                'hl': language,
                'safe': 'active'
            }
            
            results = []
            start_index = 1
            results_collected = 0
            
            async with aiohttp.ClientSession() as session:
                while results_collected < num_results and start_index <= 91:  # Max 100 results total
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
                        
                        # Process search results
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
                                    
                                    # Add additional metadata if available
                                    if 'pagemap' in item:
                                        result['metadata'] = item['pagemap']
                                    
                                    results.append(result)
                                    results_collected += 1
                                    
                                except Exception:
                                    continue
                        
                        # Check if we have more results to fetch
                        search_info = data.get('searchInformation', {})
                        total_results = int(search_info.get('totalResults', '0'))
                        
                        if results_collected >= num_results or len(data.get('items', [])) < 10:
                            break
                        
                        start_index += 10
                        
                        # Add delay between requests to be respectful
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
            
            # Social media platforms
            if any(social in domain for social in ['youtube.com', 'youtu.be']):
                return 'video'
            elif any(social in domain for social in ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com']):
                return 'social_media'
            elif any(social in domain for social in ['reddit.com', 'quora.com', 'stackoverflow.com']):
                return 'forum'
            
            # News and media
            elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com']):
                return 'news'
            
            # Academic and education
            elif any(edu in domain for edu in ['.edu', '.ac.', 'scholar.google', 'arxiv.org']):
                return 'academic'
            
            # Government and official
            elif any(gov in domain for gov in ['.gov', '.mil', '.org']):
                return 'official'
            
            # E-commerce
            elif any(shop in domain for shop in ['amazon.com', 'ebay.com', 'etsy.com']):
                return 'ecommerce'
            
            # Documentation
            elif any(doc in domain for doc in ['github.com', 'docs.', 'wiki']):
                return 'documentation'
            
            # File types
            elif any(filetype in path for filetype in ['.pdf', '.doc', '.ppt']):
                return 'document'
            
            else:
                return 'general'
                
        except Exception:
            return 'unknown'


class GoogleSearchProcessor:
    """Process Google search queries and return structured results with Custom Search API fallback"""
    
    def __init__(self):
        # Rate limiting and API clients
        self.rate_limiter = RateLimiter()
        self.custom_search_client = CustomSearchAPIClient()
        
        # Search mode configuration
        self.search_mode = os.getenv('GOOGLE_SEARCH_MODE', 'hybrid').lower()
        self.max_retries = int(os.getenv('GOOGLE_SEARCH_MAX_RETRIES', '3'))
        self.fallback_delay = float(os.getenv('GOOGLE_SEARCH_FALLBACK_DELAY', '2.0'))
        
        # Validation for search mode
        valid_modes = ['googlesearch_only', 'custom_search_only', 'hybrid']
        if self.search_mode not in valid_modes:
            logging.warning(f"Invalid GOOGLE_SEARCH_MODE '{self.search_mode}'. Using 'hybrid'")
            self.search_mode = 'hybrid'
        
        self.search_patterns = [
            # Domain-specific search patterns
            r'site:([^\s]+)',
            # File type search patterns  
            r'filetype:([^\s]+)',
            # Quote search patterns
            r'"([^"]+)"',
            # Date range patterns
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
            
            # Basic validation
            query = query.strip()
            if len(query) > 500:
                return {
                    'valid': False,
                    'error': 'Search query too long (max 500 characters)'
                }
            
            # Analyze query patterns
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
        """
        Perform Google search with flexible API selection and fallback.
        
        Supports three modes:
        - googlesearch_only: Use only googlesearch-python library
        - custom_search_only: Use only Google Custom Search API
        - hybrid: Try googlesearch-python first, fallback to Custom Search API on 429 errors
        """
        try:
            # Validate query first
            validation = self.validate_query(query)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'query': query
                }
            
            # Apply genre-specific query modifications
            enhanced_query = self._enhance_query_with_genre(query, search_genre)
            
            # Limit results to reasonable range
            num_results = max(1, min(100, num_results))
            
            # Route to appropriate search method based on configuration
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
        
        # Try googlesearch-python first
        if self.rate_limiter.can_make_request('googlesearch'):
            googlesearch_result = await self._search_with_googlesearch(
                query, num_results, language, region, search_genre, validation,
                record_rate_limit=True
            )
            attempts.append(('googlesearch-python', googlesearch_result))
            
            # If successful or non-429 error, return result
            if googlesearch_result['success'] or not self._is_rate_limit_error(googlesearch_result):
                return googlesearch_result
            
            # 429 error detected, wait and try Custom Search API
            logging.warning(f"429 error detected with googlesearch-python, falling back to Custom Search API")
            await asyncio.sleep(self.fallback_delay)
        else:
            # Rate limit prevents googlesearch request
            wait_time = self.rate_limiter.get_wait_time('googlesearch')
            attempts.append(('googlesearch-python', {
                'success': False,
                'error': f'Rate limit reached. Wait {wait_time:.1f} seconds',
                'rate_limited': True
            }))
        
        # Try Custom Search API as fallback
        if self.custom_search_client.is_configured() and self.rate_limiter.can_make_request('custom_search'):
            custom_search_result = await self._search_with_custom_api(
                query, num_results, language, region, search_genre, validation,
                record_rate_limit=True
            )
            attempts.append(('google_custom_search_api', custom_search_result))
            
            if custom_search_result['success']:
                # Add fallback information to successful result
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
        
        # Both methods failed
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
            
            # Run search in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def do_search():
                return list(search(
                    query,
                    num_results=num_results,
                    lang=language,
                    sleep_interval=1.0,  # Respectful delay between requests
                    region=region,
                    safe='active'  # Always use safe search as requested
                ))
            
            urls = await loop.run_in_executor(None, do_search)
            
            # Record successful request
            if record_rate_limit:
                self.rate_limiter.record_request('googlesearch', 'success')
            
            # Process results and try to get titles/snippets
            for i, url in enumerate(urls):
                if not url:
                    continue
                    
                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc
                    
                    # Try to extract title and snippet with a lightweight request
                    title, snippet = await self._extract_title_and_snippet(url)
                    
                    # Extract basic information
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
                    # Skip malformed URLs but continue processing
                    continue
            
            if not search_results:
                # Generate simplified query suggestions
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
            
            # Generate search statistics
            domains = [result['domain'] for result in search_results]
            unique_domains = list(set(domains))
            domain_counts = {domain: domains.count(domain) for domain in unique_domains}
            
            # Classify result types
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
            
            # Record error for rate limiting
            if record_rate_limit:
                status = '429' if '429' in error_msg or 'too many requests' in error_msg else 'error'
                self.rate_limiter.record_request('googlesearch', status)
            
            # Detect 429 errors
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
        
        # Record request for rate limiting
        if record_rate_limit:
            status = '429' if result.get('status_code') == 429 else ('success' if result['success'] else 'error')
            self.rate_limiter.record_request('custom_search', status)
        
        if result['success']:
            # Add metadata to match googlesearch format
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
            # For failed searches, add simplified query suggestions if no results found
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
    
    def get_search_configuration(self) -> Dict[str, Any]:
        """Get current search configuration and API status"""
        return {
            'search_mode': self.search_mode,
            'max_retries': self.max_retries,
            'fallback_delay': self.fallback_delay,
            'googlesearch_python': {
                'available': True,
                'rpm_limit': self.rate_limiter.rpm_limits['googlesearch'],
                'can_make_request': self.rate_limiter.can_make_request('googlesearch'),
                'wait_time': self.rate_limiter.get_wait_time('googlesearch')
            },
            'custom_search_api': {
                'available': self.custom_search_client.is_configured(),
                'configured': self.custom_search_client.is_configured(),
                'api_key_set': bool(self.custom_search_client.api_key),
                'search_engine_id_set': bool(self.custom_search_client.search_engine_id),
                'rpm_limit': self.rate_limiter.rpm_limits['custom_search'],
                'can_make_request': self.rate_limiter.can_make_request('custom_search'),
                'wait_time': self.rate_limiter.get_wait_time('custom_search')
            },
            'rate_limiting': {
                'enabled': True,
                'window_minutes': 1,
                'current_requests': {
                    'googlesearch': len([
                        req for req in self.rate_limiter.requests['googlesearch']
                        if req.timestamp > datetime.now() - timedelta(minutes=1)
                    ]),
                    'custom_search': len([
                        req for req in self.rate_limiter.requests['custom_search']
                        if req.timestamp > datetime.now() - timedelta(minutes=1)
                    ])
                }
            }
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get detailed rate limiting status for both search methods"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        def analyze_requests(method: str):
            requests = self.rate_limiter.requests[method]
            recent_requests = [req for req in requests if req.timestamp > cutoff]
            
            status_counts = {'success': 0, 'error': 0, '429': 0}
            for req in recent_requests:
                status_counts[req.status] = status_counts.get(req.status, 0) + 1
            
            return {
                'total_requests_last_minute': len(recent_requests),
                'limit': self.rate_limiter.rpm_limits[method],
                'remaining': max(0, self.rate_limiter.rpm_limits[method] - len(recent_requests)),
                'can_make_request': self.rate_limiter.can_make_request(method),
                'wait_time_seconds': self.rate_limiter.get_wait_time(method),
                'status_breakdown': status_counts,
                'error_rate': (status_counts['error'] + status_counts['429']) / max(1, len(recent_requests))
            }
        
        return {
            'googlesearch_python': analyze_requests('googlesearch'),
            'custom_search_api': analyze_requests('custom_search'),
            'timestamp': now.isoformat(),
            'recommendation': self._get_usage_recommendation()
        }
    
    def _get_usage_recommendation(self) -> str:
        """Get usage recommendation based on current state"""
        config = self.get_search_configuration()
        
        if self.search_mode == 'googlesearch_only':
            if config['googlesearch_python']['can_make_request']:
                return "Ready to search with googlesearch-python"
            else:
                wait_time = config['googlesearch_python']['wait_time']
                return f"Wait {wait_time:.1f} seconds before next googlesearch-python request"
        
        elif self.search_mode == 'custom_search_only':
            if not config['custom_search_api']['configured']:
                return "Configure Custom Search API credentials (GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID)"
            elif config['custom_search_api']['can_make_request']:
                return "Ready to search with Custom Search API"
            else:
                wait_time = config['custom_search_api']['wait_time']
                return f"Wait {wait_time:.1f} seconds before next Custom Search API request"
        
        else:  # hybrid mode
            if config['googlesearch_python']['can_make_request']:
                return "Ready to search (will try googlesearch-python first)"
            elif config['custom_search_api']['configured'] and config['custom_search_api']['can_make_request']:
                return "googlesearch-python rate limited, but Custom Search API available as fallback"
            elif not config['custom_search_api']['configured']:
                wait_time = config['googlesearch_python']['wait_time']
                return f"googlesearch-python rate limited. Wait {wait_time:.1f}s or configure Custom Search API for fallback"
            else:
                return "Both search methods rate limited. Wait for limits to reset"
    
    def _enhance_query_with_genre(self, query: str, genre: Optional[str]) -> str:
        """Enhance search query based on specified genre"""
        if not genre:
            return query
        
        # Genre-specific search enhancements (Google official operators only)
        genre_enhancements = {
            # File Types (Google official filetype: operator)
            'pdf': 'filetype:pdf',
            'documents': 'filetype:pdf OR filetype:doc OR filetype:docx',
            'presentations': 'filetype:ppt OR filetype:pptx',
            'spreadsheets': 'filetype:xls OR filetype:xlsx',

            # Language and Region (Google official lang: operator)
            'japanese': 'site:jp OR lang:ja',
            'english': 'lang:en'
        }
        
        enhancement = genre_enhancements.get(genre.lower())
        if enhancement:
            # Add genre enhancement to query
            enhanced_query = f"{query} ({enhancement})"
            return enhanced_query
        else:
            # Return original query if genre not recognized
            return query
    
    def get_available_genres(self) -> Dict[str, str]:
        """Get list of available search genres with descriptions"""
        return {
            # File Types (Google official filetype: operator)
            'pdf': 'PDF documents only',
            'documents': 'Document files (PDF, Word, etc.)',
            'presentations': 'Presentation files (PowerPoint, etc.)',
            'spreadsheets': 'Spreadsheet files (Excel, etc.)',

            # Language and Region (Google official lang: operator)
            'japanese': 'Japanese language content and .jp domains',
            'english': 'English language content'
        }
    
    async def _extract_title_and_snippet(self, url: str, timeout: int = 5) -> Tuple[str, str]:
        """Extract title and snippet from URL with lightweight HEAD/GET request"""
        try:
            # Set up headers to appear as a regular browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    # Only process if we get a successful response
                    if response.status != 200:
                        return "Unable to fetch title", "Page not accessible"
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        # For non-HTML content, generate descriptive title/snippet
                        if 'pdf' in content_type:
                            return "PDF Document", "PDF file content"
                        elif 'json' in content_type:
                            return "JSON Data", "JSON API response"
                        else:
                            return "File Content", f"Content type: {content_type}"
                    
                    # Read only first 8KB to get title and meta description
                    content_bytes = await response.content.read(8192)
                    content = content_bytes.decode('utf-8', errors='ignore')
                    
                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract title
                    title = "No title"
                    title_tag = soup.find('title')
                    if title_tag and title_tag.string:
                        title = title_tag.string.strip()
                        # Clean up title (remove extra whitespace, limit length)
                        title = ' '.join(title.split())
                        if len(title) > 100:
                            title = title[:97] + "..."
                    
                    # Extract snippet from meta description or first paragraph
                    snippet = "No description available"
                    
                    # Try meta description first
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    if not meta_desc:
                        meta_desc = soup.find('meta', attrs={'property': 'og:description'})
                    
                    if meta_desc and meta_desc.get('content'):
                        snippet = meta_desc.get('content').strip()
                    else:
                        # Fallback to first paragraph
                        paragraphs = soup.find_all('p')
                        for p in paragraphs[:3]:  # Check first 3 paragraphs
                            text = p.get_text().strip()
                            if len(text) > 20:  # Must have substantial content
                                snippet = text
                                break
                    
                    # Clean up snippet
                    snippet = ' '.join(snippet.split())
                    if len(snippet) > 200:
                        snippet = snippet[:197] + "..."
                    
                    return title, snippet
                    
        except asyncio.TimeoutError:
            return "Timeout loading page", "Page took too long to load"
        except aiohttp.ClientError:
            return "Connection error", "Unable to connect to page"
        except UnicodeDecodeError:
            return "Encoding error", "Unable to decode page content"
        except Exception as e:
            # For any other error, return generic information
            return "Unable to fetch details", f"Error: {str(e)[:50]}"
    
    def _generate_simplified_query_suggestions(self, query: str) -> List[str]:
        """Generate simplified query suggestions by reducing keywords"""
        try:
            if not query or not query.strip():
                return []
            
            # Clean and normalize query
            query = query.strip().lower()
            
            # Remove common search operators and quotes
            query_clean = re.sub(r'["\(\)\[\]]', ' ', query)
            query_clean = re.sub(r'\s+', ' ', query_clean).strip()
            
            # Split into words and filter out common stop words and operators
            stop_words = {
                'and', 'or', 'not', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                'would', 'could', 'should', 'may', 'might', 'can', 'site:', 'filetype:'
            }
            
            words = [word for word in query_clean.split() if word and len(word) > 1 and word not in stop_words]
            
            if len(words) <= 1:
                return []  # Can't simplify further
            
            suggestions = []
            
            # Strategy 1: Take first half of words
            if len(words) >= 4:
                first_half = ' '.join(words[:len(words)//2])
                if first_half:
                    suggestions.append(first_half)
            
            # Strategy 2: Take most important 2-3 words (assuming they're at the beginning)
            if len(words) >= 3:
                core_terms = ' '.join(words[:2])
                if core_terms and core_terms not in suggestions:
                    suggestions.append(core_terms)
            
            # Strategy 3: Individual high-value words (longer than 3 chars)
            single_words = [word for word in words[:3] if len(word) > 3]
            for word in single_words[:2]:
                if word not in suggestions:
                    suggestions.append(word)
            
            # Strategy 4: Try different combinations if we have 3+ words
            if len(words) >= 3:
                # Take every other word
                alt_combo = ' '.join(words[::2])
                if alt_combo and alt_combo not in suggestions and len(alt_combo.split()) > 1:
                    suggestions.append(alt_combo)
            
            # Limit to top 3 suggestions
            return suggestions[:3]
            
        except Exception:
            return []
    
    def _classify_url(self, url: str) -> str:
        """Classify URL by type based on domain and path"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            path = parsed.path
            
            # Social media platforms
            if any(social in domain for social in ['youtube.com', 'youtu.be']):
                return 'video'
            elif any(social in domain for social in ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com', 'instagram.com']):
                return 'social_media'
            elif any(social in domain for social in ['reddit.com', 'quora.com', 'stackoverflow.com']):
                return 'forum'
            
            # News and media
            elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com', 'wsj.com', 'guardian.com']):
                return 'news'
            
            # Academic and education
            elif any(edu in domain for edu in ['.edu', '.ac.', 'scholar.google', 'arxiv.org', 'researchgate']):
                return 'academic'
            
            # Government and official
            elif any(gov in domain for gov in ['.gov', '.mil', '.org']):
                return 'official'
            
            # E-commerce
            elif any(shop in domain for shop in ['amazon.com', 'ebay.com', 'etsy.com', 'shopify']):
                return 'ecommerce'
            
            # Documentation and references
            elif any(doc in domain for doc in ['github.com', 'docs.', 'documentation', 'wiki']):
                return 'documentation'
            
            # File types based on path
            elif any(filetype in path for filetype in ['.pdf', '.doc', '.ppt', '.xls']):
                return 'document'
            
            # Default classification
            else:
                if '.com' in domain:
                    return 'commercial'
                elif '.org' in domain:
                    return 'organization'
                else:
                    return 'general'
                    
        except Exception:
            return 'unknown'
    
    async def batch_search(
        self,
        queries: List[str],
        num_results_per_query: int = 10,
        max_concurrent: int = 3,
        language: str = 'en',
        region: str = 'us',
        search_genre: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform multiple Google searches in batch"""
        
        async def search_single_query(query):
            return await self.search_google(
                query=query,
                num_results=num_results_per_query,
                language=language,
                region=region,
                search_genre=search_genre
            )
        
        # Create semaphore to limit concurrent requests (be respectful to Google)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_with_semaphore(query):
            async with semaphore:
                # Add delay between requests to be respectful
                await asyncio.sleep(1.0)
                return await search_single_query(query)
        
        # Process all queries
        tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'query': queries[i],
                    'error': f'Search failed: {str(result)}'
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def analyze_search_results(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and summarize search results across multiple queries"""
        try:
            if not search_results:
                return {
                    'success': False,
                    'error': 'No search results to analyze'
                }
            
            # Aggregate statistics
            total_queries = len(search_results)
            successful_searches = sum(1 for r in search_results if r.get('success', False))
            failed_searches = total_queries - successful_searches
            
            all_results = []
            all_domains = []
            all_types = []
            
            for search_result in search_results:
                if search_result.get('success') and search_result.get('results'):
                    all_results.extend(search_result['results'])
                    all_domains.extend([r['domain'] for r in search_result['results']])
                    all_types.extend([r['type'] for r in search_result['results']])
            
            # Calculate distributions
            unique_domains = list(set(all_domains))
            domain_distribution = {domain: all_domains.count(domain) for domain in unique_domains}
            type_distribution = {rtype: all_types.count(rtype) for rtype in set(all_types)}
            
            # Find most common domains
            top_domains = sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'success': True,
                'analysis': {
                    'total_queries': total_queries,
                    'successful_searches': successful_searches,
                    'failed_searches': failed_searches,
                    'success_rate': f"{(successful_searches/total_queries*100):.1f}%" if total_queries > 0 else "0%",
                    'total_results': len(all_results),
                    'unique_domains': len(unique_domains),
                    'result_distribution': {
                        'domains': dict(top_domains),
                        'types': type_distribution,
                        'top_domains': [domain for domain, count in top_domains],
                        'most_common_type': max(type_distribution.items(), key=lambda x: x[1])[0] if type_distribution else 'none'
                    }
                },
                'summary': {
                    'queries_processed': total_queries,
                    'total_urls_found': len(all_results),
                    'unique_websites': len(unique_domains),
                    'primary_content_type': max(type_distribution.items(), key=lambda x: x[1])[0] if type_distribution else 'none'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }