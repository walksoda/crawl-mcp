"""
Google Search Analysis Mixin
Analysis and configuration methods for GoogleSearchProcessor.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup


class GoogleSearchAnalysisMixin:
    """Mixin providing analysis, configuration, and batch methods for search."""

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

        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_with_semaphore(query):
            async with semaphore:
                await asyncio.sleep(1.0)
                return await search_single_query(query)

        tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

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

            unique_domains = list(set(all_domains))
            domain_distribution = {domain: all_domains.count(domain) for domain in unique_domains}
            type_distribution = {rtype: all_types.count(rtype) for rtype in set(all_types)}

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

    def _generate_simplified_query_suggestions(self, query: str) -> List[str]:
        """Generate simplified query suggestions by reducing keywords"""
        try:
            if not query or not query.strip():
                return []

            query = query.strip().lower()

            query_clean = re.sub(r'["\(\)\[\]]', ' ', query)
            query_clean = re.sub(r'\s+', ' ', query_clean).strip()

            stop_words = {
                'and', 'or', 'not', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'site:', 'filetype:'
            }

            words = [word for word in query_clean.split() if word and len(word) > 1 and word not in stop_words]

            if len(words) <= 1:
                return []

            suggestions = []

            if len(words) >= 4:
                first_half = ' '.join(words[:len(words)//2])
                if first_half:
                    suggestions.append(first_half)

            if len(words) >= 3:
                core_terms = ' '.join(words[:2])
                if core_terms and core_terms not in suggestions:
                    suggestions.append(core_terms)

            single_words = [word for word in words[:3] if len(word) > 3]
            for word in single_words[:2]:
                if word not in suggestions:
                    suggestions.append(word)

            if len(words) >= 3:
                alt_combo = ' '.join(words[::2])
                if alt_combo and alt_combo not in suggestions and len(alt_combo.split()) > 1:
                    suggestions.append(alt_combo)

            return suggestions[:3]

        except Exception:
            return []

    async def _extract_title_and_snippet(self, url: str, timeout: int = 5) -> Tuple[str, str]:
        """Extract title and snippet from URL with lightweight HEAD/GET request"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        return "Unable to fetch title", "Page not accessible"

                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        if 'pdf' in content_type:
                            return "PDF Document", "PDF file content"
                        elif 'json' in content_type:
                            return "JSON Data", "JSON API response"
                        else:
                            return "File Content", f"Content type: {content_type}"

                    content_bytes = await response.content.read(8192)
                    content = content_bytes.decode('utf-8', errors='ignore')

                    soup = BeautifulSoup(content, 'html.parser')

                    title = "No title"
                    title_tag = soup.find('title')
                    if title_tag and title_tag.string:
                        title = title_tag.string.strip()
                        title = ' '.join(title.split())
                        if len(title) > 100:
                            title = title[:97] + "..."

                    snippet = "No description available"

                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    if not meta_desc:
                        meta_desc = soup.find('meta', attrs={'property': 'og:description'})

                    if meta_desc and meta_desc.get('content'):
                        snippet = meta_desc.get('content').strip()
                    else:
                        paragraphs = soup.find_all('p')
                        for p in paragraphs[:3]:
                            text = p.get_text().strip()
                            if len(text) > 20:
                                snippet = text
                                break

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
            return "Unable to fetch details", f"Error: {str(e)[:50]}"

    def _classify_url(self, url: str) -> str:
        """Classify URL by type based on domain and path"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            path = parsed.path

            if any(social in domain for social in ['youtube.com', 'youtu.be']):
                return 'video'
            elif any(social in domain for social in ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com', 'instagram.com']):
                return 'social_media'
            elif any(social in domain for social in ['reddit.com', 'quora.com', 'stackoverflow.com']):
                return 'forum'
            elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com', 'wsj.com', 'guardian.com']):
                return 'news'
            elif any(edu in domain for edu in ['.edu', '.ac.', 'scholar.google', 'arxiv.org', 'researchgate']):
                return 'academic'
            elif any(gov in domain for gov in ['.gov', '.mil', '.org']):
                return 'official'
            elif any(shop in domain for shop in ['amazon.com', 'ebay.com', 'etsy.com', 'shopify']):
                return 'ecommerce'
            elif any(doc in domain for doc in ['github.com', 'docs.', 'documentation', 'wiki']):
                return 'documentation'
            elif any(filetype in path for filetype in ['.pdf', '.doc', '.ppt', '.xls']):
                return 'document'
            else:
                if '.com' in domain:
                    return 'commercial'
                elif '.org' in domain:
                    return 'organization'
                else:
                    return 'general'

        except Exception:
            return 'unknown'
