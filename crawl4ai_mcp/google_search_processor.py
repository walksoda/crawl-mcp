"""
Google Search Processing Module
Handles Google search queries and result processing
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse, urljoin
from googlesearch import search
import aiohttp
import logging
from bs4 import BeautifulSoup


class GoogleSearchProcessor:
    """Process Google search queries and return structured results"""
    
    def __init__(self):
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
        """Perform Google search and return structured results with optional genre filtering"""
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
            
            # Perform search
            search_results = []
            try:
                # Run search in executor to avoid blocking
                loop = asyncio.get_event_loop()
                
                def do_search():
                    return list(search(
                        enhanced_query,
                        num_results=num_results,
                        lang=language,
                        sleep_interval=1.0,  # Respectful delay between requests
                        region=region,
                        safe='active'  # Always use safe search as requested
                    ))
                
                urls = await loop.run_in_executor(None, do_search)
                
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
                        
                    except Exception as e:
                        # Skip malformed URLs but continue processing
                        continue
                
            except Exception as search_error:
                return {
                    'success': False,
                    'error': f'Google search failed: {str(search_error)}',
                    'query': query,
                    'suggestion': 'Try a different search query or check your internet connection'
                }
            
            if not search_results:
                return {
                    'success': False,
                    'error': 'No search results found',
                    'query': query,
                    'suggestion': 'Try a broader or different search query'
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
                'enhanced_query': enhanced_query,
                'total_results': len(search_results),
                'results': search_results,
                'search_metadata': {
                    'query_info': validation,
                    'search_params': {
                        'num_results_requested': num_results,
                        'language': language,
                        'region': region,
                        'safe_search': True,  # Always enabled
                        'search_genre': search_genre,
                        'enhanced_query': enhanced_query
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
            return {
                'success': False,
                'error': f'Search processing error: {str(e)}',
                'query': query
            }
    
    def _enhance_query_with_genre(self, query: str, genre: Optional[str]) -> str:
        """Enhance search query based on specified genre"""
        if not genre:
            return query
        
        # Genre-specific search enhancements
        genre_enhancements = {
            # Academic and Educational
            'academic': 'site:edu OR site:ac.uk OR site:scholar.google.com OR filetype:pdf',
            'research': 'filetype:pdf OR site:arxiv.org OR site:researchgate.net OR "research paper"',
            'education': 'site:edu OR "tutorial" OR "course" OR "learning"',
            
            # News and Media
            'news': 'site:bbc.com OR site:cnn.com OR site:reuters.com OR site:nytimes.com OR site:guardian.com',
            'latest_news': '"breaking news" OR "latest" OR "today" site:news.google.com',
            
            # Technical and Development
            'technical': 'site:stackoverflow.com OR site:github.com OR site:docs.',
            'programming': 'site:stackoverflow.com OR site:github.com OR "code" OR "programming"',
            'documentation': 'site:docs. OR "documentation" OR "manual" OR "guide"',
            
            # Commerce and Shopping
            'shopping': 'site:amazon.com OR site:ebay.com OR "buy" OR "price" OR "review"',
            'reviews': '"review" OR "rating" OR site:amazon.com OR site:yelp.com',
            
            # Social and Community
            'forum': 'site:reddit.com OR site:quora.com OR site:stackoverflow.com OR "discussion"',
            'social': 'site:twitter.com OR site:facebook.com OR site:linkedin.com',
            
            # Media and Entertainment
            'video': 'site:youtube.com OR site:vimeo.com OR "video" OR "watch"',
            'images': 'filetype:jpg OR filetype:png OR filetype:gif OR site:flickr.com',
            
            # Government and Official
            'government': 'site:gov OR site:mil OR "official" OR "government"',
            'legal': 'site:gov OR "law" OR "legal" OR "regulation"',
            
            # File Types
            'pdf': 'filetype:pdf',
            'documents': 'filetype:pdf OR filetype:doc OR filetype:docx',
            'presentations': 'filetype:ppt OR filetype:pptx',
            'spreadsheets': 'filetype:xls OR filetype:xlsx',
            
            # Time-based
            'recent': '"2024" OR "2023" OR "recent" OR "latest"',
            'historical': 'before:2020 OR "history" OR "historical"',
            
            # Language and Region specific
            'japanese': 'site:jp OR lang:ja',
            'english': 'lang:en',
            
            # Content Quality
            'authoritative': 'site:edu OR site:gov OR site:org',
            'beginner': '"beginner" OR "introduction" OR "basics" OR "tutorial"',
            'advanced': '"advanced" OR "expert" OR "professional" OR "deep dive"'
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
            # Academic and Educational
            'academic': 'Academic and scholarly content from educational institutions',
            'research': 'Research papers, academic publications, and scientific content',
            'education': 'Educational content, tutorials, and learning materials',
            
            # News and Media
            'news': 'News articles from major news organizations',
            'latest_news': 'Breaking news and latest updates',
            
            # Technical and Development
            'technical': 'Technical documentation, Stack Overflow, and developer resources',
            'programming': 'Programming tutorials, code examples, and development guides',
            'documentation': 'Official documentation and technical manuals',
            
            # Commerce and Shopping
            'shopping': 'E-commerce sites, product pages, and shopping platforms',
            'reviews': 'Product reviews, ratings, and customer feedback',
            
            # Social and Community
            'forum': 'Forum discussions, Q&A sites, and community content',
            'social': 'Social media content and platforms',
            
            # Media and Entertainment
            'video': 'Video content from YouTube, Vimeo, and other platforms',
            'images': 'Image content and photo sharing sites',
            
            # Government and Official
            'government': 'Government websites and official information',
            'legal': 'Legal documents, laws, and regulations',
            
            # File Types
            'pdf': 'PDF documents only',
            'documents': 'Document files (PDF, Word, etc.)',
            'presentations': 'Presentation files (PowerPoint, etc.)',
            'spreadsheets': 'Spreadsheet files (Excel, etc.)',
            
            # Time-based
            'recent': 'Recent content from the last 1-2 years',
            'historical': 'Historical content and archives',
            
            # Language and Region
            'japanese': 'Japanese language content and .jp domains',
            'english': 'English language content',
            
            # Content Quality
            'authoritative': 'Authoritative sources (.edu, .gov, .org)',
            'beginner': 'Beginner-friendly and introductory content',
            'advanced': 'Advanced and expert-level content'
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
            
            # Find most common domains and types
            top_domains = sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            top_types = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)
            
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