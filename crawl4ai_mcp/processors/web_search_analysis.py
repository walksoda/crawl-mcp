"""
Web Search Analysis Mixin
Configuration, batch and analysis methods for WebSearchProcessor.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any


class WebSearchAnalysisMixin:
    """Mixin providing analysis, configuration, and batch methods for search."""

    def get_search_configuration(self) -> Dict[str, Any]:
        """Current engine selection and which engines have API keys."""
        return {
            'default_engine': self.default_engine,
            'auto_order': list(self.auto_order),
            'configured_engines': self.configured_engines(),
            'engines': {
                name: {
                    'configured': client.is_configured(),
                    'rpm_limit': self.rate_limiter.limit_for(name),
                    'can_make_request': self.rate_limiter.can_make_request(name),
                }
                for name, client in self.clients.items()
            },
        }

    async def batch_search(
        self,
        queries: List[str],
        num_results_per_query: int = 10,
        max_concurrent: int = 3,
        language: str = 'en',
        region: str = 'us',
        search_genre: Optional[str] = None,
        recent_days: Optional[int] = None,
        engine: Optional[str] = None,
        safe_search: bool = True,
        result_type: Any = 'web',
    ) -> List[Dict[str, Any]]:
        """Run several searches with limited concurrency"""

        async def search_single_query(query):
            return await self.search(
                query=query,
                num_results=num_results_per_query,
                language=language,
                region=region,
                safe_search=safe_search,
                search_genre=search_genre,
                recent_days=recent_days,
                engine=engine,
                result_type=result_type,
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
