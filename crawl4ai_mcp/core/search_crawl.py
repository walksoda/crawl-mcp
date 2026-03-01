"""Core search and crawl integration for Crawl4AI MCP Server."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models import CrawlRequest
from ..processors.google_search import GoogleSearchProcessor
from .crawler_core import _internal_crawl_url
from ..utils.time_parser import parse_time_period

# Initialize Google search processor
google_search_processor = GoogleSearchProcessor()


async def search_and_crawl(
    search_query: str,
    num_search_results: int = 5,
    crawl_top_results: int = 3,
    extract_media: bool = False,
    generate_markdown: bool = True,
    search_genre: Optional[str] = None,
    base_timeout: int = 30,
    recent_days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Google search and automatically crawl top results for full content analysis.

    Combines search discovery with full content extraction. Auto-scales timeout based on crawl count.
    Returns both search results and full page content.
    """
    try:
        # Validate parameters
        if not search_query or not search_query.strip():
            return {
                'success': False,
                'error': 'Search query cannot be empty',
                'search_query': search_query
            }

        num_search_results = max(1, min(20, num_search_results))
        crawl_top_results = max(1, min(10, min(crawl_top_results, num_search_results)))

        # Calculate dynamic timeout based on crawl count
        # Base timeout + additional time per URL (15s per additional URL after the first)
        dynamic_timeout = base_timeout + max(0, (crawl_top_results - 1) * 15)

        # Process query for OR conditions and date enhancement
        enhanced_query = search_query

        # Convert space-separated keywords to OR conditions
        keywords = search_query.split()
        if len(keywords) > 1:
            # Use OR between keywords for more flexible search
            enhanced_query = " OR ".join(keywords)

        # Optionally add date filtering for recent results
        parsed_period = parse_time_period(recent_days)
        if parsed_period:
            # parse_time_period returns either int (days) or string (date)
            if isinstance(parsed_period, int):
                # It's a number of days
                date_filter = (datetime.now() - timedelta(days=parsed_period)).strftime("%Y-%m-%d")
            else:
                # It's already a date string
                date_filter = parsed_period
            enhanced_query = f"{enhanced_query} after:{date_filter}"

        # Step 1: Perform Google search
        search_result = await google_search_processor.search_google(
            query=enhanced_query,
            num_results=num_search_results,
            search_genre=search_genre
        )

        if not search_result['success']:
            return {
                'success': False,
                'error': f"Search failed: {search_result.get('error')}",
                'search_query': search_query,
                'suggestion': search_result.get('suggestion')
            }

        search_results = search_result.get('results', [])
        if not search_results:
            return {
                'success': False,
                'error': 'No search results found',
                'search_query': search_query,
                'suggestion': 'Try a different or broader search query'
            }

        # Step 2: Crawl top results using _internal_crawl_url
        crawl_results = []
        successful_crawls = 0
        total_content_length = 0

        # Process URLs sequentially to be respectful to servers
        for i, result in enumerate(search_results[:crawl_top_results]):
            url = result['url']

            try:
                # Create crawl request
                crawl_request = CrawlRequest(
                    url=url,
                    extract_media=extract_media,
                    generate_markdown=generate_markdown,
                    timeout=dynamic_timeout,
                    wait_for_js=False,  # Default to False for faster processing
                    auto_summarize=False  # Don't auto-summarize to preserve full content
                )

                # Perform crawl
                crawl_result = await _internal_crawl_url(crawl_request)

                # Process crawl result
                crawl_data = {
                    'search_rank': i + 1,
                    'search_result': result,
                    'crawl_result': {
                        'success': crawl_result.success,
                        'url': crawl_result.url,
                        'title': crawl_result.title,
                        'content_length': len(crawl_result.markdown or crawl_result.content or '') if crawl_result.success else 0,
                        'has_content': bool(crawl_result.success and (crawl_result.markdown or crawl_result.content)),
                        'has_markdown': bool(crawl_result.success and crawl_result.markdown),
                        'has_media': bool(crawl_result.success and crawl_result.media and extract_media),
                        'error': crawl_result.error if not crawl_result.success else None,
                        'processing_method': getattr(crawl_result, 'processing_method', 'unknown')
                    }
                }

                # Include full content for successful crawls
                if crawl_result.success:
                    successful_crawls += 1
                    content_length = len(crawl_result.markdown or crawl_result.content or '')
                    total_content_length += content_length

                    crawl_data['content'] = {
                        'title': crawl_result.title,
                        'markdown': crawl_result.markdown if generate_markdown else None,
                        'html': crawl_result.content if not generate_markdown else None,
                        'media': crawl_result.media if extract_media else None,
                        'links': getattr(crawl_result, 'links', None),
                        'metadata': getattr(crawl_result, 'metadata', {})
                    }

                crawl_results.append(crawl_data)

                # Add small delay between requests to be respectful
                if i < crawl_top_results - 1:  # Don't delay after the last request
                    await asyncio.sleep(1.0)

            except Exception as e:
                crawl_results.append({
                    'search_rank': i + 1,
                    'search_result': result,
                    'crawl_result': {
                        'success': False,
                        'url': url,
                        'error': f"Crawling failed: {str(e)}",
                        'content_length': 0,
                        'has_content': False
                    }
                })

        # Step 3: Generate comprehensive summary
        failed_crawls = len(crawl_results) - successful_crawls

        return {
            'success': True,
            'search_query': search_query,
            'enhanced_query': enhanced_query,
            'search_metadata': search_result.get('search_metadata', {}),
            'crawl_summary': {
                'total_search_results': len(search_results),
                'urls_crawled': len(crawl_results),
                'successful_crawls': successful_crawls,
                'failed_crawls': failed_crawls,
                'total_content_length': total_content_length,
                'success_rate': f"{(successful_crawls/len(crawl_results)*100):.1f}%" if crawl_results else "0%",
                'average_content_length': total_content_length // successful_crawls if successful_crawls > 0 else 0,
                'timeout_used': dynamic_timeout
            },
            'search_results': search_results,
            'crawled_content': crawl_results,
            'processing_method': 'search_and_crawl_integration',
            'performance_stats': {
                'search_time': search_result.get('search_metadata', {}).get('performance', {}).get('search_time'),
                'crawl_count': len(crawl_results),
                'content_extraction_success_rate': f"{(successful_crawls/len(crawl_results)*100):.1f}%" if crawl_results else "0%"
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': f"Search and crawl error: {str(e)}",
            'search_query': search_query
        }
