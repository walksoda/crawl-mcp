"""Core search functions for Crawl4AI MCP Server."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..processors.google_search import GoogleSearchProcessor
from .crawler_summarizer import summarize_web_content

# Initialize Google search processor
google_search_processor = GoogleSearchProcessor()


async def search_google(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform Google search with genre filtering and extract structured results with metadata.

    Returns web search results with titles, snippets, URLs, and metadata.
    Supports targeted search genres for better results.

    Date Filtering:
    - recent_days: Filters to recent results (e.g., 7 for last week, 30 for last month)
    - Set to None to search all dates without filtering
    """
    try:
        # Extract parameters from request dictionary
        query = request.get('query', '')
        if not query:
            return {
                "success": False,
                "query": "",
                "error": "Query parameter is required"
            }

        num_results = max(1, min(100, request.get('num_results', 10)))
        search_genre = request.get('search_genre')
        language = request.get('language', 'en')
        region = request.get('region', 'us')
        safe_search = request.get('safe_search', True)
        recent_days = request.get('recent_days')

        # Process query for date filtering
        enhanced_query = query

        # Add date filtering if recent_days is specified
        if recent_days and recent_days > 0:
            from datetime import datetime, timedelta
            # Calculate start date for filtering
            start_date = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
            # Add date range to query for more focused results
            enhanced_query = f"{query} after:{start_date}"

        # Perform search using GoogleSearchProcessor
        result = await google_search_processor.search_google(
            query=enhanced_query,
            num_results=num_results,
            language=language,
            region=region,
            safe_search=safe_search,
            search_genre=search_genre
        )

        if result['success']:
            return {
                "success": True,
                "query": result['query'],
                "total_results": result['total_results'],
                "results": result['results'],
                "search_metadata": result['search_metadata']
            }
        else:
            return {
                "success": False,
                "query": query,
                "error": result.get('error'),
                "suggestion": result.get('suggestion')
            }

    except Exception as e:
        return {
            "success": False,
            "query": request.get('query', ''),
            "error": f"Google search error: {str(e)}"
        }


async def batch_search_google(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform multiple Google searches in batch with optional AI summarization.

    Process multiple search queries concurrently with controlled rate limiting.
    Includes batch processing statistics and result analysis.

    Date Filtering:
    - recent_days: Filters to recent results (e.g., 7 for last week, 30 for last month)
    - Set to None to search all dates without filtering

    AI Summarization:
    - Disabled by default (auto_summarize=False)
    - When enabled, summarizes all search result snippets using LLM
    - Supports multiple summary lengths and LLM providers
    """
    try:
        # Extract parameters from request dictionary
        queries = request.get('queries', [])
        if not queries:
            return {
                "success": False,
                "total_queries": 0,
                "successful_searches": 0,
                "failed_searches": 0,
                "results": [],
                "error": "Queries parameter is required and must be non-empty"
            }

        # Validate and limit parameters
        max_concurrent = max(1, min(5, request.get('max_concurrent', 3)))  # Be respectful to Google
        num_results = max(1, min(100, request.get('num_results_per_query', 10)))
        search_genre = request.get('search_genre')
        language = request.get('language', 'en')
        region = request.get('region', 'us')
        recent_days = request.get('recent_days')
        auto_summarize = request.get('auto_summarize', False)
        summary_length = request.get('summary_length', 'medium')
        llm_provider = request.get('llm_provider')
        llm_model = request.get('llm_model')

        # Process queries for date enhancement
        enhanced_queries = []
        for query in queries:
            enhanced_query = query

            # Add date filtering if recent_days is specified
            if recent_days and recent_days > 0:
                from datetime import datetime, timedelta
                # Calculate start date for filtering
                start_date = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
                # Add date range to query for more focused results
                enhanced_query = f"{query} after:{start_date}"

            enhanced_queries.append(enhanced_query)

        # Perform batch search using GoogleSearchProcessor
        batch_results = await google_search_processor.batch_search(
            queries=enhanced_queries,
            num_results_per_query=num_results,
            max_concurrent=max_concurrent,
            language=language,
            region=region,
            search_genre=search_genre
        )

        # Process results and count successes/failures
        successful = 0
        failed = 0
        processed_results = []

        for result in batch_results:
            if result.get('success', False):
                successful += 1
                processed_results.append({
                    "success": True,
                    "query": result['query'],
                    "total_results": result.get('total_results', 0),
                    "results": result.get('results', []),
                    "search_metadata": result.get('search_metadata', {})
                })
            else:
                failed += 1
                processed_results.append({
                    "success": False,
                    "query": result.get('query', ''),
                    "error": result.get('error', 'Unknown error'),
                    "suggestion": result.get('suggestion')
                })

        # Generate analysis using GoogleSearchProcessor
        analysis_result = google_search_processor.analyze_search_results(batch_results)

        # Apply AI summarization if requested
        summarization_result = None
        if auto_summarize and successful > 0:
            try:
                # Collect all result snippets and titles
                all_content = []
                for result in processed_results:
                    if result.get('success') and result.get('results'):
                        for search_result in result['results']:
                            title = search_result.get('title', '')
                            snippet = search_result.get('snippet', '')
                            url = search_result.get('url', '')
                            if title and snippet:
                                all_content.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

                if all_content:
                    combined_content = "\n".join(all_content)
                    summary_result = await summarize_web_content(
                        content=combined_content,
                        title="Batch Search Results Summary",
                        url="multiple_search_queries",
                        summary_length=summary_length,
                        llm_provider=llm_provider,
                        llm_model=llm_model
                    )

                    if summary_result.get("success"):
                        summarization_result = {
                            "summary": summary_result["summary"],
                            "original_results_count": len(all_content),
                            "compression_ratio": summary_result.get("compressed_ratio", 0),
                            "llm_provider": summary_result.get("llm_provider", "unknown"),
                            "llm_model": summary_result.get("llm_model", "unknown"),
                            "summary_length": summary_length
                        }
                    else:
                        summarization_result = {
                            "error": summary_result.get("error", "Summarization failed"),
                            "fallback_applied": True
                        }
            except Exception as e:
                summarization_result = {
                    "error": f"Summarization error: {str(e)}",
                    "fallback_applied": True
                }

        response = {
            "success": True,
            "total_queries": len(queries),
            "successful_searches": successful,
            "failed_searches": failed,
            "results": processed_results,
            "analysis": analysis_result.get('analysis') if analysis_result.get('success') else None,
            "batch_metadata": {
                "max_concurrent_used": max_concurrent,
                "num_results_per_query": num_results,
                "search_genre": search_genre,
                "language": language,
                "region": region,
                "recent_days_filter": recent_days,
                "auto_summarize_enabled": auto_summarize
            }
        }

        # Add summarization result if available
        if summarization_result:
            response["ai_summary"] = summarization_result

        return response

    except Exception as e:
        return {
            "success": False,
            "total_queries": len(request.get('queries', [])),
            "successful_searches": 0,
            "failed_searches": len(request.get('queries', [])),
            "results": [],
            "error": f"Batch search error: {str(e)}"
        }


async def get_search_genres() -> Dict[str, Any]:
    """
    Get list of available search genres for content filtering.

    Provides comprehensive information about available search genres and their capabilities.

    Parameters: None

    Returns dictionary with available genres and their descriptions.
    """
    try:
        # Get genres from GoogleSearchProcessor
        genres = google_search_processor.get_available_genres()

        # Get configuration info for additional context
        config = google_search_processor.get_search_configuration()

        return {
            "success": True,
            "total_genres": len(genres),
            "genres": genres,
            "categories": {
                "Academic & Research": [
                    {"genre": "academic", "description": genres.get("academic", "")},
                    {"genre": "research", "description": genres.get("research", "")},
                    {"genre": "education", "description": genres.get("education", "")}
                ],
                "News & Media": [
                    {"genre": "news", "description": genres.get("news", "")},
                    {"genre": "latest_news", "description": genres.get("latest_news", "")}
                ],
                "Technical & Development": [
                    {"genre": "technical", "description": genres.get("technical", "")},
                    {"genre": "programming", "description": genres.get("programming", "")},
                    {"genre": "documentation", "description": genres.get("documentation", "")}
                ],
                "Commerce & Shopping": [
                    {"genre": "shopping", "description": genres.get("shopping", "")},
                    {"genre": "reviews", "description": genres.get("reviews", "")}
                ],
                "Social & Community": [
                    {"genre": "forum", "description": genres.get("forum", "")},
                    {"genre": "social", "description": genres.get("social", "")}
                ],
                "Media & Entertainment": [
                    {"genre": "video", "description": genres.get("video", "")},
                    {"genre": "images", "description": genres.get("images", "")}
                ],
                "Government & Official": [
                    {"genre": "government", "description": genres.get("government", "")},
                    {"genre": "legal", "description": genres.get("legal", "")}
                ],
                "File Types": [
                    {"genre": "pdf", "description": genres.get("pdf", "")},
                    {"genre": "documents", "description": genres.get("documents", "")},
                    {"genre": "presentations", "description": genres.get("presentations", "")},
                    {"genre": "spreadsheets", "description": genres.get("spreadsheets", "")}
                ],
                "Time-based": [
                    {"genre": "recent", "description": genres.get("recent", "")},
                    {"genre": "historical", "description": genres.get("historical", "")}
                ],
                "Language & Region": [
                    {"genre": "japanese", "description": genres.get("japanese", "")},
                    {"genre": "english", "description": genres.get("english", "")}
                ],
                "Content Quality": [
                    {"genre": "authoritative", "description": genres.get("authoritative", "")},
                    {"genre": "beginner", "description": genres.get("beginner", "")},
                    {"genre": "advanced", "description": genres.get("advanced", "")}
                ]
            },
            "usage_examples": [
                {
                    "genre": "academic",
                    "query_example": "machine learning algorithms",
                    "enhanced_query": "machine learning algorithms (site:edu OR site:ac.uk OR site:scholar.google.com OR filetype:pdf)",
                    "description": "Find academic papers about machine learning"
                },
                {
                    "genre": "programming",
                    "query_example": "Python data structures",
                    "enhanced_query": "Python data structures (site:stackoverflow.com OR site:github.com OR \"code\" OR \"programming\")",
                    "description": "Search for Python programming tutorials and code examples"
                },
                {
                    "genre": "news",
                    "query_example": "AI breakthrough",
                    "enhanced_query": "AI breakthrough (site:bbc.com OR site:cnn.com OR site:reuters.com OR site:nytimes.com OR site:guardian.com)",
                    "description": "Get latest news articles from major news sources"
                },
                {
                    "genre": "pdf",
                    "query_example": "technical documentation",
                    "enhanced_query": "technical documentation (filetype:pdf)",
                    "description": "Find PDF documents and technical papers"
                },
                {
                    "genre": "beginner",
                    "query_example": "JavaScript basics",
                    "enhanced_query": "JavaScript basics (\"beginner\" OR \"introduction\" OR \"basics\" OR \"tutorial\")",
                    "description": "Search for beginner-friendly tutorials and guides"
                }
            ],
            "configuration": {
                "search_mode": config.get("search_mode", "hybrid"),
                "rate_limiting_enabled": True,
                "fallback_support": config.get("custom_search_api", {}).get("available", False),
                "supported_languages": ["en", "ja", "es", "fr", "de", "it", "pt", "ru", "zh", "ko"],
                "supported_regions": ["us", "uk", "jp", "de", "fr", "ca", "au", "in"]
            },
            "tips": [
                "Use specific genres to narrow down search results to relevant content types",
                "Combine multiple search terms for better results",
                "Use 'recent' genre for latest information and 'historical' for archived content",
                "Academic and research genres are excellent for in-depth, authoritative information",
                "Technical genres help find documentation and programming resources",
                "News genres provide current events and breaking news from reputable sources"
            ]
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get search genres: {str(e)}"
        }
