"""Core search functions for Crawl4AI MCP Server."""

from datetime import date, datetime
from typing import Any, Dict, Optional

from ..constants import SEARCH_ENGINES, SEARCH_GENRES
from ..processors.web_search import FILE_TYPE_GENRES, WebSearchProcessor
from ..utils.time_parser import parse_time_period
from .crawler_summarizer import summarize_web_content

# One processor shared by search_web, batch_search_web and search_and_crawl
web_search_processor = WebSearchProcessor()


def recent_days_to_int(value: Any) -> Optional[int]:
    """
    Normalize recent_days to a number of days (None when unset or invalid).

    Accepts an int or the strings parse_time_period understands
    ("7", "2w", "3mo", "1y").
    """
    parsed = parse_time_period(value)
    if parsed is None:
        return None
    if isinstance(parsed, str):
        try:
            parsed = (date.today() - datetime.strptime(parsed, "%Y-%m-%d").date()).days
        except ValueError:
            return None
    return parsed if parsed > 0 else None


async def search_web(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a web search on the selected engine and return structured results.

    Request keys: query, num_results, search_genre, language, region,
    safe_search, recent_days, engine (auto, google_custom, brave, exa),
    result_type (web, discussions).
    """
    try:
        query = request.get('query', '')
        if not query:
            return {
                "success": False,
                "query": "",
                "error": "Query parameter is required"
            }

        result = await web_search_processor.search(
            query=query,
            num_results=max(1, min(100, request.get('num_results', 10))),
            language=request.get('language', 'en'),
            region=request.get('region', 'us'),
            safe_search=request.get('safe_search', True),
            search_genre=request.get('search_genre'),
            recent_days=recent_days_to_int(request.get('recent_days')),
            engine=request.get('engine'),
            result_type=request.get('result_type') or 'web',
        )

        if result['success']:
            response = {
                "success": True,
                "query": result['query'],
                "engine": result['engine'],
                "result_type": result['result_type'],
                "total_results": result['total_results'],
                "results": result['results'],
                "search_metadata": result['search_metadata']
            }
        else:
            response = {
                "success": False,
                "query": query,
                "error": result.get('error'),
                "suggestion": result.get('suggestion'),
            }
            for key in ('error_code', 'engine', 'configured_engines', 'alternative_queries'):
                if key in result:
                    response[key] = result[key]
        for key in ('fallback_info', 'cost_dollars'):
            if key in result:
                response[key] = result[key]
        return response

    except Exception as e:
        return {
            "success": False,
            "query": request.get('query', ''),
            "error": f"Web search error: {str(e)}"
        }


async def batch_search_web(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform multiple web searches in batch with optional AI summarization.

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
        max_concurrent = max(1, min(5, request.get('max_concurrent', 3)))
        num_results = max(1, min(100, request.get('num_results_per_query', 10)))
        search_genre = request.get('search_genre')
        language = request.get('language', 'en')
        region = request.get('region', 'us')
        recent_days = request.get('recent_days')
        engine = request.get('engine')
        result_type = request.get('result_type') or 'web'
        auto_summarize = request.get('auto_summarize', False)
        summary_length = request.get('summary_length', 'medium')
        llm_provider = request.get('llm_provider')
        llm_model = request.get('llm_model')

        batch_results = await web_search_processor.batch_search(
            queries=queries,
            num_results_per_query=num_results,
            max_concurrent=max_concurrent,
            language=language,
            region=region,
            search_genre=search_genre,
            recent_days=recent_days_to_int(recent_days),
            engine=engine,
            result_type=result_type,
        )

        # Process results and count successes/failures
        successful = 0
        failed = 0
        processed_results = []

        for result in batch_results:
            if result.get('success', False):
                successful += 1
                entry = {
                    "success": True,
                    "query": result['query'],
                    "engine": result.get('engine'),
                    "total_results": result.get('total_results', 0),
                    "results": result.get('results', []),
                    "search_metadata": result.get('search_metadata', {})
                }
            else:
                failed += 1
                entry = {
                    "success": False,
                    "query": result.get('query', ''),
                    "engine": result.get('engine'),
                    "error": result.get('error', 'Unknown error'),
                    "error_code": result.get('error_code'),
                    "suggestion": result.get('suggestion')
                }
            for key in ('fallback_info', 'cost_dollars'):
                if key in result:
                    entry[key] = result[key]
            processed_results.append(entry)

        analysis_result = web_search_processor.analyze_search_results(batch_results)

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
                "engine": engine or web_search_processor.default_engine,
                "result_type": result_type,
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
    Get the supported search_genre values.

    Each genre appends a search operator to the query (file type or
    language). Any other search_genre value is rejected by the search tools.
    """
    try:
        config = web_search_processor.get_search_configuration()

        return {
            "success": True,
            "total_genres": len(SEARCH_GENRES),
            "genres": {name: spec["description"] for name, spec in SEARCH_GENRES.items()},
            "query_operators": {name: spec["operator"] for name, spec in SEARCH_GENRES.items()},
            "usage_example": {
                "genre": "pdf",
                "query_example": "technical documentation",
                "enhanced_query": f"technical documentation ({SEARCH_GENRES['pdf']['operator']})",
            },
            "configuration": {
                "default_engine": config["default_engine"],
                "auto_order": config["auto_order"],
                "configured_engines": config["configured_engines"],
                "engines": list(SEARCH_ENGINES),
                "result_types": {
                    "web": "web results (all engines)",
                    "discussions": "forum posts from Brave's discussions cluster (brave only; returned for some queries)",
                },
                "genre_support": {
                    "google_custom": "all genres (operators added to the query)",
                    "brave": "all genres (operators added to the query)",
                    "exa": (
                        "japanese (sets userLocation=JP) and english (no effect); "
                        + ", ".join(sorted(FILE_TYPE_GENRES))
                        + " are not supported, auto mode skips exa for them"
                    ),
                },
                "rate_limiting_enabled": True,
                "supported_languages": ["en", "ja", "es", "fr", "de", "it", "pt", "ru", "zh", "ko"],
                "supported_regions": ["us", "uk", "jp", "de", "fr", "ca", "au", "in"]
            },
            "notes": [
                "search_genre is optional; omit it for an unfiltered search",
                "Other search_genre values return an invalid_search_genre error",
                "Use recent_days for date filtering",
                "Pass engine in the request to pick a search engine for one call",
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get search genres: {str(e)}"
        }


# Former names, kept for callers of the search_google / batch_search_google tools
search_google = search_web
batch_search_google = batch_search_web
