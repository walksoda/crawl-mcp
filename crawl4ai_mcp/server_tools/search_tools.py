"""
Search tool registrations.
"""

from ..constants import SEARCH_ENGINES, SEARCH_GENRES
from ._shared import (
    Annotated, Dict, Field, Any,
    apply_token_limit,
    validate_content_slicing_params,
    validate_output_path,
    validate_search_genre,
    validate_search_engine,
    validate_search_result_type,
    _apply_search_content_slicing,
    _get_search_cache_key,
    _get_cached_search_result,
    _cache_search_result,
    _convert_result_to_dict,
    finalize_tool_response,
    KIND_SEARCH_JSON,
    KIND_MARKDOWN_BATCH_DICT,
    modules_unavailable_error,
    READONLY_ANNOTATIONS,
    READONLY_CLOSED_ANNOTATIONS,
)


# Text for the search_genre key in the request descriptions.
_GENRE_DESC = (
    "search_genre (optional, one of: " + ", ".join(SEARCH_GENRES)
    + "; adds a filetype/language operator to the query, other values return an error)"
)

# Text for the engine key in the request descriptions.
_ENGINE_DESC = (
    "engine (optional, one of: auto, " + ", ".join(SEARCH_ENGINES)
    + "; default from SEARCH_ENGINE, auto tries the engines that have API keys in SEARCH_ENGINE_ORDER"
    " and moves to the next one on errors other than zero results)"
)

# Text for the result_type key in the request descriptions.
_RESULT_TYPE_DESC = (
    "result_type (optional, web (default) or discussions; discussions returns forum posts from Brave's"
    " discussions cluster with the forum name, answer count, post score, question and top comment, runs on brave only,"
    " is returned for some queries only, and holds at most about a dozen posts)"
)

_SEARCH_REQUEST_DESC = (
    "Dict with: query (required), num_results, " + _GENRE_DESC + ", " + _ENGINE_DESC + ", " + _RESULT_TYPE_DESC
    + ", language, region, recent_days, content_limit (int), content_offset (int). Optional persistence keys: output_path (absolute file path, auto .json extension — full unsliced results written to disk BEFORE content_limit/content_offset slicing), include_content_in_response (bool, default False — when True keeps results in the response too, still subject to slicing), overwrite (bool, default False)."
)

_BATCH_REQUEST_DESC = (
    "Dict with: queries (max 3), num_results_per_query, " + _GENRE_DESC + ", " + _ENGINE_DESC + ", " + _RESULT_TYPE_DESC
    + ", recent_days. Optional persistence keys: output_path (absolute file path, auto .json extension — full result set written to disk), include_content_in_response (bool, default False — when True keeps results in the response too), overwrite (bool, default False)."
)


def _validate_search_request(request: Dict[str, Any]):
    """search_genre, engine and result_type checks shared by the search tools (before any network access)."""
    return (
        validate_search_genre(request.get('search_genre'))
        or validate_search_engine(request.get('engine'))
        or validate_search_result_type(request.get('result_type'))
    )


def _extract_persist_opts(request: Dict[str, Any]):
    """Pull (output_path, include_content_in_response, overwrite) from a request dict."""
    return (
        request.get('output_path'),
        bool(request.get('include_content_in_response', False)),
        bool(request.get('overwrite', False)),
    )


def register_search_tools(mcp, get_modules):
    """Register search-related MCP tools."""

    async def _search_web(request: Dict[str, Any], source_tool: str) -> Dict[str, Any]:
        request_error = _validate_search_request(request)
        if request_error:
            return request_error

        # Output path validation (Guard A)
        output_path, include_content_in_response, overwrite = _extract_persist_opts(request)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        def _persist(r: Dict[str, Any]) -> Dict[str, Any]:
            if not output_path:
                return r
            return finalize_tool_response(
                r,
                output_path=output_path,
                include_content_in_response=include_content_in_response,
                overwrite=overwrite,
                tool_kind=KIND_SEARCH_JSON,
                source_tool=source_tool,
            )

        try:
            # Extract parameters from request
            query = request.get('query', '')
            num_results = request.get('num_results', 10)

            # Extract and coerce to int (handles float/string from JSON)
            try:
                content_limit = int(request.get('content_limit', 0))
                content_offset = int(request.get('content_offset', 0))
            except (TypeError, ValueError) as e:
                return {
                    "success": False,
                    "error": f"Invalid content slicing parameter type: {str(e)}",
                    "error_code": "invalid_slicing_param_type"
                }

            # Validate content slicing parameters (always validate if non-zero)
            if content_limit != 0 or content_offset != 0:
                slicing_error = validate_content_slicing_params(content_limit, content_offset)
                if slicing_error:
                    return slicing_error

            # Check cache when slicing is requested
            cache_key = None
            if content_limit != 0 or content_offset != 0:
                cache_key = _get_search_cache_key(request)
                cached = _get_cached_search_result(cache_key)
                if cached:
                    # Apply slicing to cached result
                    import copy
                    result_copy = copy.deepcopy(cached)
                    # Persist BEFORE slicing so disk holds the full result.
                    result_copy = _persist(result_copy)
                    result_copy = _apply_search_content_slicing(result_copy, content_limit, content_offset)
                    result_copy['cache_hit'] = True
                    return apply_token_limit(result_copy, max_tokens=25000)

            # Execute search
            result = await search.search_web(request)

            # Store in cache if slicing is requested and search succeeded
            if result.get('success') and cache_key:
                import copy
                _cache_search_result(cache_key, copy.deepcopy(result))

            # Guard B: persist full (pre-slice/pre-truncate) results BEFORE
            # any slicing so disk holds the complete result set.
            result = _persist(result)

            # Apply content slicing if requested
            if content_limit != 0 or content_offset != 0:
                result = _apply_search_content_slicing(result, content_limit, content_offset)
                result['cache_hit'] = False

            # Apply token limit fallback to prevent MCP errors
            result_with_fallback = apply_token_limit(result, max_tokens=25000)

            return result_with_fallback

        except Exception as e:
            return {
                "success": False,
                "error": f"Web search error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def search_web(
        request: Annotated[Dict[str, Any], Field(description=_SEARCH_REQUEST_DESC)]
    ) -> Dict[str, Any]:
        """Search the web with Google Custom Search, Brave or Exa (engine key, or the server default). Optional search_genre filtering (file types or language; see get_search_genres). Supply output_path in the request to persist the full unsliced result set to disk as JSON and receive a slim response."""
        return await _search_web(request, "search_web")

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def search_google(
        request: Annotated[Dict[str, Any], Field(description=_SEARCH_REQUEST_DESC)]
    ) -> Dict[str, Any]:
        """Deprecated alias of search_web (same request keys and results; the engine is not limited to Google). Use search_web."""
        return await _search_web(request, "search_google")

    async def _batch_search_web(request: Dict[str, Any], source_tool: str) -> Dict[str, Any]:
        # Query limit check (MCP best practice: bounded toolsets)
        queries = request.get('queries', [])
        if len(queries) > 3:
            return {"success": False, "error": "Maximum 3 queries allowed per batch. Split into multiple calls."}

        request_error = _validate_search_request(request)
        if request_error:
            return request_error

        # Output path validation (Guard A)
        output_path, include_content_in_response, overwrite = _extract_persist_opts(request)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            result = await search.batch_search_web(request)
            if output_path:
                result = finalize_tool_response(
                    result,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_SEARCH_JSON,
                    source_tool=source_tool,
                )
            return apply_token_limit(result, max_tokens=25000)
        except Exception as e:
            return {
                "success": False,
                "error": f"Batch search error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def batch_search_web(
        request: Annotated[Dict[str, Any], Field(description=_BATCH_REQUEST_DESC)]
    ) -> Dict[str, Any]:
        """Perform multiple web searches. Max 3 queries per call. Supply output_path in the request to persist the full result set to disk as JSON and receive a slim response."""
        return await _batch_search_web(request, "batch_search_web")

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def batch_search_google(
        request: Annotated[Dict[str, Any], Field(description=_BATCH_REQUEST_DESC)]
    ) -> Dict[str, Any]:
        """Deprecated alias of batch_search_web (same request keys and results). Use batch_search_web."""
        return await _batch_search_web(request, "batch_search_google")

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def search_and_crawl(
        request: Annotated[Dict[str, Any], Field(description="Dict with: search_query (required), crawl_top_results, " + _GENRE_DESC + ", " + _ENGINE_DESC + ", " + _RESULT_TYPE_DESC + ", recent_days, generate_markdown, max_content_per_page. Optional persistence keys: output_path (absolute directory — per-page .md files + index.json, the full page bodies are written BEFORE max_content_per_page truncation; dot-containing dir names are fine), include_content_in_response (bool, default False — when True keeps crawled_pages in the response too, still subject to max_content_per_page truncation), overwrite (bool, default False). Failed pages appear in index.json with file=null.")]
    ) -> Dict[str, Any]:
        """Search the web (engine as in search_web) and crawl top results. Combines search with full content extraction. Supply output_path (directory) in the request to persist per-page markdown (unsliced) + index.json and receive a slim response."""
        request_error = _validate_search_request(request)
        if request_error:
            return request_error

        # Output path validation (Guard A)
        output_path, include_content_in_response, overwrite = _extract_persist_opts(request)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        def _persist(r: Dict[str, Any]) -> Dict[str, Any]:
            if not output_path:
                return r
            return finalize_tool_response(
                r,
                output_path=output_path,
                include_content_in_response=include_content_in_response,
                overwrite=overwrite,
                tool_kind=KIND_MARKDOWN_BATCH_DICT,
                source_tool="search_and_crawl",
            )

        try:
            # Extract parameters from request
            search_query = request.get('search_query')
            if not search_query:
                return {
                    "success": False,
                    "error": "search_query is required in request"
                }

            crawl_top_results = min(request.get('crawl_top_results', 2), 3)
            search_genre = request.get('search_genre')
            recent_days = request.get('recent_days')
            generate_markdown = request.get('generate_markdown', True)
            max_content_per_page = request.get('max_content_per_page', 5000)

            result = await search.search_and_crawl(
                search_query=search_query,
                crawl_top_results=crawl_top_results,
                search_genre=search_genre,
                recent_days=recent_days,
                generate_markdown=generate_markdown,
                engine=request.get('engine'),
                result_type=request.get('result_type') or 'web'
            )

            # Check for failed crawls and apply fallback
            if result and isinstance(result, dict) and "crawled_pages" in result:
                failed_pages = []
                for i, page in enumerate(result["crawled_pages"]):
                    if isinstance(page, dict):
                        if not page.get("success", True) or not (page.get("content") or page.get("markdown") or "").strip():
                            failed_pages.append((i, page.get("url", "")))

                # Apply fallback to failed pages
                for idx, url in failed_pages:
                    if url:
                        try:
                            fallback_result = _convert_result_to_dict(await web_crawling.crawl_url_with_fallback(
                                url=url,
                                generate_markdown=generate_markdown,
                                timeout=30
                            ))

                            if fallback_result.get("success", False):
                                fallback_result["fallback_used"] = True
                                # Update the page data; the first attempt's error no longer applies
                                result["crawled_pages"][idx].pop("error", None)
                                result["crawled_pages"][idx].update(fallback_result)

                        except Exception as fallback_error:
                            result["crawled_pages"][idx]["fallback_error"] = str(fallback_error)

                summary = result.get("crawl_summary")
                if failed_pages and isinstance(summary, dict):
                    ok = sum(1 for p in result["crawled_pages"] if isinstance(p, dict) and p.get("success"))
                    summary["successful_crawls"] = ok
                    summary["failed_crawls"] = len(result["crawled_pages"]) - ok

            # Guard B: persist BEFORE the max_content_per_page truncation
            # so disk holds the full page bodies.
            result = _persist(result)

            # Truncate content if too large
            if result and isinstance(result, dict):
                if "crawled_pages" in result:
                    for page in result["crawled_pages"]:
                        if isinstance(page, dict):
                            if page.get("content") and len(page["content"]) > max_content_per_page:
                                page["content"] = page["content"][:max_content_per_page] + "... [truncated for size limit]"
                            if page.get("markdown") and len(page["markdown"]) > max_content_per_page:
                                page["markdown"] = page["markdown"][:max_content_per_page] + "... [truncated for size limit]"

            # Apply token limit fallback before returning
            return apply_token_limit(result, max_tokens=25000)

        except Exception as e:
            # If search_and_crawl fails entirely, try with fallback crawling
            try:
                # First try to get search results only
                search_result = await search.search_web({
                    "query": request.get('search_query', ''),
                    "num_results": request.get('crawl_top_results', 2),
                    "engine": request.get('engine'),
                    "result_type": request.get('result_type')
                })

                if search_result.get("success", False) and "results" in search_result:
                    # Extract URLs and crawl with fallback
                    urls = [item.get("url", "") for item in search_result["results"] if item.get("url")]
                    crawled_pages = []

                    generate_markdown = request.get('generate_markdown', True)
                    max_content_per_page = request.get('max_content_per_page', 5000)

                    for url in urls[:request.get('crawl_top_results', 2)]:
                        try:
                            fallback_result = _convert_result_to_dict(await web_crawling.crawl_url_with_fallback(
                                url=url,
                                generate_markdown=generate_markdown,
                                timeout=30
                            ))

                            if fallback_result.get("success", False):
                                fallback_result["fallback_used"] = True
                                fallback_result["original_search_crawl_error"] = str(e)

                                # Truncate if needed
                                if fallback_result.get("content") and len(fallback_result["content"]) > max_content_per_page:
                                    fallback_result["content"] = fallback_result["content"][:max_content_per_page] + "... [truncated for size limit]"
                                if fallback_result.get("markdown") and len(fallback_result["markdown"]) > max_content_per_page:
                                    fallback_result["markdown"] = fallback_result["markdown"][:max_content_per_page] + "... [truncated for size limit]"

                            crawled_pages.append(fallback_result)

                        except Exception as individual_error:
                            crawled_pages.append({
                                "success": False,
                                "url": url,
                                "error": f"Individual crawl failed: {str(individual_error)}",
                                "original_search_crawl_error": str(e)
                            })

                    fallback_response = {
                        "success": True,
                        "query": request.get('search_query', ''),
                        "search_results": search_result.get("results", []),
                        "crawled_pages": crawled_pages,
                        "fallback_used": True,
                        "original_error": str(e)
                    }

                    # Guard B: persist even the fallback response so disk
                    # holds full per-page bodies.
                    fallback_response = _persist(fallback_response)

                    # Apply token limit fallback before returning
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                pass

            return {
                "success": False,
                "error": f"Search and crawl error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_CLOSED_ANNOTATIONS)
    async def get_search_genres() -> Dict[str, Any]:
        """Get available search genres for targeted searching."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            result = await search.get_search_genres()
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Get search genres error: {str(e)}"
            }
