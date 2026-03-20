"""
Search tool registrations.
"""

from ._shared import (
    Annotated, Dict, Field, Any,
    apply_token_limit,
    validate_content_slicing_params,
    _apply_search_content_slicing,
    _get_search_cache_key,
    _get_cached_search_result,
    _cache_search_result,
    _convert_result_to_dict,
    modules_unavailable_error,
    READONLY_ANNOTATIONS,
    READONLY_CLOSED_ANNOTATIONS,
)


def register_search_tools(mcp, get_modules):
    """Register search-related MCP tools."""

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def search_google(
        request: Annotated[Dict[str, Any], Field(description="Dict with: query (required), num_results, search_genre, language, region, recent_days")]
    ) -> Dict[str, Any]:
        """Search Google with genre filtering. Genres: academic, news, technical, commercial, social."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

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
                    result_copy = _apply_search_content_slicing(result_copy, content_limit, content_offset)
                    result_copy['cache_hit'] = True
                    return apply_token_limit(result_copy, max_tokens=25000)

            # Execute search
            result = await search.search_google(request)

            # Store in cache if slicing is requested and search succeeded
            if result.get('success') and cache_key:
                import copy
                _cache_search_result(cache_key, copy.deepcopy(result))

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
                "error": f"Google search error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def batch_search_google(
        request: Annotated[Dict[str, Any], Field(description="Dict with: queries (max 3), num_results_per_query, search_genre, recent_days")]
    ) -> Dict[str, Any]:
        """Perform multiple Google searches. Max 3 queries per call."""
        # Query limit check (MCP best practice: bounded toolsets)
        queries = request.get('queries', [])
        if len(queries) > 3:
            return {"success": False, "error": "Maximum 3 queries allowed per batch. Split into multiple calls."}

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            result = await search.batch_search_google(request)
            return apply_token_limit(result, max_tokens=25000)
        except Exception as e:
            return {
                "success": False,
                "error": f"Batch search error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def search_and_crawl(
        request: Annotated[Dict[str, Any], Field(description="Dict with: search_query (required), crawl_top_results, search_genre, recent_days")]
    ) -> Dict[str, Any]:
        """Search Google and crawl top results. Combines search with full content extraction."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

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
                generate_markdown=generate_markdown
            )

            # Check for failed crawls and apply fallback
            if result and isinstance(result, dict) and "crawled_pages" in result:
                failed_pages = []
                for i, page in enumerate(result["crawled_pages"]):
                    if isinstance(page, dict):
                        if not page.get("success", True) or not page.get("content", "").strip():
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
                                # Update the page data
                                result["crawled_pages"][idx].update(fallback_result)

                        except Exception as fallback_error:
                            result["crawled_pages"][idx]["fallback_error"] = str(fallback_error)

            # Truncate content if too large
            if result and isinstance(result, dict):
                if "crawled_pages" in result:
                    for page in result["crawled_pages"]:
                        if isinstance(page, dict):
                            if "content" in page and len(page["content"]) > max_content_per_page:
                                page["content"] = page["content"][:max_content_per_page] + "... [truncated for size limit]"
                            if "markdown" in page and len(page["markdown"]) > max_content_per_page:
                                page["markdown"] = page["markdown"][:max_content_per_page] + "... [truncated for size limit]"

            # Apply token limit fallback before returning
            return apply_token_limit(result, max_tokens=25000)

        except Exception as e:
            # If search_and_crawl fails entirely, try with fallback crawling
            try:
                # First try to get search results only
                search_result = await search.search_google({
                    "query": request.get('search_query', ''),
                    "num_results": request.get('crawl_top_results', 2)
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
                                if "content" in fallback_result and len(fallback_result["content"]) > max_content_per_page:
                                    fallback_result["content"] = fallback_result["content"][:max_content_per_page] + "... [truncated for size limit]"
                                if "markdown" in fallback_result and len(fallback_result["markdown"]) > max_content_per_page:
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
