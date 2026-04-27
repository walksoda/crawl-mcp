"""
Crawl tool registrations: crawl_url, deep_crawl_site, crawl_url_with_fallback.
"""

from ._shared import (
    Annotated, Dict, Field, Optional, Any,
    apply_token_limit,
    validate_crawl_url_params,
    validate_content_slicing_params,
    validate_output_path,
    _convert_result_to_dict,
    _process_content_fields,
    _should_trigger_fallback,
    _apply_content_slicing,
    finalize_tool_response,
    handle_screenshot_persistence,
    KIND_MARKDOWN_SINGLE,
    KIND_MARKDOWN_BATCH_DICT,
    modules_unavailable_error,
    READONLY_ANNOTATIONS,
)


def register_crawl_tools(mcp, get_modules):
    """Register crawl-related MCP tools."""

    async def _execute_fallback(
        url: str, css_selector: Optional[str], extract_media: bool,
        take_screenshot: bool, generate_markdown: bool, include_cleaned_html: bool,
        wait_for_selector: Optional[str], timeout: int, wait_for_js: bool, auto_summarize: bool,
        fallback_reason: str, original_error: Optional[str],
        content_limit: int = 0, content_offset: int = 0
    ) -> dict:
        """Execute fallback crawl with undetected browser and add diagnostics.

        Returns the fallback result WITHOUT applying content_limit /
        content_offset slicing. The caller is responsible for running
        :func:`_apply_content_slicing` after the persist hook, so that the
        on-disk copy holds the full unsliced payload.

        Note: content_limit / content_offset are still forwarded to the core
        fallback because it manages its own cache strategy keyed on the
        offset; that is independent of the response-copy slicing applied
        later in the caller.
        """
        web_crawling, _, _, _, _ = get_modules()
        fallback_result = await web_crawling.crawl_url_with_fallback(
            url=url, css_selector=css_selector, extract_media=extract_media,
            take_screenshot=take_screenshot, generate_markdown=generate_markdown,
            include_cleaned_html=include_cleaned_html,
            wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
            auto_summarize=auto_summarize, use_undetected_browser=True,
            content_limit=content_limit, content_offset=content_offset
        )

        fallback_dict = _convert_result_to_dict(fallback_result)
        fallback_dict = _process_content_fields(fallback_dict, include_cleaned_html, generate_markdown)
        # NOTE: slicing intentionally deferred to the caller so the persist
        # hook can capture the full unsliced payload to disk.

        # Always add fallback diagnostics
        fallback_dict["fallback_used"] = True
        fallback_dict["undetected_browser_used"] = True
        fallback_dict["fallback_reason"] = fallback_reason

        if original_error:
            fallback_dict["original_error"] = original_error

        return fallback_dict

    @mcp.tool()
    async def crawl_url(
        url: Annotated[str, Field(description="URL to crawl")],
        css_selector: Annotated[Optional[str], Field(description="CSS selector for extraction")] = None,
        extract_media: Annotated[bool, Field(description="Extract images/videos")] = False,
        take_screenshot: Annotated[bool, Field(description="Take screenshot")] = False,
        generate_markdown: Annotated[bool, Field(description="Generate markdown")] = True,
        include_cleaned_html: Annotated[bool, Field(description="Include cleaned HTML")] = False,
        wait_for_selector: Annotated[Optional[str], Field(description="Wait for element to load")] = None,
        timeout: Annotated[int, Field(description="Timeout in seconds")] = 60,
        wait_for_js: Annotated[bool, Field(description="Wait for JavaScript")] = False,
        auto_summarize: Annotated[bool, Field(description="Auto-summarize large content")] = False,
        use_undetected_browser: Annotated[bool, Field(description="Bypass bot detection")] = False,
        content_limit: Annotated[int, Field(description="Max characters to return (0=unlimited)")] = 0,
        content_offset: Annotated[int, Field(description="Start position for content (0-indexed)")] = 0,
        output_path: Annotated[Optional[str], Field(description="Absolute file path (auto .md extension) to persist the full unsliced markdown. When set, the response is slimmed to metadata+file path to save tokens. content_limit/content_offset still affect the response copy but not the on-disk file.")] = None,
        include_content_in_response: Annotated[bool, Field(description="When True (with output_path set), keep markdown/content in the response too. Note: the response copy is still subject to content_limit/content_offset slicing; only the on-disk file holds the full unsliced payload. Defaults to False.")] = False,
        overwrite: Annotated[bool, Field(description="Overwrite an existing output file at output_path. Defaults to False (existing files are rejected before any fetch).")] = False,
    ) -> dict:
        """Extract web page content with JavaScript support. Use wait_for_js=true for SPAs. Use content_offset/content_limit to paginate the response. Use output_path to persist the full unsliced content to disk as markdown and receive a slim metadata-only response."""
        # Input validation
        validation_error = validate_crawl_url_params(url, timeout)
        if validation_error:
            return validation_error

        # Content slicing validation
        slicing_error = validate_content_slicing_params(content_limit, content_offset)
        if slicing_error:
            return slicing_error

        # Output path validation (Guard A) — reject before any external fetch.
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, _, _, _, _ = modules

        try:
            result = await web_crawling.crawl_url(
                url=url, css_selector=css_selector, extract_media=extract_media,
                take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html,
                wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
                auto_summarize=auto_summarize, use_undetected_browser=use_undetected_browser,
                content_limit=content_limit, content_offset=content_offset
            )

            result_dict = _convert_result_to_dict(result)
            result_dict = _process_content_fields(result_dict, include_cleaned_html, generate_markdown)
            # NOTE: slicing intentionally deferred until AFTER the persist
            # hook so the on-disk copy holds the full, unsliced content.

            # Record if undetected browser was used in initial request
            if use_undetected_browser:
                result_dict["undetected_browser_used"] = True

            # Check if fallback is needed (evaluated against unsliced content
            # so a large content_offset can't spuriously trigger it).
            should_fallback, fallback_reason = _should_trigger_fallback(result_dict, generate_markdown)

            if not should_fallback:
                # Guard B: persist FIRST (pre-slice, pre-token-limit).
                result_dict = finalize_tool_response(
                    result_dict,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_SINGLE,
                    source_tool="crawl_url",
                )
                # Persist screenshot to disk (or drop with warning) before
                # apply_token_limit so the base64 blob never inflates the
                # token budget.
                result_dict = handle_screenshot_persistence(
                    result_dict,
                    output_path=output_path,
                    overwrite=overwrite,
                    source_tool="crawl_url",
                )
                # Then slice the response copy for the caller.
                result_dict = _apply_content_slicing(result_dict, content_limit, content_offset)
                return apply_token_limit(result_dict, max_tokens=25000)

            # Try fallback with undetected browser (returns UNSLICED content).
            fallback_dict = await _execute_fallback(
                url=url, css_selector=css_selector, extract_media=extract_media,
                take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html, wait_for_selector=wait_for_selector,
                timeout=timeout, wait_for_js=wait_for_js, auto_summarize=auto_summarize,
                fallback_reason=fallback_reason, original_error=None,
                content_limit=content_limit, content_offset=content_offset
            )

            fallback_dict = finalize_tool_response(
                fallback_dict,
                output_path=output_path,
                include_content_in_response=include_content_in_response,
                overwrite=overwrite,
                tool_kind=KIND_MARKDOWN_SINGLE,
                source_tool="crawl_url",
            )
            fallback_dict = handle_screenshot_persistence(
                fallback_dict,
                output_path=output_path,
                overwrite=overwrite,
                source_tool="crawl_url",
            )
            fallback_dict = _apply_content_slicing(fallback_dict, content_limit, content_offset)
            return apply_token_limit(fallback_dict, max_tokens=25000)

        except Exception as e:
            # Determine error type for better diagnostics
            error_type = type(e).__name__
            error_message = str(e)

            try:
                fallback_dict = await _execute_fallback(
                    url=url, css_selector=css_selector, extract_media=extract_media,
                    take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                    include_cleaned_html=include_cleaned_html, wait_for_selector=wait_for_selector,
                    timeout=timeout, wait_for_js=wait_for_js, auto_summarize=auto_summarize,
                    fallback_reason=f"Exception during initial crawl: {error_type}",
                    original_error=error_message,
                    content_limit=content_limit, content_offset=content_offset
                )

                fallback_dict = finalize_tool_response(
                    fallback_dict,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_SINGLE,
                    source_tool="crawl_url",
                )
                fallback_dict = handle_screenshot_persistence(
                    fallback_dict,
                    output_path=output_path,
                    overwrite=overwrite,
                    source_tool="crawl_url",
                )
                fallback_dict = _apply_content_slicing(fallback_dict, content_limit, content_offset)
                return apply_token_limit(fallback_dict, max_tokens=25000)

            except Exception as fallback_error:
                return {
                    "success": False,
                    "url": url,
                    "error": f"Both crawling methods failed",
                    "error_code": "both_methods_failed",
                    "diagnostics": {
                        "original_error": error_message,
                        "original_error_type": error_type,
                        "fallback_error": str(fallback_error),
                        "fallback_error_type": type(fallback_error).__name__
                    },
                    "retryable": "timeout" in error_message.lower() or "timeout" in str(fallback_error).lower(),
                    "suggested_fix": "Try increasing timeout or using wait_for_js=true for JavaScript-heavy pages"
                }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def deep_crawl_site(
        url: Annotated[str, Field(description="Starting URL")],
        max_depth: Annotated[int, Field(description="Link depth (1-2)")] = 2,
        max_pages: Annotated[int, Field(description="Max pages (max: 10)")] = 5,
        crawl_strategy: Annotated[str, Field(description="'bfs'|'dfs'|'best_first'")] = "bfs",
        include_external: Annotated[bool, Field(description="Follow external links")] = False,
        url_pattern: Annotated[Optional[str], Field(description="URL filter pattern")] = None,
        score_threshold: Annotated[float, Field(description="Min relevance 0-1")] = 0.0,
        extract_media: Annotated[bool, Field(description="Extract media")] = False,
        base_timeout: Annotated[int, Field(description="Timeout per page")] = 60,
        output_path: Annotated[Optional[str], Field(description="Absolute directory path to persist per-URL markdown files + index.json. Existing regular files at this path are rejected; otherwise the directory is created if missing (dot-containing names like /tmp/run.v1 are fine). When set, the response is slimmed to metadata+file paths. Failed items (success=False) are NOT written as .md but still recorded in index.json with file=null.")] = None,
        include_content_in_response: Annotated[bool, Field(description="When True (with output_path set), also include per-page content/markdown in the response items. Defaults to False so the response stays token-efficient.")] = False,
        overwrite: Annotated[bool, Field(description="Overwrite existing per-URL files inside output_path. Defaults to False (existing files cause an output_path_exists error).")] = False,
    ) -> Dict[str, Any]:
        """Crawl multiple pages from a site with configurable depth. Use output_path (directory) to persist per-URL markdown files + index.json; the response is then slimmed to metadata only."""
        # Output path validation (Guard A)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, _, _, _, _ = modules

        try:
            result = _convert_result_to_dict(await web_crawling.deep_crawl_site(
                url=url, max_depth=max_depth, max_pages=max_pages, crawl_strategy=crawl_strategy,
                include_external=include_external, url_pattern=url_pattern, score_threshold=score_threshold,
                extract_media=extract_media, base_timeout=base_timeout
            ))

            # Check if crawling was successful
            if result.get("success", True):
                result = finalize_tool_response(
                    result,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_BATCH_DICT,
                    source_tool="deep_crawl_site",
                )
                # Apply token limit fallback before returning
                return apply_token_limit(result, max_tokens=25000)

            # If deep crawl failed entirely, try with fallback strategy for the main URL
            try:
                fallback_result = _convert_result_to_dict(await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=base_timeout
                ))

                if fallback_result.get("success", False):
                    # Convert single URL result to deep crawl format
                    fallback_response = {
                        "success": True,
                        "results": [{
                            "url": url,
                            "title": fallback_result.get("title", ""),
                            "content": fallback_result.get("content", ""),
                            "markdown": fallback_result.get("markdown", ""),
                            "success": True
                        }],
                        "summary": {
                            "total_crawled": 1,
                            "successful": 1,
                            "failed": 0,
                            "fallback_used": True,
                            "note": "Used fallback crawling for main URL only due to deep crawl failure"
                        },
                        "original_error": result.get("error", "Deep crawl failed")
                    }

                    fallback_response = finalize_tool_response(
                        fallback_response,
                        output_path=output_path,
                        include_content_in_response=include_content_in_response,
                        overwrite=overwrite,
                        tool_kind=KIND_MARKDOWN_BATCH_DICT,
                        source_tool="deep_crawl_site",
                    )
                    # Apply token limit fallback before returning
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                result["fallback_error"] = str(fallback_error)

            return result

        except Exception as e:
            # If deep crawl throws an exception, try single URL fallback
            try:
                fallback_result = _convert_result_to_dict(await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=base_timeout
                ))

                if fallback_result.get("success", False):
                    fallback_response = {
                        "success": True,
                        "results": [{
                            "url": url,
                            "title": fallback_result.get("title", ""),
                            "content": fallback_result.get("content", ""),
                            "markdown": fallback_result.get("markdown", ""),
                            "success": True
                        }],
                        "summary": {
                            "total_crawled": 1,
                            "successful": 1,
                            "failed": 0,
                            "fallback_used": True,
                            "note": "Used fallback crawling for main URL only due to deep crawl exception"
                        },
                        "original_error": str(e)
                    }

                    fallback_response = finalize_tool_response(
                        fallback_response,
                        output_path=output_path,
                        include_content_in_response=include_content_in_response,
                        overwrite=overwrite,
                        tool_kind=KIND_MARKDOWN_BATCH_DICT,
                        source_tool="deep_crawl_site",
                    )
                    # Apply token limit fallback before returning
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                pass

            return {
                "success": False,
                "error": f"Deep crawl error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def crawl_url_with_fallback(
        url: Annotated[str, Field(description="URL to crawl")],
        css_selector: Annotated[Optional[str], Field(description="CSS selector")] = None,
        extract_media: Annotated[bool, Field(description="Extract media")] = False,
        take_screenshot: Annotated[bool, Field(description="Take screenshot")] = False,
        generate_markdown: Annotated[bool, Field(description="Generate markdown")] = True,
        wait_for_selector: Annotated[Optional[str], Field(description="Element to wait for")] = None,
        timeout: Annotated[int, Field(description="Timeout in seconds")] = 60,
        wait_for_js: Annotated[bool, Field(description="Wait for JavaScript")] = False,
        auto_summarize: Annotated[bool, Field(description="Auto-summarize content")] = False,
        content_limit: Annotated[int, Field(description="Max characters to return (0=unlimited)")] = 0,
        content_offset: Annotated[int, Field(description="Start position for content (0-indexed)")] = 0,
        output_path: Annotated[Optional[str], Field(description="Absolute file path (auto .md extension) to persist the full unsliced markdown. When set, the response is slimmed to metadata+file path. content_limit/content_offset still affect the response copy but not the on-disk file.")] = None,
        include_content_in_response: Annotated[bool, Field(description="When True (with output_path set), keep markdown/content in the response too. Note: the response copy is still subject to content_limit/content_offset slicing; only the on-disk file holds the full unsliced payload.")] = False,
        overwrite: Annotated[bool, Field(description="Overwrite an existing output file at output_path. Defaults to False (existing files rejected before any fetch).")] = False,
    ) -> dict:
        """Crawl with fallback strategies for anti-bot sites. Use content_offset/content_limit to paginate the response. Use output_path to persist the full unsliced content to disk as markdown and receive a slim response."""
        # Content slicing validation
        slicing_error = validate_content_slicing_params(content_limit, content_offset)
        if slicing_error:
            return slicing_error

        # Output path validation (Guard A)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, _, _, _, _ = modules

        try:
            result = await web_crawling.crawl_url_with_fallback(
                url=url, css_selector=css_selector, extract_media=extract_media,
                take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
                auto_summarize=auto_summarize,
                content_limit=content_limit, content_offset=content_offset
            )
            # Convert to dict and apply content slicing
            result_dict = _convert_result_to_dict(result)
            # Guard B: persist BEFORE slicing so disk holds full content.
            if output_path:
                result_dict = finalize_tool_response(
                    result_dict,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_SINGLE,
                    source_tool="crawl_url_with_fallback",
                )
            # Persist screenshot to disk (or drop with warning) regardless
            # of output_path, so the base64 blob never reaches the caller
            # via this tool either.
            result_dict = handle_screenshot_persistence(
                result_dict,
                output_path=output_path,
                overwrite=overwrite,
                source_tool="crawl_url_with_fallback",
            )
            result_dict = _apply_content_slicing(result_dict, content_limit, content_offset)
            return result_dict
        except Exception as e:
            return {
                "success": False,
                "error": f"Fallback crawl error: {str(e)}"
            }
