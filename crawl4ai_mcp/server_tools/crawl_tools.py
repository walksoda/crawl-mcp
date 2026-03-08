"""
Crawl tool registrations: crawl_url, deep_crawl_site, crawl_url_with_fallback.
"""

from ._shared import (
    Annotated, Dict, Field, Optional, Any,
    apply_token_limit,
    validate_crawl_url_params,
    validate_content_slicing_params,
    _convert_result_to_dict,
    _process_content_fields,
    _should_trigger_fallback,
    _apply_content_slicing,
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

        Note: content_offset is not passed to the core fallback function because
        the fallback path (crawler_fallback.py) manages its own cache strategy.
        Content slicing is applied post-hoc via _apply_content_slicing instead.
        """
        web_crawling, _, _, _, _ = get_modules()
        fallback_result = await web_crawling.crawl_url_with_fallback(
            url=url, css_selector=css_selector, extract_media=extract_media,
            take_screenshot=take_screenshot, generate_markdown=generate_markdown,
            include_cleaned_html=include_cleaned_html,
            wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
            auto_summarize=auto_summarize, use_undetected_browser=True
        )

        fallback_dict = _convert_result_to_dict(fallback_result)
        fallback_dict = _process_content_fields(fallback_dict, include_cleaned_html, generate_markdown)
        fallback_dict = _apply_content_slicing(fallback_dict, content_limit, content_offset)

        # Always add fallback diagnostics
        fallback_dict["fallback_used"] = True
        fallback_dict["undetected_browser_used"] = True
        fallback_dict["fallback_reason"] = fallback_reason

        if original_error:
            fallback_dict["original_error"] = original_error

        return fallback_dict

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
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
    ) -> dict:
        """Extract web page content with JavaScript support. Use wait_for_js=true for SPAs. Use content_offset/content_limit for pagination."""
        # Input validation
        validation_error = validate_crawl_url_params(url, timeout)
        if validation_error:
            return validation_error

        # Content slicing validation
        slicing_error = validate_content_slicing_params(content_limit, content_offset)
        if slicing_error:
            return slicing_error

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
                content_offset=content_offset
            )

            result_dict = _convert_result_to_dict(result)
            result_dict = _process_content_fields(result_dict, include_cleaned_html, generate_markdown)
            result_dict = _apply_content_slicing(result_dict, content_limit, content_offset)

            # Record if undetected browser was used in initial request
            if use_undetected_browser:
                result_dict["undetected_browser_used"] = True

            # Check if fallback is needed
            should_fallback, fallback_reason = _should_trigger_fallback(result_dict, generate_markdown)

            if not should_fallback:
                return apply_token_limit(result_dict, max_tokens=25000)

            # Try fallback with undetected browser
            fallback_dict = await _execute_fallback(
                url=url, css_selector=css_selector, extract_media=extract_media,
                take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html, wait_for_selector=wait_for_selector,
                timeout=timeout, wait_for_js=wait_for_js, auto_summarize=auto_summarize,
                fallback_reason=fallback_reason, original_error=None,
                content_limit=content_limit, content_offset=content_offset
            )

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
        base_timeout: Annotated[int, Field(description="Timeout per page")] = 60
    ) -> Dict[str, Any]:
        """Crawl multiple pages from a site with configurable depth."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, _, _, _, _ = modules

        try:
            result = await web_crawling.deep_crawl_site(
                url=url, max_depth=max_depth, max_pages=max_pages, crawl_strategy=crawl_strategy,
                include_external=include_external, url_pattern=url_pattern, score_threshold=score_threshold,
                extract_media=extract_media, base_timeout=base_timeout
            )

            # Check if crawling was successful
            if result.get("success", True):
                # Apply token limit fallback before returning
                return apply_token_limit(result, max_tokens=25000)

            # If deep crawl failed entirely, try with fallback strategy for the main URL
            try:
                fallback_result = await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=base_timeout
                )

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

                    # Apply token limit fallback before returning
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                result["fallback_error"] = str(fallback_error)

            return result

        except Exception as e:
            # If deep crawl throws an exception, try single URL fallback
            try:
                fallback_result = await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=base_timeout
                )

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
    ) -> dict:
        """Crawl with fallback strategies for anti-bot sites. Use content_offset/content_limit for pagination."""
        # Content slicing validation
        slicing_error = validate_content_slicing_params(content_limit, content_offset)
        if slicing_error:
            return slicing_error

        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, _, _, _, _ = modules

        try:
            result = await web_crawling.crawl_url_with_fallback(
                url=url, css_selector=css_selector, extract_media=extract_media,
                take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
                auto_summarize=auto_summarize
            )
            # Convert to dict and apply content slicing
            result_dict = _convert_result_to_dict(result)
            result_dict = _apply_content_slicing(result_dict, content_limit, content_offset)
            return result_dict
        except Exception as e:
            return {
                "success": False,
                "error": f"Fallback crawl error: {str(e)}"
            }
