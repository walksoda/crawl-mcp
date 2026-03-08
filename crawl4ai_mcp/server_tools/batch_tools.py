"""
Batch crawl and utility tool registrations.
"""

import asyncio
import re
import fnmatch

from ._shared import (
    Annotated, Dict, Field, List, Optional, Any,
    apply_token_limit,
    _convert_result_to_dict,
    modules_unavailable_error,
    READONLY_ANNOTATIONS,
)


def register_batch_tools(mcp, get_modules):
    """Register batch crawl and utility MCP tools."""

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def batch_crawl(
        urls: Annotated[List[str], Field(description="URLs to crawl (max 3)")],
        base_timeout: Annotated[int, Field(description="Timeout per URL (default: 30)")] = 30,
        generate_markdown: Annotated[bool, Field(description="Generate markdown (default: True)")] = True,
        extract_media: Annotated[bool, Field(description="Extract media (default: False)")] = False,
        wait_for_js: Annotated[bool, Field(description="Wait for JS (default: False)")] = False,
        max_concurrent: Annotated[int, Field(description="Max concurrent (default: 3)")] = 3
    ) -> List[Dict[str, Any]]:
        """Crawl multiple URLs with fallback. Max 3 URLs per call."""
        # URL limit check (MCP best practice: bounded toolsets)
        if len(urls) > 3:
            return [{"success": False, "error": "Maximum 3 URLs allowed per batch. Split into multiple calls."}]

        modules = get_modules()
        if not modules:
            return [{"success": False, "error": "Tool modules not available"}]
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            # Build config from individual parameters
            config = {
                "generate_markdown": generate_markdown,
                "extract_media": extract_media,
                "wait_for_js": wait_for_js,
                "max_concurrent": max_concurrent,
                "use_undetected_browser": False  # Default to False for batch
            }

            # Add timeout handling - optimized for faster response
            # Reduced timeout: base + 10s per URL (instead of base * URLs)
            total_timeout = base_timeout + (len(urls) * 10) + 30  # More reasonable timeout

            result = await asyncio.wait_for(
                utilities.batch_crawl(urls, config, base_timeout),
                timeout=total_timeout
            )

            # Check for failed crawls and apply fallback with undetected browser
            if isinstance(result, list):
                failed_urls = []
                for i, crawl_result in enumerate(result):
                    # Handle both dict and CrawlResponse objects
                    if hasattr(crawl_result, 'success'):
                        # CrawlResponse object
                        success = crawl_result.success
                        markdown = getattr(crawl_result, 'markdown', '') or ''
                    else:
                        # Dictionary object
                        success = crawl_result.get("success", True)
                        markdown = crawl_result.get("markdown", "") or ""

                    if not success or not markdown.strip():
                        failed_urls.append((i, urls[i]))

                # Apply fallback to failed URLs with undetected browser
                if failed_urls:
                    for idx, url in failed_urls:
                        try:
                            fallback_result = await web_crawling.crawl_url_with_fallback(
                                url=url,
                                generate_markdown=generate_markdown,
                                extract_media=extract_media,
                                wait_for_js=wait_for_js,
                                timeout=base_timeout,
                                use_undetected_browser=True  # Force undetected for fallback
                            )

                            # Handle both dict and CrawlResponse objects for fallback
                            if hasattr(fallback_result, 'success'):
                                # CrawlResponse object - convert to dict
                                if fallback_result.success:
                                    fallback_dict = fallback_result.dict()
                                    fallback_dict["fallback_used"] = True
                                    fallback_dict["undetected_browser_used"] = True
                                    result[idx] = fallback_dict
                            else:
                                # Dictionary object
                                if fallback_result.get("success", False):
                                    fallback_result["fallback_used"] = True
                                    fallback_result["undetected_browser_used"] = True
                                    result[idx] = fallback_result

                        except Exception as fallback_error:
                            # Handle CrawlResponse objects in error case
                            if hasattr(result[idx], 'dict'):
                                error_dict = result[idx].dict()
                                error_dict["fallback_error"] = str(fallback_error)
                                result[idx] = error_dict
                            else:
                                result[idx]["fallback_error"] = str(fallback_error)

            # Convert CrawlResponse objects to dictionaries for JSON serialization
            dict_results = []
            for crawl_result in result:
                if hasattr(crawl_result, 'dict'):
                    # CrawlResponse object
                    dict_results.append(crawl_result.dict())
                else:
                    # Already a dictionary
                    dict_results.append(crawl_result)

            # Apply token limit fallback to the entire batch result
            batch_response = {"batch_results": dict_results, "total_urls": len(urls)}
            final_result = apply_token_limit(batch_response, max_tokens=25000)

            # Return just the batch_results list if no token limits were applied
            if not final_result.get("token_limit_applied"):
                return dict_results
            else:
                # If token limits were applied, return the modified structure with metadata
                return final_result.get("batch_results", dict_results)

        except asyncio.TimeoutError:
            return [{
                "success": False,
                "error": f"Batch crawl timed out after {total_timeout} seconds"
            }]
        except Exception as e:
            # If batch crawl fails entirely, try individual fallbacks with undetected browser
            try:
                fallback_results = []
                for url in urls:
                    try:
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            generate_markdown=generate_markdown,
                            extract_media=extract_media,
                            wait_for_js=wait_for_js,
                            timeout=base_timeout,
                            use_undetected_browser=True  # Force undetected for emergency fallback
                        )
                        # Handle CrawlResponse objects in emergency fallback
                        if hasattr(fallback_result, 'success'):
                            # CrawlResponse object - convert to dict
                            fallback_dict = fallback_result.dict()
                            if fallback_result.success:
                                fallback_dict["fallback_used"] = True
                                fallback_dict["undetected_browser_used"] = True
                                fallback_dict["original_batch_error"] = str(e)
                            fallback_results.append(fallback_dict)
                        else:
                            # Dictionary object
                            if fallback_result.get("success", False):
                                fallback_result["fallback_used"] = True
                                fallback_result["undetected_browser_used"] = True
                                fallback_result["original_batch_error"] = str(e)
                            fallback_results.append(fallback_result)

                    except Exception as individual_error:
                        fallback_results.append({
                            "success": False,
                            "url": url,
                            "error": f"Individual fallback failed: {str(individual_error)}",
                            "original_batch_error": str(e)
                        })

                # Apply token limit fallback to emergency results
                batch_response = {"batch_results": fallback_results, "total_urls": len(urls)}
                final_result = apply_token_limit(batch_response, max_tokens=25000)
                return final_result.get("batch_results", fallback_results)

            except Exception:
                return [{
                    "success": False,
                    "error": f"Batch crawl error: {str(e)}"
                }]

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def multi_url_crawl(
        url_configurations: Annotated[Dict[str, Dict], Field(description="URL-config mapping (max 5 URLs). Example: {'https://site1.com': {'wait_for_js': true}}")],
        pattern_matching: Annotated[str, Field(description="Pattern: 'wildcard' or 'regex' (default: wildcard)")] = "wildcard",
        default_config: Annotated[Optional[Dict], Field(description="Default config")] = None,
        base_timeout: Annotated[int, Field(description="Timeout per URL (default: 30)")] = 30,
        max_concurrent: Annotated[int, Field(description="Max concurrent (default: 3)")] = 3
    ) -> List[Dict[str, Any]]:
        """Multi-URL crawl with pattern-based config. Max 5 URL patterns per call."""
        # URL limit check (MCP best practice: bounded toolsets)
        if len(url_configurations) > 5:
            return [{"success": False, "error": "Maximum 5 URL configurations allowed per batch. Split into multiple calls."}]

        modules = get_modules()
        if not modules:
            return [{
                "success": False,
                "error": "Tool modules not available"
            }]
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            # Extract all URLs from configurations
            all_urls = []
            for pattern in url_configurations.keys():
                # For now, treat patterns as literal URLs if they don't contain wildcards
                if pattern_matching == "wildcard" and ('*' not in pattern and '?' not in pattern):
                    all_urls.append(pattern)

            if not all_urls:
                return [{
                    "success": False,
                    "error": "No valid URLs found in configurations. Use direct URLs or wildcard patterns."
                }]

            results = []

            # Process each URL with its matched configuration
            for url in all_urls:
                matched_config = default_config or {}
                pattern_used = "default"

                # Find matching pattern and configuration
                for pattern, config in url_configurations.items():
                    pattern_matches = False

                    if pattern_matching == "wildcard":
                        pattern_matches = fnmatch.fnmatch(url, pattern) or url == pattern
                    elif pattern_matching == "regex":
                        try:
                            pattern_matches = bool(re.match(pattern, url))
                        except re.error:
                            continue

                    if pattern_matches:
                        matched_config = {**matched_config, **config}
                        pattern_used = pattern
                        break

                # Apply configuration with fallback to defaults
                crawl_config = {
                    "generate_markdown": matched_config.get("generate_markdown", True),
                    "extract_media": matched_config.get("extract_media", False),
                    "wait_for_js": matched_config.get("wait_for_js", False),
                    "timeout": matched_config.get("timeout", base_timeout),
                    "use_undetected_browser": matched_config.get("use_undetected_browser", False),
                    "css_selector": matched_config.get("css_selector")
                }

                try:
                    # Crawl with pattern-specific configuration
                    result = await web_crawling.crawl_url(
                        url=url,
                        **{k: v for k, v in crawl_config.items() if v is not None}
                    )

                    # Convert CrawlResponse to dict
                    if hasattr(result, 'model_dump'):
                        result_dict = result.model_dump()
                    elif hasattr(result, 'dict'):
                        result_dict = result.dict()
                    else:
                        result_dict = result

                    # Add configuration metadata to result
                    if result_dict.get("success", True):
                        result_dict["pattern_matched"] = pattern_used
                        result_dict["configuration_applied"] = crawl_config
                        result_dict["multi_url_config_used"] = True
                        result = result_dict
                    else:
                        # Try fallback with undetected browser if initial fails
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            **{k: v for k, v in crawl_config.items() if v is not None and k != 'use_undetected_browser'},
                            use_undetected_browser=True
                        )

                        # Convert fallback result to dict too
                        if hasattr(fallback_result, 'model_dump'):
                            result = fallback_result.model_dump()
                        elif hasattr(fallback_result, 'dict'):
                            result = fallback_result.dict()
                        else:
                            result = fallback_result

                        if result.get("success", False):
                            result["pattern_matched"] = pattern_used
                            result["configuration_applied"] = crawl_config
                            result["multi_url_config_used"] = True
                            result["fallback_used"] = True

                    results.append(result)

                except Exception as e:
                    # Error handling with pattern information
                    error_result = {
                        "success": False,
                        "url": url,
                        "error": f"Multi-URL crawl error: {str(e)}",
                        "pattern_matched": pattern_used,
                        "configuration_applied": crawl_config,
                        "multi_url_config_used": True
                    }
                    results.append(error_result)

            # Apply token limit fallback to the entire multi-URL result
            batch_response = {"multi_url_results": results, "total_urls": len(all_urls)}
            final_result = apply_token_limit(batch_response, max_tokens=25000)

            # Return just the results list if no token limits were applied
            if not final_result.get("token_limit_applied"):
                return results
            else:
                # If token limits were applied, return the modified structure
                return final_result.get("multi_url_results", results)

        except Exception as e:
            return [{
                "success": False,
                "error": f"Multi-URL configuration error: {str(e)}"
            }]

    async def get_llm_config_info() -> Dict[str, Any]:
        """Get current LLM configuration and available providers."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            result = await utilities.get_llm_config_info()
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM config info error: {str(e)}"
            }

    def get_tool_selection_guide() -> dict:
        """Get tool-to-use-case mapping guide for all available tools."""
        return {
            "web_crawling": ["crawl_url", "deep_crawl_site", "crawl_url_with_fallback", "intelligent_extract", "extract_entities", "extract_structured_data"],
            "youtube": ["extract_youtube_transcript", "batch_extract_youtube_transcripts", "get_youtube_video_info", "get_youtube_api_setup_guide"],
            "search": ["search_google", "batch_search_google", "search_and_crawl", "get_search_genres"],
            "batch": ["batch_crawl", "multi_url_crawl"],
            "files": ["process_file", "get_supported_file_formats", "enhanced_process_large_content"],
            "config": ["get_llm_config_info", "get_tool_selection_guide"],
            "diagnostics": ["get_system_diagnostics"],
            "new_v074_features": {
                "undetected_browser": "Enhanced crawl_url with use_undetected_browser parameter",
                "llm_table_extraction": "Revolutionary table extraction in extract_structured_data with use_llm_table_extraction",
                "multi_url_config": "Pattern-based configuration matching in multi_url_crawl tool",
                "intelligent_chunking": "Massive table support with adaptive chunking strategies"
            },
            "best_practices": {
                "bot_detection": "Use crawl_url with undetected browser mode for difficult sites",
                "table_data": "Enable LLM table extraction for complex table structures",
                "mixed_domains": "Use multi_url_crawl for site-specific optimization",
                "fallback_reliability": "All tools now include automatic fallback mechanisms"
            }
        }
