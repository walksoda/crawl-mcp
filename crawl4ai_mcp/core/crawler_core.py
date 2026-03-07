"""
Core crawl implementation.

Contains _internal_crawl_url - the main single-page and deep-crawl entry point,
extracted from tools/web_crawling.py.
"""

import asyncio
import os
from typing import Any, Dict, Optional

from ..models import CrawlRequest, CrawlResponse
from ..suppress_output import suppress_stdout_stderr
from ..infra.content_processors import (
    convert_media_to_list as _convert_media_to_list,
    has_meaningful_content as _has_meaningful_content,
    is_block_page as _is_block_page,
)
from ..infra.fallback_strategies import (
    normalize_cookies_to_playwright_format as _normalize_cookies_to_playwright_format,
)

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import (
    BM25ContentFilter,
    PruningContentFilter,
    LLMContentFilter,
    CacheMode,
)
from crawl4ai.chunking_strategy import (
    TopicSegmentationChunking,
    OverlappingWindowChunking,
    RegexChunking,
    SlidingWindowChunking
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

from .crawler_io import _process_response_content, _handle_youtube_url, _handle_file_url
from .crawler_summarizer import (
    summarize_web_content,
    _check_and_summarize_if_needed,
    MAX_RESPONSE_CHARS,
)


async def _internal_crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    """
    try:
        # Check if URL is a YouTube video
        youtube_result = await _handle_youtube_url(request)
        if youtube_result is not None:
            return youtube_result

        # Check if URL points to a file that should be processed with MarkItDown
        file_result = await _handle_file_url(request)
        if file_result is not None:
            return file_result

        # Setup deep crawling strategy if max_depth is specified
        deep_crawl_strategy = None
        if request.max_depth is not None and request.max_depth > 0:
            filters = []
            if request.url_pattern:
                filters.append(URLPatternFilter(patterns=[request.url_pattern]))
            if not request.include_external:
                from urllib.parse import urlparse
                domain = urlparse(request.url).netloc
                filters.append(DomainFilter(allowed_domains=[domain]))

            filter_chain = FilterChain(filters) if filters else None

            if request.crawl_strategy == "dfs":
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=request.max_depth, max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain, score_threshold=request.score_threshold
                )
            elif request.crawl_strategy == "best_first":
                deep_crawl_strategy = BestFirstCrawlingStrategy(
                    max_depth=request.max_depth, max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain, score_threshold=request.score_threshold
                )
            else:
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=request.max_depth, max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain, score_threshold=request.score_threshold
                )

        # Setup advanced content filtering
        content_filter_strategy = None
        if request.content_filter == "bm25" and request.filter_query:
            content_filter_strategy = BM25ContentFilter(user_query=request.filter_query)
        elif request.content_filter == "pruning":
            content_filter_strategy = PruningContentFilter(threshold=0.5)
        elif request.content_filter == "llm" and request.filter_query:
            content_filter_strategy = LLMContentFilter(
                instructions=f"Filter content related to: {request.filter_query}"
            )

        # Setup cache mode with freshness policy
        from ..infra.content_cache_policy import get_content_cache_policy
        policy = get_content_cache_policy()
        resolved_mode = policy.resolve_cache_mode(
            url=request.url,
            content_offset=request.content_offset,
            requested_mode=request.cache_mode,
            enable_caching=request.enable_caching,
        )
        cache_mode_map = {
            "enabled": CacheMode.ENABLED,
            "bypass": CacheMode.BYPASS,
            "disabled": CacheMode.DISABLED,
        }
        cache_mode = cache_mode_map.get(resolved_mode, CacheMode.ENABLED)

        # Configure chunking if requested
        chunking_strategy = None
        if request.chunk_content:
            step_size = max(1, int(request.chunk_size * (1 - request.overlap_rate)))
            overlap_chars = max(0, int(request.chunk_size * request.overlap_rate))

            if request.chunk_strategy == "topic":
                chunking_strategy = TopicSegmentationChunking(num_keywords=5, chunk_size=request.chunk_size)
            elif request.chunk_strategy == "regex":
                chunking_strategy = RegexChunking(patterns=[r"\n\n", r"\n#{1,6}\s", r"\n---\n"])
            elif request.chunk_strategy == "sentence":
                chunking_strategy = OverlappingWindowChunking(window_size=request.chunk_size, overlap=overlap_chars)
            else:
                chunking_strategy = SlidingWindowChunking(window_size=request.chunk_size, step=step_size)

        # Create config parameters
        config_params = {
            "css_selector": request.css_selector,
            "screenshot": request.take_screenshot,
            "wait_for": request.wait_for_selector,
            "page_timeout": request.timeout * 1000,
            "exclude_all_images": not request.extract_media,
            "verbose": False,
            "log_console": False,
            "deep_crawl_strategy": deep_crawl_strategy,
            "cache_mode": cache_mode,
            "simulate_user": request.simulate_user,
            "generate_markdown": request.generate_markdown,
        }

        js_wait_params = {}
        if request.wait_for_js:
            js_wait_params["wait_until"] = "networkidle"
            js_wait_params["delay_before_return_html"] = 2.0
            if not request.wait_for_selector:
                js_wait_params["scan_full_page"] = True

        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy

        # Build CrawlerRunConfig with backward compatibility
        import inspect
        supported_params = None
        accepts_kwargs = False
        try:
            sig = inspect.signature(CrawlerRunConfig)
            supported_params = set(sig.parameters.keys())
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    accepts_kwargs = True
                    break
        except (ValueError, TypeError):
            pass

        all_params = {**config_params, **js_wait_params}
        if content_filter_strategy:
            all_params["content_filter"] = content_filter_strategy

        unsupported_params = []
        config_fallback_used = False

        if supported_params is not None and not accepts_kwargs:
            filtered_params = {}
            for key, value in all_params.items():
                if key in supported_params:
                    filtered_params[key] = value
                else:
                    unsupported_params.append(key)
            all_params = filtered_params

        try:
            config = CrawlerRunConfig(**all_params)
        except TypeError:
            config_fallback_used = True
            try:
                minimal_params = {
                    "css_selector": request.css_selector,
                    "screenshot": request.take_screenshot,
                    "wait_for": request.wait_for_selector,
                    "page_timeout": request.timeout * 1000,
                    "verbose": False
                }
                config = CrawlerRunConfig(**minimal_params)
                unsupported_params = [k for k in all_params.keys() if k not in minimal_params]
            except TypeError:
                config = CrawlerRunConfig(page_timeout=request.timeout * 1000)
                unsupported_params = list(all_params.keys())

        # Setup browser configuration
        browser_config = {
            "headless": True,
            "verbose": False,
            "browser_type": "webkit"
        }

        if request.use_undetected_browser:
            browser_config["enable_stealth"] = True
            browser_config["browser_type"] = "chromium"

        if request.user_agent:
            browser_config["user_agent"] = request.user_agent

        headers = dict(request.headers) if request.headers else {}
        if request.auth_token:
            has_auth = any(k.lower() == "authorization" for k in headers)
            if not has_auth:
                headers["Authorization"] = f"Bearer {request.auth_token}"
        if headers:
            browser_config["headers"] = headers

        if request.cookies:
            browser_config["cookies"] = _normalize_cookies_to_playwright_format(
                request.cookies, request.url
            )

        # Execute crawl
        with suppress_stdout_stderr():
            result = None
            if request.use_undetected_browser:
                browsers_to_try = ["chromium"]
            else:
                browsers_to_try = ["webkit", "chromium"]

            for browser_type in browsers_to_try:
                try:
                    current_browser_config = browser_config.copy()
                    current_browser_config["browser_type"] = browser_type

                    async with AsyncWebCrawler(**current_browser_config) as crawler:
                        if request.execute_js and hasattr(config, 'js_code'):
                            config.js_code = request.execute_js

                        arun_params = {"url": request.url, "config": config}

                        result = await asyncio.wait_for(
                            crawler.arun(**arun_params),
                            timeout=request.timeout
                        )
                        break

                except asyncio.TimeoutError:
                    error_msg = f"Crawl timeout after {request.timeout}s for {request.url}"
                    if browser_type == browsers_to_try[-1]:
                        raise TimeoutError(error_msg)
                    continue
                except Exception as browser_error:
                    if browser_type == browsers_to_try[-1]:
                        raise browser_error
                    continue

        # Handle results
        if isinstance(result, list):
            response = await _handle_deep_crawl_list_result(result, request)
        elif hasattr(result, 'success') and result.success:
            response = await _handle_single_or_deep_result(result, request, deep_crawl_strategy)
        else:
            error_msg = "Failed to crawl URL"
            if hasattr(result, 'error_message'):
                error_msg = f"Failed to crawl URL: {result.error_message}"
            elif hasattr(result, 'error'):
                error_msg = f"Failed to crawl URL: {result.error}"
            else:
                error_msg = f"Failed to crawl URL: Unknown error (result type: {type(result)})"
            response = CrawlResponse(success=False, url=request.url, error=error_msg)

        # Record fresh fetch timestamp (BYPASS only, non-critical)
        if response.success and cache_mode == CacheMode.BYPASS:
            try:
                policy.record_fresh_fetch(request.url)
            except Exception:
                pass

        return response

    except Exception as e:
        import sys
        error_message = f"Crawling error: {str(e)}"

        if "playwright" in str(e).lower() or "browser" in str(e).lower() or "executable doesn't exist" in str(e).lower():
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)

            if is_uvx_env:
                error_message += "\n\nUVX Environment Browser Setup Required:\n" \
                    f"1. Run system diagnostics: get_system_diagnostics()\n" \
                    f"2. Manual browser installation:\n" \
                    f"   - uvx --with playwright playwright install webkit\n" \
                    f"   - Or system-wide: playwright install webkit\n" \
                    f"3. Restart Claude Desktop after installation\n" \
                    f"4. If issues persist, consider STDIO local setup\n\n" \
                    f"WebKit is lightweight (~180MB) vs Chromium (~281MB)"
            else:
                error_message += "\n\nBrowser Setup Required:\n" \
                    f"1. Install Playwright browsers:\n" \
                    f"   playwright install webkit  # Lightweight option\n" \
                    f"   playwright install chromium  # Full compatibility\n" \
                    f"2. For system dependencies: sudo apt-get install libnss3 libnspr4 libasound2\n" \
                    f"3. Run diagnostics: get_system_diagnostics()"

        return CrawlResponse(
            success=False,
            url=request.url,
            error=error_message,
            extracted_data={
                'error_type': 'browser_setup_required',
                'uvx_environment': 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable),
                'diagnostic_tool': 'get_system_diagnostics',
                'installation_commands': [
                    'playwright install webkit',
                    'playwright install chromium'
                ]
            }
        )


async def _handle_deep_crawl_list_result(result: list, request: CrawlRequest) -> CrawlResponse:
    """Handle deep crawling results returned as a list."""
    if not result:
        return CrawlResponse(
            success=False, url=request.url,
            error="No results returned from deep crawling"
        )

    all_content = []
    all_markdown = []
    all_media = []
    crawled_urls = []

    for page_result in result:
        if hasattr(page_result, 'success') and page_result.success:
            crawled_urls.append(page_result.url if hasattr(page_result, 'url') else 'unknown')
            if hasattr(page_result, 'cleaned_html') and page_result.cleaned_html:
                all_content.append(f"=== {page_result.url} ===\n{page_result.cleaned_html}")
            if hasattr(page_result, 'markdown') and page_result.markdown:
                all_markdown.append(f"=== {page_result.url} ===\n{page_result.markdown}")
            if hasattr(page_result, 'media') and page_result.media and request.extract_media:
                all_media.extend(_convert_media_to_list(page_result.media))

    if not crawled_urls:
        return CrawlResponse(
            success=False, url=request.url,
            error="Deep crawl completed but no pages were successfully crawled"
        )

    response = CrawlResponse(
        success=True,
        url=request.url,
        title=f"Deep crawl of {len(crawled_urls)} pages",
        content="\n\n".join(all_content) if all_content else "No content extracted",
        markdown="\n\n".join(all_markdown) if all_markdown else "No markdown content",
        media=all_media if request.extract_media else None,
        extracted_data={
            "crawled_pages": len(crawled_urls),
            "crawled_urls": crawled_urls,
            "processing_method": "deep_crawling"
        }
    )
    response = await _check_and_summarize_if_needed(response, request)
    return _process_response_content(response, request.include_cleaned_html)


async def _handle_single_or_deep_result(result, request: CrawlRequest, deep_crawl_strategy) -> CrawlResponse:
    """Handle successful single-page or deep crawl result object."""
    if deep_crawl_strategy and hasattr(result, 'crawled_pages'):
        return await _handle_deep_crawl_object_result(result, request)
    else:
        return await _handle_single_page_result(result, request)


async def _handle_deep_crawl_object_result(result, request: CrawlRequest) -> CrawlResponse:
    """Handle deep crawl result with crawled_pages attribute."""
    all_content = []
    all_markdown = []
    all_media = []

    for page in result.crawled_pages:
        if page.cleaned_html:
            all_content.append(f"=== {page.url} ===\n{page.cleaned_html}")
        if page.markdown:
            all_markdown.append(f"=== {page.url} ===\n{page.markdown}")
        if page.media and request.extract_media:
            all_media.extend(_convert_media_to_list(page.media))

    combined_content = "\n\n".join(all_content) if all_content else result.cleaned_html
    combined_markdown = "\n\n".join(all_markdown) if all_markdown else result.markdown
    title_to_use = (result.metadata or {}).get("title", "")
    extracted_data = {"crawled_pages": len(result.crawled_pages)} if hasattr(result, 'crawled_pages') else {}

    if request.auto_summarize and combined_content:
        estimated_tokens = len(combined_content) // 4
        if estimated_tokens > request.max_content_tokens:
            try:
                content_for_summary = combined_markdown or combined_content
                summary_result = await summarize_web_content(
                    content=content_for_summary,
                    title=title_to_use, url=request.url,
                    summary_length=request.summary_length,
                    llm_provider=request.llm_provider, llm_model=request.llm_model
                )
                if summary_result.get("success"):
                    combined_content = summary_result["summary"]
                    combined_markdown = summary_result["summary"]
                    extracted_data.update({
                        "summarization_applied": True,
                        "original_content_length": len("\n\n".join(all_content) if all_content else result.cleaned_html),
                        "original_tokens_estimate": estimated_tokens,
                        "summary_length": request.summary_length,
                        "compression_ratio": summary_result.get("compressed_ratio", 0),
                        "key_topics": summary_result.get("key_topics", []),
                        "content_type": summary_result.get("content_type", "Unknown"),
                        "main_insights": summary_result.get("main_insights", []),
                        "technical_details": summary_result.get("technical_details", []),
                        "llm_provider": summary_result.get("llm_provider", "unknown"),
                        "llm_model": summary_result.get("llm_model", "unknown"),
                        "auto_summarization_trigger": f"Deep crawl content exceeded {request.max_content_tokens} tokens"
                    })
                else:
                    extracted_data.update({
                        "summarization_attempted": True,
                        "summarization_error": summary_result.get("error", "Unknown error"),
                        "original_content_preserved": True
                    })
            except Exception as e:
                extracted_data.update({
                    "summarization_attempted": True,
                    "summarization_error": f"Exception during summarization: {str(e)}",
                    "original_content_preserved": True
                })

    response = CrawlResponse(
        success=True, url=request.url, title=title_to_use,
        content=combined_content, markdown=combined_markdown,
        media=all_media if request.extract_media else None,
        screenshot=result.screenshot if request.take_screenshot else None,
        extracted_data=extracted_data
    )
    response = await _check_and_summarize_if_needed(response, request)
    return _process_response_content(response, request.include_cleaned_html)


async def _handle_single_page_result(result, request: CrawlRequest) -> CrawlResponse:
    """Handle successful single-page crawl result."""
    content_to_use = result.cleaned_html
    markdown_to_use = result.markdown
    extracted_data = None
    title_to_use = (result.metadata or {}).get("title", "")

    if request.auto_summarize and content_to_use:
        estimated_tokens = len(content_to_use) // 4
        if estimated_tokens > request.max_content_tokens:
            try:
                content_for_summary = markdown_to_use or content_to_use
                summary_result = await summarize_web_content(
                    content=content_for_summary,
                    title=title_to_use, url=request.url,
                    summary_length=request.summary_length,
                    llm_provider=request.llm_provider, llm_model=request.llm_model
                )
                if summary_result.get("success"):
                    content_to_use = summary_result["summary"]
                    markdown_to_use = summary_result["summary"]
                    extracted_data = {
                        "summarization_applied": True,
                        "original_content_length": len(result.cleaned_html),
                        "original_tokens_estimate": estimated_tokens,
                        "summary_length": request.summary_length,
                        "compression_ratio": summary_result.get("compressed_ratio", 0),
                        "key_topics": summary_result.get("key_topics", []),
                        "content_type": summary_result.get("content_type", "Unknown"),
                        "main_insights": summary_result.get("main_insights", []),
                        "technical_details": summary_result.get("technical_details", []),
                        "llm_provider": summary_result.get("llm_provider", "unknown"),
                        "llm_model": summary_result.get("llm_model", "unknown"),
                        "auto_summarization_trigger": f"Content exceeded {request.max_content_tokens} tokens"
                    }
                else:
                    extracted_data = {
                        "summarization_attempted": True,
                        "summarization_error": summary_result.get("error", "Unknown error"),
                        "original_content_preserved": True
                    }
            except Exception as e:
                extracted_data = {
                    "summarization_attempted": True,
                    "summarization_error": f"Exception during summarization: {str(e)}",
                    "original_content_preserved": True
                }

    response = CrawlResponse(
        success=True, url=request.url, title=title_to_use,
        content=content_to_use, markdown=markdown_to_use,
        media=_convert_media_to_list(result.media) if request.extract_media else None,
        screenshot=result.screenshot if request.take_screenshot else None,
        extracted_data=extracted_data
    )
    response = await _check_and_summarize_if_needed(response, request)
    return _process_response_content(response, request.include_cleaned_html)
