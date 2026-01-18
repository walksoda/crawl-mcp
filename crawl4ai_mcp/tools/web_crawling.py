"""
Web crawling tools for Crawl4AI MCP Server.

Contains complete web crawling functionality and content extraction tools.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import (
    CrawlRequest,
    CrawlResponse,
    StructuredExtractionRequest
)

# Import required crawl4ai components
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

# Import processors and utilities
from ..file_processor import FileProcessor
from ..youtube_processor import YouTubeProcessor
from ..suppress_output import suppress_stdout_stderr

# Import refactored modules
from .session_manager import get_session_manager
from .strategy_cache import get_strategy_cache, get_fingerprint_config
from .content_processors import (
    convert_media_to_list as _convert_media_to_list,
    has_meaningful_content as _has_meaningful_content,
    is_block_page as _is_block_page,
)
from .fallback_strategies import (
    normalize_cookies_to_playwright_format as _normalize_cookies_to_playwright_format,
    static_fetch_content as _static_fetch_content,
    extract_spa_json_data as _extract_spa_json_data,
    detect_spa_framework as _detect_spa_framework,
    build_amp_url as _build_amp_url,
    try_fetch_rss_feed as _try_fetch_rss_feed,
)
from ..constants import FALLBACK_MIN_CONTENT_LENGTH, MAX_RESPONSE_TOKENS

# Initialize processors
file_processor = FileProcessor()
youtube_processor = YouTubeProcessor()


# Placeholder for summarize_web_content function
# Response size limit for MCP protocol
# MAX_RESPONSE_TOKENS imported from constants.py
MAX_RESPONSE_CHARS = MAX_RESPONSE_TOKENS * 4  # Rough estimate: 1 token ≈ 4 characters

async def summarize_web_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize web content using LLM with fallback to basic text truncation.
    """
    try:
        from ..utils.llm_extraction import LLMExtractionClient

        # Create LLM client from config
        client = LLMExtractionClient.from_config(llm_provider, llm_model)

        # Prepare summarization prompt based on summary length
        length_instructions = {
            "short": "Provide a brief 2-3 sentence summary of the main points.",
            "medium": "Provide a comprehensive summary in 1-2 paragraphs covering key points.",
            "long": "Provide a detailed summary covering all important information, insights, and context."
        }

        prompt = f"""
        Please summarize the following web content.

        Title: {title}
        URL: {url}

        {length_instructions.get(summary_length, length_instructions["medium"])}

        Focus on:
        - Main topics and key information
        - Important facts, statistics, or findings
        - Practical insights or conclusions
        - Technical details if present

        Content to summarize:
        {content[:50000]}  # Limit to prevent token overflow
        """

        system_message = "You are an expert content summarizer. Provide clear, concise summaries."

        summary_text = await client.call_llm(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,
            max_tokens=2000
        )

        if summary_text:
            # Calculate compression ratio
            original_length = len(content)
            summary_length_chars = len(summary_text)
            compression_ratio = round((1 - summary_length_chars / original_length) * 100, 2) if original_length > 0 else 0

            return {
                "success": True,
                "summary": summary_text,
                "original_length": original_length,
                "summary_length": summary_length_chars,
                "compressed_ratio": compression_ratio,
                "llm_provider": client.provider,
                "llm_model": client.model,
                "content_type": "web_content"
            }
        else:
            raise ValueError("LLM returned empty summary")

    except Exception as e:
        # Fallback to simple truncation if LLM fails
        max_chars = {
            "short": 500,
            "medium": 1500,
            "long": 3000
        }.get(summary_length, 1500)

        truncated = content[:max_chars]
        if len(content) > max_chars:
            truncated += "... [Content truncated due to size]"

        return {
            "success": False,
            "error": f"LLM summarization failed: {str(e)}. Returning truncated content.",
            "summary": truncated,
            "original_length": len(content),
            "summary_length": len(truncated),
            "compressed_ratio": round((1 - len(truncated) / len(content)) * 100, 2) if content else 0,
            "fallback_method": "truncation"
        }


def _process_response_content(response: CrawlResponse, include_cleaned_html: bool) -> CrawlResponse:
    """
    Process CrawlResponse to remove content field if include_cleaned_html is False.
    By default, only markdown is returned to reduce token usage and improve readability.
    """
    if not include_cleaned_html and hasattr(response, 'content'):
        response.content = None
    return response


# Complete implementation of internal crawl URL function
async def _internal_crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    try:
        # Check if URL is a YouTube video
        if youtube_processor.is_youtube_url(request.url):
            # ⚠️ WARNING: YouTube transcript extraction is currently experiencing issues
            # due to API specification changes. Attempting extraction but may fail.
            try:
                youtube_result = await youtube_processor.process_youtube_url(
                    url=request.url,
                    languages=["ja", "en"],  # Default language preferences
                    include_timestamps=True,
                    preserve_formatting=True,
                    include_metadata=True
                )
                
                if youtube_result['success']:
                    transcript_data = youtube_result['transcript']
                    response = CrawlResponse(
                        success=True,
                        url=request.url,
                        title=f"YouTube Video Transcript: {youtube_result['video_id']}",
                        content=transcript_data.get('full_text'),
                        markdown=transcript_data.get('clean_text'),
                        extracted_data={
                            "video_id": youtube_result['video_id'],
                            "processing_method": "youtube_transcript_api",
                            "language_info": youtube_result.get('language_info'),
                            "transcript_stats": {
                                "word_count": transcript_data.get('word_count'),
                                "segment_count": transcript_data.get('segment_count'),
                                "duration": transcript_data.get('duration_formatted')
                            },
                            "metadata": youtube_result.get('metadata')
                        }
                    )
                    response = await _check_and_summarize_if_needed(response, request)
                    return _process_response_content(response, request.include_cleaned_html)
                else:
                    # If YouTube transcript extraction fails, provide helpful error message
                    error_msg = youtube_result.get('error', 'Unknown error')
                    suggestion = youtube_result.get('suggestion', '')
                    
                    full_error = f"YouTube transcript extraction failed: {error_msg}"
                    if suggestion:
                        full_error += f"\n\nSuggestion: {suggestion}"
                    
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=full_error
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"YouTube processing error: {str(e)}"
                )
        
        # Check if URL points to a file that should be processed with MarkItDown
        if file_processor.is_supported_file(request.url):
            # Redirect to file processing for supported file formats
            try:
                file_result = await file_processor.process_file_from_url(
                    request.url,
                    max_size_mb=100  # Default size limit
                )
                
                if file_result['success']:
                    response = CrawlResponse(
                        success=True,
                        url=request.url,
                        title=file_result.get('title'),
                        content=file_result.get('content'),
                        markdown=file_result.get('content'),  # MarkItDown already provides markdown
                        extracted_data={
                            "file_type": file_result.get('file_type'),
                            "size_bytes": file_result.get('size_bytes'),
                            "is_archive": file_result.get('is_archive', False),
                            "metadata": file_result.get('metadata'),
                            "archive_contents": file_result.get('archive_contents'),
                            "processing_method": "markitdown"
                        }
                    )
                    response = await _check_and_summarize_if_needed(response, request)
                    return _process_response_content(response, request.include_cleaned_html)
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=f"File processing failed: {file_result.get('error')}"
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"File processing error: {str(e)}"
                )
        
        # Setup deep crawling strategy if max_depth is specified
        deep_crawl_strategy = None
        if request.max_depth is not None and request.max_depth > 0:
            # Create filter chain
            filters = []
            if request.url_pattern:
                filters.append(URLPatternFilter(patterns=[request.url_pattern]))
            if not request.include_external:
                # Extract domain from URL for domain filtering
                from urllib.parse import urlparse
                domain = urlparse(request.url).netloc
                filters.append(DomainFilter(allowed_domains=[domain]))
            
            filter_chain = FilterChain(filters) if filters else None
            
            # Select crawling strategy
            if request.crawl_strategy == "dfs":
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            elif request.crawl_strategy == "best_first":
                deep_crawl_strategy = BestFirstCrawlingStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            else:  # Default to BFS
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
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

        # Setup cache mode
        cache_mode = CacheMode.ENABLED
        if not request.enable_caching or request.cache_mode == "disabled":
            cache_mode = CacheMode.DISABLED
        elif request.cache_mode == "bypass":
            cache_mode = CacheMode.BYPASS

        # Configure chunking if requested
        chunking_strategy = None
        if request.chunk_content:
            step_size = max(1, int(request.chunk_size * (1 - request.overlap_rate)))
            # Calculate actual overlap amount (not step size)
            overlap_chars = max(0, int(request.chunk_size * request.overlap_rate))

            if request.chunk_strategy == "topic":
                chunking_strategy = TopicSegmentationChunking(
                    num_keywords=5,
                    chunk_size=request.chunk_size
                )
            elif request.chunk_strategy == "regex":
                # Use common text separators for regex chunking
                chunking_strategy = RegexChunking(
                    patterns=[r"\n\n", r"\n#{1,6}\s", r"\n---\n"]
                )
            elif request.chunk_strategy == "sentence":
                chunking_strategy = OverlappingWindowChunking(
                    window_size=request.chunk_size,
                    overlap=overlap_chars
                )
            else:  # sliding (default fallback)
                chunking_strategy = SlidingWindowChunking(
                    window_size=request.chunk_size,
                    step=step_size
                )

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
            # Phase 2: Add simulate_user support
            "simulate_user": request.simulate_user,
            # Phase 4: Pass generate_markdown to CrawlerRunConfig
            "generate_markdown": request.generate_markdown,
        }

        # Phase 2: Handle wait_for_js - use wait_until and delay for JS-heavy sites
        # These parameters may not be supported in older crawl4ai versions
        js_wait_params = {}
        if request.wait_for_js:
            js_wait_params["wait_until"] = "networkidle"  # Wait for network to be idle
            js_wait_params["delay_before_return_html"] = 2.0  # Additional delay for JS rendering
            if not request.wait_for_selector:
                js_wait_params["scan_full_page"] = True  # Scan full page when no specific selector
        
        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy

        # Build CrawlerRunConfig with backward compatibility
        # Use inspect.signature to filter only supported parameters
        import inspect
        supported_params = None
        accepts_kwargs = False
        try:
            sig = inspect.signature(CrawlerRunConfig)
            supported_params = set(sig.parameters.keys())
            # Check if CrawlerRunConfig accepts **kwargs (VAR_KEYWORD)
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    accepts_kwargs = True
                    break
        except (ValueError, TypeError):
            # Fallback if signature inspection fails
            pass

        # Prepare all desired parameters
        all_params = {**config_params, **js_wait_params}
        if content_filter_strategy:
            all_params["content_filter"] = content_filter_strategy

        # Track which params were filtered out for diagnostics
        unsupported_params = []
        config_fallback_used = False

        # Only filter if signature inspection worked and no **kwargs
        if supported_params is not None and not accepts_kwargs:
            filtered_params = {}
            for key, value in all_params.items():
                if key in supported_params:
                    filtered_params[key] = value
                else:
                    unsupported_params.append(key)
            all_params = filtered_params

        # Create config with filtered parameters
        try:
            config = CrawlerRunConfig(**all_params)
        except TypeError as e:
            # If signature filtering didn't work or missed something, try minimal config
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
                # Absolute minimal fallback
                config = CrawlerRunConfig(page_timeout=request.timeout * 1000)
                unsupported_params = list(all_params.keys())

        # Setup browser configuration with lightweight WebKit preference
        browser_config = {
            "headless": True,
            "verbose": False,
            "browser_type": "webkit"  # Use lightweight WebKit by default
        }
        
        # Enable undetected browser mode if requested
        if request.use_undetected_browser:
            browser_config["enable_stealth"] = True
            # Use Chromium when stealth mode is needed for better compatibility
            browser_config["browser_type"] = "chromium"
        
        if request.user_agent:
            browser_config["user_agent"] = request.user_agent

        # Build headers dict with auth_token if provided
        # NOTE: browser_config["headers"] applies to ALL requests from the page,
        # including third-party subresources. For sensitive tokens, consider using
        # route-based header injection in Playwright to limit scope to same-origin.
        headers = dict(request.headers) if request.headers else {}
        if request.auth_token:
            # Only add if not already set (preserve user-specified Authorization, case-insensitive)
            has_auth = any(k.lower() == "authorization" for k in headers)
            if not has_auth:
                headers["Authorization"] = f"Bearer {request.auth_token}"
        if headers:
            browser_config["headers"] = headers

        # Convert and add cookies in Playwright format
        if request.cookies:
            browser_config["cookies"] = _normalize_cookies_to_playwright_format(
                request.cookies, request.url
            )

        # Suppress output to avoid JSON parsing errors
        with suppress_stdout_stderr():
            # Try browsers with fallback
            result = None
            # Prioritize Chromium when stealth mode is requested for better compatibility
            if request.use_undetected_browser:
                browsers_to_try = ["chromium"]  # Only use Chromium for stealth
            else:
                browsers_to_try = ["webkit", "chromium"]
            
            for browser_type in browsers_to_try:
                try:
                    current_browser_config = browser_config.copy()
                    current_browser_config["browser_type"] = browser_type
                    
                    async with AsyncWebCrawler(**current_browser_config) as crawler:
                        # Execute custom JavaScript if provided (with compatibility check)
                        if request.execute_js and hasattr(config, 'js_code'):
                            config.js_code = request.execute_js

                        # Run crawler with config and proper timeout
                        arun_params = {"url": request.url, "config": config}

                        # Apply timeout to crawler.arun() to prevent hanging
                        result = await asyncio.wait_for(
                            crawler.arun(**arun_params),
                            timeout=request.timeout
                        )
                        break  # Success, no need to try other browsers

                except asyncio.TimeoutError:
                    # Handle timeout specifically
                    error_msg = f"Crawl timeout after {request.timeout}s for {request.url}"
                    if browser_type == browsers_to_try[-1]:
                        raise TimeoutError(error_msg)
                    continue  # Try next browser
                except Exception as browser_error:
                    # If this is the last browser to try, raise the error
                    if browser_type == browsers_to_try[-1]:
                        raise browser_error
                    # Otherwise, try the next browser
                    continue
        
        # Handle different result types (single result vs list from deep crawling)
        if isinstance(result, list):
            # Deep crawling returns a list of results
            if not result:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="No results returned from deep crawling"
                )
            
            # Process multiple results from deep crawling
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

            # Check if any pages were successfully crawled
            if not crawled_urls:
                return CrawlResponse(
                    success=False,
                    url=request.url,
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
        
        elif hasattr(result, 'success') and result.success:
            # For deep crawling, result might contain multiple pages
            if deep_crawl_strategy and hasattr(result, 'crawled_pages'):
                # Combine content from all crawled pages
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
                
                # Prepare content for potential summarization
                combined_content = "\n\n".join(all_content) if all_content else result.cleaned_html
                combined_markdown = "\n\n".join(all_markdown) if all_markdown else result.markdown
                title_to_use = (result.metadata or {}).get("title", "")
                extracted_data = {"crawled_pages": len(result.crawled_pages)} if hasattr(result, 'crawled_pages') else {}
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and combined_content:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(combined_content) // 4
                    
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = combined_markdown or combined_content
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
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
                                # Summarization failed, add error info but keep original content
                                extracted_data.update({
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                })
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data.update({
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            })
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=combined_content,
                    markdown=combined_markdown,
                    media=all_media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            else:
                # Check if auto-summarization should be applied
                content_to_use = result.cleaned_html
                markdown_to_use = result.markdown
                extracted_data = None
                title_to_use = (result.metadata or {}).get("title", "")
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and content_to_use:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(content_to_use) // 4
                    
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = markdown_to_use or content_to_use
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                content_to_use = summary_result["summary"]
                                markdown_to_use = summary_result["summary"]  # Use same summary for both
                                
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
                                # Summarization failed, add error info but keep original content
                                extracted_data = {
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                }
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data = {
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            }
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=content_to_use,
                    markdown=markdown_to_use,
                    media=_convert_media_to_list(result.media) if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            response = await _check_and_summarize_if_needed(response, request)
            return _process_response_content(response, request.include_cleaned_html)
        else:
            # Handle case where result doesn't have success attribute or failed
            error_msg = "Failed to crawl URL"
            if hasattr(result, 'error_message'):
                error_msg = f"Failed to crawl URL: {result.error_message}"
            elif hasattr(result, 'error'):
                error_msg = f"Failed to crawl URL: {result.error}"
            else:
                error_msg = f"Failed to crawl URL: Unknown error (result type: {type(result)})"
            
            return CrawlResponse(
                success=False,
                url=request.url,
                error=error_msg
            )
                
    except Exception as e:
        import sys  # Import here to ensure availability in error response
        error_message = f"Crawling error: {str(e)}"

        # Enhanced error handling for browser and UVX issues
        if "playwright" in str(e).lower() or "browser" in str(e).lower() or "executable doesn't exist" in str(e).lower():
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            
            if is_uvx_env:
                error_message += "\n\n🔧 UVX Environment Browser Setup Required:\n" \
                    f"1. Run system diagnostics: get_system_diagnostics()\n" \
                    f"2. Manual browser installation:\n" \
                    f"   - uvx --with playwright playwright install webkit\n" \
                    f"   - Or system-wide: playwright install webkit\n" \
                    f"3. Restart Claude Desktop after installation\n" \
                    f"4. If issues persist, consider STDIO local setup\n\n" \
                    f"💡 WebKit is lightweight (~180MB) vs Chromium (~281MB)"
            else:
                error_message += "\n\n🔧 Browser Setup Required:\n" \
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

async def _check_and_summarize_if_needed(
    response: CrawlResponse,
    request: CrawlRequest
) -> CrawlResponse:
    """
    Check if response content exceeds token limits and apply summarization if needed.
    Respects user-specified limits when provided.
    """
    # Skip if response failed
    if not response.success:
        return response
    
    # Skip if neither content nor markdown exists
    if not response.content and not response.markdown:
        return response
    
    # Check if already summarized
    if response.extracted_data and response.extracted_data.get("summarization_applied"):
        return response
    
    # Estimate total response size (content + markdown + metadata)
    total_chars = len(response.content or "") + len(response.markdown or "")
    
    # Determine effective token limit - user-specified takes precedence
    effective_limit_chars = MAX_RESPONSE_CHARS
    limit_source = "MCP_PROTOCOL_SAFETY"
    
    # If user explicitly set auto_summarize=True and max_content_tokens, use that instead
    if hasattr(request, 'auto_summarize') and request.auto_summarize:
        if hasattr(request, 'max_content_tokens') and request.max_content_tokens:
            # Convert user's token limit to character estimate
            user_limit_chars = request.max_content_tokens * 4
            # Use the smaller limit (user preference or protocol safety)
            if user_limit_chars < effective_limit_chars:
                effective_limit_chars = user_limit_chars
                limit_source = "USER_SPECIFIED"
    
    # Check if response exceeds the effective limit
    if total_chars > effective_limit_chars:
        try:
            # Use markdown if available, otherwise use content
            content_to_summarize = response.markdown or response.content
            
            # Determine summary length based on reduction needed
            reduction_ratio = effective_limit_chars / total_chars
            if reduction_ratio > 0.5:
                summary_length = "medium"
            elif reduction_ratio > 0.3:
                summary_length = "short"
            else:
                summary_length = "short"  # Aggressive reduction needed
            
            # Force summarization to meet token limits
            summary_result = await summarize_web_content(
                content=content_to_summarize,
                title=response.title or "",
                url=response.url,
                summary_length=summary_length,
                llm_provider=request.llm_provider if hasattr(request, 'llm_provider') else None,
                llm_model=request.llm_model if hasattr(request, 'llm_model') else None
            )
            
            if summary_result.get("success"):
                summary_text = summary_result['summary']
                
                # Build prefix based on limit source
                if limit_source == "USER_SPECIFIED":
                    prefix = f"⚠️ Content exceeded user-specified limit ({total_chars:,} chars > {effective_limit_chars:,} chars). Auto-summarized:\n\n"
                else:
                    prefix = f"⚠️ Content exceeded MCP token limit ({total_chars:,} chars > {effective_limit_chars:,} chars). Auto-summarized:\n\n"
                
                # Verify summarized content fits within limits with strict control
                final_content = f"{prefix}{summary_text}"
                if len(final_content) > effective_limit_chars:
                    # Truncate if summary still exceeds limit
                    suffix = "... [Summary truncated]"
                    available_chars = effective_limit_chars - len(prefix) - len(suffix)

                    if available_chars > 0:
                        summary_text = summary_text[:available_chars] + suffix
                    else:
                        # Extreme case: prefix alone exceeds limit, use minimal prefix
                        minimal_prefix = "[Exceeded limit] "
                        available_chars = effective_limit_chars - len(minimal_prefix) - len(suffix)
                        if available_chars > 0:
                            summary_text = summary_text[:available_chars] + suffix
                            prefix = minimal_prefix
                        else:
                            # Ultimate fallback: no content fits
                            summary_text = ""
                            prefix = f"[Content exceeded {effective_limit_chars} char limit]"

                    final_content = f"{prefix}{summary_text}"

                    # Final safety check - hard truncate if still over limit
                    if len(final_content) > effective_limit_chars:
                        final_content = final_content[:effective_limit_chars - 3] + "..."
                
                response.content = final_content
                response.markdown = final_content
                
                # Update extracted_data
                if response.extracted_data is None:
                    response.extracted_data = {}
                
                response.extracted_data.update({
                    "auto_summarization_reason": limit_source,
                    "original_size_chars": total_chars,
                    "effective_limit_chars": effective_limit_chars,
                    "user_specified_limit": limit_source == "USER_SPECIFIED",
                    "user_max_content_tokens": request.max_content_tokens if hasattr(request, 'max_content_tokens') else None,
                    "summarization_applied": True,
                    "summary_length": summary_length,
                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                    "llm_model": summary_result.get("llm_model", "unknown"),
                    "post_summary_truncated": len(f"{prefix}{summary_result['summary']}") > effective_limit_chars,
                })
            else:
                # Summarization failed, truncate content
                # Use markdown if content is None
                content_to_truncate = response.content or response.markdown or ""
                truncate_at = max(100, effective_limit_chars - 500)  # Leave room for message, minimum 100 chars
                prefix = f"⚠️ Content exceeded limit ({total_chars:,} chars > {effective_limit_chars:,} chars).\n\nSummarization failed: {summary_result.get('error', 'Unknown error')}\n\nTruncated content:\n\n"
                response.content = f"{prefix}{content_to_truncate[:truncate_at]}... [Content truncated]"
                response.markdown = response.content
                
                if response.extracted_data is None:
                    response.extracted_data = {}
                
                response.extracted_data.update({
                    "auto_truncation_reason": limit_source,
                    "original_size_chars": total_chars,
                    "effective_limit_chars": effective_limit_chars,
                    "user_specified_limit": limit_source == "USER_SPECIFIED",
                    "truncation_applied": True,
                    "summarization_attempted": True,
                    "summarization_error": summary_result.get("error", "Unknown error")
                })
                
        except Exception as e:
            # Final fallback: aggressive truncation
            # Use markdown if content is None
            content_to_truncate = response.content or response.markdown or ""
            truncate_at = max(100, effective_limit_chars - 500)  # Minimum 100 chars
            prefix = f"⚠️ Content exceeded limit ({total_chars:,} chars > {effective_limit_chars:,} chars).\n\nEmergency truncation applied due to error: {str(e)}\n\n"
            response.content = f"{prefix}{content_to_truncate[:truncate_at]}... [Content truncated]"
            response.markdown = response.content
            
            if response.extracted_data is None:
                response.extracted_data = {}
                
            response.extracted_data.update({
                "emergency_truncation_reason": limit_source,
                "original_size_chars": total_chars,
                "effective_limit_chars": effective_limit_chars,
                "user_specified_limit": limit_source == "USER_SPECIFIED",
                "truncation_error": str(e)
            })
    
    return response


async def _finalize_fallback_response(
    response: CrawlResponse,
    request_url: str,
    auto_summarize: bool = False,
    max_content_tokens: int = 15000,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> CrawlResponse:
    """
    Apply size limit and summarization to fallback responses.

    This ensures all fallback return points go through the same
    size checking and summarization as regular crawl responses.
    """
    # Create a dummy CrawlRequest with the necessary parameters
    dummy_request = CrawlRequest(
        url=request_url,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    return await _check_and_summarize_if_needed(response, dummy_request)


async def _build_json_extraction_response(
    json_data: dict,
    json_source: str,
    url: str,
    strategy_name: str,
    stage: int,
    auto_summarize: bool,
    max_content_tokens: int,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    extract_content_keys: Optional[List[str]] = None
) -> CrawlResponse:
    """
    Build CrawlResponse from extracted JSON data.
    
    Common helper for Stage 1 (static JSON extraction) and Stage 7 (JSON extraction fallback).
    
    Args:
        json_data: Extracted JSON data
        json_source: Source identifier (e.g., '__NEXT_DATA__')
        url: Original request URL
        strategy_name: Name of the strategy (e.g., 'static_json_extraction', 'json_extraction')
        stage: Fallback stage number (1 or 7)
        auto_summarize: Whether to auto-summarize large content
        max_content_tokens: Token limit for auto-summarization
        llm_provider: LLM provider for summarization
        llm_model: LLM model for summarization
        extract_content_keys: Optional list of priority keys to extract content from
            (e.g., ['props', 'pageProps', 'data']). If None, uses full JSON.
    
    Returns:
        CrawlResponse with formatted JSON content
    """
    content_text = ""
    
    # Try to extract content from priority keys if specified
    if isinstance(json_data, dict) and extract_content_keys:
        for key in extract_content_keys:
            if key in json_data:
                content_text = json.dumps(json_data[key], indent=2, ensure_ascii=False)
                break
    
    # Fallback to full JSON if no priority key found or not specified
    if not content_text:
        content_text = json.dumps(json_data, indent=2, ensure_ascii=False)
    
    response = CrawlResponse(
        success=True,
        url=url,
        markdown=f"# Extracted JSON Data ({json_source})\n\n```json\n{content_text[:50000]}\n```",
        content=content_text,
        extracted_data={
            "fallback_strategy_used": strategy_name,
            "fallback_stage": stage,
            "json_source": json_source,
            "site_type_detected": "spa_with_json"
        }
    )
    
    response = await _finalize_fallback_response(
        response, url, auto_summarize, max_content_tokens, llm_provider, llm_model
    )
    return response


# Other internal functions for specialized extraction
async def _internal_intelligent_extract(
    url: str,
    extraction_goal: str,
    content_filter: str = "bm25",
    filter_query: Optional[str] = None,
    chunk_content: bool = False,
    use_llm: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal intelligent extract implementation.
    Uses LLM and content filtering for targeted extraction.
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(
            url=url,
            content_filter=content_filter,
            filter_query=filter_query,
            chunk_content=chunk_content,
            generate_markdown=True,
            include_cleaned_html=True
        )

        crawl_result = await _internal_crawl_url(request)

        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }

        # If LLM processing is disabled, return the crawled content
        if not use_llm:
            return {
                "url": url,
                "success": True,
                "extracted_content": crawl_result.content,
                "extraction_goal": extraction_goal,
                "processing_method": "basic_crawl_only"
            }

        # Implement LLM-based intelligent extraction
        try:
            from ..utils.llm_extraction import LLMExtractionClient

            # Create LLM client from config
            client = LLMExtractionClient.from_config(llm_provider, llm_model)

            # Prepare extraction prompt
            extraction_prompt = f"""
            You are an expert content analyst. Your task is to extract specific information from web content based on the extraction goal.

            EXTRACTION GOAL: {extraction_goal}

            INSTRUCTIONS:
            - Focus specifically on information relevant to the extraction goal
            - Extract concrete data, statistics, quotes, and specific details
            - Maintain accuracy and preserve exact information from the source
            - Organize findings in a structured, easy-to-understand format
            - If the content doesn't contain relevant information, clearly state that

            {f"ADDITIONAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}

            Please provide a JSON response with the following structure:
            {{
                "extracted_data": "The specific information extracted according to the goal",
                "key_findings": ["List", "of", "main", "findings"],
                "relevant_quotes": ["Important", "quotes", "from", "source"],
                "statistics_data": ["Numerical", "data", "and", "statistics"],
                "sources_references": ["References", "to", "specific", "sections"],
                "extraction_confidence": "High/Medium/Low - confidence in extraction quality",
                "missing_information": ["Information", "sought", "but", "not", "found"]
            }}

            CONTENT TO ANALYZE:
            {crawl_result.content[:50000]}  # Limit content to prevent token overflow
            """

            system_message = "You are an expert content analyst specializing in precise information extraction."

            extracted_content = await client.call_llm(
                prompt=extraction_prompt,
                system_message=system_message,
                temperature=0.1,
                max_tokens=4000
            )

            # Parse JSON response
            if extracted_content:
                try:
                    extraction_data = client.parse_json_response(extracted_content)

                    return {
                        "url": url,
                        "success": True,
                        "extraction_goal": extraction_goal,
                        "extracted_data": extraction_data.get("extracted_data", ""),
                        "key_findings": extraction_data.get("key_findings", []),
                        "relevant_quotes": extraction_data.get("relevant_quotes", []),
                        "statistics_data": extraction_data.get("statistics_data", []),
                        "sources_references": extraction_data.get("sources_references", []),
                        "extraction_confidence": extraction_data.get("extraction_confidence", "Medium"),
                        "missing_information": extraction_data.get("missing_information", []),
                        "processing_method": "llm_intelligent_extraction",
                        "llm_provider": client.provider,
                        "llm_model": client.model,
                        "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                        "custom_instructions_used": bool(custom_instructions)
                    }

                except (json.JSONDecodeError, AttributeError) as e:
                    # Fallback: treat as plain text extraction
                    return {
                        "url": url,
                        "success": True,
                        "extraction_goal": extraction_goal,
                        "extracted_data": str(extracted_content),
                        "key_findings": [],
                        "relevant_quotes": [],
                        "statistics_data": [],
                        "sources_references": [],
                        "extraction_confidence": "Medium",
                        "missing_information": [],
                        "processing_method": "llm_intelligent_extraction_fallback",
                        "llm_provider": client.provider,
                        "llm_model": client.model,
                        "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                        "custom_instructions_used": bool(custom_instructions),
                        "json_parse_error": str(e)
                    }
            else:
                return {
                    "url": url,
                    "success": False,
                    "error": "LLM extraction returned empty result"
                }

        except Exception as llm_error:
            # LLM processing failed, return crawled content with error info
            return {
                "url": url,
                "success": True,  # Still return success since we have crawled content
                "extraction_goal": extraction_goal,
                "extracted_data": crawl_result.content,
                "key_findings": [],
                "relevant_quotes": [],
                "statistics_data": [],
                "sources_references": [],
                "extraction_confidence": "Low",
                "missing_information": [],
                "processing_method": "crawl_fallback_due_to_llm_error",
                "llm_error": str(llm_error),
                "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                "custom_instructions_used": bool(custom_instructions)
            }

    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Intelligent extraction error: {str(e)}"
        }




def _regex_worker(pattern: str, text: str, flags: int, result_queue) -> None:
    """
    Worker function for regex execution in separate process.
    
    This runs in an isolated process to enable true timeout via process termination.
    
    Args:
        pattern: Regular expression pattern
        text: Text to search
        flags: Regex compilation flags
        result_queue: multiprocessing.Queue to put results
    """
    import re
    try:
        compiled = re.compile(pattern, flags)
        matches = compiled.findall(text)
        result_queue.put(("success", matches))
    except Exception as e:
        result_queue.put(("error", str(e)))

def _safe_regex_findall(pattern: str, text: str, timeout: float = 5.0) -> List[str]:
    """
    Execute re.findall with true timeout protection using multiprocessing.
    
    This provides genuine timeout protection against ReDoS attacks by running
    the regex in a separate process that can be forcefully terminated.
    
    Args:
        pattern: Regular expression pattern to match
        text: Text to search in
        timeout: Maximum execution time in seconds (default: 5.0)
    
    Returns:
        List of matches found
        
    Raises:
        ValueError: If the pattern is invalid or too long
        TimeoutError: If regex execution exceeds timeout
    """
    import re
    import multiprocessing
    
    # Validate pattern first
    try:
        re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")
    
    # Safety: limit pattern complexity to mitigate DoS risk
    if len(pattern) > 500:
        raise ValueError(f"Pattern too long ({len(pattern)} chars, max 500)")
    
    # Use multiprocessing for true timeout with process termination
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_regex_worker,
        args=(pattern, text, re.IGNORECASE, result_queue)
    )
    
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        # Process exceeded timeout - terminate it
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            # Force kill if terminate didn't work
            process.kill()
            process.join()
        raise TimeoutError(f"Regex execution timed out after {timeout}s (possible ReDoS pattern)")
    
    if result_queue.empty():
        raise RuntimeError("Regex worker returned no result")
    
    status, result = result_queue.get_nowait()
    if status == "error":
        raise ValueError(f"Regex execution error: {result}")
    
    return result


async def _internal_extract_entities(
    url: str,
    entity_types: List[str],
    custom_patterns: Optional[Dict[str, str]] = None,
    include_context: bool = True,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Internal extract entities implementation using regex patterns.
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(url=url, generate_markdown=True, include_cleaned_html=True)
        crawl_result = await _internal_crawl_url(request)

        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }

        # Use content if available, fallback to markdown for entity extraction
        content = crawl_result.content or crawl_result.markdown or ""
        entities = {}
        pattern_errors = {}

        # Define regex patterns for common entity types
        patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phones": r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
            "urls": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            "dates": r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b',
            "ips": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "prices": r'[$£€¥]?\s?\d+(?:[.,]\d{2,3})*(?:[.,]\d{2})?',
            "credit_cards": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "coordinates": r'[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?)'
        }
        
        # Add custom patterns if provided
        if custom_patterns:
            patterns.update(custom_patterns)
        
        # Extract entities for each requested type using safe regex with timeout
        for entity_type in entity_types:
            if entity_type in patterns:
                try:
                    matches = _safe_regex_findall(patterns[entity_type], content, timeout=5.0)
                    if matches:
                        # Deduplicate if requested
                        if deduplicate:
                            matches = list(set(matches))
                        entities[entity_type] = matches
                except ValueError as e:
                    # Invalid regex pattern (likely from custom_patterns)
                    pattern_errors[entity_type] = f"Invalid pattern: {str(e)}"
                    entities[entity_type] = []
                except TimeoutError as e:
                    # Regex execution timed out (possible ReDoS)
                    pattern_errors[entity_type] = f"Pattern timeout: {str(e)}"
                    entities[entity_type] = []
        
        result = {
            "url": url,
            "success": True,
            "entities": entities,
            "entity_types_requested": entity_types,
            "processing_method": "regex_extraction",
            "content_length": len(content),
            "total_entities_found": sum(len(v) for v in entities.values())
        }
        
        # Include pattern errors if any occurred
        if pattern_errors:
            result["pattern_errors"] = pattern_errors
        
        return result
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Entity extraction error: {str(e)}"
        }


async def _internal_llm_extract_entities(
    url: str,
    entity_types: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    include_context: bool = True,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Internal LLM extract entities implementation using AI-powered named entity recognition.

    Supports both standard entity types (emails, phones, etc.) and advanced NER
    (people, organizations, locations, custom entities).
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(url=url, generate_markdown=True, include_cleaned_html=True)
        crawl_result = await _internal_crawl_url(request)

        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }

        # Use content if available, fallback to markdown for entity extraction
        content = crawl_result.content or crawl_result.markdown or ""
        if not content.strip():
            return {
                "url": url,
                "success": True,
                "entities": {},
                "entity_types_requested": entity_types,
                "processing_method": "llm_extraction",
                "content_length": 0,
                "total_entities_found": 0,
                "note": "No content found to extract entities from"
            }

        from ..utils.llm_extraction import LLMExtractionClient

        # Create LLM client from config
        client = LLMExtractionClient.from_config(provider, model)

        # Define entity types and their descriptions
        entity_descriptions = {
            "emails": "Email addresses (e.g., user@example.com)",
            "phones": "Phone numbers in various formats",
            "urls": "Web URLs and links",
            "dates": "Dates in various formats",
            "ips": "IP addresses",
            "prices": "Prices and monetary amounts",
            "credit_cards": "Credit card numbers",
            "coordinates": "Geographic coordinates (latitude, longitude)",
            "social_media": "Social media handles and profiles",
            "people": "Names of people, individuals, persons",
            "organizations": "Company names, institutions, organizations",
            "locations": "Places, cities, countries, geographic locations",
            "products": "Product names, brands, models",
            "events": "Events, conferences, meetings, occasions"
        }

        # Build entity types description for prompt
        requested_entities = []
        for entity_type in entity_types:
            description = entity_descriptions.get(entity_type, f"Custom entity type: {entity_type}")
            requested_entities.append(f"- {entity_type}: {description}")

        entity_types_text = "\n".join(requested_entities)

        # Prepare extraction prompt
        extraction_prompt = f"""
You are an expert entity extraction specialist. Extract all instances of the specified entity types from the given web content.

ENTITY TYPES TO EXTRACT:
{entity_types_text}

EXTRACTION INSTRUCTIONS:
- Extract ALL instances of each specified entity type from the content
- Maintain exact accuracy - extract entities exactly as they appear in the source
- For each entity type, provide a list of unique entities found
- If context is requested, include a brief surrounding text snippet for each entity
- Remove duplicates within each entity type
- If no entities of a specific type are found, return an empty list for that type
- Return results in valid JSON format

{f"ADDITIONAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}

Please provide a JSON response with the following structure:
{{
    "entities": {{
        "entity_type_1": [
            {{
                "value": "extracted_entity_text",
                "context": "surrounding text context (if requested)",
                "confidence": "High/Medium/Low"
            }}
        ],
        "entity_type_2": [...]
    }},
    "extraction_summary": {{
        "total_entities_found": number,
        "entity_types_found": ["list", "of", "types", "with", "results"],
        "entity_types_empty": ["list", "of", "types", "with", "no", "results"],
        "extraction_confidence": "High/Medium/Low"
    }}
}}

WEB CONTENT TO ANALYZE:
{content[:40000]}  # Limit content to prevent token overflow
"""

        system_message = "You are an expert entity extraction specialist focused on accuracy and comprehensive extraction."

        extracted_content = await client.call_llm(
            prompt=extraction_prompt,
            system_message=system_message,
            temperature=0.1,
            max_tokens=4000
        )

        # Parse JSON response
        if extracted_content:
            try:
                extraction_result = client.parse_json_response(extracted_content)

                # Process entities to match expected format
                processed_entities = {}
                for entity_type, entities_list in extraction_result.get("entities", {}).items():
                    if entity_type in entity_types:
                        if include_context and isinstance(entities_list, list) and entities_list:
                            # Keep full entity objects with context if requested
                            processed_entities[entity_type] = entities_list
                        else:
                            # Extract just the values if no context requested
                            if isinstance(entities_list, list):
                                values = []
                                for entity in entities_list:
                                    if isinstance(entity, dict):
                                        values.append(entity.get('value', str(entity)))
                                    else:
                                        values.append(str(entity))
                                processed_entities[entity_type] = list(set(values)) if deduplicate else values
                            else:
                                processed_entities[entity_type] = entities_list

                summary = extraction_result.get("extraction_summary", {})

                return {
                    "url": url,
                    "success": True,
                    "entities": processed_entities,
                    "entity_types_requested": entity_types,
                    "processing_method": "llm_extraction",
                    "llm_provider": client.provider,
                    "llm_model": client.model,
                    "content_length": len(content),
                    "total_entities_found": summary.get("total_entities_found", sum(len(v) for v in processed_entities.values())),
                    "extraction_confidence": summary.get("extraction_confidence", "Medium"),
                    "entity_types_found": summary.get("entity_types_found", list(processed_entities.keys())),
                    "entity_types_empty": summary.get("entity_types_empty", [et for et in entity_types if et not in processed_entities]),
                    "include_context": include_context,
                    "deduplicated": deduplicate
                }

            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback: treat as plain text
                return {
                    "url": url,
                    "success": True,
                    "entities": {"raw_extraction": [str(extracted_content)]},
                    "entity_types_requested": entity_types,
                    "processing_method": "llm_extraction_fallback",
                    "llm_provider": client.provider,
                    "llm_model": client.model,
                    "content_length": len(content),
                    "total_entities_found": 1,
                    "extraction_confidence": "Low",
                    "json_parse_error": str(e),
                    "note": f"JSON parsing failed, returned raw LLM output: {str(e)}"
                }
        else:
            return {
                "url": url,
                "success": False,
                "error": "LLM entity extraction returned empty result"
            }

    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"LLM entity extraction error: {str(e)}"
        }


async def _internal_extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Internal extract structured data implementation.
    Supports both CSS selector and LLM-based extraction methods.
    """
    try:
        # First crawl the URL to get the content
        crawl_request = CrawlRequest(
            url=request.url,
            generate_markdown=True,
            include_cleaned_html=True
        )

        crawl_result = await _internal_crawl_url(crawl_request)

        if not crawl_result.success:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Failed to crawl URL for structured extraction: {crawl_result.error}"
            )

        extracted_data = {}

        if request.extraction_type == "css" and request.css_selectors:
            # CSS selector-based extraction
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(crawl_result.content, 'html.parser')

                for field_name, css_selector in request.css_selectors.items():
                    elements = soup.select(css_selector)
                    if elements:
                        if len(elements) == 1:
                            # Single element
                            extracted_data[field_name] = elements[0].get_text(strip=True)
                        else:
                            # Multiple elements
                            extracted_data[field_name] = [elem.get_text(strip=True) for elem in elements]
                    else:
                        extracted_data[field_name] = None

                return CrawlResponse(
                    success=True,
                    url=request.url,
                    title=crawl_result.title,
                    content=crawl_result.content,
                    markdown=crawl_result.markdown,
                    extracted_data={
                        "structured_data": extracted_data,
                        "extraction_method": "css_selectors",
                        "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                        "extracted_fields": list(extracted_data.keys())
                    }
                )

            except ImportError:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="BeautifulSoup4 not installed. Install with: pip install beautifulsoup4"
                )

        elif request.extraction_type == "llm":
            # LLM-based extraction
            try:
                from ..utils.llm_extraction import LLMExtractionClient

                # Create LLM client from config
                client = LLMExtractionClient.from_config(request.llm_provider, request.llm_model)

                # Prepare schema description
                schema_description = ""
                if request.extraction_schema:
                    schema_items = []
                    for field, description in request.extraction_schema.items():
                        schema_items.append(f"- {field}: {description}")
                    schema_description = "\n".join(schema_items)

                # Prepare extraction prompt
                structured_prompt = f"""
                You are an expert data extraction specialist. Extract structured data from the given web content according to the specified schema.

                SCHEMA FIELDS TO EXTRACT:
                {schema_description}

                EXTRACTION INSTRUCTIONS:
                - Extract information for each field in the schema
                - Maintain accuracy and preserve exact information from the source
                - If a field's information is not found, set it to null
                - Return data in valid JSON format matching the schema structure
                - Focus on extracting concrete, factual information

                {f"ADDITIONAL INSTRUCTIONS: {request.instruction}" if request.instruction else ""}

                Please provide a JSON response with the following structure:
                {{
                    "structured_data": {{
                        // Fields matching the requested schema
                    }},
                    "extraction_confidence": "High/Medium/Low",
                    "found_fields": ["list", "of", "successfully", "extracted", "fields"],
                    "missing_fields": ["list", "of", "fields", "not", "found"],
                    "additional_context": "Any relevant context or notes about the extraction"
                }}

                WEB CONTENT TO ANALYZE:
                {crawl_result.content[:40000]}  # Limit content to prevent token overflow
                """

                system_message = "You are an expert data extraction specialist focused on accuracy and structured output."

                extracted_content = await client.call_llm(
                    prompt=structured_prompt,
                    system_message=system_message,
                    temperature=0.1,
                    max_tokens=4000
                )

                # Parse JSON response
                if extracted_content:
                    try:
                        extraction_result = client.parse_json_response(extracted_content)

                        return CrawlResponse(
                            success=True,
                            url=request.url,
                            title=crawl_result.title,
                            content=crawl_result.content,
                            markdown=crawl_result.markdown,
                            extracted_data={
                                "structured_data": extraction_result.get("structured_data", {}),
                                "extraction_method": "llm_based",
                                "extraction_confidence": extraction_result.get("extraction_confidence", "Medium"),
                                "found_fields": extraction_result.get("found_fields", []),
                                "missing_fields": extraction_result.get("missing_fields", []),
                                "additional_context": extraction_result.get("additional_context", ""),
                                "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "llm_provider": client.provider,
                                "llm_model": client.model,
                                "custom_instruction_used": bool(request.instruction)
                            }
                        )

                    except (json.JSONDecodeError, AttributeError) as e:
                        # Fallback: treat as plain text
                        return CrawlResponse(
                            success=True,
                            url=request.url,
                            title=crawl_result.title,
                            content=crawl_result.content,
                            markdown=crawl_result.markdown,
                            extracted_data={
                                "structured_data": {"raw_extraction": str(extracted_content)},
                                "extraction_method": "llm_based_fallback",
                                "extraction_confidence": "Low",
                                "found_fields": [],
                                "missing_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "additional_context": f"JSON parsing failed: {str(e)}",
                                "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "llm_provider": client.provider,
                                "llm_model": client.model,
                                "json_parse_error": str(e)
                            }
                        )
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error="LLM structured extraction returned empty result"
                    )

            except Exception as llm_error:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"LLM structured extraction failed: {str(llm_error)}"
                )

        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Unsupported extraction type: {request.extraction_type}. Supported types: 'css', 'llm'"
            )

    except Exception as e:
        return CrawlResponse(
            success=False,
            url=request.url,
            error=f"Structured data extraction error: {str(e)}"
        )


# MCP Tool implementations
async def crawl_url(
    url: Annotated[str, Field(description="Target URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    include_cleaned_html: Annotated[bool, Field(description="Include cleaned HTML in content field (default: False)")] = False,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Maximum crawling depth, None for single page (default: None)")] = None,
    max_pages: Annotated[Optional[int], Field(description="Maximum number of pages to crawl (default: 10)")] = 10,
    include_external: Annotated[bool, Field(description="Whether to follow external domain links (default: False)")] = False,
    crawl_strategy: Annotated[str, Field(description="Crawling strategy: 'bfs', 'dfs', or 'best_first' (default: 'bfs')")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL pattern filter e.g. '*docs*' (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum score for URLs to be crawled (default: 0.3)")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="Content filter: 'bm25', 'pruning', 'llm' (default: None)")] = None,
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 content filtering (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Whether to chunk large content (default: False)")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunking strategy: 'topic', 'regex', 'sentence' (default: 'topic')")] = "topic",
    chunk_size: Annotated[int, Field(description="Maximum chunk size in tokens (default: 1000)")] = 1000,
    overlap_rate: Annotated[float, Field(description="Overlap rate between chunks 0.0-1.0 (default: 0.1)")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="Custom user agent string (default: None)")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="Custom HTTP headers (default: None)")] = None,
    enable_caching: Annotated[bool, Field(description="Whether to enable caching (default: True)")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode: 'enabled', 'disabled', 'bypass' (default: 'enabled')")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JavaScript code to execute (default: None)")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate human-like browsing behavior (default: False)")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Use undetected browser mode to bypass bot detection (default: False)")] = False,
    auth_token: Annotated[Optional[str], Field(description="Authentication token (default: None)")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Custom cookies (default: None)")] = None,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None
) -> CrawlResponse:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.

    Core web crawling tool with comprehensive configuration options.
    Essential for SPAs: set wait_for_js=true for JavaScript-heavy sites.

    By default, returns markdown content only. Set include_cleaned_html=True to also
    receive the cleaned HTML content field for scenarios requiring both formats.
    """
    # Create CrawlRequest object from individual parameters
    request = CrawlRequest(
        url=url,
        css_selector=css_selector,
        extract_media=extract_media,
        take_screenshot=take_screenshot,
        generate_markdown=generate_markdown,
        include_cleaned_html=include_cleaned_html,
        wait_for_selector=wait_for_selector,
        timeout=timeout,
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        crawl_strategy=crawl_strategy,
        url_pattern=url_pattern,
        score_threshold=score_threshold,
        content_filter=content_filter,
        filter_query=filter_query,
        chunk_content=chunk_content,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap_rate=overlap_rate,
        user_agent=user_agent,
        headers=headers,
        enable_caching=enable_caching,
        cache_mode=cache_mode,
        execute_js=execute_js,
        wait_for_js=wait_for_js,
        simulate_user=simulate_user,
        use_undetected_browser=use_undetected_browser,
        auth_token=auth_token,
        cookies=cookies,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        summary_length=summary_length,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    return await _internal_crawl_url(request)


async def deep_crawl_site(
    url: Annotated[str, Field(description="Starting URL")],
    max_depth: Annotated[int, Field(description="Link depth to follow")] = 2,
    max_pages: Annotated[int, Field(description="Max pages (max: 10)")] = 5,
    crawl_strategy: Annotated[str, Field(description="'bfs'|'dfs'|'best_first'")] = "bfs",
    include_external: Annotated[bool, Field(description="Follow external links")] = False,
    url_pattern: Annotated[Optional[str], Field(description="URL wildcard filter")] = None,
    score_threshold: Annotated[float, Field(description="Min relevance 0-1")] = 0.0,
    extract_media: Annotated[bool, Field(description="Extract media")] = False,
    base_timeout: Annotated[int, Field(description="Timeout per page (sec)")] = 60
) -> Dict[str, Any]:
    """Crawl multiple pages from a site with configurable depth."""
    # Use crawl_url with deep crawling enabled
    request = CrawlRequest(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        crawl_strategy=crawl_strategy,
        include_external=include_external,
        url_pattern=url_pattern,
        score_threshold=score_threshold,
        extract_media=extract_media,
        timeout=base_timeout,
        generate_markdown=True
    )
    
    result = await _internal_crawl_url(request)
    
    # Convert CrawlResponse to dict for consistency with legacy API
    return {
        "success": result.success,
        "url": result.url,
        "title": result.title,
        "content": result.content,
        "markdown": result.markdown,
        "media": result.media,
        "extracted_data": result.extracted_data,
        "error": result.error,
        "processing_method": "deep_crawling"
    }


async def intelligent_extract(
    url: Annotated[str, Field(description="URL to extract from")],
    extraction_goal: Annotated[str, Field(description="What data to extract")],
    content_filter: Annotated[str, Field(description="'bm25'|'pruning'|'llm'")] = "bm25",
    filter_query: Annotated[Optional[str], Field(description="BM25 keywords")] = None,
    chunk_content: Annotated[bool, Field(description="Split large content")] = False,
    use_llm: Annotated[bool, Field(description="Use LLM")] = True,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
    llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
    custom_instructions: Annotated[Optional[str], Field(description="Extra LLM guidance")] = None
) -> Dict[str, Any]:
    """Extract specific data from web pages using LLM."""
    return await _internal_intelligent_extract(
        url=url,
        extraction_goal=extraction_goal,
        content_filter=content_filter,
        filter_query=filter_query,
        chunk_content=chunk_content,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
        custom_instructions=custom_instructions
    )


async def extract_entities(
    url: Annotated[str, Field(description="URL to extract from")],
    entity_types: Annotated[List[str], Field(description="Types: emails|phones|urls|dates|ips|prices|social_media")],
    custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns")] = None,
    include_context: Annotated[bool, Field(description="Include surrounding text")] = True,
    deduplicate: Annotated[bool, Field(description="Remove duplicates")] = True,
    use_llm: Annotated[bool, Field(description="Use LLM for NER")] = False,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
    llm_model: Annotated[Optional[str], Field(description="LLM model")] = None
) -> Dict[str, Any]:
    """Extract entities (emails, phones, etc.) from web pages."""
    if use_llm:
        # Use LLM-based extraction for all entity types when requested
        return await _internal_llm_extract_entities(
            url=url,
            entity_types=entity_types,
            provider=llm_provider,
            model=llm_model,
            custom_instructions=None,  # Could be added as parameter in future
            include_context=include_context,
            deduplicate=deduplicate
        )
    else:
        # Use regex-based extraction
        return await _internal_extract_entities(
            url=url,
            entity_types=entity_types,
            custom_patterns=custom_patterns,
            include_context=include_context,
            deduplicate=deduplicate
        )


async def extract_structured_data(
    request: Annotated[StructuredExtractionRequest, Field(description="Extraction request with URL and schema")]
) -> CrawlResponse:
    """Extract structured data using CSS selectors or LLM."""
    return await _internal_extract_structured_data(request)


async def crawl_url_with_fallback(
    url: Annotated[str, Field(description="URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector")] = None,
    extract_media: Annotated[bool, Field(description="Extract media")] = False,
    take_screenshot: Annotated[bool, Field(description="Take screenshot")] = False,
    generate_markdown: Annotated[bool, Field(description="Generate markdown")] = True,
    include_cleaned_html: Annotated[bool, Field(description="Include HTML")] = False,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for element")] = None,
    timeout: Annotated[int, Field(description="Timeout (sec)")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Crawl depth")] = None,
    max_pages: Annotated[Optional[int], Field(description="Max pages")] = 10,
    include_external: Annotated[bool, Field(description="Follow external")] = False,
    crawl_strategy: Annotated[str, Field(description="'bfs'|'dfs'|'best_first'")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL filter")] = None,
    score_threshold: Annotated[float, Field(description="Min score")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="'bm25'|'pruning'|'llm'")] = None,
    filter_query: Annotated[Optional[str], Field(description="Filter keywords")] = None,
    chunk_content: Annotated[bool, Field(description="Chunk content")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunk strategy")] = "topic",
    chunk_size: Annotated[int, Field(description="Chunk tokens")] = 1000,
    overlap_rate: Annotated[float, Field(description="Chunk overlap")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="User agent")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="HTTP headers")] = None,
    enable_caching: Annotated[bool, Field(description="Enable cache")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JS to execute")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JS")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate user")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Bypass bot detection")] = False,
    auth_token: Annotated[Optional[str], Field(description="Auth token")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Cookies")] = None,
    auto_summarize: Annotated[bool, Field(description="Auto summarize")] = False,
    max_content_tokens: Annotated[int, Field(description="Max tokens")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
    llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
    # Phase 6: Session management
    use_session: Annotated[bool, Field(description="Use stored session")] = False,
    save_session: Annotated[bool, Field(description="Save session on success")] = False,
    session_ttl_hours: Annotated[int, Field(description="Session TTL in hours")] = 24,
    # Phase 7: Strategy caching
    use_strategy_cache: Annotated[bool, Field(description="Use cached strategy for domain")] = True,
    save_strategy: Annotated[bool, Field(description="Save successful strategy")] = True,
    strategy_ttl_days: Annotated[int, Field(description="Strategy cache TTL in days")] = 7,
    # Phase 8: Fingerprint evasion
    use_stealth_mode: Annotated[bool, Field(description="Enable fingerprint evasion")] = False,
    fingerprint_profile: Annotated[Optional[str], Field(description="Browser profile: chrome_windows|chrome_mac|firefox_windows|safari_mac|chrome_mobile|safari_mobile")] = None
) -> CrawlResponse:
    """Crawl with multiple fallback strategies for anti-bot sites.

    Phase 3: Multi-stage fallback with 7 stages:
    1. Static fast path - HTTP fetch without browser
    2. Normal headless - Minimal browser overhead
    3. Chromium + stealth - JS wait with realistic behavior
    4. User behavior - Scroll, click simulation
    5. Mobile agent - iOS Safari user agent
    6. AMP/RSS - Alternative sources (AMP pages, RSS feeds)
    7. JSON extraction - Extract from __NEXT_DATA__ etc.

    Phase 6: Session persistence
    - use_session: Load stored cookies/localStorage for the domain
    - save_session: Save session data on successful crawl
    - session_ttl_hours: Session expiration time (default 24 hours)

    Phase 7: Strategy caching
    - use_strategy_cache: Start from the best known strategy for domain
    - save_strategy: Record successful strategy for future use
    - strategy_ttl_days: Cache expiration time (default 7 days)

    Phase 8: Fingerprint evasion
    - use_stealth_mode: Enable browser fingerprint evasion scripts
    - fingerprint_profile: Specific browser profile to emulate
    - save_strategy: Record successful strategy for future use
    - strategy_ttl_days: Cache expiration time (default 7 days)
    """
    import asyncio
    import random
    from urllib.parse import urlparse

    # Detect site characteristics for optimized strategy selection
    domain = urlparse(url).netloc.lower()
    is_hn = 'ycombinator.com' in domain
    is_reddit = 'reddit.com' in domain
    is_social_media = any(site in domain for site in ['twitter.com', 'facebook.com', 'linkedin.com'])
    is_news_site = any(site in domain for site in ['cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com'])

    # ========================================
    # Phase 6: Session Management
    # ========================================
    session_manager = get_session_manager() if (use_session or save_session) else None
    stored_session = None
    session_cookies = None

    if use_session and session_manager:
        stored_session = session_manager.get_session(url)
        if stored_session:
            print(f"Session loaded for domain: {domain}")
            # Extract cookies from stored session
            session_cookies = stored_session.get("cookies", [])
            # Convert to dict format if needed for merging with user cookies
            if session_cookies and not cookies:
                # Use session cookies directly
                cookies = {c["name"]: c["value"] for c in session_cookies if "name" in c and "value" in c}
            elif session_cookies and cookies:
                # Merge: user cookies take precedence
                merged = {c["name"]: c["value"] for c in session_cookies if "name" in c and "value" in c}
                merged.update(cookies)
                cookies = merged

    def _save_session_on_success(result: CrawlResponse) -> CrawlResponse:
        """Save session data after successful crawl if save_session is enabled."""
        if save_session and session_manager and result.success:
            # Try to extract cookies from result or use provided cookies
            cookies_to_save = []
            parsed = urlparse(url)

            # Check if result has cookies from crawl (in extracted_data)
            if result.extracted_data and isinstance(result.extracted_data, dict):
                crawl_cookies = result.extracted_data.get("cookies", [])
                if crawl_cookies and isinstance(crawl_cookies, list):
                    for cookie in crawl_cookies:
                        if isinstance(cookie, dict) and "name" in cookie and "value" in cookie:
                            cookies_to_save.append({
                                "name": cookie["name"],
                                "value": cookie["value"],
                                "domain": cookie.get("domain", parsed.netloc),
                                "path": cookie.get("path", "/"),
                                "httpOnly": cookie.get("httpOnly", False),
                                "secure": cookie.get("secure", parsed.scheme == "https"),
                                "sameSite": cookie.get("sameSite", "Lax")
                            })

            # Add user-provided cookies (these take precedence)
            if cookies:
                existing_names = {c["name"] for c in cookies_to_save}
                for name, value in cookies.items():
                    if name not in existing_names:
                        cookies_to_save.append({
                            "name": name,
                            "value": value,
                            "domain": parsed.netloc,
                            "path": "/",
                            "httpOnly": False,
                            "secure": parsed.scheme == "https",
                            "sameSite": "Lax"
                        })
                    else:
                        # Update existing cookie with user value
                        for c in cookies_to_save:
                            if c["name"] == name:
                                c["value"] = value
                                break

            if cookies_to_save:
                session_manager.save_session(
                    url=url,
                    cookies=cookies_to_save,
                    ttl_hours=session_ttl_hours
                )
                print(f"Session saved for domain: {domain}")

                # Add session info to extracted_data
                if result.extracted_data is None:
                    result.extracted_data = {}
                result.extracted_data["session_saved"] = True
                result.extracted_data["session_domain"] = domain
                result.extracted_data["session_cookies_count"] = len(cookies_to_save)

        return result

    # ========================================
    # Phase 7: Strategy Caching
    # ========================================
    strategy_cache = get_strategy_cache() if (use_strategy_cache or save_strategy) else None
    cached_strategy = None
    recommended_stages = [1, 2, 3, 4, 5, 6, 7]  # Default order

    if use_strategy_cache and strategy_cache:
        cached_strategy = strategy_cache.get_best_strategy(url)
        if cached_strategy:
            recommended_stages = strategy_cache.get_recommended_stages(url)
            print(f"Strategy cache hit for {domain}: best_stage={cached_strategy.get('start_stage')}, "
                  f"skip={cached_strategy.get('skip_stages', [])}, "
                  f"success_count={cached_strategy.get('success_count', 0)}")

    def _record_strategy_success(stage: int, strategy_name: str, response_time: float = None):
        """Record successful strategy to cache."""
        if save_strategy and strategy_cache:
            strategy_cache.record_success(
                url=url,
                stage=stage,
                strategy_name=strategy_name,
                response_time=response_time,
                ttl_days=strategy_ttl_days
            )
            print(f"Strategy recorded: stage={stage}, strategy={strategy_name}")

    def _record_strategy_failure(stage: int, strategy_name: str, error: str = None):
        """Record failed strategy to cache."""
        if save_strategy and strategy_cache:
            strategy_cache.record_failure(
                url=url,
                stage=stage,
                strategy_name=strategy_name,
                error=error
            )

    # ========================================
    # Phase 8: Fingerprint Evasion
    # ========================================
    stealth_config = None
    stealth_js = None
    stealth_user_agent = None
    stealth_headers = None

    if use_stealth_mode:
        stealth_config = get_fingerprint_config(fingerprint_profile)
        # Validate stealth_config and fallback to default if invalid
        if not stealth_config or not isinstance(stealth_config, dict):
            stealth_config = get_fingerprint_config("chrome_win")  # Default fallback
        # Ensure required keys exist with defaults
        stealth_js = stealth_config.get("stealth_js", "")
        stealth_user_agent = stealth_config.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        stealth_headers = stealth_config.get("headers", {})
        print(f"Stealth mode enabled: profile={stealth_config.get('profile_name', 'fallback')}")

        # IMPORTANT: For consistency, stealth mode always uses profile's UA and headers
        # User-provided values are ignored to maintain fingerprint integrity
        # (UA, headers, and JS must all match the same profile)
        if user_agent and user_agent != stealth_user_agent:
            print(f"  Warning: user_agent ignored in stealth mode for consistency")
        user_agent = stealth_user_agent

        if headers:
            # Only merge non-fingerprint headers (e.g., auth headers)
            # Fingerprint-related headers from profile take precedence
            user_non_fingerprint = {k: v for k, v in headers.items()
                                    if not k.lower().startswith(('sec-ch-', 'accept', 'user-agent'))}
            stealth_headers = {**stealth_headers, **user_non_fingerprint}
        headers = stealth_headers

    # Common user agents for different strategies
    realistic_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]

    realistic_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }

    # Site-specific CSS selector optimization
    if is_hn and not css_selector:
        css_selector = ".fatitem, .athing, .comtr"

    # ========================================
    # Stage 1: Static fast path (no browser)
    # ========================================
    # Inject auth_token into headers for static fetch (if provided)
    # Only add if not already set (preserve user-specified Authorization, case-insensitive)
    static_fetch_headers = dict(headers) if headers else {}
    if auth_token:
        has_auth = any(k.lower() == "authorization" for k in static_fetch_headers)
        if not has_auth:
            static_fetch_headers["Authorization"] = f"Bearer {auth_token}"

    print(f"Stage 1/7: Attempting static HTTP fetch for {url}")
    static_success, static_html, static_error = await _static_fetch_content(
        url, headers=static_fetch_headers, timeout=min(timeout, 15)
    )

    if static_success and static_html:
        # Step 1: Try JSON extraction FIRST (works for Next.js, Nuxt, etc. even in SPA mode)
        json_success, json_data, json_source = _extract_spa_json_data(static_html)
        if json_success and json_data:
            # Check for block page indicators in the raw HTML
            if _is_block_page(static_html):
                print("  Block page detected in static HTML, skipping JSON extraction")
            else:
                # Record success for Phase 7
                _record_strategy_success(1, "static_json_extraction")
                # Return JSON extracted data with size limit check
                json_response = await _build_json_extraction_response(
                    json_data=json_data,
                    json_source=json_source,
                    url=url,
                    strategy_name="static_json_extraction",
                    stage=1,
                    auto_summarize=auto_summarize,
                    max_content_tokens=max_content_tokens,
                    llm_provider=llm_provider,
                    llm_model=llm_model
                )
                return _save_session_on_success(json_response)

        # Step 2: Check for SPA indicators
        spa_framework, spa_selector = _detect_spa_framework(static_html)

        if spa_framework:
            # SPA detected - proceed to browser-based stages
            print(f"  SPA detected ({spa_framework}), proceeding to browser-based stages")
            if not wait_for_selector and spa_selector:
                wait_for_selector = spa_selector
        else:
            # Step 3: Not SPA - try HTML extraction if content is substantial
            if len(static_html) > 5000 and '<noscript>' not in static_html.lower():
                # Use crawl4ai for HTML to markdown conversion only
                try:
                    result = await crawl_url(
                        url=url,
                        css_selector=css_selector,
                        generate_markdown=generate_markdown,
                        include_cleaned_html=include_cleaned_html,
                        timeout=timeout,
                        wait_for_js=False,
                        cache_mode="enabled"
                    )
                    has_content, content_source = _has_meaningful_content(result, min_length=100)
                    if has_content:
                        # Check for block page indicators
                        content_text = " ".join([
                            result.markdown or "",
                            result.content or "",
                            getattr(result, 'raw_content', None) or ""
                        ])
                        if _is_block_page(content_text):
                            print("  Block page detected in static fast path, skipping")
                        else:
                            # Record success for Phase 7
                            _record_strategy_success(1, "static_fast_path")
                            if result.extracted_data is None:
                                result.extracted_data = {}
                            result.extracted_data.update({
                                "fallback_strategy_used": "static_fast_path",
                                "fallback_stage": 1,
                                "content_source": content_source
                            })
                            return _save_session_on_success(result)
                except Exception:
                    pass

    # ========================================
    # Build browser-based fallback strategies
    # ========================================
    strategies = []

    # Stage 2: Normal headless (minimal browser overhead)
    # Phase 8: In stealth mode, use profile UA/headers/JS for consistency
    stage2_ua = user_agent if use_stealth_mode else (user_agent or realistic_user_agents[0])
    stage2_headers = headers if use_stealth_mode else (headers or {})
    stage2_execute_js = stealth_js if use_stealth_mode else None
    strategies.append({
        "name": "normal_headless",
        "stage": 2,
        "params": {
            "user_agent": stage2_ua,
            "headers": stage2_headers,
            "wait_for_js": False,
            "simulate_user": False,
            "timeout": min(timeout, 20),
            "css_selector": css_selector,
            "wait_for_selector": None,
            "cache_mode": "enabled",
            "execute_js": stage2_execute_js
        }
    })

    # Stage 3: Chromium + stealth (JS wait with realistic behavior)
    # Phase 8: Add stealth JS if enabled and use consistent UA/headers
    stage3_execute_js = stealth_js if use_stealth_mode else None
    stage3_ua = user_agent if use_stealth_mode else (user_agent or random.choice(realistic_user_agents))
    stage3_headers = headers if use_stealth_mode else {**realistic_headers, **(headers or {})}
    strategies.append({
        "name": "chromium_stealth",
        "stage": 3,
        "params": {
            "user_agent": stage3_ua,
            "headers": stage3_headers,
            "wait_for_js": True,
            "simulate_user": False,
            "timeout": max(timeout, 30),
            "css_selector": css_selector,
            "wait_for_selector": wait_for_selector or (".fatitem" if is_hn else None),
            "cache_mode": "bypass",
            "execute_js": stage3_execute_js
        }
    })

    # Stage 4: User behavior simulation (scroll, click)
    # Phase 8: In stealth mode, keep same UA/headers as Stage 3 for consistency
    stage4_behavior_js = """
        // Simulate human-like interaction
        window.scrollTo(0, document.body.scrollHeight / 3);
        await new Promise(r => setTimeout(r, 500));
        window.scrollTo(0, document.body.scrollHeight / 2);
        await new Promise(r => setTimeout(r, 500));
        window.scrollTo(0, 0);
    """ if is_hn or is_social_media else (execute_js or "")
    stage4_execute_js = (stealth_js + "\n" + stage4_behavior_js) if use_stealth_mode and stealth_js else stage4_behavior_js
    # In stealth mode, use same UA as Stage 3; otherwise pick different UA
    stage4_ua = user_agent if use_stealth_mode else random.choice([ua for ua in realistic_user_agents if ua != strategies[-1]["params"]["user_agent"]])
    stage4_headers = headers if use_stealth_mode else {**realistic_headers, **(headers or {})}
    strategies.append({
        "name": "user_behavior",
        "stage": 4,
        "params": {
            "user_agent": stage4_ua,
            "headers": stage4_headers,
            "wait_for_js": True,
            "simulate_user": True,
            "timeout": timeout + 30,
            "css_selector": css_selector,
            "execute_js": stage4_execute_js if stage4_execute_js.strip() else None,
            "cache_mode": "disabled"
        }
    })

    # Stage 5: Mobile user agent
    # Phase 8: Use safari_mobile profile for consistency (UA is iOS Safari)
    mobile_stealth_js = None
    mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    mobile_headers = {**realistic_headers}
    # Safari doesn't send Sec-CH-UA headers, so don't include them
    if use_stealth_mode:
        mobile_config = get_fingerprint_config("safari_mobile")  # Use matching Safari profile
        mobile_stealth_js = mobile_config["stealth_js"]
        mobile_ua = mobile_config["user_agent"]
        mobile_headers = mobile_config["headers"]
    strategies.append({
        "name": "mobile_agent",
        "stage": 5,
        "params": {
            "user_agent": mobile_ua,
            "headers": mobile_headers,
            "wait_for_js": True,
            "simulate_user": False,
            "timeout": timeout,
            "css_selector": css_selector,
            "cache_mode": "bypass",
            "execute_js": mobile_stealth_js
        }
    })

    last_error = None
    total_stages = 7  # Stages: 1=static, 2-5=browser, 6=AMP/RSS, 7=JSON

    # ========================================
    # Phase 7: Apply strategy cache to execution order
    # ========================================
    if use_strategy_cache and cached_strategy:
        skip_stages = set(cached_strategy.get("skip_stages", []))
        start_stage = cached_strategy.get("start_stage", 1)

        # Sort strategies based on recommended order
        # Prioritize the cached best stage, skip known failed stages
        def strategy_sort_key(s):
            stage = s.get("stage", 99)
            if stage in skip_stages:
                return 1000 + stage  # Move failed stages to end
            if stage == start_stage:
                return 0  # Prioritize best known stage
            # Maintain relative order for others based on recommended_stages
            try:
                return recommended_stages.index(stage)
            except ValueError:
                return 500 + stage

        strategies.sort(key=strategy_sort_key)

        # Log the adjusted order
        adjusted_order = [s.get("stage") for s in strategies]
        print(f"Strategy order adjusted by cache: {adjusted_order} (skip: {list(skip_stages)})")

    for i, strategy in enumerate(strategies):
        try:
            # Add random delay between attempts (1-5 seconds)
            if i > 0:
                delay = random.uniform(1, 5)
                await asyncio.sleep(delay)

            stage_num = strategy.get("stage", i + 2)  # Stage 1 was static fetch
            print(f"Stage {stage_num}/{total_stages}: Attempting {strategy['name']}")

            # Start timing for response_time tracking
            import time as time_module
            stage_start_time = time_module.time()

            # Prepare strategy-specific parameters
            strategy_params = {
                "url": url,
                "css_selector": strategy["params"].get("css_selector", css_selector),
                "extract_media": extract_media,
                "take_screenshot": take_screenshot,
                "generate_markdown": generate_markdown,
                "include_cleaned_html": include_cleaned_html,
                "wait_for_selector": strategy["params"].get("wait_for_selector", wait_for_selector),
                "timeout": strategy["params"].get("timeout", timeout),
                "max_depth": None,  # Force single page to avoid deep crawling issues
                "max_pages": max_pages,
                "include_external": include_external,
                "crawl_strategy": crawl_strategy,
                "url_pattern": url_pattern,
                "score_threshold": score_threshold,
                "content_filter": content_filter,
                "filter_query": filter_query,
                "chunk_content": chunk_content,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "overlap_rate": overlap_rate,
                "user_agent": strategy["params"].get("user_agent", user_agent),
                "headers": strategy["params"].get("headers", headers),
                "enable_caching": enable_caching,
                "cache_mode": strategy["params"].get("cache_mode", cache_mode),
                "execute_js": strategy["params"].get("execute_js", execute_js),
                "wait_for_js": strategy["params"].get("wait_for_js", wait_for_js),
                "simulate_user": strategy["params"].get("simulate_user", simulate_user),
                "use_undetected_browser": use_undetected_browser,
                "auth_token": auth_token,
                "cookies": cookies,
                "auto_summarize": auto_summarize,
                "max_content_tokens": max_content_tokens,
                "summary_length": summary_length,
                "llm_provider": llm_provider,
                "llm_model": llm_model
            }
            
            # Attempt crawl with current strategy
            result = await crawl_url(**strategy_params)

            # Check explicit failure first - preserve actual error message
            if not result.success:
                actual_error = getattr(result, 'error', None) or "Unknown error"
                last_error = f"Strategy {strategy['name']}: {actual_error}"
                # Record failure for Phase 7
                _record_strategy_failure(stage_num, strategy["name"], actual_error)
                continue

            # Check if we got meaningful content (markdown, content, or raw_content)
            has_content, content_source = _has_meaningful_content(
                result, min_length=FALLBACK_MIN_CONTENT_LENGTH
            )

            # Additional check for block pages (check all fields, not just first non-empty)
            if has_content:
                raw_content = getattr(result, 'raw_content', None) or ""
                # Concatenate all fields to ensure block indicators aren't missed
                content_text = " ".join([
                    result.markdown or "",
                    result.content or "",
                    raw_content
                ]).lower()
                if _is_block_page(content_text):
                    has_content = False
                    print(f"  Block page detected, skipping strategy {strategy['name']}")
                    _record_strategy_failure(stage_num, strategy["name"], "block_page_detected")

            if has_content:
                # Calculate response time and record success for Phase 7
                response_time = time_module.time() - stage_start_time
                _record_strategy_success(stage_num, strategy["name"], response_time)
                # Add strategy info to extracted_data
                if result.extracted_data is None:
                    result.extracted_data = {}
                result.extracted_data.update({
                    "fallback_strategy_used": strategy["name"],
                    "fallback_stage": strategy.get("stage", i + 2),
                    "total_stages": total_stages,
                    "site_type_detected": "hackernews" if is_hn else "reddit" if is_reddit else "social_media" if is_social_media else "news" if is_news_site else "general",
                    "content_source": content_source
                })
                return _save_session_on_success(result)

            last_error = f"Strategy {strategy['name']}: No meaningful content in markdown/content/raw_content"
            # Record failure for Phase 7 (no content)
            _record_strategy_failure(stage_num, strategy["name"], "no_meaningful_content")

        except Exception as e:
            last_error = f"Strategy {strategy['name']}: {str(e)}"
            print(f"Strategy {strategy['name']} failed: {e}")
            # Record failure for Phase 7
            _record_strategy_failure(strategy.get("stage", i + 2), strategy["name"], str(e))
            continue

    # ========================================
    # Stage 5 Alternative: AMP/RSS fallback
    # ========================================
    print(f"Stage 6/{total_stages}: Attempting AMP/RSS fallback")

    # Try AMP version
    amp_url = _build_amp_url(url)
    if amp_url:
        try:
            amp_result = await crawl_url(
                url=amp_url,
                css_selector=css_selector,
                generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html,
                timeout=min(timeout, 20),
                wait_for_js=False
            )
            has_content, content_source = _has_meaningful_content(amp_result, min_length=100)
            if has_content:
                # Check for block page indicators
                amp_content_text = " ".join([
                    amp_result.markdown or "",
                    amp_result.content or "",
                    getattr(amp_result, 'raw_content', None) or ""
                ])
                if _is_block_page(amp_content_text):
                    print("  Block page detected in AMP result, skipping")
                else:
                    # Record success for Phase 7
                    _record_strategy_success(6, "amp_page")
                    if amp_result.extracted_data is None:
                        amp_result.extracted_data = {}
                    amp_result.extracted_data.update({
                        "fallback_strategy_used": "amp_page",
                        "fallback_stage": 6,
                        "original_url": url,
                        "amp_url_used": amp_url,
                        "content_source": content_source
                    })
                    return _save_session_on_success(amp_result)
        except Exception as e:
            print(f"AMP fallback failed: {e}")
            _record_strategy_failure(6, "amp_page", str(e))

    # Try RSS/Atom feed
    try:
        rss_success, feed_url, feed_items = await _try_fetch_rss_feed(url)
        if rss_success and feed_items:
            # Format RSS items as markdown
            markdown_content = f"# RSS Feed Content\n\nFeed URL: {feed_url}\n\n"
            for item in feed_items[:20]:  # Limit to 20 items
                if item.get('title'):
                    markdown_content += f"## {item['title']}\n"
                if item.get('link'):
                    markdown_content += f"[Link]({item['link']})\n\n"
                if item.get('description'):
                    markdown_content += f"{item['description']}\n\n"
                markdown_content += "---\n\n"

            # Check for block page indicators in RSS content
            if _is_block_page(markdown_content):
                print("  Block page detected in RSS content, skipping")
            else:
                # Record success for Phase 7
                _record_strategy_success(6, "rss_feed")
                rss_response = CrawlResponse(
                    success=True,
                    url=url,
                    markdown=markdown_content,
                    content=json.dumps(feed_items, ensure_ascii=False),
                    extracted_data={
                        "fallback_strategy_used": "rss_feed",
                        "fallback_stage": 6,
                        "feed_url": feed_url,
                        "item_count": len(feed_items),
                        "content_source": "rss"
                    }
                )
                rss_response = await _finalize_fallback_response(
                    rss_response, url, auto_summarize, max_content_tokens, llm_provider, llm_model
                )
                return _save_session_on_success(rss_response)
    except Exception as e:
        print(f"RSS fallback failed: {e}")
        _record_strategy_failure(6, "rss_feed", str(e))

    # ========================================
    # Stage 7: JSON extraction from cached static HTML
    # ========================================
    print(f"Stage 7/{total_stages}: Attempting JSON extraction from static HTML")

    if static_success and static_html:
        json_success, json_data, json_source = _extract_spa_json_data(static_html)
        if json_success and json_data:
            # Record success for Phase 7
            _record_strategy_success(7, "json_extraction")
            # Use priority keys for content extraction
            stage7_response = await _build_json_extraction_response(
                json_data=json_data,
                json_source=json_source,
                url=url,
                strategy_name="json_extraction",
                stage=7,
                auto_summarize=auto_summarize,
                max_content_tokens=max_content_tokens,
                llm_provider=llm_provider,
                llm_model=llm_model,
                extract_content_keys=['props', 'pageProps', 'data', 'content', 'article', 'post']
            )
            return _save_session_on_success(stage7_response)

    # Record failure for Stage 7 if static HTML exists but JSON extraction failed
    if static_success and static_html:
        _record_strategy_failure(7, "json_extraction", "no_json_data")

    # All strategies failed, return error with details
    all_strategies = ["static_fast_path"] + [s["name"] for s in strategies] + ["amp_page", "rss_feed", "json_extraction"]
    return CrawlResponse(
        success=False,
        url=url,
        error=f"All {total_stages} fallback stages failed. Last error: {last_error}. "
              f"This site may have strong anti-bot protection. "
              f"Stages attempted: {', '.join(all_strategies)}",
        extracted_data={
            "fallback_strategies_attempted": all_strategies,
            "total_stages": total_stages,
            "site_type_detected": "hackernews" if is_hn else "reddit" if is_reddit else "social_media" if is_social_media else "news" if is_news_site else "general",
            "static_fetch_result": "success" if static_success else f"failed: {static_error}",
            "recommendations": [
                "Try accessing the site manually to check if it's available",
                "Consider using the site's API if available",
                "Try accessing during off-peak hours",
                "Use a VPN if the site is geo-blocked",
                "Check if the site has a mobile or AMP version"
            ]
        }
    )


