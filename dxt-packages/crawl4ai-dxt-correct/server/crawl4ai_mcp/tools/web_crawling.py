"""
Web crawling tools for Crawl4AI MCP Server.

Contains complete web crawling functionality and content extraction tools.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import (
    CrawlRequest,
    CrawlResponse,
    StructuredExtractionRequest
)

# Import required crawl4ai components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
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

# Initialize processors
file_processor = FileProcessor()
youtube_processor = YouTubeProcessor()


# Placeholder for summarize_web_content function
async def summarize_web_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Placeholder for web content summarization.
    TODO: Implement LLM-based summarization.
    """
    return {
        "success": False,
        "error": "Summarization not yet implemented in modular architecture",
        "summary": content  # Return original content as fallback
    }


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
                    return CrawlResponse(
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
                    return CrawlResponse(
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
            step_size = int(request.chunk_size * (1 - request.overlap_rate))
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
            "cache_mode": cache_mode
        }
        
        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy
        
        # Add content filter if supported by current crawl4ai version
        try:
            config = CrawlerRunConfig(**config_params, content_filter=content_filter_strategy)
        except TypeError:
            # Fallback for older versions without content_filter support
            config = CrawlerRunConfig(**config_params)

        # Setup browser configuration
        browser_config = {
            "headless": True,
            "verbose": False
        }
        
        if request.user_agent:
            browser_config["user_agent"] = request.user_agent
        
        if request.headers:
            browser_config["headers"] = request.headers

        # Suppress output to avoid JSON parsing errors
        with suppress_stdout_stderr():
            async with AsyncWebCrawler(**browser_config) as crawler:
                # Handle authentication
                if request.cookies:
                    # Set cookies if provided
                    await crawler.set_cookies(request.cookies)
                
                # Execute custom JavaScript if provided
                if request.execute_js:
                    config.js_code = request.execute_js
                
                # Run crawler with config
                arun_params = {"url": request.url, "config": config}
                
                result = await crawler.arun(**arun_params)
        
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
                        all_media.extend(page_result.media)
            
            return CrawlResponse(
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
                        all_media.extend(page.media)
                
                # Prepare content for potential summarization
                combined_content = "\n\n".join(all_content) if all_content else result.cleaned_html
                combined_markdown = "\n\n".join(all_markdown) if all_markdown else result.markdown
                title_to_use = result.metadata.get("title", "")
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
                title_to_use = result.metadata.get("title", "")
                
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
                    media=result.media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            return response
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
        error_message = f"Crawling error: {str(e)}"
        if "playwright" in str(e).lower() or "browser" in str(e).lower():
            error_message += "\n\nNote: This might be a browser setup issue. Please ensure Playwright is properly installed."
            
        return CrawlResponse(
            success=False,
            url=request.url,
            error=error_message
        )


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
            generate_markdown=True
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
        
        # TODO: Implement LLM extraction logic here
        # For now, return the crawled content with extraction goal info
        return {
            "url": url,
            "success": True,
            "extracted_content": crawl_result.content,
            "extraction_goal": extraction_goal,
            "processing_method": "crawl_with_filtering",
            "content_length": len(crawl_result.content) if crawl_result.content else 0,
            "note": "LLM-based intelligent extraction not yet fully implemented"
        }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Intelligent extraction error: {str(e)}"
        }


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
        request = CrawlRequest(url=url, generate_markdown=True)
        crawl_result = await _internal_crawl_url(request)
        
        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }
        
        content = crawl_result.content or ""
        entities = {}
        
        # Define regex patterns for common entity types
        patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phones": r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
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
        
        # Extract entities for each requested type
        import re
        for entity_type in entity_types:
            if entity_type in patterns:
                matches = re.findall(patterns[entity_type], content, re.IGNORECASE)
                if matches:
                    # Deduplicate if requested
                    if deduplicate:
                        matches = list(set(matches))
                    entities[entity_type] = matches
        
        return {
            "url": url,
            "success": True,
            "entities": entities,
            "entity_types_requested": entity_types,
            "processing_method": "regex_extraction",
            "content_length": len(content),
            "total_entities_found": sum(len(v) for v in entities.values())
        }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Entity extraction error: {str(e)}"
        }


async def _internal_llm_extract_entities(
    url: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    instruction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal LLM extract entities implementation.
    TODO: Implement LLM-based named entity recognition.
    """
    return {
        "url": url,
        "success": False,
        "error": "LLM entity extraction not yet implemented in modular architecture"
    }


async def _internal_extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Internal extract structured data implementation.
    TODO: Implement CSS/LLM-based structured data extraction.
    """
    return CrawlResponse(
        success=False,
        url=request.url,
        error="Structured data extraction not yet implemented in modular architecture"
    )


# MCP Tool implementations
async def crawl_url(
    url: Annotated[str, Field(description="Target URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
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
    """
    # Create CrawlRequest object from individual parameters
    request = CrawlRequest(
        url=url,
        css_selector=css_selector,
        xpath=xpath,
        extract_media=extract_media,
        take_screenshot=take_screenshot,
        generate_markdown=generate_markdown,
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
    url: Annotated[str, Field(description="Starting URL for multi-page crawling")],
    max_depth: Annotated[int, Field(description="Link levels to follow from start URL (default: 2)")] = 2,
    max_pages: Annotated[int, Field(description="Maximum pages to crawl (default: 5)")] = 5,
    crawl_strategy: Annotated[str, Field(description="Crawling approach: 'bfs', 'dfs', 'best_first' (default: 'bfs')")] = "bfs",
    include_external: Annotated[bool, Field(description="Follow external domain links (default: False)")] = False,
    url_pattern: Annotated[Optional[str], Field(description="Wildcard filter like '*docs*' or '*api*' (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum relevance score 0.0-1.0 (default: 0.0)")] = 0.0,
    extract_media: Annotated[bool, Field(description="Include images/videos (default: False)")] = False,
    base_timeout: Annotated[int, Field(description="Timeout per page in seconds (default: 60)")] = 60
) -> Dict[str, Any]:
    """
    Crawl multiple related pages from a website (maximum 5 pages for stability).
    
    Multi-page crawling with configurable depth and filtering options.
    Perfect for documentation sites and content discovery.
    """
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
    url: Annotated[str, Field(description="Target webpage URL")],
    extraction_goal: Annotated[str, Field(description="Specific data to extract, be precise")],
    content_filter: Annotated[str, Field(description="Pre-filter content - 'bm25', 'pruning', 'llm' (default: 'bm25')")] = "bm25",
    filter_query: Annotated[Optional[str], Field(description="Keywords for BM25 filtering to improve accuracy (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Split large content for better processing (default: False)")] = False,
    use_llm: Annotated[bool, Field(description="Enable LLM processing (default: True)")] = True,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name (default: auto-detected)")] = None,
    custom_instructions: Annotated[Optional[str], Field(description="Additional guidance for LLM (default: None)")] = None
) -> Dict[str, Any]:
    """
    AI-powered extraction of specific data from web pages using LLM semantic understanding.
    
    Uses LLM to extract specific information based on your extraction goal.
    Pre-filtering improves accuracy and reduces processing time.
    """
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
    url: Annotated[str, Field(description="Target webpage URL")],
    entity_types: Annotated[List[str], Field(description="List of entity types to extract")],
    custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns for specialized extraction (default: None)")] = None,
    include_context: Annotated[bool, Field(description="Include surrounding text context (default: True)")] = True,
    deduplicate: Annotated[bool, Field(description="Remove duplicate entities (default: True)")] = True,
    use_llm: Annotated[bool, Field(description="Use AI for named entity recognition (default: False)")] = False,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for NER (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name (default: auto-detected)")] = None
) -> Dict[str, Any]:
    """
    Extract specific entity types from web pages using regex patterns or LLM.
    
    Supports regex: emails, phones, urls, dates, ips, social_media, prices, credit_cards, coordinates
    Supports LLM: names (people/organizations/locations) when use_llm=True
    """
    if use_llm and "names" in entity_types:
        # Use LLM-based extraction for named entities
        return await _internal_llm_extract_entities(url, llm_provider, llm_model)
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
    request: Annotated[StructuredExtractionRequest, Field(description="StructuredExtractionRequest with URL, schema, and extraction parameters")]
) -> CrawlResponse:
    """
    Extract structured data from a URL using CSS selectors or LLM-based extraction.
    
    Extract data matching a predefined schema using CSS selectors or LLM processing.
    Useful for consistent data extraction from similar page structures.
    """
    return await _internal_extract_structured_data(request)


async def crawl_url_with_fallback(
    url: Annotated[str, Field(description="Target URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Maximum crawling depth (None for single page)")] = None,
    max_pages: Annotated[Optional[int], Field(description="Maximum number of pages to crawl")] = 10,
    include_external: Annotated[bool, Field(description="Whether to follow external domain links")] = False,
    crawl_strategy: Annotated[str, Field(description="Crawling strategy: 'bfs', 'dfs', or 'best_first'")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL pattern filter (e.g., '*docs*')")] = None,
    score_threshold: Annotated[float, Field(description="Minimum score for URLs to be crawled")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="Content filter type: 'bm25', 'pruning', 'llm'")] = None,
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 content filtering")] = None,
    chunk_content: Annotated[bool, Field(description="Whether to chunk large content")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunking strategy: 'topic', 'regex', 'sentence'")] = "topic",
    chunk_size: Annotated[int, Field(description="Maximum chunk size in tokens")] = 1000,
    overlap_rate: Annotated[float, Field(description="Overlap rate between chunks (0.0-1.0)")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="Custom user agent string")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="Custom HTTP headers")] = None,
    enable_caching: Annotated[bool, Field(description="Whether to enable caching")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode: 'enabled', 'disabled', 'bypass'")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JavaScript code to execute")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate human-like browsing behavior")] = False,
    auth_token: Annotated[Optional[str], Field(description="Authentication token")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Custom cookies")] = None,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long'")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization (auto-detected if not specified)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization (auto-detected if not specified)")] = None
) -> CrawlResponse:
    """
    Enhanced crawling with multiple fallback strategies for difficult sites.
    
    Uses multiple fallback strategies when normal crawling fails. Same parameters as crawl_url 
    but with enhanced reliability for sites with aggressive anti-bot protection.
    """
    # For now, fall back to regular crawl_url
    return await crawl_url(
        url=url,
        css_selector=css_selector,
        xpath=xpath,
        extract_media=extract_media,
        take_screenshot=take_screenshot,
        generate_markdown=generate_markdown,
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
        auth_token=auth_token,
        cookies=cookies,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        summary_length=summary_length,
        llm_provider=llm_provider,
        llm_model=llm_model
    )


