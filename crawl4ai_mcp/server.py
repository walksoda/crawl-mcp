"""
Crawl4AI MCP Server

A Model Context Protocol server that provides web crawling and content extraction
capabilities using the crawl4ai library.
"""

import asyncio
import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler
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
from .strategies import (
    CustomCssExtractionStrategy,
    XPathExtractionStrategy,
    create_extraction_strategy,
)
from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from .suppress_output import suppress_stdout_stderr
from .file_processor import FileProcessor
from .youtube_processor import YouTubeProcessor
from .google_search_processor import GoogleSearchProcessor


class CrawlRequest(BaseModel):
    """Request model for crawling operations."""
    url: str = Field(..., description="URL to crawl")
    css_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    xpath: Optional[str] = Field(None, description="XPath selector for content extraction")
    extract_media: bool = Field(False, description="Whether to extract media files")
    take_screenshot: bool = Field(False, description="Whether to take a screenshot")
    generate_markdown: bool = Field(True, description="Whether to generate markdown")
    wait_for_selector: Optional[str] = Field(None, description="Wait for specific element")
    timeout: int = Field(60, description="Request timeout in seconds")
    
    # Deep crawling parameters
    max_depth: Optional[int] = Field(None, description="Maximum crawling depth (None for single page)")
    max_pages: Optional[int] = Field(10, description="Maximum number of pages to crawl")
    include_external: bool = Field(False, description="Whether to follow external domain links")
    crawl_strategy: str = Field("bfs", description="Crawling strategy: 'bfs', 'dfs', or 'best_first'")
    url_pattern: Optional[str] = Field(None, description="URL pattern filter (e.g., '*docs*')")
    score_threshold: float = Field(0.3, description="Minimum score for URLs to be crawled")
    
    # Advanced content processing
    content_filter: Optional[str] = Field(None, description="Content filter type: 'bm25', 'pruning', 'llm'")
    filter_query: Optional[str] = Field(None, description="Query for BM25 content filtering")
    chunk_content: bool = Field(False, description="Whether to chunk large content")
    chunk_strategy: str = Field("topic", description="Chunking strategy: 'topic', 'regex', 'sentence'")
    chunk_size: int = Field(1000, description="Maximum chunk size in tokens")
    overlap_rate: float = Field(0.1, description="Overlap rate between chunks (0.0-1.0)")
    
    # Browser configuration
    user_agent: Optional[str] = Field(None, description="Custom user agent string")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")
    enable_caching: bool = Field(True, description="Whether to enable caching")
    cache_mode: str = Field("enabled", description="Cache mode: 'enabled', 'disabled', 'bypass'")
    
    # JavaScript and interaction
    execute_js: Optional[str] = Field(None, description="JavaScript code to execute")
    wait_for_js: bool = Field(False, description="Wait for JavaScript to complete")
    simulate_user: bool = Field(False, description="Simulate human-like browsing behavior")
    
    # Authentication
    auth_token: Optional[str] = Field(None, description="Authentication token")
    cookies: Optional[Dict[str, str]] = Field(None, description="Custom cookies")


class StructuredExtractionRequest(BaseModel):
    """Request model for structured data extraction."""
    url: str = Field(..., description="URL to crawl")
    schema: Dict[str, Any] = Field(..., description="JSON schema for extraction")
    extraction_type: str = Field("css", description="Type of extraction: 'css' or 'llm'")
    css_selectors: Optional[Dict[str, str]] = Field(None, description="CSS selectors for each field")
    llm_provider: Optional[str] = Field("openai", description="LLM provider for LLM-based extraction")
    llm_model: Optional[str] = Field("gpt-3.5-turbo", description="LLM model name")
    instruction: Optional[str] = Field(None, description="Custom instruction for LLM extraction")


class FileProcessRequest(BaseModel):
    """Request model for file processing operations."""
    url: str = Field(..., description="URL of the file to process (PDF, Office, ZIP)")
    max_size_mb: int = Field(100, description="Maximum file size in MB")
    extract_all_from_zip: bool = Field(True, description="Whether to extract all files from ZIP archives")
    include_metadata: bool = Field(True, description="Whether to include file metadata")


class FileProcessResponse(BaseModel):
    """Response model for file processing operations."""
    success: bool
    url: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    size_bytes: Optional[int] = None
    is_archive: bool = False
    content: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    archive_contents: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class YouTubeTranscriptRequest(BaseModel):
    """Request model for YouTube transcript extraction."""
    url: str = Field(..., description="YouTube video URL")
    languages: Optional[List[str]] = Field(["ja", "en"], description="Preferred languages in order of preference")
    translate_to: Optional[str] = Field(None, description="Target language for translation")
    include_timestamps: bool = Field(True, description="Include timestamps in transcript")
    preserve_formatting: bool = Field(True, description="Preserve original formatting")
    include_metadata: bool = Field(True, description="Include video metadata")


class YouTubeTranscriptResponse(BaseModel):
    """Response model for YouTube transcript extraction."""
    success: bool
    url: Optional[str] = None
    video_id: Optional[str] = None
    transcript: Optional[Dict[str, Any]] = None
    language_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_method: Optional[str] = None
    error: Optional[str] = None


class YouTubeBatchRequest(BaseModel):
    """Request model for batch YouTube transcript extraction."""
    urls: List[str] = Field(..., description="List of YouTube video URLs")
    languages: Optional[List[str]] = Field(["ja", "en"], description="Preferred languages in order of preference")
    translate_to: Optional[str] = Field(None, description="Target language for translation")
    include_timestamps: bool = Field(True, description="Include timestamps in transcript")
    max_concurrent: int = Field(3, description="Maximum concurrent requests (1-10)")


class YouTubeBatchResponse(BaseModel):
    """Response model for batch YouTube transcript extraction."""
    success: bool
    total_urls: int
    successful_extractions: int
    failed_extractions: int
    results: List[YouTubeTranscriptResponse]
    processing_summary: Optional[Dict[str, Any]] = None


class GoogleSearchRequest(BaseModel):
    """Request model for Google search operations."""
    query: str = Field(..., description="Search query")
    num_results: int = Field(10, description="Number of results to return (1-100)")
    language: str = Field("en", description="Search language (e.g., 'en', 'ja')")
    region: str = Field("us", description="Search region (e.g., 'us', 'jp')")
    search_genre: Optional[str] = Field(None, description="Search genre for content filtering (e.g., 'academic', 'news', 'technical')")
    safe_search: bool = Field(True, description="Enable safe search filtering (always enabled for safety)")


class GoogleSearchResponse(BaseModel):
    """Response model for Google search operations."""
    success: bool
    query: Optional[str] = None
    total_results: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    search_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GoogleBatchSearchRequest(BaseModel):
    """Request model for batch Google search operations."""
    queries: List[str] = Field(..., description="List of search queries")
    num_results_per_query: int = Field(10, description="Number of results per query (1-100)")
    max_concurrent: int = Field(3, description="Maximum concurrent searches (1-5)")
    language: str = Field("en", description="Search language (e.g., 'en', 'ja')")
    region: str = Field("us", description="Search region (e.g., 'us', 'jp')")
    search_genre: Optional[str] = Field(None, description="Search genre for content filtering")


class GoogleBatchSearchResponse(BaseModel):
    """Response model for batch Google search operations."""
    success: bool
    total_queries: int
    successful_searches: int
    failed_searches: int
    results: List[GoogleSearchResponse]
    analysis: Optional[Dict[str, Any]] = None




class CrawlResponse(BaseModel):
    """Response model for crawling operations."""
    success: bool
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    markdown: Optional[str] = None
    media: Optional[List[Dict[str, str]]] = None
    screenshot: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Suppress crawl4ai verbose output completely
logging.getLogger("crawl4ai").setLevel(logging.CRITICAL)
logging.getLogger("playwright").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Disable all logging to console for crawl4ai related modules
for logger_name in ["crawl4ai", "playwright", "asyncio", "urllib3"]:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False

# Initialize FastMCP server
mcp = FastMCP("Crawl4AI MCP Server")

# Initialize FileProcessor for MarkItDown integration
file_processor = FileProcessor()

# Initialize YouTubeProcessor for transcript extraction (youtube-transcript-api v1.1.0+)
youtube_processor = YouTubeProcessor()

# Initialize GoogleSearchProcessor for search functionality
google_search_processor = GoogleSearchProcessor()


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

        config = CrawlerRunConfig(
            css_selector=request.css_selector,
            screenshot=request.take_screenshot,
            wait_for=request.wait_for_selector,
            page_timeout=request.timeout * 1000,
            exclude_all_images=not request.extract_media,
            verbose=False,  # Disable verbose output
            log_console=False,  # Disable console logging
            deep_crawl_strategy=deep_crawl_strategy,
        )
        
        # Setup advanced content filtering
        content_filter_strategy = None
        if request.content_filter == "bm25" and request.filter_query:
            content_filter_strategy = BM25ContentFilter(query=request.filter_query)
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
            from crawl4ai.chunking_strategy import SlidingWindowChunking
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
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=result.metadata.get("title"),
                    content="\n\n".join(all_content) if all_content else result.cleaned_html,
                    markdown="\n\n".join(all_markdown) if all_markdown else result.markdown,
                    media=all_media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data={"crawled_pages": len(result.crawled_pages)} if hasattr(result, 'crawled_pages') else None
                )
            else:
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=result.metadata.get("title"),
                    content=result.cleaned_html,
                    markdown=result.markdown,
                    media=result.media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
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
            error_message += "\n\nSuggestion: Install Playwright dependencies with: sudo apt-get install libnss3 libnspr4 libasound2"
        
        return CrawlResponse(
            success=False,
            url=request.url,
            error=error_message
        )


@mcp.tool
async def crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    return await _internal_crawl_url(request)


@mcp.tool
async def deep_crawl_site(
    url: str,
    max_depth: int = 2,
    max_pages: int = 5,  # Conservative default
    crawl_strategy: str = "bfs",
    include_external: bool = False,
    url_pattern: Optional[str] = None,
    score_threshold: float = 0.0,  # More permissive default
    extract_media: bool = False
) -> Dict[str, Any]:
    """
    Perform deep crawling of a website with specified depth and strategy.
    
    Args:
        url: Starting URL for deep crawling
        max_depth: Maximum crawling depth (limited to 2 for stability)
        max_pages: Maximum number of pages to crawl (limited to 5 for stability)
        crawl_strategy: Strategy - 'bfs', 'dfs', or 'best_first'
        include_external: Whether to follow external domain links
        url_pattern: URL pattern filter (e.g., '*docs*', '*blog*')
        score_threshold: Minimum score for URLs to be crawled (0.0 = permissive)
        extract_media: Whether to extract media files (always disabled for performance)
        
    Returns:
        Dictionary with crawled pages information and site map
    """
    try:
        # Create filter chain - always include domain filter for stability
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        filters = []
        if url_pattern:
            filters.append(URLPatternFilter(patterns=[url_pattern]))
        
        # Always add domain filter unless explicitly allowing external links
        if not include_external:
            filters.append(DomainFilter(allowed_domains=[domain]))
        else:
            # Even when allowing external, add a broad domain filter for stability
            # This prevents the filter_chain=None error
            filters.append(DomainFilter(allowed_domains=[domain, "*"]))
        
        # Always create a filter chain to avoid NoneType errors
        filter_chain = FilterChain(filters)
        
        # Conservative resource limits for MCP stability
        if max_pages > 5:  # Much more conservative limit
            max_pages = 5
        if max_depth > 2:  # Limit depth for stability
            max_depth = 2
        
        # Use more permissive score threshold for better link following
        effective_score_threshold = min(score_threshold, 0.0)  # Cap at 0.0 for better results
        
        # Select crawling strategy with corrected parameters
        if crawl_strategy == "dfs":
            strategy = DFSDeepCrawlStrategy(
                max_depth=max_depth,
                max_pages=max_pages,  # Now uses the limited value
                include_external=include_external,
                filter_chain=filter_chain,
                score_threshold=effective_score_threshold
            )
        elif crawl_strategy == "best_first":
            strategy = BestFirstCrawlingStrategy(
                max_depth=max_depth,
                max_pages=max_pages,  # Now uses the limited value
                include_external=include_external,
                filter_chain=filter_chain,
                score_threshold=effective_score_threshold
            )
        else:  # Default to BFS
            strategy = BFSDeepCrawlStrategy(
                max_depth=max_depth,
                max_pages=max_pages,  # Now uses the limited value
                include_external=include_external,
                filter_chain=filter_chain,
                score_threshold=effective_score_threshold
            )

        # Add timeout and resource limits for deep crawling
        config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            exclude_all_images=True,  # Always exclude images for performance
            verbose=False,  # Always disable to prevent MCP connection issues
            log_console=False,  # Always disable to prevent MCP connection issues
            page_timeout=60000,  # 60 seconds per page for slow sites
        )
        
        # Enhanced browser configuration
        browser_config = {
            "headless": True,
            "verbose": False,
            "viewport_width": 1280,
            "viewport_height": 720,
            "user_agent": "Mozilla/5.0 (compatible; Crawl4AI-DeepCrawler/1.0)",
            "accept_downloads": False,
            "ignore_https_errors": True,
        }
        
        # Configuration info for debugging if needed
        config_info = {
            "strategy_used": crawl_strategy,
            "max_depth_actual": max_depth,
            "max_pages_actual": max_pages,
            "score_threshold_actual": effective_score_threshold,
            "include_external": include_external,
            "url_pattern": url_pattern
        }
        
        try:
            with suppress_stdout_stderr():
                async with AsyncWebCrawler(**browser_config) as crawler:
                    # Reduced overall timeout for MCP stability (90 seconds)
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=config),
                        timeout=90
                    )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Deep crawling operation timed out after 90 seconds",
                "starting_url": url,
                "timeout_reason": "Operation exceeded maximum allowed time",
                "config_info": config_info
            }
        except Exception as crawler_error:
            return {
                "success": False,
                "error": f"Browser/crawler error: {str(crawler_error)}",
                "starting_url": url,
                "error_type": type(crawler_error).__name__,
                "config_info": config_info
            }
        
        # Handle different result types from deep crawling
        site_map = {
            "starting_url": url,
            "strategy_used": crawl_strategy,
            "total_pages_crawled": 0,
            "pages": [],
            "site_structure": {},
            "content_summary": ""
        }
        
        # Check if result is a list (multiple pages) or single result object
        if isinstance(result, list):
            # Deep crawling returns a list of CrawlResult objects
            site_map["total_pages_crawled"] = len(result)
            
            for page_result in result:
                if hasattr(page_result, 'success') and page_result.success:
                    page_info = {
                        "url": page_result.url,
                        "title": page_result.metadata.get("title", "No title") if page_result.metadata else "No title",
                        "content_length": len(page_result.cleaned_html) if page_result.cleaned_html else 0,
                        "links_found": len(page_result.links.get("internal", [])) if hasattr(page_result, 'links') and page_result.links else 0,
                        "depth": getattr(page_result, 'depth', 0),
                        "content_preview": (page_result.cleaned_html[:200] + "...") if page_result.cleaned_html else "",
                        "markdown_preview": (page_result.markdown[:200] + "...") if page_result.markdown else ""
                    }
                    site_map["pages"].append(page_info)
                else:
                    # Add failed page info
                    failed_page_info = {
                        "url": getattr(page_result, 'url', 'Unknown URL'),
                        "title": "Failed to crawl",
                        "content_length": 0,
                        "error": getattr(page_result, 'error_message', 'Unknown error'),
                        "content_preview": "",
                        "markdown_preview": ""
                    }
                    site_map["pages"].append(failed_page_info)
            
            # Create content summary
            successful_pages = [p for p in site_map["pages"] if "error" not in p]
            total_content = sum(p["content_length"] for p in successful_pages)
            site_map["content_summary"] = f"Crawled {len(successful_pages)} of {site_map['total_pages_crawled']} pages with {total_content} characters of content"
            
            return site_map
            
        elif hasattr(result, 'success'):
            # Single result object
            if result.success:
                site_map["total_pages_crawled"] = 1
                site_map["pages"].append({
                    "url": result.url,
                    "title": result.metadata.get("title", "No title") if result.metadata else "No title",
                    "content_length": len(result.cleaned_html) if result.cleaned_html else 0,
                    "content_preview": (result.cleaned_html[:200] + "...") if result.cleaned_html else "",
                    "markdown_preview": (result.markdown[:200] + "...") if result.markdown else ""
                })
                site_map["content_summary"] = f"Single page crawled with {site_map['pages'][0]['content_length']} characters of content"
                return site_map
            else:
                return {
                    "success": False,
                    "error": f"Deep crawling failed: {result.error_message}",
                    "starting_url": url,
                    "config_info": config_info
                }
        else:
            # Unknown result type
            return {
                "success": False,
                "error": f"Unknown result type from crawler: {type(result)}. Result: {str(result)[:500]}",
                "starting_url": url,
                "config_info": config_info
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Deep crawling error: {str(e)}",
            "starting_url": url,
            "error_type": type(e).__name__
        }


@mcp.tool
async def get_llm_config_info() -> Dict[str, Any]:
    """
    Get information about the current LLM configuration.
    
    Returns:
        Dictionary with LLM configuration details including available providers and models
    """
    try:
        from .config import config_manager
        
        if not config_manager.llm_config:
            return {
                "success": False,
                "error": "No LLM configuration loaded"
            }
        
        # Get available models by provider
        available_models = config_manager.list_available_models()
        
        # Check API key status for each provider
        provider_status = {}
        for provider_name, provider_config in config_manager.llm_config.providers.items():
            api_key_available = bool(config_manager.get_api_key(provider_name))
            has_direct_key = provider_config.api_key is not None
            has_env_key = provider_config.api_key_env is not None
            
            provider_status[provider_name] = {
                "api_key_required": has_direct_key or has_env_key or provider_name not in ['ollama'],
                "api_key_direct": has_direct_key,
                "api_key_env_var": provider_config.api_key_env,
                "api_key_available": api_key_available,
                "base_url": provider_config.base_url,
                "models": provider_config.models
            }
        
        return {
            "success": True,
            "default_provider": config_manager.get_default_provider(),
            "default_model": config_manager.get_default_model(),
            "providers": provider_status,
            "config_source": "MCP configuration"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get LLM config info: {str(e)}",
            "error_type": type(e).__name__
        }


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
    Perform intelligent content extraction with advanced filtering and AI analysis.
    
    Args:
        url: URL to extract content from
        extraction_goal: Description of what to extract (e.g., "product information", "article summary")
        content_filter: Filter type - 'bm25', 'pruning', 'llm', or 'none'
        filter_query: Query for content filtering (required for BM25)
        chunk_content: Whether to chunk large content for better processing
        use_llm: Whether to use LLM for intelligent extraction
        llm_provider: LLM provider (auto-detected from config if not specified)
        llm_model: Specific model to use (auto-detected from config if not specified)
        custom_instructions: Custom instructions for extraction
        
    Returns:
        Dictionary with extracted content and metadata
    """
    try:
        from .config import get_llm_config, config_manager
        
        # Setup content filter
        content_filter_strategy = None
        if content_filter == "bm25" and filter_query:
            content_filter_strategy = BM25ContentFilter(query=filter_query)
        elif content_filter == "pruning":
            content_filter_strategy = PruningContentFilter(threshold=0.5)
        elif content_filter == "llm" and use_llm:
            instructions = custom_instructions or f"Extract content related to: {extraction_goal}"
            llm_config = get_llm_config(llm_provider, llm_model)
            # Use ONLY llm_config parameter, avoid mixing with legacy params
            content_filter_strategy = LLMContentFilter(
                llm_config=llm_config,
                instruction=instructions,
                verbose=False
            )

        # Setup extraction strategy
        extraction_strategy = None
        if use_llm:
            schema = {
                "extracted_content": extraction_goal,
                "summary": "Brief summary of extracted content",
                "key_points": "List of key points",
                "metadata": "Additional relevant metadata"
            }
            
            instructions = custom_instructions or f"""
            Extract content related to: {extraction_goal}
            Provide a structured analysis including:
            1. Main content relevant to the goal
            2. Summary of findings
            3. Key points in a list format
            4. Any relevant metadata
            """
            
            llm_config = get_llm_config(llm_provider, llm_model)
            # Use ONLY llm_config parameter, avoid mixing with legacy params
            extraction_strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                schema=schema,
                extraction_type="schema",
                instruction=instructions,
                verbose=False
            )

        # Configure chunking
        chunking_strategy = None
        if chunk_content:
            from crawl4ai.chunking_strategy import SlidingWindowChunking
            chunking_strategy = SlidingWindowChunking(
                window_size=1000,
                step=900  # 10% overlap
            )

        # Setup crawler configuration
        config_params = {
            "extraction_strategy": extraction_strategy,
            "verbose": False,
            "log_console": False,
        }
        
        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy
        
        # Add content filter if supported by current crawl4ai version
        try:
            config = CrawlerRunConfig(**config_params, content_filter=content_filter_strategy)
        except TypeError:
            # Fallback for older versions without content_filter support
            config = CrawlerRunConfig(**config_params)

        with suppress_stdout_stderr():
            async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)

        if result.success:
            response_data = {
                "url": url,
                "extraction_goal": extraction_goal,
                "success": True,
                "content": {
                    "raw_content": result.cleaned_html[:2000] + "..." if len(result.cleaned_html or "") > 2000 else result.cleaned_html,
                    "markdown": result.markdown[:2000] + "..." if len(result.markdown or "") > 2000 else result.markdown,
                    "extracted_data": None,
                    "chunks": []
                },
                "metadata": {
                    "title": result.metadata.get("title") if result.metadata else None,
                    "content_length": len(result.cleaned_html) if result.cleaned_html else 0,
                    "filter_used": content_filter,
                    "llm_used": use_llm,
                    "chunked": chunk_content
                }
            }

            # Add extracted structured data if available
            if result.extracted_content:
                try:
                    response_data["content"]["extracted_data"] = json.loads(result.extracted_content)
                except json.JSONDecodeError:
                    response_data["content"]["extracted_data"] = result.extracted_content

            # Add chunk information if chunking was used
            if chunk_content and hasattr(result, 'content_chunks'):
                response_data["content"]["chunks"] = [
                    {
                        "index": i,
                        "content": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        "length": len(chunk)
                    }
                    for i, chunk in enumerate(result.content_chunks or [])
                ]

            return response_data

        else:
            return {
                "url": url,
                "extraction_goal": extraction_goal,
                "success": False,
                "error": f"Extraction failed: {result.error_message}",
                "metadata": {"filter_used": content_filter, "llm_used": use_llm}
            }

    except Exception as e:
        return {
            "url": url,
            "extraction_goal": extraction_goal,
            "success": False,
            "error": f"Intelligent extraction error: {str(e)}",
            "metadata": {"filter_used": content_filter, "llm_used": use_llm}
        }


@mcp.tool
async def intelligent_extract(
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
    Perform intelligent content extraction with advanced filtering and AI analysis.
    
    Args:
        url: URL to extract content from
        extraction_goal: Description of what to extract (e.g., "product information", "article summary")
        content_filter: Filter type - 'bm25', 'pruning', 'llm', or 'none'
        filter_query: Query for content filtering (required for BM25)
        chunk_content: Whether to chunk large content for better processing
        use_llm: Whether to use LLM for intelligent extraction
        llm_provider: LLM provider (auto-detected from config if not specified)
        llm_model: Specific model to use (auto-detected from config if not specified)
        custom_instructions: Custom instructions for extraction
        
    Returns:
        Dictionary with extracted content and metadata
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


async def _internal_extract_entities(
    url: str,
    entity_types: List[str],
    custom_patterns: Optional[Dict[str, str]] = None,
    include_context: bool = True,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Extract specific entities (emails, phones, URLs, dates, etc.) from web content using regex patterns.
    
    Args:
        url: URL to extract entities from
        entity_types: Types of entities to extract (emails, phones, urls, dates, social_media, etc.)
        custom_patterns: Custom regex patterns for entity extraction
        include_context: Whether to include surrounding context for each entity
        deduplicate: Whether to remove duplicate entities
        
    Returns:
        Dictionary with extracted entities organized by type
    """
    try:
        # Build extraction patterns
        patterns = {}
        
        # Built-in patterns for common entities
        builtin_patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phones": r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            "urls": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            "ips": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "social_media": r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+',
            "prices": r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)',
            "credit_cards": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "coordinates": r'[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)(?:\.\d+)?)'
        }
        
        # Add requested built-in patterns
        for entity_type in entity_types:
            if entity_type in builtin_patterns:
                patterns[entity_type] = builtin_patterns[entity_type]
        
        # Add custom patterns
        if custom_patterns:
            patterns.update(custom_patterns)
        
        if not patterns:
            return {
                "url": url,
                "success": False,
                "error": "No valid entity types or patterns provided",
                "available_types": list(builtin_patterns.keys())
            }

        # Use the same successful configuration as intelligent_extract
        config = CrawlerRunConfig(
            verbose=False,
            log_console=False
        )

        with suppress_stdout_stderr():
            async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)

        if result.success:
            import re
            extracted_entities = {}
            
            # Get text content for pattern matching
            content = result.cleaned_html or result.markdown or ""
            
            # Process each entity type pattern
            for entity_type, pattern in patterns.items():
                try:
                    # Find all matches
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    entity_list = []
                    seen = set() if deduplicate else None
                    
                    for match in matches:
                        entity_value = match.group(0)
                        
                        if deduplicate:
                            if entity_value in seen:
                                continue
                            seen.add(entity_value)
                        
                        entity_info = {"value": entity_value}
                        
                        # Add context if requested
                        if include_context:
                            start_pos = max(0, match.start() - 50)
                            end_pos = min(len(content), match.end() + 50)
                            entity_info["context"] = content[start_pos:end_pos]
                            entity_info["position"] = match.start()
                        
                        entity_list.append(entity_info)
                    
                    extracted_entities[entity_type] = {
                        "count": len(entity_list),
                        "entities": entity_list
                    }
                    
                except re.error as e:
                    extracted_entities[entity_type] = {
                        "count": 0,
                        "entities": [],
                        "error": f"Invalid regex pattern: {e}"
                    }

            return {
                "url": url,
                "success": True,
                "entity_types_requested": entity_types,
                "total_entities_found": sum(data.get("count", 0) for data in extracted_entities.values()),
                "entities": extracted_entities,
                "metadata": {
                    "title": result.metadata.get("title") if result.metadata else None,
                    "content_length": len(result.cleaned_html) if result.cleaned_html else 0,
                    "deduplicated": deduplicate,
                    "context_included": include_context
                }
            }

        else:
            return {
                "url": url,
                "success": False,
                "error": f"Entity extraction failed: {result.error_message}",
                "entity_types_requested": entity_types
            }

    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Entity extraction error: {str(e)}",
            "entity_types_requested": entity_types
        }


@mcp.tool
async def extract_entities(
    url: str,
    entity_types: List[str],
    custom_patterns: Optional[Dict[str, str]] = None,
    include_context: bool = True,
    deduplicate: bool = True,
    use_llm: bool = False,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract entities from web content using regex patterns or LLM analysis.
    
    Args:
        url: URL to extract entities from
        entity_types: Types of entities to extract
            - For regex: emails, phones, urls, dates, social_media, etc.
            - For LLM: names (people, organizations, locations, misc entities)
        custom_patterns: Custom regex patterns for entity extraction (regex mode only)
        include_context: Whether to include surrounding context for each entity
        deduplicate: Whether to remove duplicate entities
        use_llm: If True, use LLM for named entity recognition instead of regex
        llm_provider: LLM provider to use (openai, anthropic, ollama) when use_llm=True
        llm_model: LLM model to use when use_llm=True
        
    Returns:
        Dictionary with extracted entities organized by type
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


async def _internal_llm_extract_entities(
    url: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    instruction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal implementation for LLM-based entity extraction using direct LiteLLM approach
    """
    try:
        from .config import config_manager
        import json
        import litellm
        
        # Create LLM configuration
        llm_config = config_manager.create_llm_config(provider, model)
        
        # First, crawl the webpage to get content
        config = CrawlerRunConfig(
            verbose=False,
            log_console=False
        )
        
        with suppress_stdout_stderr():
            async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)
        
        if not result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {result.error_message}",
                "extraction_method": "llm_direct"
            }
        
        # Get content for LLM processing
        content = result.cleaned_html or result.markdown or result.html or ""
        
        if not content:
            return {
                "url": url,
                "success": False,
                "error": "No content found on the webpage",
                "extraction_method": "llm_direct"
            }
        
        # Limit content size to avoid token limits and reduce processing time
        # GPT-4.1 models support 1M token context but become less reliable with more tokens
        max_content_length = 6000  # Optimized for GPT-4.1 models
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Default instruction if not provided
        if not instruction:
            instruction = """Extract named entities from this text. Return JSON:
            {"entities": [{"name": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|MISC"}]}
            Only extract clearly identifiable entities."""
        
        # Prepare LiteLLM call
        messages = [
            {"role": "user", "content": f"{instruction}\n\nWeb page content:\n{content}"}
        ]
        
        # Configure LiteLLM
        if llm_config.api_token:
            litellm.api_key = llm_config.api_token
        if llm_config.base_url:
            litellm.api_base = llm_config.base_url
            
        # Call LLM with GPT-4.1 optimized settings
        response = await litellm.acompletion(
            model=llm_config.provider,
            messages=messages,
            max_tokens=500,  # Increased for GPT-4.1's better output capacity
            temperature=0.1, # Slightly more creative for better extractions
            timeout=25,      # 25 second timeout
        )
        
        llm_response = response.choices[0].message.content
        
        # Parse LLM response
        try:
            # Try to extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = llm_response[json_start:json_end]
                extracted_json = json.loads(json_text)
            else:
                # If no JSON found, try to parse the entire response
                extracted_json = json.loads(llm_response)
            
            entities = extracted_json.get("entities", [])
            
            # Organize entities by type
            entities_by_type = {}
            for entity in entities:
                entity_type = entity.get("type", "MISC")
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append({
                    "name": entity.get("name", ""),
                    "context": entity.get("context", ""),
                    "confidence": entity.get("confidence", 1.0)
                })
            
            return {
                "url": url,
                "success": True,
                "entities_by_type": entities_by_type,
                "total_entities": len(entities),
                "llm_provider": llm_config.provider,
                "extraction_method": "llm_direct",
                "content_length": len(content)
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_llm_response": llm_response[:500] + "..." if len(llm_response) > 500 else llm_response,
                "extraction_method": "llm_direct"
            }
            
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"LLM entity extraction error: {str(e)}",
            "extraction_method": "llm_direct"
        }




async def _internal_extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """Internal implementation for structured data extraction"""
    try:
        if request.extraction_type == "css" and request.css_selectors:
            strategy = JsonCssExtractionStrategy(request.css_selectors)
        elif request.extraction_type == "llm":
            # Create LLM config using our config manager
            from .config import config_manager
            llm_config = config_manager.create_llm_config(
                provider=request.llm_provider,
                model=request.llm_model
            )
            
            strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                schema=request.schema,
                extraction_type="schema",
                instruction=request.instruction or "Extract data according to the provided schema.",
                chunk_token_threshold=2000,  # Smaller chunks for faster processing
                apply_chunking=True,
                verbose=False
            )
        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error="Invalid extraction type or missing CSS selectors"
            )
            
        config = CrawlerRunConfig(
            extraction_strategy=strategy,
            page_timeout=30000,  # 30 seconds in milliseconds
            verbose=False,
            log_console=False,
        )
        
        # First crawl without extraction to get content
        with suppress_stdout_stderr():
            async with AsyncWebCrawler(verbose=False) as crawler:
                initial_result = await crawler.arun(url=request.url)
        
        if not initial_result.success:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Failed to crawl URL: {initial_result.error_message}"
            )
        
        # For LLM extraction, truncate content to avoid token limits
        if request.extraction_type == "llm":
            # Get content and truncate to manageable size
            content = initial_result.cleaned_html or initial_result.markdown or initial_result.html or ""
            max_content_length = 10000  # Increased for GPT-4.1's 1M token context
            
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Create a simple text-based extraction instead of using the full HTML
            import litellm
            from .config import config_manager
            
            llm_config = config_manager.create_llm_config(
                provider=request.llm_provider,
                model=request.llm_model
            )
            
            # Configure LiteLLM
            if llm_config.api_token:
                litellm.api_key = llm_config.api_token
            if llm_config.base_url:
                litellm.api_base = llm_config.base_url
            
            # Create prompt for structured extraction
            instruction = request.instruction or "Extract data according to the provided schema."
            schema_str = json.dumps(request.schema, indent=2)
            
            prompt = f"""{instruction}

Schema to follow:
{schema_str}

Web page content:
{content}

Return valid JSON that matches the schema."""
            
            try:
                # Call LLM directly with GPT-4.1 optimized settings
                response = await litellm.acompletion(
                    model=llm_config.provider,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1200, # Increased for GPT-4.1's better output capacity
                    temperature=0.1, # Slightly more creative for better structured output
                    timeout=30,      # 30 second timeout for complex schemas
                )
                
                llm_response = response.choices[0].message.content
                
                # Parse LLM response
                try:
                    # Try to extract JSON from the response
                    json_start = llm_response.find('{')
                    json_end = llm_response.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_text = llm_response[json_start:json_end]
                        extracted_data = json.loads(json_text)
                    else:
                        # If no JSON found, try to parse the entire response
                        extracted_data = json.loads(llm_response)
                    
                    return CrawlResponse(
                        success=True,
                        url=request.url,
                        title=initial_result.metadata.get("title"),
                        content=initial_result.cleaned_html,
                        markdown=initial_result.markdown,
                        extracted_data=extracted_data,
                    )
                    
                except (json.JSONDecodeError, KeyError) as e:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=f"Failed to parse LLM response as JSON: {str(e)}",
                        content=initial_result.cleaned_html,
                        markdown=initial_result.markdown
                    )
                    
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"LLM extraction error: {str(e)}",
                    content=initial_result.cleaned_html,
                    markdown=initial_result.markdown
                )
        
        # For CSS extraction, use the original approach
        else:
            with suppress_stdout_stderr():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=request.url, config=config)
            
            if result.success:
                extracted_data = None
                if result.extracted_content:
                    try:
                        parsed_content = json.loads(result.extracted_content)
                        # Handle case where result is a list instead of dict
                        if isinstance(parsed_content, list) and len(parsed_content) > 0:
                            extracted_data = parsed_content[0]  # Take first item
                        elif isinstance(parsed_content, dict):
                            extracted_data = parsed_content
                    except (json.JSONDecodeError, TypeError):
                        extracted_data = {"raw_content": result.extracted_content}
                
                return CrawlResponse(
                    success=True,
                    url=request.url,
                    title=result.metadata.get("title"),
                    content=result.cleaned_html,
                    markdown=result.markdown,
                    extracted_data=extracted_data,
                )
            else:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"Failed to extract data: {result.error_message}"
                )
                
    except Exception as e:
        return CrawlResponse(
            success=False,
            url=request.url,
            error=f"Extraction error: {str(e)}"
        )


@mcp.tool
async def extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Extract structured data from a URL using CSS selectors or LLM-based extraction.
    
    Args:
        request: StructuredExtractionRequest with URL, schema, and extraction parameters
        
    Returns:
        CrawlResponse with extracted structured data
    """
    return await _internal_extract_structured_data(request)


@mcp.tool
async def batch_crawl(urls: List[str], config: Optional[Dict[str, Any]] = None) -> List[CrawlResponse]:
    """
    Crawl multiple URLs in batch.
    
    Args:
        urls: List of URLs to crawl
        config: Optional configuration parameters
        
    Returns:
        List of CrawlResponse objects for each URL
    """
    results = []
    
    try:
        with suppress_stdout_stderr():
            async with AsyncWebCrawler(verbose=False) as crawler:
                for url in urls:
                    try:
                        default_config = {"verbose": False, "log_console": False}
                        crawl_config = CrawlerRunConfig(**{**default_config, **(config or {})})
                        result = await crawler.arun(url=url, config=crawl_config)
                        
                        if result.success:
                            response = CrawlResponse(
                                success=True,
                                url=url,
                                title=result.metadata.get("title"),
                                content=result.cleaned_html,
                                markdown=result.markdown,
                            )
                        else:
                            response = CrawlResponse(
                                success=False,
                                url=url,
                                error=f"Failed to crawl: {result.error_message}"
                            )
                            
                        results.append(response)
                        
                    except Exception as e:
                        results.append(CrawlResponse(
                            success=False,
                            url=url,
                            error=f"Error crawling {url}: {str(e)}"
                        ))
                    
    except Exception as e:
        # If crawler setup fails, return error for all URLs
        for url in urls:
            results.append(CrawlResponse(
                success=False,
                url=url,
                error=f"Crawler initialization error: {str(e)}"
            ))
    
    return results


@mcp.tool
async def crawl_url_with_fallback(request: CrawlRequest) -> CrawlResponse:
    """
    Enhanced crawling with multiple fallback strategies using crawl4ai.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    # Try different crawling strategies in order of preference
    strategies = [
        # Strategy 1: Full browser with JavaScript
        {
            "browser_type": "chromium",
            "headless": True,
            "config": CrawlerRunConfig(
                css_selector=request.css_selector,
                screenshot=request.take_screenshot,
                wait_for=request.wait_for_selector,
                page_timeout=request.timeout * 1000,
                exclude_all_images=not request.extract_media,
                js_only=False
            )
        },
        # Strategy 2: Simplified browser mode
        {
            "browser_type": "chromium", 
            "headless": True,
            "config": CrawlerRunConfig(
                page_timeout=60000,
                exclude_all_images=True,
                remove_overlay_elements=True,
                js_only=False,
                verbose=False
            )
        },
        # Strategy 3: Minimal mode
        {
            "browser_type": "chromium",
            "headless": True, 
            "config": CrawlerRunConfig(
                page_timeout=60000,
                exclude_all_images=True,
                exclude_external_links=True,
                remove_overlay_elements=True,
                js_only=False,
                verbose=False,
                wait_for_images=False
            )
        }
    ]
    
    last_error = None
    
    for i, strategy in enumerate(strategies):
        try:
            with suppress_stdout_stderr():
                async with AsyncWebCrawler(
                    browser_type=strategy["browser_type"],
                    headless=strategy["headless"],
                    verbose=False
                ) as crawler:
                    result = await crawler.arun(url=request.url, config=strategy["config"])
            
            if result.success:
                return CrawlResponse(
                    success=True,
                    url=request.url,
                    title=result.metadata.get("title"),
                    content=result.cleaned_html,
                    markdown=result.markdown,
                    media=result.media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                )
            else:
                last_error = f"Strategy {i+1} failed: {result.error_message}"
                    
        except Exception as e:
            last_error = f"Strategy {i+1} error: {str(e)}"
            continue
    
    # If all strategies failed
    error_message = f"All crawling strategies failed. Last error: {last_error}"
    if "playwright" in str(last_error).lower():
        error_message += "\n\nTo fix: Install browser dependencies with 'sudo apt-get install libnss3 libnspr4 libasound2'"
    
    return CrawlResponse(
        success=False,
        url=request.url,
        error=error_message
    )


@mcp.tool
async def process_file(request: FileProcessRequest) -> FileProcessResponse:
    """
    Process various file formats (PDF, Office, ZIP) and convert to Markdown using MarkItDown.
    
    Args:
        request: FileProcessRequest containing file URL and processing parameters
        
    Returns:
        FileProcessResponse with processed content and metadata
    """
    try:
        # Process the file
        result = await file_processor.process_file_from_url(
            request.url,
            max_size_mb=request.max_size_mb
        )
        
        if result['success']:
            response = FileProcessResponse(
                success=True,
                url=result.get('url'),
                filename=result.get('filename'),
                file_type=result.get('file_type'),
                size_bytes=result.get('size_bytes'),
                is_archive=result.get('is_archive', False),
                content=result.get('content'),
                title=result.get('title'),
                metadata=result.get('metadata') if request.include_metadata else None,
                archive_contents=result.get('archive_contents') if result.get('is_archive') and request.extract_all_from_zip else None
            )
        else:
            response = FileProcessResponse(
                success=False,
                url=request.url,
                error=result.get('error'),
                file_type=result.get('file_type')
            )
            
        return response
        
    except Exception as e:
        return FileProcessResponse(
            success=False,
            url=request.url,
            error=f"File processing error: {str(e)}"
        )


@mcp.tool
async def get_supported_file_formats() -> Dict[str, Any]:
    """
    Get list of supported file formats for file processing.
    
    Returns:
        Dictionary with supported file formats and descriptions
    """
    try:
        return {
            "success": True,
            "supported_formats": file_processor.supported_extensions,
            "categories": {
                "pdf": {
                    "description": "PDF Documents",
                    "extensions": [".pdf"],
                    "features": ["Text extraction", "Structure preservation", "Metadata extraction"]
                },
                "microsoft_office": {
                    "description": "Microsoft Office Documents",
                    "extensions": [".docx", ".pptx", ".xlsx", ".xls"],
                    "features": ["Content extraction", "Table processing", "Slide content", "Cell data"]
                },
                "archives": {
                    "description": "Archive Files",
                    "extensions": [".zip"],
                    "features": ["Multi-file extraction", "Nested processing", "Format detection"]
                },
                "web_and_text": {
                    "description": "Web and Text Formats",
                    "extensions": [".html", ".htm", ".txt", ".md", ".csv", ".rtf"],
                    "features": ["HTML parsing", "Text processing", "CSV structure", "Rich text"]
                },
                "ebooks": {
                    "description": "eBook Formats",
                    "extensions": [".epub"],
                    "features": ["Chapter extraction", "Metadata", "Content structure"]
                }
            },
            "max_file_size": "100MB (configurable)",
            "output_format": "Markdown",
            "additional_features": [
                "Automatic file type detection",
                "Metadata extraction",
                "ZIP archive processing",
                "Error handling and recovery",
                "Size limit protection"
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving format information: {str(e)}"
        }


@mcp.tool
async def extract_youtube_transcript(request: YouTubeTranscriptRequest) -> YouTubeTranscriptResponse:
    """
    Extract transcript from a YouTube video using youtube-transcript-api.
    
    This function uses the youtube-transcript-api library for simple and reliable transcript extraction
    without complex authentication requirements.
    
    Features:
    - Simple youtube-transcript-api integration
    - Support for multiple languages with automatic fallback
    - Automatic language preference handling
    - Both manual and auto-generated captions
    - Basic video information extraction
    
    No Setup Required:
    - Works directly with public YouTube videos that have transcripts
    - No API keys or authentication needed
    - Automatically handles language preferences
    
    Args:
        request: YouTubeTranscriptRequest containing URL and extraction parameters
        
    Returns:
        YouTubeTranscriptResponse with transcript data or error information
    """
    try:
        # Check if URL is valid YouTube URL
        if not youtube_processor.is_youtube_url(request.url):
            return YouTubeTranscriptResponse(
                success=False,
                url=request.url,
                error="URL is not a valid YouTube video URL"
            )
        
        # Process with youtube-transcript-api
        result = await youtube_processor.process_youtube_url(
            url=request.url,
            languages=request.languages,
            translate_to=request.translate_to,
            include_timestamps=request.include_timestamps,
            preserve_formatting=request.preserve_formatting,
            include_metadata=request.include_metadata
        )
        
        if result['success']:
            transcript_data = result['transcript']
            language_info = result['language_info']
            
            return YouTubeTranscriptResponse(
                success=True,
                url=result['url'],
                video_id=result['video_id'],
                transcript={
                    'full_text': transcript_data['full_text'],
                    'clean_text': transcript_data['clean_text'],
                    'segments': transcript_data.get('segments', []),
                    'segment_count': transcript_data.get('segment_count', 0),
                    'word_count': transcript_data.get('word_count', 0),
                    'duration_seconds': transcript_data.get('duration_seconds', 0),
                    'duration_formatted': transcript_data.get('duration_formatted', '0s')
                },
                language_info={
                    'source_language': language_info['source_language'],
                    'final_language': language_info['final_language'],
                    'is_translated': language_info['is_translated']
                },
                processing_method=result['processing_method'],
                metadata=result.get('metadata')
            )
        else:
            return YouTubeTranscriptResponse(
                success=False,
                url=request.url,
                error=result.get('error', 'Unknown error during transcript extraction')
            )
                
    except Exception as e:
        return YouTubeTranscriptResponse(
            success=False,
            url=request.url,
            error=f"YouTube transcript processing error: {str(e)}"
        )


@mcp.tool
async def batch_extract_youtube_transcripts(request: YouTubeBatchRequest) -> YouTubeBatchResponse:
    """
    Extract transcripts from multiple YouTube videos using youtube-transcript-api.
    
    Processes multiple YouTube URLs concurrently with controlled rate limiting
    to avoid overwhelming YouTube's servers.
    
    Features:
    - Concurrent processing with configurable limits
    - youtube-transcript-api integration
    - Comprehensive error handling per video
    - Batch processing statistics
    - No authentication required
    
    Args:
        request: YouTubeBatchRequest containing URLs and extraction parameters
        
    Returns:
        YouTubeBatchResponse with individual results and batch statistics
    """
    try:
        # Validate and limit concurrent requests
        max_concurrent = min(max(1, request.max_concurrent), 5)  # Conservative limit for stability
        
        # Create individual transcript requests
        async def process_single_url(url: str) -> YouTubeTranscriptResponse:
            single_request = YouTubeTranscriptRequest(
                url=url,
                languages=request.languages,
                translate_to=request.translate_to,
                include_timestamps=request.include_timestamps
            )
            return await extract_youtube_transcript(single_request)
        
        # Process URLs with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(url: str) -> YouTubeTranscriptResponse:
            async with semaphore:
                return await process_single_url(url)
        
        # Execute all requests concurrently
        tasks = [process_with_semaphore(url) for url in request.urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        response_results = []
        successful_extractions = 0
        failed_extractions = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                response_results.append(YouTubeTranscriptResponse(
                    success=False,
                    url=request.urls[i],
                    error=f"Processing exception: {str(result)}"
                ))
                failed_extractions += 1
            else:
                response_results.append(result)
                if result.success:
                    successful_extractions += 1
                else:
                    failed_extractions += 1
        
        # Determine overall success
        overall_success = successful_extractions > 0
        
        return YouTubeBatchResponse(
            success=overall_success,
            results=response_results,
            total_urls=len(request.urls),
            successful_extractions=successful_extractions,
            failed_extractions=failed_extractions,
            processing_summary={
                'processing_method': 'youtube_transcript_api_batch',
                'total_processed': len(request.urls),
                'success_rate': f"{(successful_extractions/len(request.urls)*100):.1f}%" if request.urls else "0%",
                'concurrent_limit': max_concurrent
            }
        )
        
    except Exception as e:
        # Return error response for all URLs
        response_results = []
        for url in request.urls:
            response_results.append(YouTubeTranscriptResponse(
                success=False,
                url=url,
                error=f"Batch processing error: {str(e)}"
            ))
        
        return YouTubeBatchResponse(
            success=False,
            results=response_results,
            total_urls=len(request.urls),
            successful_extractions=0,
            failed_extractions=len(request.urls),
            processing_summary={
                'processing_method': 'youtube_transcript_api_batch',
                'error': f"Batch processing failed: {str(e)}"
            }
        )


@mcp.tool
async def get_youtube_video_info(video_url: str) -> Dict[str, Any]:
    """
    Get YouTube video information using youtube-transcript-api.
    
    Retrieves basic video information and transcript availability
    without requiring authentication. This provides transcript language
    availability and basic video details.
    
    Features:
    - youtube-transcript-api integration
    - Transcript availability and language information
    - No authentication required
    - Simple and reliable access to public video data
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Dictionary with video information and transcript details
    """
    try:
        if not youtube_processor.is_youtube_url(video_url):
            return {
                'success': False,
                'error': 'URL is not a valid YouTube video URL',
                'url': video_url
            }
        
        video_id = youtube_processor.extract_video_id(video_url)
        if not video_id:
            return {
                'success': False,
                'error': 'Could not extract video ID from URL',
                'url': video_url
            }
        
        # Get video info using youtube-transcript-api
        video_info = youtube_processor.get_video_info(video_id)
        video_info['url'] = video_url
        video_info['success'] = not video_info.get('error')
        
        # Add processing method information
        if video_info['success']:
            video_info['api_method'] = 'youtube_transcript_api'
            video_info['data_source'] = 'youtube_transcript_api_v1.1.0+'
        
        return video_info
        
    except Exception as e:
        return {
            'success': False,
            'url': video_url,
            'error': f"Failed to get video info: {str(e)}",
            'api_method': 'youtube_transcript_api'
        }


@mcp.tool
async def get_youtube_api_setup_guide() -> Dict[str, Any]:
    """
    Get setup information for youtube-transcript-api integration.
    
    Provides information about the current youtube-transcript-api setup,
    which requires no authentication or API keys for basic transcript extraction.
    
    Returns:
        Dictionary with setup information, capabilities, and usage tips
    """
    try:
        # Check if youtube-transcript-api is available
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            youtube_transcript_available = True
            youtube_transcript_version = getattr(YouTubeTranscriptApi, '__version__', 'unknown')
        except ImportError:
            youtube_transcript_available = False
            youtube_transcript_version = None
        
        # Current configuration status
        config_status = {
            'youtube_transcript_api_installed': youtube_transcript_available,
            'youtube_transcript_api_version': youtube_transcript_version,
            'authentication_required': False,
            'api_keys_required': False,
            'setup_required': not youtube_transcript_available
        }
        
        # Determine setup completeness
        setup_complete = youtube_transcript_available
        
        return {
            'success': True,
            'setup_complete': setup_complete,
            'current_configuration': config_status,
            'setup_guide': {
                'title': 'YouTube Transcript API Setup (Simple)',
                'description': 'youtube-transcript-api requires no authentication for public video transcripts',
                'installation': {
                    'step_1': 'Install the package: pip install youtube-transcript-api',
                    'step_2': 'No additional configuration needed',
                    'step_3': 'Ready to extract transcripts from public YouTube videos'
                },
                'capabilities': {
                    'transcript_extraction': 'Extract transcripts from public YouTube videos',
                    'multi_language': 'Support for multiple languages with automatic fallback',
                    'auto_generated': 'Access both manual and auto-generated captions',
                    'translation': 'Basic translation support for available transcript languages',
                    'no_quota_limits': 'No API quotas or rate limits (within reasonable usage)'
                },
                'limitations': {
                    'public_videos_only': 'Only works with public videos that have transcripts',
                    'no_metadata': 'Limited video metadata (no view counts, descriptions, etc.)',
                    'transcript_dependent': 'Requires videos to have captions/transcripts available'
                }
            },
            'next_steps': {
                'if_not_setup': [
                    '1. Install youtube-transcript-api: pip install youtube-transcript-api',
                    '2. No additional configuration needed',
                    '3. Start extracting transcripts immediately'
                ],
                'if_setup_complete': [
                    '1. YouTube transcript extraction is ready to use',
                    '2. Call extract_youtube_transcript with any public YouTube URL',
                    '3. No authentication or API keys needed'
                ]
            },
            'troubleshooting': {
                'common_issues': {
                    'import_error': 'Install youtube-transcript-api: pip install youtube-transcript-api',
                    'no_transcript_found': 'Video may not have transcripts available or may be private',
                    'transcripts_disabled': 'Video owner has disabled transcripts for this video',
                    'video_unavailable': 'Video may be private, deleted, or region-restricted'
                }
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to get setup guide: {str(e)}"
        }


@mcp.tool
async def search_google(request: GoogleSearchRequest) -> GoogleSearchResponse:
    """
    Perform Google search and return structured results.
    
    Args:
        request: GoogleSearchRequest with search query and parameters
        
    Returns:
        GoogleSearchResponse with search results and metadata
    """
    try:
        # Validate max_concurrent for individual searches (limit to reasonable range)
        num_results = max(1, min(100, request.num_results))
        
        # Perform search
        result = await google_search_processor.search_google(
            query=request.query,
            num_results=num_results,
            language=request.language,
            region=request.region,
            safe_search=request.safe_search,
            search_genre=request.search_genre
        )
        
        if result['success']:
            return GoogleSearchResponse(
                success=True,
                query=result['query'],
                total_results=result['total_results'],
                results=result['results'],
                search_metadata=result['search_metadata']
            )
        else:
            return GoogleSearchResponse(
                success=False,
                query=request.query,
                error=result.get('error')
            )
            
    except Exception as e:
        return GoogleSearchResponse(
            success=False,
            query=request.query,
            error=f"Google search error: {str(e)}"
        )


@mcp.tool
async def batch_search_google(request: GoogleBatchSearchRequest) -> GoogleBatchSearchResponse:
    """
    Perform multiple Google searches in batch with analysis.
    
    Args:
        request: GoogleBatchSearchRequest with multiple queries and parameters
        
    Returns:
        GoogleBatchSearchResponse with batch results and analysis
    """
    try:
        # Validate and limit parameters
        max_concurrent = max(1, min(5, request.max_concurrent))  # Be respectful to Google
        num_results = max(1, min(100, request.num_results_per_query))
        
        # Perform batch search
        batch_results = await google_search_processor.batch_search(
            queries=request.queries,
            num_results_per_query=num_results,
            max_concurrent=max_concurrent,
            language=request.language,
            region=request.region,
            search_genre=request.search_genre
        )
        
        # Convert results to response objects
        response_results = []
        successful = 0
        failed = 0
        
        for result in batch_results:
            if result['success']:
                successful += 1
                response_results.append(GoogleSearchResponse(
                    success=True,
                    query=result['query'],
                    total_results=result['total_results'],
                    results=result['results'],
                    search_metadata=result['search_metadata']
                ))
            else:
                failed += 1
                response_results.append(GoogleSearchResponse(
                    success=False,
                    query=result['query'],
                    error=result.get('error')
                ))
        
        # Generate analysis
        analysis = google_search_processor.analyze_search_results(batch_results)
        
        return GoogleBatchSearchResponse(
            success=True,
            total_queries=len(request.queries),
            successful_searches=successful,
            failed_searches=failed,
            results=response_results,
            analysis=analysis.get('analysis') if analysis.get('success') else None
        )
        
    except Exception as e:
        return GoogleBatchSearchResponse(
            success=False,
            total_queries=len(request.queries),
            successful_searches=0,
            failed_searches=len(request.queries),
            results=[],
            analysis={"error": f"Batch search failed: {str(e)}"}
        )


@mcp.tool
async def search_and_crawl(
    search_query: str,
    num_search_results: int = 5,
    crawl_top_results: int = 3,
    extract_media: bool = False,
    generate_markdown: bool = True,
    search_genre: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform Google search and crawl the top results for comprehensive content analysis.
    
    Args:
        search_query: Google search query
        num_search_results: Number of search results to retrieve (1-20)
        crawl_top_results: Number of top results to crawl (1-10)
        extract_media: Whether to extract media from crawled pages
        generate_markdown: Whether to generate markdown content
        search_genre: Optional search genre for content filtering
        
    Returns:
        Dictionary with search results and crawled content
    """
    try:
        # Validate parameters
        num_search_results = max(1, min(20, num_search_results))
        crawl_top_results = max(1, min(10, min(crawl_top_results, num_search_results)))
        
        # Step 1: Perform Google search
        search_result = await google_search_processor.search_google(
            query=search_query,
            num_results=num_search_results,
            search_genre=search_genre
        )
        
        if not search_result['success']:
            return {
                'success': False,
                'error': f"Search failed: {search_result.get('error')}",
                'search_query': search_query
            }
        
        search_results = search_result['results']
        if not search_results:
            return {
                'success': False,
                'error': 'No search results found',
                'search_query': search_query
            }
        
        # Step 2: Crawl top results
        top_urls = [result['url'] for result in search_results[:crawl_top_results]]
        
        crawl_results = []
        for i, url in enumerate(top_urls):
            try:
                crawl_request = CrawlRequest(
                    url=url,
                    extract_media=extract_media,
                    generate_markdown=generate_markdown,
                    timeout=30
                )
                
                crawl_result = await _internal_crawl_url(crawl_request)
                
                # Add search ranking to crawl result
                crawl_data = {
                    'search_rank': i + 1,
                    'search_result': search_results[i],
                    'crawl_result': {
                        'success': crawl_result.success,
                        'url': crawl_result.url,
                        'title': crawl_result.title,
                        'content_length': len(crawl_result.content or ''),
                        'has_content': bool(crawl_result.content),
                        'has_markdown': bool(crawl_result.markdown),
                        'has_media': bool(crawl_result.media),
                        'error': crawl_result.error
                    }
                }
                
                # Include full content for successful crawls
                if crawl_result.success:
                    crawl_data['content'] = {
                        'title': crawl_result.title,
                        'content': crawl_result.content,
                        'markdown': crawl_result.markdown,
                        'media': crawl_result.media if extract_media else None
                    }
                
                crawl_results.append(crawl_data)
                
            except Exception as e:
                crawl_results.append({
                    'search_rank': i + 1,
                    'search_result': search_results[i],
                    'crawl_result': {
                        'success': False,
                        'url': url,
                        'error': f"Crawling failed: {str(e)}"
                    }
                })
        
        # Step 3: Generate summary
        successful_crawls = sum(1 for r in crawl_results if r['crawl_result']['success'])
        total_content_length = sum(
            r['crawl_result'].get('content_length', 0) 
            for r in crawl_results 
            if r['crawl_result']['success']
        )
        
        return {
            'success': True,
            'search_query': search_query,
            'search_metadata': search_result['search_metadata'],
            'crawl_summary': {
                'total_search_results': len(search_results),
                'urls_crawled': len(crawl_results),
                'successful_crawls': successful_crawls,
                'failed_crawls': len(crawl_results) - successful_crawls,
                'total_content_length': total_content_length,
                'success_rate': f"{(successful_crawls/len(crawl_results)*100):.1f}%" if crawl_results else "0%"
            },
            'search_results': search_results,
            'crawled_content': crawl_results,
            'processing_method': 'search_and_crawl_integration'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Search and crawl error: {str(e)}",
            'search_query': search_query
        }


@mcp.tool
async def get_search_genres() -> Dict[str, Any]:
    """
    Get list of available search genres for content filtering.
    
    Returns:
        Dictionary with available genres and their descriptions
    """
    try:
        genres = google_search_processor.get_available_genres()
        
        return {
            "success": True,
            "total_genres": len(genres),
            "genres": genres,
            "categories": {
                "Academic & Educational": ["academic", "research", "education"],
                "News & Media": ["news", "latest_news"],
                "Technical & Development": ["technical", "programming", "documentation"],
                "Commerce & Shopping": ["shopping", "reviews"],
                "Social & Community": ["forum", "social"],
                "Media & Entertainment": ["video", "images"],
                "Government & Official": ["government", "legal"],
                "File Types": ["pdf", "documents", "presentations", "spreadsheets"],
                "Time-based": ["recent", "historical"],
                "Language & Region": ["japanese", "english"],
                "Content Quality": ["authoritative", "beginner", "advanced"]
            },
            "usage_examples": [
                {"genre": "academic", "example": "Find academic papers about machine learning"},
                {"genre": "programming", "example": "Search for Python programming tutorials and code examples"},
                {"genre": "news", "example": "Get latest news articles from major news sources"},
                {"genre": "pdf", "example": "Find PDF documents and research papers"},
                {"genre": "beginner", "example": "Search for beginner-friendly tutorials and guides"}
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get search genres: {str(e)}"
        }


@mcp.resource("uri://crawl4ai/config")
async def get_crawler_config() -> str:
    """
    Get the default crawler configuration options.
    
    Returns:
        JSON string with configuration options
    """
    config_options = {
        "browser_config": {
            "headless": True,
            "viewport_width": 1920,
            "viewport_height": 1080,
            "user_agent": "Mozilla/5.0 (compatible; Crawl4AI/1.0)",
        },
        "crawler_config": {
            "timeout": 30,
            "extract_media": False,
            "screenshot": False,
            "generate_markdown": True,
            "remove_overlay_elements": True,
        },
        "extraction_strategies": {
            "css": "Use CSS selectors for targeted extraction",
            "xpath": "Use XPath expressions for complex queries",
            "llm": "Use LLM-based extraction with custom schemas",
        }
    }
    
    return json.dumps(config_options, indent=2)


@mcp.resource("uri://crawl4ai/examples")
async def get_usage_examples() -> str:
    """
    Get usage examples for the Crawl4AI MCP server.
    
    Returns:
        JSON string with example requests
    """
    examples = {
        "basic_crawl": {
            "description": "Basic URL crawling",
            "request": {
                "url": "https://example.com",
                "generate_markdown": True,
                "take_screenshot": False
            }
        },
        "css_extraction": {
            "description": "Extract specific content using CSS selectors",
            "request": {
                "url": "https://news.ycombinator.com",
                "css_selector": ".storylink"
            }
        },
        "structured_extraction": {
            "description": "Extract structured data with schema",
            "request": {
                "url": "https://example-store.com/product/123",
                "schema": {
                    "name": "Product name",
                    "price": "Product price",
                    "description": "Product description"
                },
                "extraction_type": "css",
                "css_selectors": {
                    "name": "h1.product-title",
                    "price": ".price",
                    "description": ".description"
                }
            }
        }
    }
    
    return json.dumps(examples, indent=2)


@mcp.resource("uri://crawl4ai/file-processing")
async def get_file_processing_examples() -> str:
    """
    Get usage examples for file processing with MarkItDown.
    
    Returns:
        JSON string with file processing examples
    """
    examples = {
        "pdf_processing": {
            "description": "Process PDF document and extract text content",
            "request": {
                "url": "https://example.com/document.pdf",
                "max_size_mb": 50,
                "include_metadata": True
            }
        },
        "office_document": {
            "description": "Process Microsoft Word document",
            "request": {
                "url": "https://example.com/report.docx",
                "max_size_mb": 25,
                "include_metadata": True
            }
        },
        "excel_spreadsheet": {
            "description": "Process Excel spreadsheet",
            "request": {
                "url": "https://example.com/data.xlsx",
                "max_size_mb": 30,
                "include_metadata": True
            }
        },
        "powerpoint_presentation": {
            "description": "Process PowerPoint presentation",
            "request": {
                "url": "https://example.com/slides.pptx",
                "max_size_mb": 40,
                "include_metadata": True
            }
        },
        "zip_archive": {
            "description": "Process ZIP archive containing multiple files",
            "request": {
                "url": "https://example.com/documents.zip",
                "max_size_mb": 100,
                "extract_all_from_zip": True,
                "include_metadata": True
            }
        },
        "large_file_processing": {
            "description": "Process large file with custom size limit",
            "request": {
                "url": "https://example.com/large-document.pdf",
                "max_size_mb": 200,
                "include_metadata": False
            }
        }
    }
    
    return json.dumps(examples, indent=2)


@mcp.prompt
def crawl_website_prompt(url: str, extraction_type: str = "basic"):
    """
    Create a prompt for crawling a website with specific instructions.
    
    Args:
        url: The URL to crawl
        extraction_type: Type of extraction (basic, structured, content)
    
    Returns:
        List of prompt messages for crawling
    """
    if extraction_type == "basic":
        content = f"""URLをクローリングしてコンテンツを取得してください: {url}

基本的なクローリングを実行し、以下の情報を取得してください：
- ページタイトル
- メインコンテンツ
- Markdown形式のテキスト

crawl_urlツールを使用してください。"""

    elif extraction_type == "structured":
        content = f"""構造化データ抽出のためにURLをクローリングしてください: {url}

以下の手順で実行してください：
1. ページの構造を分析
2. 適切なCSSセレクターまたはXPathを特定
3. extract_structured_dataツールを使用して構造化データを抽出

抽出したいデータの種類を指定してください（例：記事タイトル、価格、説明文など）。"""

    elif extraction_type == "content":
        content = f"""コンテンツ分析のためにURLをクローリングしてください: {url}

以下の分析を行ってください：
1. ページの主要コンテンツを抽出
2. 重要な見出しやセクションを特定
3. メディアファイル（画像、動画など）があれば一覧化
4. ページの構造と内容を要約

crawl_urlツールを使用し、extract_media=trueに設定してください。"""

    else:
        content = f"""URLをクローリングしてください: {url}

利用可能な抽出タイプ：
- basic: 基本的なコンテンツ取得
- structured: 構造化データ抽出
- content: 詳細なコンテンツ分析

適切なツールを選択して実行してください。"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


@mcp.prompt
def analyze_crawl_results_prompt(crawl_data: str):
    """
    Create a prompt for analyzing crawl results.
    
    Args:
        crawl_data: JSON string of crawl results
    
    Returns:
        List of prompt messages for analysis
    """
    content = f"""以下のクローリング結果を分析してください：

{crawl_data}

分析項目：
1. 取得したコンテンツの概要
2. 主要な情報やキーポイント
3. データの構造と品質
4. 追加で抽出すべき情報があるか
5. 結果の有用性と改善点

詳細な分析レポートを提供してください。"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


@mcp.prompt
def batch_crawl_setup_prompt(urls: str):
    """
    Create a prompt for setting up batch crawling.
    
    Args:
        urls: Comma-separated list of URLs
    
    Returns:
        List of prompt messages for batch crawling
    """
    url_list = [url.strip() for url in urls.split(",")]
    
    content = f"""複数のURLを一括でクローリングしてください：

対象URL（{len(url_list)}件）：
{chr(10).join(f"- {url}" for url in url_list)}

batch_crawlツールを使用して以下を実行してください：
1. 全URLのコンテンツを取得
2. 各ページの基本情報を収集
3. 結果を比較・分析
4. 共通点や相違点を特定
5. 統合レポートを作成

効率的な一括処理を行い、結果をまとめて報告してください。"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


@mcp.prompt
def process_file_prompt(file_url: str, file_type: str = "auto"):
    """
    Create a prompt for processing files with MarkItDown.
    
    Args:
        file_url: URL of the file to process
        file_type: Type of file processing (auto, pdf, office, zip)
    
    Returns:
        List of prompt messages for file processing
    """
    if file_type == "pdf":
        content = f"""PDFファイルを処理してMarkdown形式に変換してください: {file_url}

以下の機能を使用してPDFを処理してください：
1. process_fileツールを使用してPDFをダウンロード・処理
2. テキスト内容をMarkdown形式で抽出
3. 文書の構造（見出し、段落、リストなど）を保持
4. メタデータ（タイトル、作成者、作成日など）を取得
5. 処理結果を分析・要約

PDFの内容を理解しやすい形式で提示してください。"""

    elif file_type == "office":
        content = f"""Microsoft Officeファイルを処理してMarkdown形式に変換してください: {file_url}

以下の手順で処理してください：
1. process_fileツールを使用してOfficeファイルを処理
2. 文書の種類に応じた適切な抽出を実行
   - Word: テキスト、見出し、表、画像キャプション
   - Excel: シート名、セルデータ、表形式
   - PowerPoint: スライドタイトル、コンテンツ、ノート
3. 元の構造とフォーマットを可能な限り保持
4. メタデータと追加情報を含める

処理した内容を構造化された形式で提示してください。"""

    elif file_type == "zip":
        content = f"""ZIPアーカイブを処理して中身のファイルを分析してください: {file_url}

以下の処理を実行してください：
1. process_fileツールを使用してZIPファイルを処理
2. アーカイブ内の全ファイルを抽出・分析
3. 各ファイルの種類と内容を特定
4. サポートされているファイル形式をMarkdownに変換
5. ファイル構造とディレクトリ階層を可視化
6. 処理できなかったファイルがあれば理由を説明

全体的な分析結果と各ファイルの詳細を報告してください。"""

    else:  # auto
        content = f"""ファイルを自動検出して適切に処理してください: {file_url}

以下の手順で処理してください：
1. get_supported_file_formatsツールでサポート形式を確認
2. process_fileツールを使用してファイルを処理
3. ファイル形式に応じた最適な抽出を実行
4. 内容をMarkdown形式で構造化
5. メタデータと追加情報を取得
6. 処理結果を分析・要約

ファイルの種類と内容に応じた詳細な分析を提供してください。

利用可能なツール：
- process_file: ファイル処理とMarkdown変換
- get_supported_file_formats: サポート形式の確認"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


def main():
    """Main entry point for the MCP server."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Crawl4AI MCP Server")
        print("Usage: python -m crawl4ai_mcp.server [--transport TRANSPORT] [--host HOST] [--port PORT]")
        print("Transports: stdio (default), streamable-http, sse")
        return
    
    # Parse command line arguments
    transport = "stdio"
    host = "127.0.0.1"
    port = 8000
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1
    
    # Run the server
    if transport == "stdio":
        mcp.run()
    elif transport == "streamable-http" or transport == "http":
        mcp.run(transport="streamable-http", host=host, port=port)
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        print(f"Unknown transport: {transport}")
        print("Available transports: stdio, streamable-http, sse")
        sys.exit(1)


if __name__ == "__main__":
    main()