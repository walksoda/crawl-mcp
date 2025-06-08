"""
Crawl4AI MCP Server

A Model Context Protocol server that provides web crawling and content extraction
capabilities using the crawl4ai library.
"""

import asyncio
import json
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
    RegexExtractionStrategy,
    create_extraction_strategy,
)
from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from .suppress_output import suppress_stdout_stderr
from .file_processor import FileProcessor


class CrawlRequest(BaseModel):
    """Request model for crawling operations."""
    url: str = Field(..., description="URL to crawl")
    css_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    xpath: Optional[str] = Field(None, description="XPath selector for content extraction")
    extract_media: bool = Field(False, description="Whether to extract media files")
    take_screenshot: bool = Field(False, description="Whether to take a screenshot")
    generate_markdown: bool = Field(True, description="Whether to generate markdown")
    wait_for_selector: Optional[str] = Field(None, description="Wait for specific element")
    timeout: int = Field(30, description="Request timeout in seconds")
    
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


@mcp.tool
async def crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    try:
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
        chunking_config = {}
        if request.chunk_content:
            chunking_config = {
                "chunk_token_threshold": request.chunk_size,
                "overlap_rate": request.overlap_rate
            }

        # Create config without content_filter (will be applied separately)
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
            **chunking_config
        }
        
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
                
                result = await crawler.arun(url=request.url, config=config)
        
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
async def deep_crawl_site(
    url: str,
    max_depth: int = 2,
    max_pages: int = 20,
    crawl_strategy: str = "bfs",
    include_external: bool = False,
    url_pattern: Optional[str] = None,
    score_threshold: float = 0.3,
    extract_media: bool = False
) -> Dict[str, Any]:
    """
    Perform deep crawling of a website with specified depth and strategy.
    
    Args:
        url: Starting URL for deep crawling
        max_depth: Maximum crawling depth (recommended: 1-3)
        max_pages: Maximum number of pages to crawl
        crawl_strategy: Strategy - 'bfs', 'dfs', or 'best_first'
        include_external: Whether to follow external domain links
        url_pattern: URL pattern filter (e.g., '*docs*', '*blog*')
        score_threshold: Minimum score for URLs to be crawled (0.0-1.0)
        extract_media: Whether to extract media files
        
    Returns:
        Dictionary with crawled pages information and site map
    """
    try:
        # Create filter chain
        filters = []
        if url_pattern:
            filters.append(URLPatternFilter(patterns=[url_pattern]))
        if not include_external:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            filters.append(DomainFilter(allowed_domains=[domain]))
        
        filter_chain = FilterChain(filters) if filters else None
        
        # Select crawling strategy
        if crawl_strategy == "dfs":
            strategy = DFSDeepCrawlStrategy(
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=include_external,
                filter_chain=filter_chain,
                score_threshold=score_threshold
            )
        elif crawl_strategy == "best_first":
            strategy = BestFirstCrawlingStrategy(
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=include_external,
                filter_chain=filter_chain,
                score_threshold=score_threshold
            )
        else:  # Default to BFS
            strategy = BFSDeepCrawlStrategy(
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=include_external,
                filter_chain=filter_chain,
                score_threshold=score_threshold
            )

        config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            exclude_all_images=not extract_media,
            verbose=False,
            log_console=False,
        )
        
        with suppress_stdout_stderr():
            async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)
        
        if result.success:
            site_map = {
                "starting_url": url,
                "strategy_used": crawl_strategy,
                "total_pages_crawled": 0,
                "pages": [],
                "site_structure": {},
                "content_summary": ""
            }
            
            if hasattr(result, 'crawled_pages'):
                site_map["total_pages_crawled"] = len(result.crawled_pages)
                
                for page in result.crawled_pages:
                    page_info = {
                        "url": page.url,
                        "title": page.metadata.get("title", "No title") if page.metadata else "No title",
                        "content_length": len(page.cleaned_html) if page.cleaned_html else 0,
                        "links_found": len(page.links.get("internal", [])) if hasattr(page, 'links') else 0,
                        "depth": getattr(page, 'depth', 0),
                        "content_preview": (page.cleaned_html[:200] + "...") if page.cleaned_html else "",
                        "markdown_preview": (page.markdown[:200] + "...") if page.markdown else ""
                    }
                    site_map["pages"].append(page_info)
                
                # Create content summary
                total_content = sum(p["content_length"] for p in site_map["pages"])
                site_map["content_summary"] = f"Crawled {site_map['total_pages_crawled']} pages with {total_content} characters of content"
            else:
                # Single page result
                site_map["total_pages_crawled"] = 1
                site_map["pages"].append({
                    "url": url,
                    "title": result.metadata.get("title", "No title") if result.metadata else "No title",
                    "content_length": len(result.cleaned_html) if result.cleaned_html else 0,
                    "content_preview": (result.cleaned_html[:200] + "...") if result.cleaned_html else "",
                    "markdown_preview": (result.markdown[:200] + "...") if result.markdown else ""
                })
            
            return site_map
        else:
            return {
                "success": False,
                "error": f"Deep crawling failed: {result.error_message}",
                "starting_url": url
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Deep crawling error: {str(e)}",
            "starting_url": url
        }


@mcp.tool
async def intelligent_extract(
    url: str,
    extraction_goal: str,
    content_filter: str = "bm25",
    filter_query: Optional[str] = None,
    chunk_content: bool = False,
    use_llm: bool = True,
    llm_provider: str = "openai",
    llm_model: str = "gpt-3.5-turbo",
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
        llm_provider: LLM provider (openai, claude, etc.)
        llm_model: Specific model to use
        custom_instructions: Custom instructions for extraction
        
    Returns:
        Dictionary with extracted content and metadata
    """
    try:
        # Setup content filter
        content_filter_strategy = None
        if content_filter == "bm25" and filter_query:
            content_filter_strategy = BM25ContentFilter(query=filter_query)
        elif content_filter == "pruning":
            content_filter_strategy = PruningContentFilter(threshold=0.5)
        elif content_filter == "llm" and use_llm:
            instructions = custom_instructions or f"Extract content related to: {extraction_goal}"
            content_filter_strategy = LLMContentFilter(
                provider=llm_provider,
                model=llm_model,
                instructions=instructions
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
            
            extraction_strategy = LLMExtractionStrategy(
                provider=llm_provider,
                model=llm_model,
                schema=schema,
                extraction_type="schema",
                instruction=instructions
            )

        # Configure chunking
        chunking_config = {}
        if chunk_content:
            chunking_config = {
                "chunk_token_threshold": 1000,
                "overlap_rate": 0.1
            }

        # Setup crawler configuration
        config_params = {
            "extraction_strategy": extraction_strategy,
            "verbose": False,
            "log_console": False,
            **chunking_config
        }
        
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
async def extract_entities(
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

        # Create regex extraction strategy
        extraction_strategy = RegexExtractionStrategy(patterns=patterns)
        
        config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            verbose=False,
            log_console=False,
        )

        with suppress_stdout_stderr():
            async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
                result = await crawler.arun(url=url, config=config)

        if result.success:
            extracted_entities = {}
            
            if result.extracted_content:
                try:
                    raw_extracted = json.loads(result.extracted_content)
                    
                    # Process each entity type
                    for entity_type, matches in raw_extracted.items():
                        if matches:
                            entity_list = []
                            seen = set() if deduplicate else None
                            
                            for match in matches:
                                entity_value = match if isinstance(match, str) else match.get('match', str(match))
                                
                                if deduplicate:
                                    if entity_value in seen:
                                        continue
                                    seen.add(entity_value)
                                
                                entity_info = {"value": entity_value}
                                
                                # Add context if requested and available
                                if include_context and isinstance(match, dict):
                                    entity_info["context"] = match.get('context', '')
                                    entity_info["position"] = match.get('position', -1)
                                
                                entity_list.append(entity_info)
                            
                            extracted_entities[entity_type] = {
                                "count": len(entity_list),
                                "entities": entity_list
                            }
                        else:
                            extracted_entities[entity_type] = {"count": 0, "entities": []}
                
                except json.JSONDecodeError:
                    extracted_entities = {"raw_content": result.extracted_content}

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
async def extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Extract structured data from a URL using CSS selectors or LLM-based extraction.
    
    Args:
        request: StructuredExtractionRequest with URL, schema, and extraction parameters
        
    Returns:
        CrawlResponse with extracted structured data
    """
    try:
        if request.extraction_type == "css" and request.css_selectors:
            strategy = JsonCssExtractionStrategy(request.css_selectors)
        elif request.extraction_type == "llm":
            strategy = LLMExtractionStrategy(
                provider=request.llm_provider,
                api_token=None,  # Should be set via environment variable
                schema=request.schema,
                extraction_type="schema",
                model=request.llm_model,
            )
        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error="Invalid extraction type or missing CSS selectors"
            )
            
        config = CrawlerRunConfig(
            extraction_strategy=strategy,
            timeout=30,
            verbose=False,
            log_console=False,
        )
        
        with suppress_stdout_stderr():
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=request.url, config=config)
        
        if result.success:
            extracted_data = json.loads(result.extracted_content) if result.extracted_content else None
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
                page_timeout=15000,
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
                page_timeout=10000,
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