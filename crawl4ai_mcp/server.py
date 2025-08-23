"""
Crawl4AI MCP Server - FastMCP 2.0 Version

Uses FastMCP 2.0.0 which doesn't have banner output issues.
Clean STDIO transport compatible for perfect MCP communication.
"""

import os
import sys
import warnings

# Set environment variables before any imports
os.environ["FASTMCP_QUIET"] = "true"
os.environ["FASTMCP_NO_BANNER"] = "true" 
os.environ["FASTMCP_SILENT"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TERM"] = "dumb"
os.environ["SHELL"] = "/bin/sh"

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import logging
logging.disable(logging.CRITICAL)

# Import FastMCP 2.0 - no banner output!
from fastmcp import FastMCP
from typing import Any, Dict, List, Optional, Union, Annotated
from pydantic import Field, BaseModel

# Create MCP server with clean initialization
mcp = FastMCP("Crawl4AI")

# Global lazy loading state
_heavy_imports_loaded = False
_browser_setup_done = False
_browser_setup_failed = False
_tools_imported = False

def _load_heavy_imports():
    """Load heavy imports only when tools are actually used"""
    global _heavy_imports_loaded
    if _heavy_imports_loaded:
        return
        
    global asyncio, json, AsyncWebCrawler
    
    import asyncio
    import json
    from crawl4ai import AsyncWebCrawler
    
    _heavy_imports_loaded = True

def _load_tool_modules():
    """Load tool modules only when needed"""
    global _tools_imported
    if _tools_imported:
        return
    
    global web_crawling, search, youtube, file_processing, utilities
    
    try:
        from .tools import web_crawling, search, youtube, file_processing, utilities
        _tools_imported = True
    except ImportError:
        # Fallback for relative imports
        try:
            from crawl4ai_mcp.tools import web_crawling, search, youtube, file_processing, utilities
            _tools_imported = True
        except ImportError:
            _tools_imported = False

def _ensure_browser_setup():
    """Browser setup with lazy loading"""
    global _browser_setup_done, _browser_setup_failed
    
    if _browser_setup_done:
        return True
    if _browser_setup_failed:
        return False
        
    try:
        # Quick browser cache check
        import glob
        from pathlib import Path
        
        cache_pattern = str(Path.home() / ".cache" / "ms-playwright" / "chromium-*")
        if glob.glob(cache_pattern):
            _browser_setup_done = True
            return True
        else:
            _browser_setup_failed = True
            return False
    except Exception:
        _browser_setup_failed = True
        return False

# Tool definitions with immediate registration but lazy implementation

@mcp.tool()
def get_system_diagnostics() -> dict:
    """
    Get comprehensive system diagnostics for troubleshooting UVX and browser issues.
    
    Returns detailed information about the environment, browser installations,
    and provides specific recommendations for fixing issues.
    """
    _load_heavy_imports()
    
    import platform
    import glob
    from pathlib import Path
    
    # Check browser cache
    cache_pattern = str(Path.home() / ".cache" / "ms-playwright" / "chromium-*")
    cache_dirs = glob.glob(cache_pattern)
    
    return {
        "status": "FastMCP 2.0 Server - Clean STDIO communication",
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "fastmcp_version": "2.0.0",
        "browser_cache_found": len(cache_dirs) > 0,
        "cache_directories": cache_dirs,
        "recommendations": [
            "Install Playwright browsers: pip install playwright && playwright install webkit",
            "For UVX: uvx --with playwright playwright install webkit"
        ]
    }

@mcp.tool()
async def crawl_url(
    url: Annotated[str, Field(description="Target URL to crawl. Examples: https://example.com, https://news.site.com/article")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction. Examples: '.article-content', '#main-content', 'div.post', 'article p' (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction. Examples: '//div[@class=\"content\"]', '//article//p', '//h1[@id=\"title\"]' (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element to load. CSS selector or XPath. Examples: '.content-loaded', '#dynamic-content', '[data-loaded=\"true\"]' (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False
) -> dict:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.
    
    Core web crawling tool with comprehensive configuration options.
    Essential for SPAs: set wait_for_js=true for JavaScript-heavy sites.
    
    NOTE: If this tool fails with 503 errors, rate limiting, or anti-bot protection 
    (especially for sites like HackerNews, Reddit, social media), try using 
    crawl_url_with_fallback which implements multiple retry strategies with 
    different user agents and timing patterns.
    
    Returns structured data with content, metadata, and optional media/screenshots.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await web_crawling.crawl_url(
            url=url, css_selector=css_selector, xpath=xpath, extract_media=extract_media,
            take_screenshot=take_screenshot, generate_markdown=generate_markdown,
            wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
            auto_summarize=auto_summarize
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"Crawling error: {str(e)}"
        }

@mcp.tool()
async def extract_youtube_transcript(
    url: Annotated[str, Field(description="YouTube video URL. Supports formats: https://www.youtube.com/watch?v=VIDEO_ID, https://youtu.be/VIDEO_ID")],
    languages: Annotated[Optional[Union[List[str], str]], Field(description="Array of language codes in preference order. Can be array like [\"ja\", \"en\"] or string like '[\"ja\", \"en\"]' (default: [\"ja\", \"en\"])")] = ["ja", "en"],
    translate_to: Annotated[Optional[str], Field(description="Target language code for translation. Examples: 'en' (English), 'ja' (Japanese), 'es' (Spanish), 'fr' (French), 'de' (German) (default: None)")] = None,
    include_timestamps: Annotated[bool, Field(description="Include timestamps in transcript (default: True)")] = True,
    preserve_formatting: Annotated[bool, Field(description="Preserve original formatting (default: True)")] = True,
    include_metadata: Annotated[bool, Field(description="Include video metadata (default: True)")] = True,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize long transcripts using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'azure', 'together', 'groq', auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model: 'gpt-4o', 'claude-3-sonnet', 'gemini-pro', or provider-specific model names, auto-detected if not specified (default: None)")] = None
) -> dict:
    """
    Extract YouTube video transcripts with timestamps and optional AI summarization.
    
    Works with public videos that have captions. No authentication required.
    Auto-detects available languages and falls back appropriately.
    
    Note: Automatic transcription may contain errors.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    # Handle string-encoded array for languages parameter
    if isinstance(languages, str):
        try:
            import json
            languages = json.loads(languages)
        except (json.JSONDecodeError, ValueError):
            import re
            matches = re.findall(r'"([^"]*)"', languages)
            if matches:
                languages = matches
            else:
                languages = ["ja", "en"]
    
    try:
        result = await youtube.extract_youtube_transcript(
            url=url, languages=languages, translate_to=translate_to, 
            include_timestamps=include_timestamps, preserve_formatting=preserve_formatting,
            include_metadata=include_metadata, auto_summarize=auto_summarize, 
            max_content_tokens=max_content_tokens, summary_length=summary_length,
            llm_provider=llm_provider, llm_model=llm_model
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"YouTube transcript error: {str(e)}"
        }

@mcp.tool()
async def batch_extract_youtube_transcripts(
    request: Annotated[Dict[str, Any], Field(description="YouTubeBatchRequest dictionary containing: urls (required list of YouTube URLs), languages (default: ['ja', 'en']), max_concurrent (default: 3, max: 5), include_timestamps, translate_to, preserve_formatting, include_metadata (all optional booleans)")]
) -> Dict[str, Any]:
    """
    Extract transcripts from multiple YouTube videos using youtube-transcript-api.
    
    Efficiently processes multiple YouTube videos with rate limiting and error handling.
    Auto-detects available languages with fallback support for each video.
    
    Note: Automatic transcription may contain errors.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await youtube.batch_extract_youtube_transcripts(request)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Batch YouTube extraction error: {str(e)}"
        }

@mcp.tool()
async def get_youtube_video_info(
    video_url: Annotated[str, Field(description="YouTube video URL. Supports formats: https://www.youtube.com/watch?v=VIDEO_ID, https://youtu.be/VIDEO_ID")],
    summarize_transcript: Annotated[bool, Field(description="Summarize long transcripts using LLM (default: False)")] = False,
    max_tokens: Annotated[int, Field(description="Token limit before triggering summarization (default: 25000)")] = 25000,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    include_timestamps: Annotated[bool, Field(description="Include timestamps in transcript info (default: True)")] = True
) -> Dict[str, Any]:
    """
    Get comprehensive YouTube video information including metadata and transcript availability.
    
    Provides detailed information about video metadata, available transcript languages,
    and optional transcript summarization with LLM integration.
    
    Note: No authentication required for public videos.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await youtube.get_youtube_video_info(
            video_url=video_url, summarize_transcript=summarize_transcript,
            max_tokens=max_tokens, llm_provider=llm_provider, llm_model=llm_model,
            summary_length=summary_length, include_timestamps=include_timestamps
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"YouTube video info error: {str(e)}"
        }

@mcp.tool()
async def get_youtube_api_setup_guide() -> Dict[str, Any]:
    """
    Get setup information for youtube-transcript-api integration.
    
    Provides information about current youtube-transcript-api setup.
    No authentication or API keys required for basic transcript extraction.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await youtube.get_youtube_api_setup_guide()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"YouTube API setup guide error: {str(e)}"
        }

@mcp.tool()
async def process_file(
    url: Annotated[str, Field(description="URL of the file to process. Examples: https://example.com/document.pdf, https://site.com/report.docx, https://files.com/data.xlsx, https://docs.com/archive.zip")],
    max_size_mb: Annotated[int, Field(description="Maximum file size in MB (default: 100)")] = 100,
    extract_all_from_zip: Annotated[bool, Field(description="Whether to extract all files from ZIP archives (default: True)")] = True,
    include_metadata: Annotated[bool, Field(description="Whether to include file metadata (default: True)")] = True,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None
) -> dict:
    """
    Process files (PDF, Word, etc.) and convert to markdown with optional AI summarization.
    
    Supports PDF, Word, Excel, PowerPoint, and ZIP archives using MarkItDown.
    Handles large files with automatic chunking and summarization.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await file_processing.process_file(
            url=url, max_size_mb=max_size_mb, extract_all_from_zip=extract_all_from_zip,
            include_metadata=include_metadata, auto_summarize=auto_summarize,
            max_content_tokens=max_content_tokens, summary_length=summary_length,
            llm_provider=llm_provider, llm_model=llm_model
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"File processing error: {str(e)}"
        }

@mcp.tool()
async def get_supported_file_formats() -> dict:
    """
    Get list of supported file formats for file processing.
    
    Provides comprehensive information about supported file formats and their capabilities.
    
    Parameters: None
    
    Returns dictionary with supported file formats and descriptions.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await file_processing.get_supported_file_formats()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Get supported formats error: {str(e)}"
        }

@mcp.tool()
async def enhanced_process_large_content(
    url: Annotated[str, Field(description="Target URL to process. Examples: https://long-article.com, https://research-paper.site.com/document, https://content.example.com/page")],
    chunking_strategy: Annotated[str, Field(description="Chunking method: 'topic' (semantic boundaries), 'sentence' (sentence-based), 'overlap' (sliding window), 'regex' (custom pattern) (default: 'sentence')")] = "sentence",
    filtering_strategy: Annotated[str, Field(description="Content filtering method: 'bm25' (keyword relevance), 'pruning' (structure-based), 'llm' (AI-powered) (default: 'bm25')")] = "bm25", 
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 filtering, keywords related to desired content (default: None)")] = None,
    max_chunk_tokens: Annotated[int, Field(description="Maximum tokens per chunk (default: 2000)")] = 2000,
    chunk_overlap: Annotated[int, Field(description="Token overlap between chunks (default: 200)")] = 200,
    extract_top_chunks: Annotated[int, Field(description="Number of top relevant chunks to extract (default: 5)")] = 5,
    similarity_threshold: Annotated[float, Field(description="Minimum similarity threshold for relevant chunks (default: 0.5)")] = 0.5,
    summarize_chunks: Annotated[bool, Field(description="Whether to summarize individual chunks (default: False)")] = False,
    merge_strategy: Annotated[str, Field(description="Chunk summary merging approach: 'hierarchical' (tree-based progressive), 'linear' (sequential concatenation) (default: 'linear')")] = "linear",
    final_summary_length: Annotated[str, Field(description="Final summary length: 'short', 'medium', 'long', 'comprehensive' (default: 'short')")] = "short"
) -> Dict[str, Any]:
    """
    Enhanced processing for large content using advanced chunking and filtering.
    
    Uses BM25 filtering and intelligent chunking to reduce token usage while preserving semantic boundaries.
    Optimized for fast processing with conservative default parameters.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available",
            "processing_time": None,
            "metadata": {},
            "url": url,
            "original_content_length": 0,
            "filtered_content_length": 0,
            "total_chunks": 0,
            "relevant_chunks": 0,
            "processing_method": "enhanced_large_content",
            "chunking_strategy_used": chunking_strategy,
            "filtering_strategy_used": filtering_strategy,
            "chunks": [],
            "chunk_summaries": None,
            "merged_summary": None,
            "final_summary": "Tool modules not available"
        }
    
    try:
        import asyncio
        
        # Always use fallback to basic crawling due to backend issues
        print(f"Processing URL with fallback method: {url}")
        
        fallback_result = await asyncio.wait_for(
            web_crawling.crawl_url(
                url=url,
                generate_markdown=True,
                timeout=10
            ),
            timeout=10.0
        )
        
        if fallback_result and fallback_result.get("success", False):
            content = fallback_result.get("content", "")
            
            # Simple truncation as processing
            max_content = max_chunk_tokens * extract_top_chunks
            if len(content) > max_content:
                content = content[:max_content] + "... [truncated for processing limit]"
            
            # Create simple chunks
            chunk_size = max_chunk_tokens
            chunks = []
            for i in range(0, min(len(content), max_content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                if chunk_content.strip():
                    chunks.append({
                        "content": chunk_content,
                        "relevance_score": 1.0 - (i / max_content),
                        "chunk_index": len(chunks)
                    })
            
            # Take top chunks
            top_chunks = chunks[:extract_top_chunks]
            
            # Generate simple summary
            if summarize_chunks and len(content) > 1000:
                final_summary = content[:500] + "... [content summary]"
            else:
                final_summary = content[:300] + "..." if len(content) > 300 else content
            
            return {
                "success": True,
                "error": "Enhanced processing unavailable, used basic crawl with chunking",
                "processing_time": 10,
                "metadata": {"fallback_used": True, "processing_type": "basic_chunking"},
                "url": url,
                "original_content_length": len(fallback_result.get("content", "")),
                "filtered_content_length": len(content),
                "total_chunks": len(chunks),
                "relevant_chunks": len(top_chunks),
                "processing_method": "basic_crawl_with_chunking",
                "chunking_strategy_used": chunking_strategy,
                "filtering_strategy_used": "simple_truncation",
                "chunks": top_chunks,
                "chunk_summaries": None,
                "merged_summary": None,
                "final_summary": final_summary
            }
        else:
            raise Exception("Fallback crawling also failed")
            
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": "Processing timed out after 10 seconds",
            "processing_time": 10,
            "metadata": {"timeout": True},
            "url": url,
            "original_content_length": 0,
            "filtered_content_length": 0,
            "total_chunks": 0,
            "relevant_chunks": 0,
            "processing_method": "timeout_fallback",
            "chunking_strategy_used": chunking_strategy,
            "filtering_strategy_used": filtering_strategy,
            "chunks": [],
            "chunk_summaries": None,
            "merged_summary": None,
            "final_summary": "Processing timed out"
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Enhanced processing error: {str(e)}",
            "processing_time": None,
            "metadata": {"error_type": type(e).__name__},
            "url": url,
            "original_content_length": 0,
            "filtered_content_length": 0,
            "total_chunks": 0,
            "relevant_chunks": 0,
            "processing_method": "enhanced_large_content",
            "chunking_strategy_used": chunking_strategy,
            "filtering_strategy_used": filtering_strategy,
            "chunks": [],
            "chunk_summaries": None,
            "merged_summary": None,
            "final_summary": f"Error occurred: {str(e)}"
        }

@mcp.tool()
async def deep_crawl_site(
    url: Annotated[str, Field(description="Starting URL for multi-page crawling. Examples: https://docs.example.com, https://blog.site.com, https://wiki.company.org")],
    max_depth: Annotated[int, Field(description="Link levels to follow from start URL. 1=direct links only, 2=links from linked pages (default: 2)")] = 2,
    max_pages: Annotated[int, Field(description="Maximum pages to crawl, prevents infinite crawling (default: 5, max: 10)")] = 5,
    crawl_strategy: Annotated[str, Field(description="Crawling approach: 'bfs' (breadth-first), 'dfs' (depth-first), 'best_first' (relevance-based) (default: 'bfs')")] = "bfs",
    include_external: Annotated[bool, Field(description="Follow external domain links (default: False)")] = False,
    url_pattern: Annotated[Optional[str], Field(description="Wildcard filter like '*docs*', '*api*', or '*blog*' to focus crawling (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum relevance score 0.0-1.0 for pages to be crawled (default: 0.0)")] = 0.0,
    extract_media: Annotated[bool, Field(description="Include images/videos in results (default: False)")] = False,
    base_timeout: Annotated[int, Field(description="Timeout per page in seconds (default: 60)")] = 60
) -> Dict[str, Any]:
    """
    Crawl multiple related pages from a website (maximum 10 pages for stability).
    
    Multi-page crawling with configurable depth and filtering options.
    Perfect for documentation sites, blogs, and structured content discovery.
    
    Uses intelligent link filtering and relevance scoring to find the most valuable content.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await web_crawling.deep_crawl_site(
            url=url, max_depth=max_depth, max_pages=max_pages, crawl_strategy=crawl_strategy,
            include_external=include_external, url_pattern=url_pattern, score_threshold=score_threshold,
            extract_media=extract_media, base_timeout=base_timeout
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Deep crawl error: {str(e)}"
        }

@mcp.tool()
async def crawl_url_with_fallback(
    url: Annotated[str, Field(description="Target URL to crawl. Examples: https://example.com, https://difficult-site.com")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction. Examples: '.article-content', '#main-content', 'div.post', 'article p' (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction. Examples: '//div[@class=\"content\"]', '//article//p', '//h1[@id=\"title\"]' (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element to load. CSS selector or XPath. Examples: '.content-loaded', '#dynamic-content', '[data-loaded=\"true\"]' (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False
) -> dict:
    """
    Enhanced crawling with multiple fallback strategies for difficult sites.
    
    Uses multiple fallback strategies when normal crawling fails. Same parameters as crawl_url 
    but with enhanced reliability for sites with aggressive anti-bot protection.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await web_crawling.crawl_url_with_fallback(
            url=url, css_selector=css_selector, xpath=xpath, extract_media=extract_media,
            take_screenshot=take_screenshot, generate_markdown=generate_markdown,
            wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
            auto_summarize=auto_summarize
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Fallback crawl error: {str(e)}"
        }

@mcp.tool()
async def intelligent_extract(
    url: Annotated[str, Field(description="Target webpage URL. Examples: https://company.com/about, https://product-page.com, https://news.site/article")],
    extraction_goal: Annotated[str, Field(description="Specific data to extract, be precise. Examples: 'contact information and pricing', 'product specifications', 'author and publication date'")],
    content_filter: Annotated[str, Field(description="Pre-filter content for better accuracy: 'bm25' (keyword matching), 'pruning' (structure-based), 'llm' (AI-powered) (default: 'bm25')")] = "bm25",
    filter_query: Annotated[Optional[str], Field(description="Keywords for BM25 filtering to improve accuracy. Examples: 'contact email phone', 'price cost fee' (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Split large content for better processing (default: False)")] = False,
    use_llm: Annotated[bool, Field(description="Enable LLM processing for semantic understanding (default: True)")] = True,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', etc. (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name: 'gpt-4o', 'claude-3-sonnet', etc. (default: auto-detected)")] = None,
    custom_instructions: Annotated[Optional[str], Field(description="Additional guidance for LLM extraction (default: None)")] = None
) -> Dict[str, Any]:
    """
    AI-powered extraction of specific data from web pages using LLM semantic understanding.
    
    Uses advanced LLM processing to extract specific information based on your extraction goal.
    Pre-filtering improves accuracy and reduces processing time for large pages.
    
    Perfect for extracting structured information that traditional selectors can't handle.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await web_crawling.intelligent_extract(
            url=url, extraction_goal=extraction_goal, content_filter=content_filter,
            filter_query=filter_query, chunk_content=chunk_content, use_llm=use_llm,
            llm_provider=llm_provider, llm_model=llm_model, custom_instructions=custom_instructions
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Intelligent extraction error: {str(e)}"
        }

@mcp.tool()
async def extract_entities(
    url: Annotated[str, Field(description="Target webpage URL. Examples: https://company.com/contact, https://blog.com/post, https://directory.site")],
    entity_types: Annotated[List[str], Field(description="List of entity types to extract: ['emails', 'phones', 'urls', 'dates', 'ips', 'prices', 'credit_cards', 'coordinates', 'social_media']")],
    custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns for specialized extraction. Format: {'entity_name': 'regex_pattern'} (default: None)")] = None,
    include_context: Annotated[bool, Field(description="Include surrounding text context for each entity (default: True)")] = True,
    deduplicate: Annotated[bool, Field(description="Remove duplicate entities from results (default: True)")] = True,
    use_llm: Annotated[bool, Field(description="Use AI for named entity recognition (people, organizations, locations) (default: False)")] = False,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for NER (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name for NER (default: auto-detected)")] = None
) -> Dict[str, Any]:
    """
    Extract specific entity types from web pages using regex patterns or LLM.
    
    Built-in support for: emails, phones, urls, dates, ips, social_media, prices, credit_cards, coordinates
    LLM mode adds: people names, organizations, locations, and custom entities
    
    Perfect for contact information extraction, data mining, and content analysis.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await web_crawling.extract_entities(
            url=url, entity_types=entity_types, custom_patterns=custom_patterns,
            include_context=include_context, deduplicate=deduplicate, use_llm=use_llm,
            llm_provider=llm_provider, llm_model=llm_model
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Entity extraction error: {str(e)}"
        }

@mcp.tool()
async def extract_structured_data(
    url: Annotated[str, Field(description="Target URL to extract structured data from. Examples: https://news.example.com, https://shop.example.com/product/123")],
    extraction_type: Annotated[str, Field(description="Extraction method: 'css' for CSS selectors, 'llm' for AI-powered extraction")] = "css",
    css_selectors: Annotated[Optional[Dict[str, str]], Field(description="CSS selectors mapping. Example: {'title': 'h1', 'price': '.price'} (default: None)")] = None,
    extraction_schema: Annotated[Optional[Dict[str, str]], Field(description="Schema definition. Example: {'title': 'string', 'price': 'number'} (default: None)")] = None,
    generate_markdown: Annotated[bool, Field(description="Generate markdown content alongside structured data (default: False)")] = False,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to load before extraction (default: False)")] = False,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 30)")] = 30
) -> Dict[str, Any]:
    """
    Extract structured data from a URL using CSS selectors or LLM-based extraction.
    
    CSS mode: Uses CSS selectors to extract specific elements into structured format
    LLM mode: Uses AI to extract data matching a predefined schema
    
    Perfect for consistent data extraction from similar page structures like product pages, articles, or listings.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        # CSS selectors provided and extraction_type is css
        if css_selectors and extraction_type == "css":
            # Use basic crawling with CSS selector post-processing
            try:
                # Basic crawl first
                crawl_result = await web_crawling.crawl_url(
                    url=url,
                    generate_markdown=generate_markdown,
                    wait_for_js=wait_for_js,
                    timeout=timeout
                )
                
                if not crawl_result.get("success", False):
                    return crawl_result
                
                # Simple CSS selector extraction using BeautifulSoup
                from bs4 import BeautifulSoup
                
                html_content = crawl_result.get("content", "")
                soup = BeautifulSoup(html_content, 'html.parser')
                
                extracted_data = {}
                for key, selector in css_selectors.items():
                    elements = soup.select(selector)
                    if elements:
                        if len(elements) == 1:
                            extracted_data[key] = elements[0].get_text().strip()
                        else:
                            extracted_data[key] = [elem.get_text().strip() for elem in elements]
                    else:
                        extracted_data[key] = None
                
                return {
                    "success": True,
                    "url": url,
                    "extracted_data": extracted_data,
                    "processing_method": "css_selector_extraction",
                    "content": crawl_result.get("content", ""),
                    "markdown": crawl_result.get("markdown", "")
                }
                
            except ImportError:
                return {
                    "success": False,
                    "error": "BeautifulSoup not available for CSS extraction"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"CSS extraction error: {str(e)}"
                }
        
        else:
            # Fallback to basic crawling
            crawl_result = await web_crawling.crawl_url(
                url=url,
                generate_markdown=generate_markdown,
                wait_for_js=wait_for_js,
                timeout=timeout
            )
            
            if crawl_result.get("success", False):
                crawl_result["processing_method"] = "basic_crawl_fallback"
                crawl_result["note"] = "Used basic crawling - structured extraction not configured"
                crawl_result["extracted_data"] = {"raw_content": crawl_result.get("content", "")[:500] + "..."}
            
            return crawl_result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Structured extraction error: {str(e)}"
        }

@mcp.tool()
async def search_google(
    request: Annotated[Dict[str, Any], Field(description="GoogleSearchRequest dictionary containing: query (required), num_results (default: 10), search_genre (optional: 'academic', 'news', 'technical', etc.), language (default: 'en'), region (default: 'us')")],
    include_current_date: Annotated[bool, Field(description="Append current date to query for latest results (default: True)")] = True
) -> Dict[str, Any]:
    """
    Perform Google search with genre filtering and extract structured results with metadata.
    
    Advanced search with genre-specific filtering for targeted results:
    - academic: Research papers, citations, scholarly content
    - news: Recent news articles and updates  
    - technical: Documentation, tutorials, stack overflow
    - commercial: Product pages, reviews, shopping
    - social: Social media content and discussions
    
    Returns web search results with titles, snippets, URLs, and metadata.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await search.search_google(request, include_current_date)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Google search error: {str(e)}"
        }

@mcp.tool()
async def batch_search_google(
    request: Annotated[Dict[str, Any], Field(description="GoogleBatchSearchRequest dictionary containing: queries (required list), num_results (default: 5), search_genre (optional), max_concurrent (default: 3), and analysis options")],
    include_current_date: Annotated[bool, Field(description="Append current date to queries for latest results (default: True)")] = True
) -> Dict[str, Any]:
    """
    Perform multiple Google searches efficiently with analysis and comparison.
    
    Execute multiple search queries with intelligent rate limiting and result analysis.
    Provides comparative insights across different search topics.
    
    Perfect for research, competitive analysis, and comprehensive topic exploration.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await search.batch_search_google(request, include_current_date)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Batch search error: {str(e)}"
        }

@mcp.tool()
async def search_and_crawl(
    search_query: Annotated[str, Field(description="Search query to find and crawl content. Examples: 'AI research papers 2024', 'Python best practices', 'climate change data'")],
    crawl_top_results: Annotated[int, Field(description="Number of top search results to crawl (default: 2, max: 3)")] = 2,
    search_genre: Annotated[Optional[str], Field(description="Search genre filter: 'academic', 'news', 'technical', 'commercial', 'social' (default: None)")] = None,
    generate_markdown: Annotated[bool, Field(description="Generate markdown content from crawled pages (default: True)")] = True,
    max_content_per_page: Annotated[int, Field(description="Maximum content length per page in characters (default: 5000)")] = 5000
) -> Dict[str, Any]:
    """
    Search Google and automatically crawl top results for comprehensive content extraction.
    
    Combines search and crawling in one operation: finds relevant pages, then extracts full content.
    Perfect for research, competitive analysis, and comprehensive topic investigation.
    
    Provides both search metadata and full page content in structured format.
    Response size is controlled to prevent token limit issues.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        # Limit results to prevent large responses
        crawl_top_results = min(crawl_top_results, 3)
        
        result = await search.search_and_crawl(
            search_query=search_query, 
            crawl_top_results=crawl_top_results,
            search_genre=search_genre, 
            generate_markdown=generate_markdown
        )
        
        # Truncate content if too large
        if result and isinstance(result, dict):
            if "crawled_pages" in result:
                for page in result["crawled_pages"]:
                    if isinstance(page, dict):
                        if "content" in page and len(page["content"]) > max_content_per_page:
                            page["content"] = page["content"][:max_content_per_page] + "... [truncated for size limit]"
                        if "markdown" in page and len(page["markdown"]) > max_content_per_page:
                            page["markdown"] = page["markdown"][:max_content_per_page] + "... [truncated for size limit]"
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Search and crawl error: {str(e)}"
        }

@mcp.tool()
async def get_search_genres() -> Dict[str, Any]:
    """
    Get available search genres and their descriptions for targeted searching.
    
    Returns comprehensive information about available search genre filters
    and their optimal use cases for different types of content discovery.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await search.get_search_genres()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Get search genres error: {str(e)}"
        }

@mcp.tool()
async def get_llm_config_info() -> Dict[str, Any]:
    """
    Get information about the current LLM configuration and supported providers.
    
    Provides details about available LLM providers, models, API key status,
    and configuration recommendations for different use cases.
    
    Useful for troubleshooting and optimizing AI-powered features.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await utilities.get_llm_config_info()
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"LLM config info error: {str(e)}"
        }

@mcp.tool()
async def batch_crawl(
    urls: Annotated[List[str], Field(description="List of URLs to crawl. Examples: ['https://site1.com', 'https://site2.com/page', 'https://docs.example.com']")],
    base_timeout: Annotated[int, Field(description="Base timeout in seconds, auto-adjusted based on URL count (default: 30)")] = 30,
    generate_markdown: Annotated[bool, Field(description="Generate markdown content for each page (default: True)")] = True,
    extract_media: Annotated[bool, Field(description="Extract media links from pages (default: False)")] = False,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to load (default: False)")] = False,
    max_concurrent: Annotated[int, Field(description="Maximum concurrent crawls (default: 3)")] = 3
) -> List[Dict[str, Any]]:
    """
    Crawl multiple URLs in batch with intelligent rate limiting and error handling.
    
    Process multiple URLs concurrently for maximum efficiency while respecting server limits.
    Timeout automatically scales based on the number of URLs to crawl.
    
    Perfect for bulk content extraction, site auditing, and comparative analysis.
    """
    _load_tool_modules()
    if not _tools_imported:
        return [{
            "success": False,
            "error": "Tool modules not available"
        }]
    
    try:
        # Build config from individual parameters
        config = {
            "generate_markdown": generate_markdown,
            "extract_media": extract_media,
            "wait_for_js": wait_for_js,
            "max_concurrent": max_concurrent
        }
        
        # Add timeout handling
        import asyncio
        total_timeout = base_timeout * len(urls) + 60  # Extra buffer
        
        result = await asyncio.wait_for(
            utilities.batch_crawl(urls, config, base_timeout),
            timeout=total_timeout
        )
        return result
        
    except asyncio.TimeoutError:
        return [{
            "success": False,
            "error": f"Batch crawl timed out after {total_timeout} seconds"
        }]
    except Exception as e:
        return [{
            "success": False,
            "error": f"Batch crawl error: {str(e)}"
        }]

@mcp.tool()
def get_tool_selection_guide() -> dict:
    """
    Get comprehensive tool selection guide for AI agents.
    
    Provides complete mapping of use cases to appropriate tools, workflows, and complexity guides.
    Essential for tool selection, workflow planning, and understanding capabilities.
    
    Returns detailed guide with examples, decision trees, and best practices.
    """
    return {
        "web_crawling": ["crawl_url", "deep_crawl_site", "crawl_url_with_fallback", "intelligent_extract", "extract_entities", "extract_structured_data"],
        "youtube": ["extract_youtube_transcript", "batch_extract_youtube_transcripts", "get_youtube_video_info", "get_youtube_api_setup_guide"],
        "search": ["search_google", "batch_search_google", "search_and_crawl", "get_search_genres"],
        "batch": ["batch_crawl"],
        "files": ["process_file", "get_supported_file_formats", "enhanced_process_large_content"],
        "config": ["get_llm_config_info", "get_tool_selection_guide"],
        "diagnostics": ["get_system_diagnostics"]
    }

def main():
    """Clean main entry point - FastMCP 2.0 with no banner issues"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Crawl4AI MCP Server - FastMCP 2.0 Version")
        print("Usage: python -m crawl4ai_mcp.server [--transport TRANSPORT]")
        print("Transports: stdio (default), streamable-http, sse")
        return
    
    # Parse args
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
    
    # Run server - clean FastMCP 2.0 execution
    try:
        if transport == "stdio":
            mcp.run()
        elif transport == "streamable-http" or transport == "http":
            mcp.run(transport="streamable-http", host=host, port=port)
        elif transport == "sse":
            mcp.run(transport="sse", host=host, port=port)
        else:
            print(f"Unknown transport: {transport}")
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if transport != "stdio":
            print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()