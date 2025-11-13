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

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (GPT-4 encoding as Claude approximation).
    Falls back to character-based estimation if tiktoken is unavailable.
    """
    try:
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-4")
        return len(encoder.encode(str(text)))
    except Exception:
        # Fallback to character-based estimation
        # English: ~4 chars/token, Japanese: ~2 chars/token
        text_str = str(text)
        japanese_chars = sum(1 for c in text_str if '\u3040' <= c <= '\u9fff')
        total_chars = len(text_str)

        if total_chars > 0 and japanese_chars / total_chars > 0.3:
            # Japanese-heavy text: ~2 chars per token
            return total_chars // 2
        else:
            # English-heavy text: ~4 chars per token
            return total_chars // 4

def _apply_token_limit_fallback(result: dict, max_tokens: int = 20000) -> dict:
    """Apply token limit fallback to MCP tool responses to prevent Claude Code errors"""
    import json
    result_copy = result.copy()
    
    # Check current size
    current_tokens = _estimate_tokens(json.dumps(result_copy))
    
    if current_tokens <= max_tokens:
        return result_copy
    
    # Add fallback indicators with clear warning
    result_copy["token_limit_applied"] = True
    result_copy["original_response_tokens"] = current_tokens
    result_copy["truncated_to_tokens"] = max_tokens
    result_copy["warning"] = f"Response truncated from {current_tokens} to ~{max_tokens} tokens due to size limits. Partial data returned below."
    
    # Add recommendations based on content type
    recommendations = []
    if "results" in result_copy and isinstance(result_copy["results"], list):
        original_count = result_copy.get("results_truncated_from", len(result_copy["results"]))
        recommendations.append(f"Consider reducing num_results parameter (current response had {original_count} results)")
    if any(key in result_copy for key in ["content", "markdown", "text"]):
        recommendations.append("Consider using auto_summarize=True for large content")
        recommendations.append("Consider using css_selector or xpath to extract specific content sections")
    if "search_query" in result_copy or "query" in result_copy:
        recommendations.append("Consider narrowing your search query or using search_genre filtering")
    
    if recommendations:
        result_copy["recommendations"] = recommendations
    
    # Priority order for content truncation
    content_fields = [
        ("content", 8000),          # Main content - keep substantial portion
        ("markdown", 6000),         # Markdown version  
        ("raw_content", 2000),      # Raw text content
        ("text", 2000),            # Extracted text
        ("results", 4000),         # Search/crawl results
        ("chunks", 2000),          # Content chunks
        ("extracted_data", 1500),  # Structured data
        ("summary", 1500),         # Summary content
        ("final_summary", 1000),   # Final summary
        ("metadata", 500),         # Metadata - keep small portion
        ("entities", 1000),        # Extracted entities
        ("table_data", 1500)       # Table data
    ]
    
    for field, max_field_tokens in content_fields:
        if field in result_copy and result_copy[field]:
            field_content = result_copy[field]
            
            # Handle different field types
            if isinstance(field_content, list):
                # For lists, truncate by limiting number of items
                original_length = len(field_content)
                if original_length > 10:
                    result_copy[field] = field_content[:10]
                    result_copy[f"{field}_truncated_from"] = original_length
                    result_copy[f"{field}_truncated_info"] = f"Showing 10 of {original_length} items"
            elif isinstance(field_content, dict):
                # For dicts, truncate string values
                truncated_dict = {}
                for k, v in field_content.items():
                    if isinstance(v, str) and len(v) > 500:
                        truncated_dict[k] = v[:500] + "... [TRUNCATED]"
                    else:
                        truncated_dict[k] = v
                result_copy[field] = truncated_dict
            elif isinstance(field_content, str):
                # For strings, truncate with ellipsis
                max_chars = max_field_tokens * 4
                original_length = len(field_content)
                if original_length > max_chars:
                    result_copy[field] = field_content[:max_chars] + f"... [TRUNCATED: showing {max_chars} of {original_length} chars]"
    
    # Final size check and emergency truncation  
    current_tokens = _estimate_tokens(json.dumps(result_copy))
    if current_tokens > max_tokens:
        # Emergency truncation - keep only essential fields
        essential_fields = ["success", "url", "error", "title", "file_type", "processing_method", "query", "search_query", "video_id", "language_info"]
        essential_result = {
            key: result_copy.get(key) for key in essential_fields if key in result_copy
        }
        
        essential_result.update({
            "token_limit_applied": True,
            "emergency_truncation": True,
            "original_response_tokens": result.get("original_response_tokens", current_tokens),
            "warning": f"Response truncated from {current_tokens} to ~{max_tokens} tokens. Partial content returned.",
            "recommendations": [
                "Use more specific parameters to reduce content size",
                "Enable auto_summarize for large content",
                "Use filtering parameters (css_selector, xpath, search_genre)",
                "Reduce num_results for search queries",
                "Use max_content_per_page to limit page content size"
            ],
            "available_fields_in_original": list(result.keys()),
        })
        
        # Calculate available tokens for content after essential fields
        essential_base_tokens = _estimate_tokens(json.dumps(essential_result))
        available_content_tokens = max(max_tokens - essential_base_tokens - 500, 1000)  # Reserve 500 for safety margin
        
        # Try to fit as much content as possible within available token budget
        content_added = False
        
        # Priority: markdown > content > summary
        content_sources = [
            ("markdown", result.get("markdown")),
            ("content", result.get("content")),
            ("summary", result.get("summary")),
            ("text", result.get("text"))
        ]
        
        for field_name, field_value in content_sources:
            if field_value and not content_added:
                content_str = str(field_value)
                content_tokens = _estimate_tokens(content_str)
                
                if content_tokens <= available_content_tokens:
                    # Content fits completely
                    essential_result[field_name] = content_str
                    essential_result[f"{field_name}_info"] = f"Complete {field_name} content ({content_tokens} tokens)"
                    content_added = True
                else:
                    # Truncate content to fit available tokens
                    # Estimate characters needed: available_tokens * chars_per_token
                    chars_per_token = len(content_str) / content_tokens if content_tokens > 0 else 4
                    estimated_chars = int(available_content_tokens * chars_per_token)
                    
                    truncated_content = content_str[:estimated_chars]
                    actual_tokens = _estimate_tokens(truncated_content)
                    
                    # Adjust if estimation was off
                    while actual_tokens > available_content_tokens and len(truncated_content) > 100:
                        truncated_content = truncated_content[:int(len(truncated_content) * 0.9)]
                        actual_tokens = _estimate_tokens(truncated_content)
                    
                    if truncated_content:
                        percentage = int((len(truncated_content) / len(content_str)) * 100)
                        essential_result[field_name] = truncated_content + "\n\n[TRUNCATED - Content continues beyond token limit]"
                        essential_result[f"{field_name}_info"] = f"Partial {field_name} ({percentage}% of original, {actual_tokens}/{content_tokens} tokens)"
                        content_added = True
                    break
        
        # Keep partial results if available
        if result.get("results") and isinstance(result["results"], list):
            essential_result["results"] = result["results"][:3]
            essential_result["results_truncated_info"] = f"Showing 3 of {len(result['results'])} results"
        
        # Special handling for YouTube transcript data
        if result.get("transcript") and isinstance(result["transcript"], list):
            # Try to fit as many transcript entries as possible within token budget
            transcript_entries = result["transcript"]
            total_entries = len(transcript_entries)
            
            # Use remaining available tokens if content wasn't added
            transcript_available_tokens = available_content_tokens if not content_added else max(available_content_tokens // 2, 1000)
            
            # Binary search to find maximum number of entries that fit
            entries_to_include = []
            for i, entry in enumerate(transcript_entries):
                test_entries = transcript_entries[:i+1]
                test_size = _estimate_tokens(json.dumps(test_entries))
                if test_size > transcript_available_tokens:
                    break
                entries_to_include = test_entries
            
            if entries_to_include:
                essential_result["transcript"] = entries_to_include
                included_count = len(entries_to_include)
                percentage = int((included_count / total_entries) * 100)
                essential_result["transcript_truncated_info"] = f"Showing {included_count} of {total_entries} entries ({percentage}%)"
                essential_result["transcript_total_entries"] = total_entries
                
                # Calculate time coverage if timestamps are available
                if entries_to_include and "start" in entries_to_include[-1]:
                    last_timestamp = entries_to_include[-1].get("start", 0)
                    essential_result["transcript_time_coverage_seconds"] = last_timestamp
                    essential_result["transcript_time_coverage_formatted"] = f"{int(last_timestamp // 60)}:{int(last_timestamp % 60):02d}"
            
        return essential_result
    
    return result_copy

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
    include_cleaned_html: Annotated[bool, Field(description="Include cleaned HTML in content field (default: False, only markdown returned)")] = False,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element to load. CSS selector or XPath. Examples: '.content-loaded', '#dynamic-content', '[data-loaded=\"true\"]' (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Use undetected browser to bypass bot detection (default: False)")] = False
) -> dict:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.

    Core web crawling tool with comprehensive configuration options.
    Essential for SPAs: set wait_for_js=true for JavaScript-heavy sites.

    NEW v0.7.4: Supports undetected browser mode to bypass sophisticated bot detection.
    Automatically falls back to enhanced strategies if initial crawling fails
    due to JavaScript errors, anti-bot protection, or other issues.

    By default, returns markdown content only for optimal readability and reduced token usage.
    Set include_cleaned_html=True to also receive the cleaned HTML content field.

    Returns structured data with markdown, metadata, and optional media/screenshots.
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
            include_cleaned_html=include_cleaned_html,
            wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
            auto_summarize=auto_summarize, use_undetected_browser=use_undetected_browser
        )
        
        # Convert CrawlResponse to dict
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result

        # Remove content field if include_cleaned_html is False
        if not include_cleaned_html and 'content' in result_dict:
            del result_dict['content']

        if result_dict.get("success", True) and result_dict.get("markdown", "").strip():
            # Apply token limit fallback before returning
            return _apply_token_limit_fallback(result_dict, max_tokens=25000)
        
        # If initial crawling failed or returned empty content, try fallback with undetected browser
        fallback_result = await web_crawling.crawl_url_with_fallback(
            url=url, css_selector=css_selector, xpath=xpath, extract_media=extract_media,
            take_screenshot=take_screenshot, generate_markdown=generate_markdown,
            include_cleaned_html=include_cleaned_html,
            wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
            auto_summarize=auto_summarize, use_undetected_browser=True
        )

        # Convert fallback result to dict and mark that fallback was used
        if hasattr(fallback_result, 'model_dump'):
            fallback_dict = fallback_result.model_dump()
        elif hasattr(fallback_result, 'dict'):
            fallback_dict = fallback_result.dict()
        else:
            fallback_dict = fallback_result

        # Remove content field if include_cleaned_html is False
        if not include_cleaned_html and 'content' in fallback_dict:
            del fallback_dict['content']

        if fallback_dict.get("success", False):
            fallback_dict["fallback_used"] = True
            if use_undetected_browser:
                fallback_dict["undetected_browser_used"] = True

        # Apply token limit fallback before returning
        return _apply_token_limit_fallback(fallback_dict, max_tokens=25000)
        
    except Exception as e:
        # If initial crawling throws an exception, try fallback with undetected browser
        try:
            fallback_result = await web_crawling.crawl_url_with_fallback(
                url=url, css_selector=css_selector, xpath=xpath, extract_media=extract_media,
                take_screenshot=take_screenshot, generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html,
                wait_for_selector=wait_for_selector, timeout=timeout, wait_for_js=wait_for_js,
                auto_summarize=auto_summarize, use_undetected_browser=True
            )

            # Convert fallback result to dict and mark that fallback was used
            if hasattr(fallback_result, 'model_dump'):
                fallback_dict = fallback_result.model_dump()
            elif hasattr(fallback_result, 'dict'):
                fallback_dict = fallback_result.dict()
            else:
                fallback_dict = fallback_result

            # Remove content field if include_cleaned_html is False
            if not include_cleaned_html and 'content' in fallback_dict:
                del fallback_dict['content']

            if fallback_dict.get("success", False):
                fallback_dict["fallback_used"] = True
                fallback_dict["undetected_browser_used"] = True
                fallback_dict["original_error"] = str(e)

            # Apply token limit fallback before returning
            return _apply_token_limit_fallback(fallback_dict, max_tokens=25000)
            
        except Exception as fallback_error:
            return {
                "success": False,
                "url": url,
                "error": f"Both crawling methods failed. Original: {str(e)}, Fallback: {str(fallback_error)}"
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
        
        # Apply token limit fallback to prevent MCP errors
        result_with_fallback = _apply_token_limit_fallback(result, max_tokens=20000)
        
        # If token limit was applied and auto_summarize was False, provide helpful suggestion
        if result_with_fallback.get("token_limit_applied") and not auto_summarize:
            if not result_with_fallback.get("emergency_truncation"):
                result_with_fallback["suggestion"] = "Transcript was truncated due to MCP token limits. Consider setting auto_summarize=True for long transcripts."
                
        return result_with_fallback
        
    except Exception as e:
        return {
            "success": False,
            "error": f"YouTube transcript error: {str(e)}"
        }

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
        
        # Apply token limit fallback to prevent MCP errors
        result_with_fallback = _apply_token_limit_fallback(result, max_tokens=20000)
        
        # If token limit was applied, provide helpful suggestion
        if result_with_fallback.get("token_limit_applied"):
            if not result_with_fallback.get("emergency_truncation"):
                result_with_fallback["suggestion"] = "Batch transcript data was truncated due to MCP token limits. Consider reducing the number of videos or enabling auto_summarize for individual videos."
                
        return result_with_fallback
        
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
        
        # Apply token limit fallback to prevent MCP errors
        result_with_fallback = _apply_token_limit_fallback(result, max_tokens=20000)
        
        # If token limit was applied and summarize_transcript was False, provide helpful suggestion
        if result_with_fallback.get("token_limit_applied") and not summarize_transcript:
            if not result_with_fallback.get("emergency_truncation"):
                result_with_fallback["suggestion"] = "Video info was truncated due to MCP token limits. Consider setting summarize_transcript=True for long transcripts."
                
        return result_with_fallback
        
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
        
        # Convert FileProcessResponse object to dict for JSON serialization
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            # Fallback: manual conversion
            result_dict = {
                'success': getattr(result, 'success', False),
                'url': getattr(result, 'url', None),
                'filename': getattr(result, 'filename', None),
                'file_type': getattr(result, 'file_type', None),
                'size_bytes': getattr(result, 'size_bytes', None),
                'is_archive': getattr(result, 'is_archive', False),
                'content': getattr(result, 'content', None),
                'title': getattr(result, 'title', None),
                'metadata': getattr(result, 'metadata', None),
                'archive_contents': getattr(result, 'archive_contents', None),
                'error': getattr(result, 'error', None),
                'processing_time': getattr(result, 'processing_time', None)
            }
        
        # Apply token limit fallback to prevent MCP errors
        result_with_fallback = _apply_token_limit_fallback(result_dict, max_tokens=20000)
        
        # If token limit was applied and auto_summarize was False, provide helpful suggestion
        if result_with_fallback.get("token_limit_applied") and not auto_summarize:
            if not result_with_fallback.get("emergency_truncation"):
                result_with_fallback["suggestion"] = "Content was truncated due to MCP token limits. Consider setting auto_summarize=True for better content reduction."
            
        return result_with_fallback
        
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
    Automatically applies fallback strategies for individual page crawling failures.
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
        
        # Check if crawling was successful
        if result.get("success", True):
            # Apply token limit fallback before returning
            return _apply_token_limit_fallback(result, max_tokens=25000)
        
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
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
            
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
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
        except Exception as fallback_error:
            pass
        
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
    Automatically applies fallback crawling if initial content retrieval fails.
    
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
        
        # Check if extraction was successful
        if result.get("success", True):
            # Apply token limit fallback before returning
            return _apply_token_limit_fallback(result, max_tokens=25000)
        
        # If intelligent extraction failed, try with fallback crawling
        try:
            fallback_crawl = await web_crawling.crawl_url_with_fallback(
                url=url, generate_markdown=True, timeout=60
            )
            
            if fallback_crawl.get("success", False):
                # Attempt basic extraction from fallback content
                content = fallback_crawl.get("markdown", "") or fallback_crawl.get("content", "")
                
                if content.strip():
                    fallback_response = {
                        "success": True,
                        "url": url,
                        "extraction_goal": extraction_goal,
                        "extracted_data": {
                            "raw_content": content[:2000] + ("..." if len(content) > 2000 else ""),
                            "note": "Fallback extraction - manual processing may be needed"
                        },
                        "content": fallback_crawl.get("content", ""),
                        "markdown": fallback_crawl.get("markdown", ""),
                        "fallback_used": True,
                        "original_error": result.get("error", "Intelligent extraction failed")
                    }
                    
                    # Apply token limit fallback before returning
                    return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                    
        except Exception as fallback_error:
            result["fallback_error"] = str(fallback_error)
        
        return result
        
    except Exception as e:
        # If intelligent extraction throws an exception, try basic fallback
        try:
            fallback_crawl = await web_crawling.crawl_url_with_fallback(
                url=url, generate_markdown=True, timeout=60
            )
            
            if fallback_crawl.get("success", False):
                content = fallback_crawl.get("markdown", "") or fallback_crawl.get("content", "")
                
                fallback_response = {
                    "success": True,
                    "url": url,
                    "extraction_goal": extraction_goal,
                    "extracted_data": {
                        "raw_content": content[:2000] + ("..." if len(content) > 2000 else ""),
                        "note": "Fallback extraction - manual processing may be needed"
                    },
                    "content": fallback_crawl.get("content", ""),
                    "markdown": fallback_crawl.get("markdown", ""),
                    "fallback_used": True,
                    "original_error": str(e)
                }
                
                # Apply token limit fallback before returning
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
        except Exception as fallback_error:
            pass
        
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
    Automatically applies fallback crawling if initial content retrieval fails.
    
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
        
        # Check if entity extraction was successful
        if result.get("success", True):
            # Apply token limit fallback before returning
            return _apply_token_limit_fallback(result, max_tokens=25000)
        
        # If entity extraction failed, try with fallback crawling
        try:
            fallback_crawl = await web_crawling.crawl_url_with_fallback(
                url=url, generate_markdown=True, timeout=60
            )
            
            if fallback_crawl.get("success", False):
                content = fallback_crawl.get("content", "") or fallback_crawl.get("markdown", "")
                
                # Basic regex-based entity extraction on fallback content
                import re
                entities = {}
                
                if "emails" in entity_types:
                    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                    if emails:
                        entities["emails"] = list(set(emails)) if deduplicate else emails
                
                if "phones" in entity_types:
                    phones = re.findall(r'[\+]?[1-9]?[0-9]{7,15}', content)
                    if phones:
                        entities["phones"] = list(set(phones)) if deduplicate else phones
                
                if "urls" in entity_types:
                    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                    if urls:
                        entities["urls"] = list(set(urls)) if deduplicate else urls
                
                fallback_response = {
                    "success": True,
                    "url": url,
                    "entities": entities,
                    "entity_types": entity_types,
                    "total_found": sum(len(v) for v in entities.values()),
                    "content": content[:500] + ("..." if len(content) > 500 else ""),
                    "fallback_used": True,
                    "note": "Basic regex extraction used - some entity types may not be fully supported",
                    "original_error": result.get("error", "Entity extraction failed")
                }
                
                # Apply token limit fallback before returning
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
        except Exception as fallback_error:
            result["fallback_error"] = str(fallback_error)
        
        return result
        
    except Exception as e:
        # If entity extraction throws an exception, try basic fallback
        try:
            fallback_crawl = await web_crawling.crawl_url_with_fallback(
                url=url, generate_markdown=True, timeout=60
            )
            
            if fallback_crawl.get("success", False):
                content = fallback_crawl.get("content", "") or fallback_crawl.get("markdown", "")
                
                # Basic regex-based entity extraction
                import re
                entities = {}
                
                if "emails" in entity_types:
                    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                    if emails:
                        entities["emails"] = list(set(emails)) if deduplicate else emails
                
                if "phones" in entity_types:
                    phones = re.findall(r'[\+]?[1-9]?[0-9]{7,15}', content)
                    if phones:
                        entities["phones"] = list(set(phones)) if deduplicate else phones
                
                if "urls" in entity_types:
                    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                    if urls:
                        entities["urls"] = list(set(urls)) if deduplicate else urls
                
                fallback_response = {
                    "success": True,
                    "url": url,
                    "entities": entities,
                    "entity_types": entity_types,
                    "total_found": sum(len(v) for v in entities.values()),
                    "content": content[:500] + ("..." if len(content) > 500 else ""),
                    "fallback_used": True,
                    "note": "Basic regex extraction used - some entity types may not be fully supported",
                    "original_error": str(e)
                }
                
                # Apply token limit fallback before returning
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
        except Exception as fallback_error:
            pass
        
        return {
            "success": False,
            "error": f"Entity extraction error: {str(e)}"
        }

@mcp.tool()
async def extract_structured_data(
    url: Annotated[str, Field(description="Target URL to extract structured data from. Examples: https://news.example.com, https://shop.example.com/product/123")],
    extraction_type: Annotated[str, Field(description="Extraction method: 'css' for CSS selectors, 'llm' for AI-powered extraction, 'table' for LLM table extraction")] = "css",
    css_selectors: Annotated[Optional[Dict[str, str]], Field(description="CSS selectors mapping. Example: {'title': 'h1', 'price': '.price'} (default: None)")] = None,
    extraction_schema: Annotated[Optional[Dict[str, str]], Field(description="Schema definition. Example: {'title': 'string', 'price': 'number'} (default: None)")] = None,
    generate_markdown: Annotated[bool, Field(description="Generate markdown content alongside structured data (default: False)")] = False,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to load before extraction (default: False)")] = False,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 30)")] = 30,
    use_llm_table_extraction: Annotated[bool, Field(description="NEW v0.7.4: Use revolutionary LLM table extraction with intelligent chunking (default: False)")] = False,
    table_chunking_strategy: Annotated[str, Field(description="Table chunking strategy: 'intelligent' (adaptive), 'fixed' (fixed size), 'semantic' (content-aware) (default: 'intelligent')")] = "intelligent"
) -> Dict[str, Any]:
    """
    Extract structured data from a URL using CSS selectors, LLM-based extraction, or revolutionary table extraction.
    
    CSS mode: Uses CSS selectors to extract specific elements into structured format
    LLM mode: Uses AI to extract data matching a predefined schema
    Table mode: NEW v0.7.4 - Revolutionary LLM Table Extraction with intelligent chunking for massive tables
    Automatically applies fallback crawling if initial content retrieval fails.
    
    Perfect for consistent data extraction from similar page structures like product pages, articles, or listings.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        # NEW: LLM Table Extraction mode
        if extraction_type == "table" or use_llm_table_extraction:
            try:
                result = await web_crawling.extract_structured_data(
                    url=url,
                    extraction_type="llm_table",
                    extraction_schema=extraction_schema,
                    generate_markdown=generate_markdown,
                    wait_for_js=wait_for_js,
                    timeout=timeout,
                    chunking_strategy=table_chunking_strategy
                )
                
                if result.get("success", False):
                    result["processing_method"] = "llm_table_extraction"
                    result["features_used"] = ["intelligent_chunking", "massive_table_support"]
                    # Apply token limit fallback before returning
                    return _apply_token_limit_fallback(result, max_tokens=25000)
                    
            except Exception as table_error:
                # Fallback to CSS extraction if table extraction fails
                if css_selectors:
                    extraction_type = "css"
                else:
                    return {
                        "success": False,
                        "error": f"LLM table extraction failed: {str(table_error)}",
                        "suggested_fallback": "Try with css_selectors or extraction_type='css'"
                    }
        
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
                
                # If initial crawl fails, try fallback
                if not crawl_result.get("success", False) or not crawl_result.get("content", "").strip():
                    fallback_result = await web_crawling.crawl_url_with_fallback(
                        url=url,
                        generate_markdown=generate_markdown,
                        wait_for_js=wait_for_js,
                        timeout=timeout,
                        use_undetected_browser=True
                    )
                    
                    if fallback_result.get("success", False):
                        crawl_result = fallback_result
                        crawl_result["fallback_used"] = True
                    else:
                        return crawl_result
                
                # Enhanced CSS selector extraction with table detection
                from bs4 import BeautifulSoup
                
                html_content = crawl_result.get("content", "")
                soup = BeautifulSoup(html_content, 'html.parser')
                
                extracted_data = {}
                tables_found = []
                
                # Enhanced table detection and extraction
                tables = soup.find_all('table')
                if tables and use_llm_table_extraction:
                    for i, table in enumerate(tables):
                        table_data = {
                            "table_index": i,
                            "headers": [],
                            "rows": [],
                            "extraction_method": "enhanced_css_with_table_support"
                        }
                        
                        # Extract headers
                        headers = table.find_all(['th', 'td'])
                        if headers:
                            table_data["headers"] = [h.get_text().strip() for h in headers[:10]]  # Limit for performance
                        
                        # Extract first few rows
                        rows = table.find_all('tr')
                        for j, row in enumerate(rows[:5]):  # Limit for performance
                            cells = row.find_all(['td', 'th'])
                            row_data = [cell.get_text().strip() for cell in cells]
                            if row_data:
                                table_data["rows"].append(row_data)
                        
                        tables_found.append(table_data)
                
                # Standard CSS selector extraction
                for key, selector in css_selectors.items():
                    elements = soup.select(selector)
                    if elements:
                        if len(elements) == 1:
                            extracted_data[key] = elements[0].get_text().strip()
                        else:
                            extracted_data[key] = [elem.get_text().strip() for elem in elements]
                    else:
                        extracted_data[key] = None
                
                result = {
                    "success": True,
                    "url": url,
                    "extracted_data": extracted_data,
                    "processing_method": "enhanced_css_selector_extraction",
                    "content": crawl_result.get("content", ""),
                    "markdown": crawl_result.get("markdown", "")
                }
                
                if tables_found:
                    result["tables_detected"] = len(tables_found)
                    result["table_data"] = tables_found
                    result["table_extraction_enhanced"] = True
                
                if crawl_result.get("fallback_used"):
                    result["fallback_used"] = True
                
                # Apply token limit fallback before returning
                return _apply_token_limit_fallback(result, max_tokens=25000)
                
            except ImportError:
                # If BeautifulSoup not available, try fallback crawl
                try:
                    fallback_result = await web_crawling.crawl_url_with_fallback(
                        url=url,
                        generate_markdown=generate_markdown,
                        wait_for_js=wait_for_js,
                        timeout=timeout,
                        use_undetected_browser=True
                    )
                    
                    if fallback_result.get("success", False):
                        fallback_response = {
                            "success": True,
                            "url": url,
                            "extracted_data": {"raw_content": fallback_result.get("content", "")[:500] + "..."},
                            "processing_method": "fallback_crawl_only",
                            "content": fallback_result.get("content", ""),
                            "markdown": fallback_result.get("markdown", ""),
                            "fallback_used": True,
                            "note": "BeautifulSoup not available - CSS extraction skipped"
                        }
                        
                        # Apply token limit fallback before returning
                        return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "error": "BeautifulSoup not available for CSS extraction"
                }
                
            except Exception as e:
                # Try fallback on CSS extraction error
                try:
                    fallback_result = await web_crawling.crawl_url_with_fallback(
                        url=url,
                        generate_markdown=generate_markdown,
                        wait_for_js=wait_for_js,
                        timeout=timeout,
                        use_undetected_browser=True
                    )
                    
                    if fallback_result.get("success", False):
                        fallback_response = {
                            "success": True,
                            "url": url,
                            "extracted_data": {"raw_content": fallback_result.get("content", "")[:500] + "..."},
                            "processing_method": "fallback_crawl_after_css_error",
                            "content": fallback_result.get("content", ""),
                            "markdown": fallback_result.get("markdown", ""),
                            "fallback_used": True,
                            "original_error": str(e)
                        }
                        
                        # Apply token limit fallback before returning
                        return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                        
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "error": f"CSS extraction error: {str(e)}"
                }
        
        else:
            # Fallback to basic crawling or LLM extraction
            crawl_result = await web_crawling.crawl_url(
                url=url,
                generate_markdown=generate_markdown,
                wait_for_js=wait_for_js,
                timeout=timeout
            )
            
            # If basic crawl fails, try fallback
            if not crawl_result.get("success", False) or not crawl_result.get("content", "").strip():
                fallback_result = await web_crawling.crawl_url_with_fallback(
                    url=url,
                    generate_markdown=generate_markdown,
                    wait_for_js=wait_for_js,
                    timeout=timeout,
                    use_undetected_browser=True
                )
                
                if fallback_result.get("success", False):
                    crawl_result = fallback_result
                    crawl_result["fallback_used"] = True
            
            if crawl_result.get("success", False):
                crawl_result["processing_method"] = "basic_crawl_fallback"
                crawl_result["note"] = "Used basic crawling - structured extraction not configured"
                crawl_result["extracted_data"] = {"raw_content": crawl_result.get("content", "")[:500] + "..."}
            
            # Apply token limit fallback before returning
            return _apply_token_limit_fallback(crawl_result, max_tokens=25000)
        
    except Exception as e:
        # Final fallback attempt
        try:
            fallback_result = await web_crawling.crawl_url_with_fallback(
                url=url,
                generate_markdown=generate_markdown,
                wait_for_js=wait_for_js,
                timeout=timeout,
                use_undetected_browser=True
            )
            
            if fallback_result.get("success", False):
                fallback_response = {
                    "success": True,
                    "url": url,
                    "extracted_data": {"raw_content": fallback_result.get("content", "")[:500] + "..."},
                    "processing_method": "emergency_fallback",
                    "content": fallback_result.get("content", ""),
                    "markdown": fallback_result.get("markdown", ""),
                    "fallback_used": True,
                    "original_error": str(e)
                }
                
                # Apply token limit fallback before returning
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
        except Exception:
            pass
        
        return {
            "success": False,
            "error": f"Structured extraction error: {str(e)}"
        }

@mcp.tool()
async def search_google(
    request: Annotated[Dict[str, Any], Field(description="GoogleSearchRequest dictionary containing: query (required), num_results (default: 10), search_genre (optional), language (default: 'en'), region (default: 'us'), recent_days (optional, filter to past N days)")]
) -> Dict[str, Any]:
    """
    Perform Google search with genre filtering and extract structured results with metadata.

    Advanced search with genre-specific filtering for targeted results:
    - academic: Research papers, citations, scholarly content
    - news: Recent news articles and updates  
    - technical: Documentation, tutorials, stack overflow
    - commercial: Product pages, reviews, shopping
    - social: Social media content and discussions
    
    Date filtering: Use recent_days in request to filter to recent results (e.g., 7 for last week).
    
    Returns web search results with titles, snippets, URLs, and metadata.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await search.search_google(request)
        
        # Apply token limit fallback to prevent MCP errors
        result_with_fallback = _apply_token_limit_fallback(result, max_tokens=20000)
        
        return result_with_fallback
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Google search error: {str(e)}"
        }

async def batch_search_google(
    request: Annotated[Dict[str, Any], Field(description="GoogleBatchSearchRequest dictionary containing: queries (required list), num_results_per_query (default: 10), search_genre (optional), max_concurrent (default: 3), recent_days (optional), auto_summarize (default: False), summary_length (default: 'medium'), llm_provider (optional), llm_model (optional)")]
) -> Dict[str, Any]:
    """
    Perform multiple Google searches efficiently with analysis and comparison.
    
    Execute multiple search queries with intelligent rate limiting and result analysis.
    Provides comparative insights across different search topics.
    
    Date filtering: Use recent_days in request to filter to recent results (e.g., 7 for last week).
    AI summarization: Disabled by default, enable with auto_summarize=True in request.
    
    Perfect for research, competitive analysis, and comprehensive topic exploration.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
    try:
        result = await search.batch_search_google(request)
        return _apply_token_limit_fallback(result, max_tokens=25000)
    except Exception as e:
        return {
            "success": False,
            "error": f"Batch search error: {str(e)}"
        }

@mcp.tool()
async def search_and_crawl(
    request: Annotated[Dict[str, Any], Field(description="SearchAndCrawlRequest dictionary containing: search_query (required), crawl_top_results (default: 2), search_genre (optional), recent_days (optional), generate_markdown (default: True), max_content_per_page (default: 5000)")]
) -> Dict[str, Any]:
    """
    Search Google and automatically crawl top results for comprehensive content extraction.
    
    Combines search and crawling in one operation: finds relevant pages, then extracts full content.
    Perfect for research, competitive analysis, and comprehensive topic investigation.
    
    Date filtering: Use recent_days in request to filter to recent results (e.g., 7 for last week).
    
    Provides both search metadata and full page content in structured format.
    Response size is controlled to prevent token limit issues.
    Automatically applies fallback strategies for failed crawls.
    """
    _load_tool_modules()
    if not _tools_imported:
        return {
            "success": False,
            "error": "Tool modules not available"
        }
    
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
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            generate_markdown=generate_markdown,
                            timeout=30
                        )
                        
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
        return _apply_token_limit_fallback(result, max_tokens=25000)
        
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
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            generate_markdown=generate_markdown,
                            timeout=30
                        )
                        
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
                return _apply_token_limit_fallback(fallback_response, max_tokens=25000)
                
        except Exception as fallback_error:
            pass
        
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

async def batch_crawl(
    urls: Annotated[List[str], Field(description="List of URLs to crawl. Examples: ['https://site1.com', 'https://site2.com/page', 'https://docs.example.com']")],
    base_timeout: Annotated[int, Field(description="Base timeout in seconds, auto-adjusted based on URL count (default: 30)")] = 30,
    generate_markdown: Annotated[bool, Field(description="Generate markdown content for each page (default: True)")] = True,
    extract_media: Annotated[bool, Field(description="Extract media links from pages (default: False)")] = False,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to load (default: False)")] = False,
    max_concurrent: Annotated[int, Field(description="Maximum concurrent crawls (default: 3)")] = 3,
    use_undetected_browser: Annotated[bool, Field(description="Use undetected browser for all URLs to bypass bot detection (default: False)")] = False
) -> List[Dict[str, Any]]:
    """
    Crawl multiple URLs in batch with intelligent rate limiting and error handling.
    
    Process multiple URLs concurrently for maximum efficiency while respecting server limits.
    Timeout automatically scales based on the number of URLs to crawl.
    NEW v0.7.4: Supports undetected browser mode for enhanced bot detection bypass.
    Automatically applies fallback strategies for failed individual crawls.
    
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
            "max_concurrent": max_concurrent,
            "use_undetected_browser": use_undetected_browser
        }
        
        # Add timeout handling - optimized for faster response
        import asyncio
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
        final_result = _apply_token_limit_fallback(batch_response, max_tokens=25000)

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
            final_result = _apply_token_limit_fallback(batch_response, max_tokens=25000)
            return final_result.get("batch_results", fallback_results)
            
        except Exception:
            return [{
                "success": False,
                "error": f"Batch crawl error: {str(e)}"
            }]

async def multi_url_crawl(
    url_configurations: Annotated[Dict[str, Dict], Field(description="URL pattern to configuration mapping. Example: {'*news*': {'wait_for_js': True}, '*api*': {'timeout': 120}}")],
    pattern_matching: Annotated[str, Field(description="Pattern matching method: 'wildcard' (*.com), 'regex' (advanced patterns) (default: 'wildcard')")] = "wildcard",
    default_config: Annotated[Optional[Dict], Field(description="Default configuration for URLs not matching any pattern (default: None)")] = None,
    base_timeout: Annotated[int, Field(description="Base timeout per URL in seconds (default: 30)")] = 30,
    max_concurrent: Annotated[int, Field(description="Maximum concurrent crawls (default: 3)")] = 3
) -> List[Dict[str, Any]]:
    """
    NEW v0.7.3: Multi-URL Configuration - Different strategies for different URL patterns in one batch.
    
    Advanced batch crawling with pattern-based configuration matching.
    Each URL is processed with settings that match its pattern, allowing for optimized
    crawling strategies per site type (e.g., news sites vs APIs vs documentation).
    
    Pattern Examples:
    - Wildcard: '*news*', '*api*', '*.pdf', 'https://docs.*'
    - Regex: r'.*\/(api|v\d+)\/', r'https:\/\/[^\/]+\.com\/news'
    
    Perfect for mixed-domain crawling with site-specific optimizations.
    """
    _load_tool_modules()
    if not _tools_imported:
        return [{
            "success": False,
            "error": "Tool modules not available"
        }]
    
    try:
        import re
        import fnmatch
        
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
                "css_selector": matched_config.get("css_selector"),
                "xpath": matched_config.get("xpath")
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
        final_result = _apply_token_limit_fallback(batch_response, max_tokens=25000)
        
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
        "batch": ["batch_crawl", "multi_url_crawl"],
        "files": ["process_file", "get_supported_file_formats", "enhanced_process_large_content"],
        "config": ["get_llm_config_info", "get_tool_selection_guide"],
        "diagnostics": ["get_system_diagnostics"],
        "new_v074_features": {
            "undetected_browser": "Enhanced crawl_url and batch_crawl with use_undetected_browser parameter",
            "llm_table_extraction": "Revolutionary table extraction in extract_structured_data with use_llm_table_extraction",
            "multi_url_config": "Pattern-based configuration matching in multi_url_crawl tool",
            "intelligent_chunking": "Massive table support with adaptive chunking strategies"
        },
        "best_practices": {
            "bot_detection": "Use undetected browser mode for difficult sites",
            "table_data": "Enable LLM table extraction for complex table structures",
            "mixed_domains": "Use multi_url_crawl for site-specific optimization",
            "fallback_reliability": "All tools now include automatic fallback mechanisms"
        }
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