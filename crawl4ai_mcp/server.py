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
from pydantic import Field

# Create MCP server with clean initialization
mcp = FastMCP("Crawl4AI")

# Global lazy loading state
_heavy_imports_loaded = False
_browser_setup_done = False
_browser_setup_failed = False

def _load_heavy_imports():
    """Load heavy imports only when tools are actually used"""
    global _heavy_imports_loaded
    if _heavy_imports_loaded:
        return
        
    global asyncio, json, BaseModel, AsyncWebCrawler
    
    import asyncio
    import json
    from pydantic import BaseModel
    from crawl4ai import AsyncWebCrawler
    
    _heavy_imports_loaded = True

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
    """Get system diagnostics for troubleshooting browser setup"""
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
    url: Annotated[str, Field(description="URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction")] = None,
    wait_for_selector: Annotated[Optional[str], Field(description="CSS selector to wait for")] = None,
    timeout: Annotated[int, Field(description="Timeout in seconds")] = 60,
    extract_media: Annotated[bool, Field(description="Extract media files")] = False,
    take_screenshot: Annotated[bool, Field(description="Take screenshot")] = False
) -> dict:
    """Crawl a URL and extract content"""
    _load_heavy_imports()
    
    if not _ensure_browser_setup():
        return {
            "success": False,
            "error": "Browser setup required. Install Playwright: pip install playwright && playwright install webkit"
        }
    
    try:
        # Import crawler config classes
        from crawl4ai import CrawlerRunConfig
        
        config = CrawlerRunConfig(
            css_selector=css_selector,
            wait_for=wait_for_selector,
            page_timeout=timeout * 1000,
            screenshot=take_screenshot,
            exclude_all_images=not extract_media,
            verbose=False
        )
        
        async with AsyncWebCrawler(headless=True, browser_type="webkit") as crawler:
            result = await crawler.arun(url, config=config)
            
            return {
                "success": result.success,
                "url": url,
                "title": result.metadata.get("title", "") if result.success else "",
                "content": result.cleaned_html if result.success else "",
                "markdown": result.markdown if result.success else "",
                "error": None if result.success else "Crawling failed"
            }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"Crawling error: {str(e)}"
        }

@mcp.tool()
async def extract_youtube_transcript(
    url: Annotated[str, Field(description="YouTube video URL")],
    languages: Annotated[List[str], Field(description="Language preferences")] = ["ja", "en"],
    include_timestamps: Annotated[bool, Field(description="Include timestamps")] = True
) -> dict:
    """Extract YouTube video transcript"""
    _load_heavy_imports()
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        
        # Extract video ID
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if not video_id_match:
            return {"success": False, "error": "Invalid YouTube URL"}
        
        video_id = video_id_match.group(1)
        
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        
        # Format transcript
        if include_timestamps:
            formatted = "\n".join([f"[{int(entry['start'])}s] {entry['text']}" for entry in transcript])
        else:
            formatted = " ".join([entry['text'] for entry in transcript])
        
        return {
            "success": True,
            "video_id": video_id,
            "transcript": formatted,
            "language": transcript[0].get('language_code', 'unknown') if transcript else 'unknown'
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"YouTube transcript error: {str(e)}"
        }

@mcp.tool()
async def search_google(
    query: Annotated[str, Field(description="Search query")],
    num_results: Annotated[int, Field(description="Number of results")] = 10
) -> dict:
    """Perform Google search"""
    _load_heavy_imports()
    
    try:
        from googlesearch import search
        
        results = []
        for i, url in enumerate(search(query, num_results=num_results)):
            if i >= num_results:
                break
            results.append({
                "url": url,
                "title": f"Search result {i+1}",
                "snippet": "Search result snippet"
            })
        
        return {
            "success": True,
            "query": query,
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Search error: {str(e)}"
        }

@mcp.tool()
async def process_file(
    url: Annotated[str, Field(description="File URL to process")],
    max_size_mb: Annotated[int, Field(description="Maximum file size in MB")] = 100
) -> dict:
    """Process files (PDF, Word, etc.) and convert to markdown"""
    _load_heavy_imports()
    
    try:
        from markitdown import MarkItDown
        import aiohttp
        
        # Download file
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return {"success": False, "error": f"Failed to download: {response.status}"}
                
                content = await response.read()
                
                # Check size
                size_mb = len(content) / (1024 * 1024)
                if size_mb > max_size_mb:
                    return {"success": False, "error": f"File too large: {size_mb:.1f}MB > {max_size_mb}MB"}
                
                # Process with MarkItDown
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    
                    try:
                        markitdown = MarkItDown()
                        result = markitdown.convert(temp_file.name)
                        
                        return {
                            "success": True,
                            "url": url,
                            "content": result.text_content,
                            "file_size_mb": round(size_mb, 2)
                        }
                    finally:
                        os.unlink(temp_file.name)
                        
    except Exception as e:
        return {
            "success": False,
            "error": f"File processing error: {str(e)}"
        }

@mcp.tool()
def get_supported_file_formats() -> dict:
    """Get list of supported file formats"""
    return {
        "supported_formats": {
            "documents": [".pdf", ".docx", ".pptx", ".xlsx"],
            "archives": [".zip"],
            "text": [".txt", ".md", ".csv"]
        },
        "processors": {
            "pdf": "pdfminer-six",
            "docx": "mammoth", 
            "xlsx": "openpyxl"
        }
    }

@mcp.tool()
def get_tool_selection_guide() -> dict:
    """Get tool selection guide"""
    return {
        "web_crawling": ["crawl_url"],
        "youtube": ["extract_youtube_transcript"],
        "search": ["search_google"],
        "files": ["process_file", "get_supported_file_formats"],
        "diagnostics": ["get_system_diagnostics"]
    }

def main():
    """Clean main entry point - FastMCP 2.0 with no banner issues"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Crawl4AI MCP Server - FastMCP 2.0 Version")
        print("Usage: python -m crawl4ai_mcp.server_v2 [--transport TRANSPORT]")
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