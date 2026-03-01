"""
MCP tool registrations for Crawl4AI MCP Server.

Thin adapter layer: each module registers @mcp.tool() handlers
that delegate to core/ business logic via server_helpers.
"""

from .crawl_tools import register_crawl_tools
from .extraction_tools import register_extraction_tools
from .youtube_tools import register_youtube_tools
from .file_tools import register_file_tools
from .search_tools import register_search_tools
from .batch_tools import register_batch_tools


def register_all_tools(mcp, get_modules):
    """Register all MCP tools on the given FastMCP instance.

    Args:
        mcp: FastMCP server instance.
        get_modules: Callable returning (web_crawling, search, youtube, file_processing, utilities).
    """
    register_crawl_tools(mcp, get_modules)
    register_extraction_tools(mcp, get_modules)
    register_youtube_tools(mcp, get_modules)
    register_file_tools(mcp, get_modules)
    register_search_tools(mcp, get_modules)
    register_batch_tools(mcp, get_modules)
