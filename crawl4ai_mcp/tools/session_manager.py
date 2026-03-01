"""Session management for Crawl4AI MCP Server.

This module is a backward-compatibility facade.
All implementation has been moved to crawl4ai_mcp.infra.session.
"""

from ..infra.session import (  # noqa: F401
    SessionManager,
    get_session_manager,
    extract_cookies_from_result,
)
