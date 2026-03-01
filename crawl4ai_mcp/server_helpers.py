"""Server helper functions for Crawl4AI MCP Server.

This module is a backward-compatibility facade.
Functions have been moved to:
- crawl4ai_mcp.infra.browser (browser setup, heavy imports)
- crawl4ai_mcp.server (tool module loading: _load_tool_modules,
  get_tool_modules, is_tools_imported — NOT re-exported here
  to avoid circular imports)
- crawl4ai_mcp.infra.diagnostics (system diagnostics)
- crawl4ai_mcp.middleware.content_slicing (content slicing)
- crawl4ai_mcp.middleware.search_slicing (search content slicing)
- crawl4ai_mcp.middleware.cache (search cache)
- crawl4ai_mcp.middleware.response_transform (result conversion)
"""

# Browser setup (from infra)
from .infra.browser import (  # noqa: F401
    _load_heavy_imports,
    _ensure_browser_setup,
    is_heavy_imports_loaded,
    is_browser_setup_done,
)

# Diagnostics (from infra)
from .infra.diagnostics import get_system_diagnostics  # noqa: F401

# Response transform (from middleware)
from .middleware.response_transform import (  # noqa: F401
    _convert_result_to_dict,
    _process_content_fields,
)

# Fallback trigger - kept here as it's a simple utility
from typing import Tuple


def _should_trigger_fallback(
    result_dict: dict,
    generate_markdown: bool
) -> Tuple[bool, str]:
    """
    Determine if fallback should be triggered and return reason.

    Args:
        result_dict: The crawl result dictionary
        generate_markdown: Whether markdown generation was requested

    Returns:
        Tuple of (should_trigger, reason_message)
    """
    # Check explicit failure
    if not result_dict.get("success", True):
        error_msg = result_dict.get("error", "Unknown error")
        return True, f"Initial crawl failed: {error_msg}"

    # Check for meaningful content based on what was requested
    has_markdown = bool(result_dict.get("markdown", "").strip())
    has_content = bool(result_dict.get("content", "").strip())
    has_raw_content = bool(result_dict.get("raw_content", "").strip())

    # If markdown was requested but is empty, and no other content exists
    if generate_markdown and not has_markdown:
        if has_content or has_raw_content:
            return False, ""
        return True, "Empty markdown and no alternative content available"

    # If markdown was not requested, check for any content
    if not generate_markdown and not has_content and not has_raw_content:
        return True, "No content available in response"

    return False, ""


# Content slicing (from middleware)
from .middleware.content_slicing import _apply_content_slicing  # noqa: F401, E402

# Search content slicing (from middleware)
from .middleware.search_slicing import (  # noqa: F401, E402
    _apply_search_content_slicing,
)

# Search cache (from middleware)
from .middleware.cache import (  # noqa: F401, E402
    _CacheEntry,
    _get_search_cache_key,
    _cleanup_expired_cache,
    _get_cached_search_result,
    _cache_search_result,
    clear_search_cache,
    _SEARCH_CACHE_MAX_SIZE,
    _SEARCH_CACHE_TTL_SECONDS,
    _search_result_cache,
)
