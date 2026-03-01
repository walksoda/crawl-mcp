"""Fallback strategies for Crawl4AI MCP Server.

This module is a backward-compatibility facade.
All implementation has been moved to crawl4ai_mcp.infra.fallback_strategies.
"""

from ..infra.fallback_strategies import (  # noqa: F401
    normalize_cookies_to_playwright_format,
    static_fetch_content,
    extract_spa_json_data,
    detect_spa_framework,
    build_amp_url,
    try_fetch_rss_feed,
    get_fallback_stage_info,
    _normalize_cookies_to_playwright_format,
    _static_fetch_content,
    _extract_spa_json_data,
    _detect_spa_framework,
    _build_amp_url,
    _try_fetch_rss_feed,
)
