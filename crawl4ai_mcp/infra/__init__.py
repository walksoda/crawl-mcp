"""Infrastructure package for Crawl4AI MCP Server.

Provides browser management, session persistence, strategy caching,
content processing utilities, and fallback strategies.
"""

from .browser import (
    _load_heavy_imports,
    _ensure_browser_setup,
    is_heavy_imports_loaded,
    is_browser_setup_done,
)
from .diagnostics import get_system_diagnostics
from .session import SessionManager, get_session_manager, extract_cookies_from_result
from .strategy_cache import StrategyCache, get_strategy_cache
from .content_cache_policy import ContentCachePolicy, get_content_cache_policy
from .fingerprint import FingerprintProfile, get_fingerprint_config

__all__ = [
    "_load_heavy_imports",
    "_ensure_browser_setup",
    "is_heavy_imports_loaded",
    "is_browser_setup_done",
    "get_system_diagnostics",
    "SessionManager",
    "get_session_manager",
    "extract_cookies_from_result",
    "StrategyCache",
    "get_strategy_cache",
    "ContentCachePolicy",
    "get_content_cache_policy",
    "FingerprintProfile",
    "get_fingerprint_config",
]
