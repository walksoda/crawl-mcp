"""Browser setup and heavy import loading for Crawl4AI MCP Server.

Handles heavy import loading and browser cache detection.
"""

import glob
from pathlib import Path

# Global state for lazy loading
_heavy_imports_loaded = False
_browser_setup_done = False
_browser_setup_failed = False

# Global module references (heavy imports only)
asyncio = None
json = None
AsyncWebCrawler = None


def _load_heavy_imports() -> None:
    """Load heavy imports only when tools are actually used."""
    global _heavy_imports_loaded
    if _heavy_imports_loaded:
        return

    global asyncio, json, AsyncWebCrawler

    import asyncio as _asyncio
    import json as _json
    from crawl4ai import AsyncWebCrawler as _AsyncWebCrawler

    asyncio = _asyncio
    json = _json
    AsyncWebCrawler = _AsyncWebCrawler

    _heavy_imports_loaded = True


def _ensure_browser_setup() -> bool:
    """
    Browser setup with lazy loading.

    Returns:
        True if browser is set up, False otherwise.
    """
    global _browser_setup_done, _browser_setup_failed

    if _browser_setup_done:
        return True
    if _browser_setup_failed:
        return False

    try:
        # Quick browser cache check
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


def is_heavy_imports_loaded() -> bool:
    """Check if heavy imports are loaded."""
    return _heavy_imports_loaded


def is_browser_setup_done() -> bool:
    """Check if browser setup is done."""
    return _browser_setup_done
