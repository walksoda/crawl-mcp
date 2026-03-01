"""System diagnostics for Crawl4AI MCP Server.

Provides system information for troubleshooting browser
and environment issues.
"""

import glob
import platform
from pathlib import Path

from .browser import _load_heavy_imports


def get_system_diagnostics() -> dict:
    """Get system diagnostics for troubleshooting browser and environment issues."""
    _load_heavy_imports()

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
