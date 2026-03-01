"""Module check middleware for Crawl4AI MCP Server.

Ensures heavy imports and browser are initialized before tool execution.
"""

from .pipeline import Middleware, PipelineContext
from ..infra.browser import _load_heavy_imports, _ensure_browser_setup


class ModuleCheckMiddleware(Middleware):
    """Ensures required modules are loaded before tool execution."""

    async def before(self, ctx: PipelineContext) -> None:
        _load_heavy_imports()
        _ensure_browser_setup()
        return None
