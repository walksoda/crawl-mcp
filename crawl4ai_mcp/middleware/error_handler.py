"""Error handling middleware for Crawl4AI MCP Server.

Catches exceptions from handler execution and returns
structured error responses.
"""

import traceback

from .pipeline import Middleware, PipelineContext


class ErrorHandlingMiddleware(Middleware):
    """Catches exceptions and returns structured error dicts.

    This middleware should typically be the last in the chain
    so it wraps all other middleware and the handler.
    """

    async def after(self, ctx: PipelineContext) -> None:
        # This middleware works differently - it wraps the handler
        # The actual error catching is done in the pipeline execute
        pass
