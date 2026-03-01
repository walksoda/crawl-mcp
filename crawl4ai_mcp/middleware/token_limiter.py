"""Token limit middleware for Crawl4AI MCP Server.

Applies token limits to tool responses to prevent context overflow.
Supports both dict and list results.
"""

from .pipeline import Middleware, PipelineContext
from ..utils.token_utils import apply_token_limit


class TokenLimitMiddleware(Middleware):
    """Applies token limits to tool responses.

    For dict results: applies apply_token_limit directly.
    For list results (when supports_list=True): applies to each element.
    """

    def __init__(self, max_tokens: int = 25000, supports_list: bool = False):
        self._max_tokens = max_tokens
        self._supports_list = supports_list

    async def after(self, ctx: PipelineContext) -> None:
        if isinstance(ctx.result, dict):
            ctx.result = apply_token_limit(ctx.result, self._max_tokens)
        elif isinstance(ctx.result, list) and self._supports_list:
            ctx.result = [
                apply_token_limit(r, self._max_tokens) if isinstance(r, dict) else r
                for r in ctx.result
            ]
