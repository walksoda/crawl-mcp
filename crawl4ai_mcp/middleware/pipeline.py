"""Middleware pipeline for Crawl4AI MCP Server.

Provides the core pipeline infrastructure for composing middleware
that handle cross-cutting concerns (validation, slicing, token limits, etc.).
"""

from typing import Any, Callable, Optional, List
from dataclasses import dataclass, field


@dataclass
class PipelineContext:
    """Context object passed through the middleware pipeline.

    Attributes:
        params: Tool parameters from the MCP call
        result: Execution result (set after handler runs)
        metadata: Shared data between middlewares
        tool_name: Name of the tool being executed (for conditional logic)
    """
    params: dict
    result: Optional[Any] = None
    metadata: dict = field(default_factory=dict)
    tool_name: str = ""


class Middleware:
    """Base class for pipeline middleware.

    Subclasses override before() and/or after() to implement
    cross-cutting concerns.
    """

    async def before(self, ctx: PipelineContext) -> Optional[Any]:
        """Pre-processing hook.

        Called before the handler executes.
        Return a value to short-circuit the pipeline (early return).
        Return None to continue.
        """
        return None

    async def after(self, ctx: PipelineContext) -> None:
        """Post-processing hook.

        Called after the handler executes.
        Modify ctx.result in place to transform the response.
        """
        pass


class ToolPipeline:
    """Executes a handler through a chain of middleware.

    Middleware are executed in order:
    1. before() hooks run in declaration order
    2. The handler executes
    3. after() hooks run in declaration order

    If any before() hook returns a non-None value, the pipeline
    short-circuits and returns that value (handler and remaining
    before hooks are skipped, but after hooks still run).
    """

    def __init__(self, *middlewares: Middleware):
        self._middlewares: List[Middleware] = list(middlewares)

    async def execute(
        self,
        handler: Callable,
        params: dict,
        tool_name: str = ""
    ) -> Any:
        """Execute the handler through the middleware chain.

        Args:
            handler: The async function to execute
            params: Parameters to pass to the handler
            tool_name: Name of the tool (for conditional middleware logic)

        Returns:
            The result from the handler or from a short-circuiting middleware
        """
        ctx = PipelineContext(params=params, tool_name=tool_name)

        # Run before hooks
        for mw in self._middlewares:
            early_result = await mw.before(ctx)
            if early_result is not None:
                ctx.result = early_result
                # Still run after hooks for cleanup
                for after_mw in self._middlewares:
                    await after_mw.after(ctx)
                return ctx.result

        # Execute handler
        ctx.result = await handler(**ctx.params)

        # Run after hooks
        for mw in self._middlewares:
            await mw.after(ctx)

        return ctx.result
