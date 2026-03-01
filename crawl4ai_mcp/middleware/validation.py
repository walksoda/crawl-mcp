"""Validation middleware for Crawl4AI MCP Server.

Validates tool input parameters (URL, timeout, content slicing params)
before the handler executes.
"""

from typing import Optional, Any

from .pipeline import Middleware, PipelineContext
from ..validators import (
    validate_url,
    validate_timeout,
    validate_content_slicing_params,
)


class ValidationMiddleware(Middleware):
    """Validates input parameters before handler execution.

    Configurable validation steps:
    - url: Validate the 'url' parameter
    - timeout: Validate the 'timeout' parameter
    - content_slicing: Validate content_limit and content_offset
    """

    def __init__(
        self,
        url: bool = False,
        timeout: bool = False,
        content_slicing: bool = False,
    ):
        self._validate_url = url
        self._validate_timeout = timeout
        self._validate_content_slicing = content_slicing

    async def before(self, ctx: PipelineContext) -> Optional[Any]:
        params = ctx.params

        if self._validate_url and 'url' in params:
            error = validate_url(params['url'])
            if error:
                return error

        if self._validate_timeout and 'timeout' in params:
            error = validate_timeout(params['timeout'])
            if error:
                return error

        if self._validate_content_slicing:
            content_limit = params.get('content_limit', 0)
            content_offset = params.get('content_offset', 0)
            if content_limit is not None or content_offset is not None:
                error = validate_content_slicing_params(
                    content_limit or 0,
                    content_offset or 0
                )
                if error:
                    return error

        return None
