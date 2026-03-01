"""Content slicing middleware for Crawl4AI MCP Server.

Applies content offset/limit slicing to crawl results
(markdown/content fields).
"""

from typing import Optional

from .pipeline import Middleware, PipelineContext


class ContentSlicingMiddleware(Middleware):
    """Applies content slicing to crawl result dicts.

    Slices 'markdown' and 'content' fields based on
    content_offset and content_limit parameters.
    Adds slicing_info to the result.
    """

    async def after(self, ctx: PipelineContext) -> None:
        if not isinstance(ctx.result, dict):
            return

        content_limit = ctx.params.get('content_limit', 0) or 0
        content_offset = ctx.params.get('content_offset', 0) or 0

        # Skip if no slicing requested
        if content_limit == 0 and content_offset == 0:
            return

        ctx.result = _apply_content_slicing(
            ctx.result, content_limit, content_offset
        )


def _apply_content_slicing(
    result_dict: dict,
    content_limit: int,
    content_offset: int
) -> dict:
    """
    Apply content slicing to retrieve partial content.

    Args:
        result_dict: The crawl result dictionary
        content_limit: Maximum characters to return (0=unlimited)
        content_offset: Start position for content (0-indexed)

    Returns:
        Result dictionary with slicing_info added

    Note:
        slicing_info reflects the state after slicing but before token limit.
        The actual returned content may be further truncated by token limits.
    """
    # Defensive: ensure non-negative values
    content_limit = max(0, content_limit)
    content_offset = max(0, content_offset)

    slicing_info = {}

    for field in ['markdown', 'content']:
        if field in result_dict and result_dict[field]:
            text = result_dict[field]
            original_length = len(text)

            # Apply slicing
            sliced = text[content_offset:]
            if content_limit > 0:
                sliced = sliced[:content_limit]
            result_dict[field] = sliced

            # Calculate effective_limit: the actual upper bound applied
            if content_limit > 0:
                effective_limit = content_limit
            else:
                effective_limit = max(0, original_length - content_offset)

            slicing_info[field] = {
                'original_length': original_length,
                'offset': content_offset,
                'limit': content_limit,  # 0 means unlimited
                'effective_limit': effective_limit,
                'returned_length': len(sliced),
                'offset_exceeded': content_offset >= original_length
            }
        else:
            # Field does not exist or is empty - still return info for consistency
            slicing_info[field] = {
                'present': False,
                'offset': content_offset,
                'limit': content_limit
            }

    result_dict['slicing_info'] = slicing_info
    return result_dict
