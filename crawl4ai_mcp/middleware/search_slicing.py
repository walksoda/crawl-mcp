"""Search content slicing middleware for Crawl4AI MCP Server.

Combines search result snippets into a content field and applies
offset/limit slicing specific to search results.
"""

from .pipeline import Middleware, PipelineContext


class SearchSlicingMiddleware(Middleware):
    """Applies content slicing to search results.

    Combines individual result snippets into a single 'content' field
    and applies offset/limit slicing.
    """

    async def after(self, ctx: PipelineContext) -> None:
        if not isinstance(ctx.result, dict):
            return

        content_limit = ctx.params.get('content_limit', 0) or 0
        content_offset = ctx.params.get('content_offset', 0) or 0

        # Skip if no slicing requested
        if content_limit == 0 and content_offset == 0:
            return

        ctx.result = _apply_search_content_slicing(
            ctx.result, content_limit, content_offset
        )


def _apply_search_content_slicing(
    result_dict: dict,
    content_limit: int,
    content_offset: int
) -> dict:
    """
    Apply content slicing to search results by combining snippets.

    Args:
        result_dict: The search result dictionary
        content_limit: Maximum characters to return (0=unlimited)
        content_offset: Start position for content (0-indexed)

    Returns:
        Result dictionary with content field and slicing_info added
    """
    # Defensive: ensure non-negative values
    content_limit = max(0, content_limit)
    content_offset = max(0, content_offset)

    # Null-safe handling of results
    results = result_dict.get('results') or []
    if not isinstance(results, list):
        results = []
    content_parts = []

    for i, item in enumerate(results, 1):
        # Skip non-dict items to prevent AttributeError
        if not isinstance(item, dict):
            continue
        title = item.get('title', '') or ''
        snippet = item.get('snippet', '') or ''
        url = item.get('url', '') or ''
        part = f"[{i}] {title}\n{url}\n{snippet}\n"
        content_parts.append(part)

    combined_content = "\n".join(content_parts)
    original_length = len(combined_content)

    # Apply slicing
    sliced = combined_content[content_offset:]
    if content_limit > 0:
        sliced = sliced[:content_limit]

    result_dict['content'] = sliced

    # Calculate effective_limit
    if content_limit > 0:
        effective_limit = content_limit
    else:
        effective_limit = max(0, original_length - content_offset)

    result_dict['slicing_info'] = {
        'content': {
            'original_length': original_length,
            'offset': content_offset,
            'limit': content_limit,
            'effective_limit': effective_limit,
            'returned_length': len(sliced),
            'offset_exceeded': content_offset >= original_length,
            'source': 'combined_search_snippets',
            'result_count': len(results)
        }
    }

    return result_dict
