"""Response transform middleware for Crawl4AI MCP Server.

Converts CrawlResponse objects to dicts and processes content fields
(removing HTML when markdown is available, etc.).
"""

from typing import Optional, Any

from .pipeline import Middleware, PipelineContext


class ResponseTransformMiddleware(Middleware):
    """Converts CrawlResponse to dict and processes content fields.

    Applied after the handler to normalize response format.
    """

    async def after(self, ctx: PipelineContext) -> None:
        if ctx.result is None:
            return

        # Convert to dict if needed
        ctx.result = _convert_result_to_dict(ctx.result)

        if not isinstance(ctx.result, dict):
            return

        # Process content fields based on params
        include_cleaned_html = ctx.params.get('include_cleaned_html', False)
        generate_markdown = ctx.params.get('generate_markdown', True)

        ctx.result = _process_content_fields(
            ctx.result, include_cleaned_html, generate_markdown
        )


def _convert_result_to_dict(result) -> dict:
    """Convert CrawlResponse or similar object to dict."""
    if hasattr(result, 'model_dump'):
        return result.model_dump()
    elif hasattr(result, 'dict'):
        return result.dict()
    return result


def _process_content_fields(
    result_dict: dict,
    include_cleaned_html: bool,
    generate_markdown: bool
) -> dict:
    """Process content fields based on flags and add warnings if needed."""
    # Preserve existing warnings from crawler
    warnings = result_dict.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [warnings] if warnings else []

    # Handle content field based on include_cleaned_html flag
    if not include_cleaned_html and 'content' in result_dict:
        # Only remove if markdown is available or generate_markdown is True
        if generate_markdown and result_dict.get("markdown", "").strip():
            del result_dict['content']
            warnings.append(
                "HTML content removed to save tokens. Use include_cleaned_html=true to include it."
            )
        elif not generate_markdown:
            # Keep content when markdown generation is disabled
            warnings.append("HTML content preserved because generate_markdown=false.")

    if warnings:
        result_dict["warnings"] = warnings

    return result_dict
