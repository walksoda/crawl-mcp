"""Regression test for enhanced_process_large_content's content-field read.

The fallback path of ``enhanced_process_large_content`` used to call
``fallback_result.get("content", "")``, but ``crawl_url`` populates
``markdown`` and leaves ``content=None`` for the markdown-rendering path.
``dict.get(key, default)`` returns ``None`` (not the default) when the key
exists with a None value, so the next ``len(content)`` call crashed every
invocation with ``TypeError: object of type 'NoneType' has no len()``.

This test pins the new behavior: read ``markdown`` first, then ``content``,
defaulting to "". The exception path is gone and chunks come back populated.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai_mcp.server_tools.file_tools import register_file_tools


class FakeMcp:
    """Minimal FastMCP stand-in that captures decorated tools.

    Mirrors the style used by tests/unit/test_tool_persist_wiring.py.
    """

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        if args and callable(args[0]):
            fn = args[0]
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _modules_tuple(web_crawling, file_processing=None):
    """Construct the 5-tuple of modules register_file_tools expects."""
    return (
        web_crawling,
        MagicMock(),                     # search
        MagicMock(),                     # youtube
        file_processing or MagicMock(),  # file_processing
        MagicMock(),                     # utilities
    )


def _register(web_crawling) -> FakeMcp:
    fm = FakeMcp()
    register_file_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))
    return fm


# Long-enough text to comfortably split into more than one chunk.
SAMPLE_MARKDOWN = (
    "# HTTP\n\n"
    "The Hypertext Transfer Protocol is an application protocol for "
    "distributed, collaborative, hypermedia information systems. "
    + ("HTTP forms the foundation of data communication for the web. " * 30)
)


def _crawl_response(*, markdown: str | None, content: str | None) -> Any:
    """Return a SimpleNamespace shaped like a CrawlResponse model.

    The function under test passes the result through
    ``_convert_result_to_dict``, so any object with these attributes
    (or a dict) works.
    """
    resp = MagicMock()
    resp.success = True
    resp.url = "https://example.org/doc"
    resp.title = "Doc"
    resp.markdown = markdown
    resp.content = content
    resp.error = None
    resp.processing_time = 0.1
    resp.metadata = {}
    resp.media = None
    resp.screenshot = None
    resp.extracted_data = {}
    resp.model_dump = MagicMock(return_value={
        "success": True,
        "url": "https://example.org/doc",
        "title": "Doc",
        "markdown": markdown,
        "content": content,
        "error": None,
        "processing_time": 0.1,
        "metadata": {},
        "media": None,
        "screenshot": None,
        "extracted_data": {},
    })
    return resp


class TestEnhancedLargeContentReadsMarkdown:
    """The fallback path must use markdown when content is None."""

    @pytest.mark.asyncio
    async def test_markdown_only_response_does_not_crash(self):
        """The legacy bug: content=None -> len(None) crash. Must succeed now."""
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(
            return_value=_crawl_response(markdown=SAMPLE_MARKDOWN, content=None)
        )

        fm = _register(web_crawling)
        tool = fm.tools["enhanced_process_large_content"]

        result = await tool(
            url="https://example.org/doc",
            chunking_strategy="sentence",
            filtering_strategy="bm25",
            max_chunk_tokens=200,
            extract_top_chunks=2,
            summarize_chunks=False,
        )

        assert result["success"] is True, f"unexpected error: {result.get('error')}"
        # The bug always reported original_content_length=0; with markdown
        # used we must see the real length.
        assert result["original_content_length"] == len(SAMPLE_MARKDOWN)
        assert result["total_chunks"] >= 1
        assert len(result["chunks"]) >= 1
        # The wrapper still records error_type=TypeError for the legacy
        # crash; ensure that's gone.
        assert result.get("metadata", {}).get("error_type") != "TypeError"

    @pytest.mark.asyncio
    async def test_content_only_response_still_works(self):
        """Backward compatibility: callers that populate `content` still work."""
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(
            return_value=_crawl_response(markdown=None, content=SAMPLE_MARKDOWN)
        )

        fm = _register(web_crawling)
        tool = fm.tools["enhanced_process_large_content"]

        result = await tool(
            url="https://example.org/doc",
            max_chunk_tokens=200,
            extract_top_chunks=2,
            summarize_chunks=False,
        )

        assert result["success"] is True
        assert result["original_content_length"] == len(SAMPLE_MARKDOWN)
        assert len(result["chunks"]) >= 1

    @pytest.mark.asyncio
    async def test_both_none_returns_clean_failure_not_crash(self):
        """When neither field is populated, fail cleanly — no len(None)."""
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(
            return_value=_crawl_response(markdown=None, content=None)
        )

        fm = _register(web_crawling)
        tool = fm.tools["enhanced_process_large_content"]

        result = await tool(
            url="https://example.org/doc",
            max_chunk_tokens=200,
            extract_top_chunks=2,
            summarize_chunks=False,
        )

        # Whatever shape the empty-content path returns, it must NOT be
        # the legacy crash signature.
        assert result.get("metadata", {}).get("error_type") != "TypeError"
        assert "object of type 'NoneType' has no len()" not in (result.get("error") or "")
