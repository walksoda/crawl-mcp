"""Large pages must not be LLM-summarized unless auto_summarize=True.

Without opt-in, the full content reaches the tool layer, which persists it
(output_path) and then truncates the response copy with apply_token_limit.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai_mcp.core import crawler_core, crawler_summarizer
from crawl4ai_mcp.models import CrawlRequest
from crawl4ai_mcp.server_tools.crawl_tools import register_crawl_tools
from crawl4ai_mcp.utils.token_utils import estimate_tokens

from .test_tool_persist_wiring import FakeMcp, _modules_tuple

URL = "https://example.invalid/large"
# Over the old 400,000-char MCP_PROTOCOL_SAFETY threshold
LARGE_MARKDOWN = "word " * 120_000


@pytest.fixture
def summarizer(monkeypatch):
    mock = AsyncMock(return_value={"success": True, "summary": "summary"})
    monkeypatch.setattr(crawler_summarizer, "summarize_web_content", mock)
    monkeypatch.setattr(crawler_core, "summarize_web_content", mock)
    return mock


def _page(markdown):
    return SimpleNamespace(
        cleaned_html=f"<div>{markdown}</div>", markdown=markdown,
        metadata={"title": "Large"}, media=None, screenshot=None,
    )


async def test_large_page_is_returned_in_full_without_opt_in(summarizer):
    response = await crawler_core._handle_single_page_result(
        _page(LARGE_MARKDOWN), CrawlRequest(url=URL)
    )

    summarizer.assert_not_called()
    assert response.markdown == LARGE_MARKDOWN
    assert not (response.extracted_data or {}).get("summarization_applied")


async def test_large_page_is_summarized_with_opt_in(summarizer):
    response = await crawler_core._handle_single_page_result(
        _page(LARGE_MARKDOWN), CrawlRequest(url=URL, auto_summarize=True)
    )

    summarizer.assert_called_once()
    assert response.extracted_data["summarization_applied"] is True


def _register_fallback_tool(markdown):
    web_crawling = MagicMock()
    web_crawling.crawl_url_with_fallback = AsyncMock(return_value={
        "success": True, "url": URL, "title": "Large",
        "markdown": markdown, "content": markdown,
    })
    fm = FakeMcp()
    register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))
    return fm.tools["crawl_url_with_fallback"]


async def test_fallback_tool_response_is_token_limited():
    tool = _register_fallback_tool(LARGE_MARKDOWN)

    result = await tool(url=URL)

    assert result["token_limit_applied"] is True
    assert estimate_tokens(json.dumps(result)) <= 25000
    assert any("output_path" in r for r in result["recommendations"])


async def test_fallback_tool_persists_full_content(tmp_path):
    tool = _register_fallback_tool(LARGE_MARKDOWN)
    out = tmp_path / "large.md"

    result = await tool(url=URL, output_path=str(out))

    assert result["persisted"] is True
    assert LARGE_MARKDOWN in out.read_text(encoding="utf-8")
