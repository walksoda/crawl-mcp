"""search_and_crawl end to end through the real core and tool layer.

Only the search engine and the page crawler are stubbed. The earlier tests
mocked the core's return value with a crawled_pages shape the core never
produced, so output_path, re-crawl and truncation were broken unnoticed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai_mcp.core import search_crawl
from crawl4ai_mcp.server_tools.search_tools import register_search_tools
from crawl4ai_mcp.tools import search as search_facade

HITS = [
    {"rank": 1, "url": "https://ok.example/a", "domain": "ok.example", "title": "A", "snippet": "sa", "type": "general"},
    {"rank": 2, "url": "https://bad.example/b", "domain": "bad.example", "title": "B", "snippet": "sb", "type": "general"},
]


class FakeMcp:
    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _crawl(request):
    if "bad.example" in request.url:
        return SimpleNamespace(success=False, url=request.url, title=None, markdown=None, content=None,
                               media=None, error="HTTP 403", links=None, metadata=None,
                               processing_method="crawl4ai")
    return SimpleNamespace(success=True, url=request.url, title="Page A", markdown="# A\n" + "x" * 9000,
                           content="<html/>", media=None, error=None, links={"internal": []},
                           metadata={"lang": "en"}, processing_method="crawl4ai")


@pytest.fixture
def stubbed(monkeypatch):
    processor = MagicMock()
    processor.search = AsyncMock(return_value={
        "success": True, "engine": "brave", "result_type": "web", "results": HITS,
        "search_metadata": {"search_params": {"engine": "brave"}},
    })
    monkeypatch.setattr(search_crawl, "web_search_processor", processor)
    monkeypatch.setattr(search_crawl, "_internal_crawl_url", AsyncMock(side_effect=_crawl))
    monkeypatch.setattr(search_crawl.asyncio, "sleep", AsyncMock())

    web_crawling = MagicMock()
    web_crawling.crawl_url_with_fallback = AsyncMock(return_value={
        "success": True, "url": "https://bad.example/b", "title": "Page B", "markdown": "# B recovered",
    })
    fm = FakeMcp()
    register_search_tools(fm, lambda: (web_crawling, search_facade, MagicMock(), MagicMock(), MagicMock()))
    return fm.tools["search_and_crawl"], processor, web_crawling


@pytest.mark.asyncio
async def test_core_returns_flat_crawled_pages(stubbed):
    result = await search_crawl.search_and_crawl("q", crawl_top_results=2, engine="brave")

    assert "crawled_content" not in result
    a, b = result["crawled_pages"]
    assert a["url"] == "https://ok.example/a"
    assert a["title"] == "Page A"
    assert a["success"] is True
    assert a["markdown"].startswith("# A")
    assert a["search_rank"] == 1
    assert a["search_result"]["snippet"] == "sa"
    assert "error" not in a
    assert b["success"] is False
    assert b["error"] == "HTTP 403"
    assert b["title"] == "B"
    assert result["crawl_summary"]["successful_crawls"] == 1


@pytest.mark.asyncio
async def test_tool_recrawls_failed_page_and_truncates(stubbed):
    tool, _, web_crawling = stubbed
    result = await tool(request={"search_query": "q", "crawl_top_results": 2, "max_content_per_page": 100})

    web_crawling.crawl_url_with_fallback.assert_awaited_once()
    assert web_crawling.crawl_url_with_fallback.await_args.kwargs["url"] == "https://bad.example/b"
    a, b = result["crawled_pages"]
    assert a["markdown"].endswith("... [truncated for size limit]")
    assert len(a["markdown"]) < 200
    assert b["success"] is True
    assert b["fallback_used"] is True
    assert b["markdown"] == "# B recovered"
    assert "error" not in b
    assert result["crawl_summary"]["successful_crawls"] == 2
    assert result["crawl_summary"]["failed_crawls"] == 0


@pytest.mark.asyncio
async def test_tool_output_path_writes_pages(stubbed, tmp_path):
    tool, _, web_crawling = stubbed
    web_crawling.crawl_url_with_fallback = AsyncMock(return_value={"success": False, "error": "still blocked"})
    out = tmp_path / "pages"
    result = await tool(request={"search_query": "q", "crawl_top_results": 2, "max_content_per_page": 100,
                                 "output_path": str(out)})

    assert result.get("success") is not False, result
    index = json.loads((out / "index.json").read_text())
    entries = index["items"] if isinstance(index, dict) and "items" in index else index
    files = {e["url"]: e.get("file") for e in entries}
    assert files["https://bad.example/b"] is None
    page_file = out / files["https://ok.example/a"]
    text = page_file.read_text()
    # Disk holds the full page, written before max_content_per_page truncation.
    assert "x" * 9000 in text
    assert 'url: "https://ok.example/a"' in text
    # Slim response: bodies are left out unless include_content_in_response.
    assert all("markdown" not in p for p in result["crawled_pages"])


@pytest.mark.asyncio
async def test_tool_passes_result_type_and_engine_to_search(stubbed):
    tool, processor, _ = stubbed
    await tool(request={"search_query": "q", "crawl_top_results": 1, "engine": "brave",
                        "result_type": "discussions"})
    kwargs = processor.search.await_args.kwargs
    assert kwargs["engine"] == "brave"
    assert kwargs["result_type"] == "discussions"
