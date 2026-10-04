"""deep_crawl_site returns per-page results so output_path can persist them."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from crawl4ai_mcp.core.crawler_core import _handle_deep_crawl_list_result
from crawl4ai_mcp.models import CrawlRequest, CrawlResponse
from crawl4ai_mcp.server_tools.crawl_tools import register_crawl_tools

from .test_tool_persist_wiring import FakeMcp, _modules_tuple

URL = "https://example.invalid/"


def _page(n, success=True):
    return SimpleNamespace(
        url=f"{URL}p{n}", success=success, markdown=f"# Page {n}" if success else None,
        cleaned_html=f"<h1>Page {n}</h1>", media=None, metadata={"title": f"Page {n}"},
        error_message=None if success else "HTTP 500",
    )


async def test_core_returns_per_page_results():
    response = await _handle_deep_crawl_list_result([_page(1), _page(2), _page(3, success=False)], CrawlRequest(url=URL))

    pages = response.model_dump()["crawled_pages"]
    assert [p["url"] for p in pages] == [f"{URL}p1", f"{URL}p2", f"{URL}p3"]
    assert [p["success"] for p in pages] == [True, True, False]
    assert pages[0]["markdown"] == "# Page 1" and pages[0]["title"] == "Page 1"
    assert pages[2]["markdown"] is None and pages[2]["error"] == "HTTP 500"


def test_other_responses_have_no_crawled_pages_key():
    assert "crawled_pages" not in CrawlResponse(success=True, url=URL).model_dump()


def _tool(pages):
    web_crawling = MagicMock()
    web_crawling.deep_crawl_site = AsyncMock(return_value={
        "success": True, "url": URL, "title": "Deep crawl of 2 pages",
        "markdown": "=== p1 ===\n# Page 1\n\n=== p2 ===\n# Page 2", "content": None,
        "crawled_pages": pages, "processing_method": "deep_crawling",
    })
    fm = FakeMcp()
    register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))
    return fm.tools["deep_crawl_site"]


def _pages():
    return [
        {"url": f"{URL}p1", "title": "Page 1", "success": True, "markdown": "# Page 1", "error": None},
        {"url": f"{URL}p2", "title": "Page 2", "success": False, "markdown": None, "error": "HTTP 500"},
    ]


async def test_output_path_writes_one_file_per_page(tmp_path):
    out = tmp_path / "deep"

    result = await _tool(_pages())(url=URL, max_depth=1, max_pages=2, output_path=str(out))

    assert result["persisted"] is True
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [item["success"] for item in index["items"]] == [True, False]
    assert "# Page 1" in (out / index["items"][0]["file"].split("/")[-1]).read_text(encoding="utf-8")
    assert index["items"][1]["file"] is None
    # Slim response: no merged or per-page bodies
    assert "markdown" not in result
    assert all("markdown" not in page for page in result["crawled_pages"])


async def test_without_output_path_pages_keep_only_metadata():
    result = await _tool(_pages())(url=URL, max_depth=1, max_pages=2)

    assert "# Page 1" in result["markdown"]
    assert [page["url"] for page in result["crawled_pages"]] == [f"{URL}p1", f"{URL}p2"]
    assert all("markdown" not in page for page in result["crawled_pages"])
