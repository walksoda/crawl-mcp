"""Tool-level integration tests for the file-persistence wiring.

These tests register each MCP tool against a lightweight fake ``mcp`` object
so we can invoke the decorated coroutine directly with mocked core modules.
The goal is to prove that every return path of every tool we wired passes
through ``finalize_tool_response`` correctly — no live network, no FastMCP
stack. The pure persistence logic is already exercised by
``tests/unit/test_file_persistence.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai_mcp.server_tools.batch_tools import register_batch_tools
from crawl4ai_mcp.server_tools.crawl_tools import register_crawl_tools
from crawl4ai_mcp.server_tools.extraction_tools import register_extraction_tools
from crawl4ai_mcp.server_tools.file_tools import register_file_tools
from crawl4ai_mcp.server_tools.search_tools import register_search_tools
from crawl4ai_mcp.server_tools.youtube_tools import register_youtube_tools


# ---------------------------------------------------------------------------
# Fake MCP registry
# ---------------------------------------------------------------------------


class FakeMcp:
    """Minimal stand-in for FastMCP that captures decorated tools."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        # FastMCP's decorator can be used with or without args; handle both.
        if args and callable(args[0]):
            fn = args[0]
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _modules_tuple(web_crawling=None, search=None, youtube=None, file_processing=None, utilities=None):
    """Construct the 5-tuple of modules the registers expect."""
    return (
        web_crawling or MagicMock(),
        search or MagicMock(),
        youtube or MagicMock(),
        file_processing or MagicMock(),
        utilities or MagicMock(),
    )


# ---------------------------------------------------------------------------
# crawl_url wiring
# ---------------------------------------------------------------------------


class TestCrawlUrlWiring:
    def _register(self, web_crawling):
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))
        return fm

    @pytest.mark.asyncio
    async def test_output_path_rejects_relative_before_fetch(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(side_effect=AssertionError("should not be called"))
        fm = self._register(web_crawling)

        out = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path="relative/out.md",  # relative → early reject
        )
        assert out["success"] is False
        assert out["error_code"] == "output_path_not_absolute"
        web_crawling.crawl_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_path_persists_and_strips(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "Example",
            "markdown": "# Example\n\n" + "body content " * 500,
            "content": "raw body",
        })
        fm = self._register(web_crawling)

        out_file = tmp_path / "article.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
        )
        assert result["persisted"] is True
        assert result["output_files"] == [str(out_file)]
        assert "markdown" not in result
        assert "content" not in result
        # Disk has the full body
        written = out_file.read_text(encoding="utf-8")
        assert "body content" in written
        assert 'url: "https://example.com"' in written

    @pytest.mark.asyncio
    async def test_include_content_keeps_markdown(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "E",
            "markdown": "# x",
            "content": "c",
        })
        fm = self._register(web_crawling)
        out_file = tmp_path / "a.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
            include_content_in_response=True,
        )
        assert result["content_included_in_response"] is True
        assert "markdown" in result

    @pytest.mark.asyncio
    async def test_fallback_path_persists(self, tmp_path):
        """Normal crawl succeeds but triggers fallback via empty markdown."""
        web_crawling = MagicMock()
        # First call returns empty-markdown → should_fallback triggers.
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "Empty",
            "markdown": "",
            "content": "",
        })
        # Fallback returns full content.
        web_crawling.crawl_url_with_fallback = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "Recovered",
            "markdown": "# Recovered",
            "content": "body",
        })
        fm = self._register(web_crawling)
        out_file = tmp_path / "out.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        assert out_file.exists()
        assert "Recovered" in out_file.read_text()

    @pytest.mark.asyncio
    async def test_exception_fallback_path_persists(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(side_effect=RuntimeError("boom"))
        web_crawling.crawl_url_with_fallback = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "T",
            "markdown": "# Fallback",
            "content": "c",
        })
        fm = self._register(web_crawling)
        out_file = tmp_path / "out.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        assert "Fallback" in out_file.read_text()

    @pytest.mark.asyncio
    async def test_persist_captures_full_content_despite_content_limit(self, tmp_path):
        """Regression: disk must hold the full unsliced markdown even when
        the caller also asked for content_limit/content_offset slicing.
        Previously persist ran AFTER _apply_content_slicing and the on-disk
        copy was truncated to match the response."""
        web_crawling = MagicMock()
        full_md = "# Huge\n" + ("line\n" * 1000)
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "T",
            "markdown": full_md,
            "content": "raw",
        })
        fm = self._register(web_crawling)
        out_file = tmp_path / "full.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
            include_content_in_response=True,  # so we can see the sliced response
            content_limit=50,
        )
        # Disk: ALL 1000 lines survive
        disk = out_file.read_text(encoding="utf-8")
        assert disk.count("line\n") == 1000
        # Response: markdown sliced to 50 chars (slicing_info confirms)
        assert result["slicing_info"]["markdown"]["limit"] == 50
        assert len(result["markdown"]) <= 50

    @pytest.mark.asyncio
    async def test_persist_captures_full_content_via_fallback_with_content_limit(self, tmp_path):
        """Same regression, but through the empty-markdown fallback path."""
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "Empty",
            "markdown": "",
            "content": "",
        })
        full_md = "# Recovered\n" + ("line\n" * 500)
        web_crawling.crawl_url_with_fallback = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "Recovered",
            "markdown": full_md,
            "content": "raw",
        })
        fm = self._register(web_crawling)
        out_file = tmp_path / "fb.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
            include_content_in_response=True,
            content_limit=30,
        )
        disk = out_file.read_text(encoding="utf-8")
        assert disk.count("line\n") == 500
        assert result["slicing_info"]["markdown"]["limit"] == 30

    @pytest.mark.asyncio
    async def test_empty_output_path_is_noop(self, tmp_path):
        """Regression: output_path='' must pass through without attempting
        to persist (mirrors validate_output_path's empty-string acceptance)."""
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "T",
            "markdown": "# X",
            "content": "c",
        })
        fm = self._register(web_crawling)
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path="",
        )
        assert "persisted" not in result
        assert "markdown" in result  # no stripping happened
        # No file should have been created anywhere
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_error_not_persisted(self, tmp_path):
        """When both crawl and fallback fail, no file should be written."""
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(side_effect=RuntimeError("boom"))
        web_crawling.crawl_url_with_fallback = AsyncMock(
            side_effect=RuntimeError("also boom")
        )
        fm = self._register(web_crawling)
        out_file = tmp_path / "out.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
        )
        assert result["success"] is False
        assert not out_file.exists()


# ---------------------------------------------------------------------------
# deep_crawl_site wiring
# ---------------------------------------------------------------------------


class TestDeepCrawlSiteWiring:
    @pytest.mark.asyncio
    async def test_normal_success_persists_batch(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.deep_crawl_site = AsyncMock(return_value={
            "success": True,
            "results": [
                {"success": True, "url": "https://a.com/x", "title": "A", "markdown": "# A"},
                {"success": True, "url": "https://b.com/y", "title": "B", "markdown": "# B"},
            ],
            "summary": {"total_crawled": 2},
        })
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        out_dir = tmp_path / "deep"
        result = await fm.tools["deep_crawl_site"](
            url="https://a.com",
            output_path=str(out_dir),
        )
        assert result.get("persisted") is True
        # Two per-URL md files + index.json
        assert (out_dir / "index.json").exists()
        md_files = list(out_dir.glob("*.md"))
        assert len(md_files) == 2
        # Response copy has results[] with metadata but no markdown body
        for item in result["results"]:
            assert "markdown" not in item
            assert item["url"]

    @pytest.mark.asyncio
    async def test_relative_output_path_rejected(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.deep_crawl_site = AsyncMock(side_effect=AssertionError("should not fire"))
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        result = await fm.tools["deep_crawl_site"](
            url="https://a.com",
            output_path="not/absolute",
        )
        assert result["success"] is False
        assert result["error_code"] == "output_path_not_absolute"
        web_crawling.deep_crawl_site.assert_not_called()


# ---------------------------------------------------------------------------
# crawl_url_with_fallback wiring
# ---------------------------------------------------------------------------


class TestCrawlUrlWithFallbackWiring:
    @pytest.mark.asyncio
    async def test_persists_full_before_slicing(self, tmp_path):
        web_crawling = MagicMock()
        full_md = "# Title\n" + ("line\n" * 1000)
        web_crawling.crawl_url_with_fallback = AsyncMock(return_value={
            "success": True,
            "url": "https://example.com",
            "title": "T",
            "markdown": full_md,
            "content": "raw",
        })
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        out_file = tmp_path / "art.md"
        result = await fm.tools["crawl_url_with_fallback"](
            url="https://example.com",
            output_path=str(out_file),
            content_limit=100,  # Slicing is applied AFTER persist
        )
        # Disk holds the full untruncated markdown
        body = out_file.read_text(encoding="utf-8")
        assert body.count("line\n") == 1000
        # Response has persistence metadata
        assert result.get("persisted") is True


# ---------------------------------------------------------------------------
# batch_crawl wiring (list shape)
# ---------------------------------------------------------------------------


class TestBatchCrawlWiring:
    @pytest.mark.asyncio
    async def test_preserves_list_shape_and_adds_output_file(self, tmp_path):
        web_crawling = MagicMock()
        # utilities.batch_crawl is awaited; emulate with a plain list of dicts.
        utilities = MagicMock()
        utilities.batch_crawl = AsyncMock(return_value=[
            {"success": True, "url": "https://a.com", "title": "A", "markdown": "# A", "content": "c"},
            {"success": True, "url": "https://b.com", "title": "B", "markdown": "# B", "content": "c"},
        ])
        fm = FakeMcp()
        register_batch_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling, utilities=utilities))

        out_dir = tmp_path / "bc"
        result = await fm.tools["batch_crawl"](
            urls=["https://a.com", "https://b.com"],
            output_path=str(out_dir),
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for item in result:
            assert "output_file" in item
            assert Path(item["output_file"]).exists()
            # markdown stripped
            assert "markdown" not in item
        assert (out_dir / "index.json").exists()

    @pytest.mark.asyncio
    async def test_relative_path_rejected_returns_list(self, tmp_path):
        utilities = MagicMock()
        utilities.batch_crawl = AsyncMock(side_effect=AssertionError("should not fire"))
        fm = FakeMcp()
        register_batch_tools(fm, lambda: _modules_tuple(utilities=utilities))
        result = await fm.tools["batch_crawl"](
            urls=["https://a.com"],
            output_path="rel/dir",
        )
        # List shape preserved
        assert isinstance(result, list)
        assert result[0]["error_code"] == "output_path_not_absolute"
        utilities.batch_crawl.assert_not_called()


# ---------------------------------------------------------------------------
# multi_url_crawl wiring (list shape)
# ---------------------------------------------------------------------------


class TestMultiUrlCrawlWiring:
    @pytest.mark.asyncio
    async def test_list_shape_persist(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://a.com",
            "title": "A",
            "markdown": "# A body",
            "content": "raw",
        })
        fm = FakeMcp()
        register_batch_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        out_dir = tmp_path / "muc"
        result = await fm.tools["multi_url_crawl"](
            url_configurations={"https://a.com": {}},
            output_path=str(out_dir),
        )
        assert isinstance(result, list)
        assert "output_file" in result[0]
        assert (out_dir / "index.json").exists()


# ---------------------------------------------------------------------------
# process_file wiring
# ---------------------------------------------------------------------------


class TestProcessFileWiring:
    @pytest.mark.asyncio
    async def test_persists_converted_markdown(self, tmp_path):
        file_processing = MagicMock()
        # process_file expects a Pydantic-like object with model_dump().
        payload = {
            "success": True,
            "url": "https://example.com/doc.pdf",
            "filename": "doc.pdf",
            "file_type": "pdf",
            "content": "# Title\n\n" + "paragraph " * 500,
            "title": "Doc",
            "metadata": {"pages": 10},
        }
        fake_response = MagicMock()
        fake_response.model_dump = MagicMock(return_value=payload)
        file_processing.process_file = AsyncMock(return_value=fake_response)
        fm = FakeMcp()
        register_file_tools(fm, lambda: _modules_tuple(file_processing=file_processing))

        out_file = tmp_path / "doc.md"
        result = await fm.tools["process_file"](
            url="https://example.com/doc.pdf",
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        assert "paragraph" in out_file.read_text()
        assert "content" not in result  # stripped


# ---------------------------------------------------------------------------
# extract_youtube_transcript + comments wiring
# ---------------------------------------------------------------------------


class TestYouTubeTranscriptWiring:
    @pytest.mark.asyncio
    async def test_transcript_persist_before_token_limit(self, tmp_path):
        youtube = MagicMock()
        long_transcript = "transcript line " * 5000  # way over 25k tokens
        youtube.extract_youtube_transcript = AsyncMock(return_value={
            "success": True,
            "url": "https://youtube.com/watch?v=x",
            "video_id": "x",
            "title": "Video",
            "content": long_transcript,
        })
        fm = FakeMcp()
        register_youtube_tools(fm, lambda: _modules_tuple(youtube=youtube))

        out_file = tmp_path / "t.md"
        result = await fm.tools["extract_youtube_transcript"](
            url="https://youtube.com/watch?v=x",
            output_path=str(out_file),
        )
        # Disk has the FULL transcript (token limit would have truncated it)
        disk = out_file.read_text(encoding="utf-8")
        assert disk.count("transcript line") == 5000
        # Response is slim
        assert result.get("persisted") is True
        assert "content" not in result


class TestYouTubeCommentsWiring:
    @pytest.mark.asyncio
    async def test_persist_before_internal_truncate(self, tmp_path):
        youtube = MagicMock()
        big_comments = [{"author": f"u{i}", "text": "lorem ipsum dolor sit amet " * 20} for i in range(500)]
        youtube.extract_youtube_comments = AsyncMock(return_value={
            "success": True,
            "url": "https://youtube.com/watch?v=x",
            "video_id": "x",
            "title": "Video",
            "extracted_data": {
                "comments": big_comments,
                "comment_count": 500,
            },
        })
        fm = FakeMcp()
        register_youtube_tools(fm, lambda: _modules_tuple(youtube=youtube))

        out_file = tmp_path / "c.json"
        result = await fm.tools["extract_youtube_comments"](
            url="https://youtube.com/watch?v=x",
            output_path=str(out_file),
        )
        # Disk has ALL 500 comments (internal truncate would have halved them)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(loaded["extracted_data"]["comments"]) == 500
        # Response does not contain the comments array
        assert result.get("persisted") is True
        assert "comments" not in result["extracted_data"]
        assert result["extracted_data"]["comment_count"] == 500


class TestGetYoutubeVideoInfoWiring:
    @pytest.mark.asyncio
    async def test_persists(self, tmp_path):
        youtube = MagicMock()
        youtube.get_youtube_video_info = AsyncMock(return_value={
            "success": True,
            "url": "https://youtube.com/watch?v=x",
            "title": "V",
            "content": "long transcript " * 1000,
        })
        fm = FakeMcp()
        register_youtube_tools(fm, lambda: _modules_tuple(youtube=youtube))

        out_file = tmp_path / "info.md"
        result = await fm.tools["get_youtube_video_info"](
            video_url="https://youtube.com/watch?v=x",
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        assert "long transcript" in out_file.read_text()


class TestBatchYoutubeTranscriptsWiring:
    @pytest.mark.asyncio
    async def test_persist_via_request_dict(self, tmp_path):
        youtube = MagicMock()
        youtube.batch_extract_youtube_transcripts = AsyncMock(return_value={
            "success": True,
            "results": [
                {"success": True, "url": "https://y/a", "title": "A", "content": "A body"},
                {"success": True, "url": "https://y/b", "title": "B", "content": "B body"},
            ],
        })
        fm = FakeMcp()
        register_youtube_tools(fm, lambda: _modules_tuple(youtube=youtube))
        out_dir = tmp_path / "ytbatch"
        result = await fm.tools["batch_extract_youtube_transcripts"]({
            "urls": ["https://y/a", "https://y/b"],
            "output_path": str(out_dir),
        })
        assert result.get("persisted") is True
        assert (out_dir / "index.json").exists()
        md_files = list(out_dir.glob("*.md"))
        assert len(md_files) == 2


# ---------------------------------------------------------------------------
# extraction tools wiring — spot-check the simplest success path
# ---------------------------------------------------------------------------


class TestExtractionWiring:
    @pytest.mark.asyncio
    async def test_intelligent_extract_normal_success_persists(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.intelligent_extract = AsyncMock(return_value={
            "success": True,
            "url": "https://x.com",
            "extracted_data": {"k": "v"},
            "content": "raw " * 2000,
            "markdown": "md " * 2000,
        })
        fm = FakeMcp()
        register_extraction_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        out_file = tmp_path / "ie.json"
        result = await fm.tools["intelligent_extract"](
            url="https://x.com",
            extraction_goal="k",
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        loaded = json.loads(out_file.read_text())
        assert loaded["extracted_data"] == {"k": "v"}
        assert "content" not in result
        assert "markdown" not in result

    @pytest.mark.asyncio
    async def test_extract_entities_persists(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.extract_entities = AsyncMock(return_value={
            "success": True,
            "url": "https://x.com",
            "entities": {"emails": ["a@b.com"]},
            "entity_types": ["emails"],
            "total_found": 1,
        })
        fm = FakeMcp()
        register_extraction_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        out_file = tmp_path / "ent.json"
        result = await fm.tools["extract_entities"](
            url="https://x.com",
            entity_types=["emails"],
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        loaded = json.loads(out_file.read_text())
        assert loaded["entities"] == {"emails": ["a@b.com"]}

    @pytest.mark.asyncio
    async def test_extract_structured_data_css_path_persists(self, tmp_path):
        """CSS-selector success path (line 444 in extraction_tools.py)."""
        web_crawling = MagicMock()
        # inner crawl returns HTML so BeautifulSoup can parse it
        web_crawling.crawl_url = AsyncMock(return_value={
            "success": True,
            "url": "https://x.com",
            "content": "<html><body><h1>Title</h1><span class='price'>$10</span></body></html>",
            "markdown": "# Title\n\n$10",
        })
        fm = FakeMcp()
        register_extraction_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))

        out_file = tmp_path / "sd.json"
        result = await fm.tools["extract_structured_data"](
            url="https://x.com",
            extraction_type="css",
            css_selectors={"name": "h1", "price": ".price"},
            output_path=str(out_file),
        )
        assert result.get("persisted") is True
        loaded = json.loads(out_file.read_text())
        assert loaded["extracted_data"]["name"] == "Title"
        assert loaded["extracted_data"]["price"] == "$10"


# ---------------------------------------------------------------------------
# search tools wiring
# ---------------------------------------------------------------------------


class TestSearchGoogleWiring:
    @pytest.mark.asyncio
    async def test_persists_full_results(self, tmp_path):
        search = MagicMock()
        big_results = [{"title": f"t{i}", "url": f"https://e.com/{i}", "snippet": "s"} for i in range(50)]
        search.search_google = AsyncMock(return_value={
            "success": True,
            "query": "python",
            "results": big_results,
        })
        fm = FakeMcp()
        register_search_tools(fm, lambda: _modules_tuple(search=search))

        out_file = tmp_path / "s.json"
        result = await fm.tools["search_google"]({
            "query": "python",
            "output_path": str(out_file),
        })
        # Disk has 50 results
        loaded = json.loads(out_file.read_text())
        assert len(loaded["results"]) == 50
        # Response has persistence metadata, results stripped
        assert result.get("persisted") is True
        assert "results" not in result


class TestBatchSearchGoogleWiring:
    @pytest.mark.asyncio
    async def test_persists(self, tmp_path):
        search = MagicMock()
        search.batch_search_google = AsyncMock(return_value={
            "success": True,
            "results": {"q1": [{"title": "t"}], "q2": [{"title": "u"}]},
        })
        fm = FakeMcp()
        register_search_tools(fm, lambda: _modules_tuple(search=search))
        out_file = tmp_path / "bs.json"
        result = await fm.tools["batch_search_google"]({
            "queries": ["q1", "q2"],
            "output_path": str(out_file),
        })
        assert result.get("persisted") is True
        assert out_file.exists()


class TestSearchAndCrawlWiring:
    @pytest.mark.asyncio
    async def test_persists_before_truncate(self, tmp_path):
        """search_and_crawl should persist full page bodies before the
        per-page max_content_per_page truncation runs."""
        search = MagicMock()
        big_body = "body content " * 1000  # Much larger than default max_content_per_page
        search.search_and_crawl = AsyncMock(return_value={
            "success": True,
            "query": "q",
            "search_results": [{"url": "https://a.com", "title": "A"}],
            "crawled_pages": [
                {
                    "success": True,
                    "url": "https://a.com",
                    "title": "A",
                    "content": big_body,
                    "markdown": big_body,
                }
            ],
        })
        fm = FakeMcp()
        register_search_tools(fm, lambda: _modules_tuple(search=search))
        out_dir = tmp_path / "sc"
        result = await fm.tools["search_and_crawl"]({
            "search_query": "q",
            "output_path": str(out_dir),
            "max_content_per_page": 200,  # Would truncate in the response
        })
        assert result.get("persisted") is True
        # Disk holds the FULL body, not truncated
        md_files = list(out_dir.glob("*.md"))
        assert len(md_files) == 1
        disk = md_files[0].read_text()
        # "body content " appears 1000 times in disk payload
        assert disk.count("body content") == 1000
