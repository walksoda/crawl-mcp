"""Tests for routing remote URLs between file processing and the browser."""

import pytest

from crawl4ai_mcp.core import crawler_fallback, crawler_io
from crawl4ai_mcp.models import CrawlRequest
from crawl4ai_mcp.processors.file_processor import FileProcessor
from crawl4ai_mcp.server_helpers import _should_trigger_fallback


@pytest.fixture(scope="module")
def processor():
    return FileProcessor()


class TestIsDocumentUrl:
    @pytest.mark.parametrize("url", [
        "https://example.com/report.pdf",
        "https://example.com/files/Deck.PPTX",
        "https://raw.githubusercontent.com/org/repo/main/docs/Intent%20Trust%20Domain.md",
        "https://arxiv.org/pdf/2401.12345",
    ])
    def test_documents(self, processor, url):
        assert processor.is_document_url(url) is True

    @pytest.mark.parametrize("url", [
        "https://www.example.com/news/2026/june/launch.html",
        "http://classic.austlii.edu.au/au/legis/cth/consol_act/s478.1.htm",
        "https://example.com/blog/post",
        "https://example.com/page.php?format=html",
        "https://arxiv.org/abs/2401.12345",
    ])
    def test_web_pages(self, processor, url):
        assert processor.is_document_url(url) is False

    def test_html_still_supported_for_process_file(self, processor):
        # process_file keeps accepting .html; only crawl routing changes
        assert processor.is_supported_file("https://example.com/page.html") is True


class TestHandleFileUrl:
    @pytest.mark.asyncio
    async def test_remote_html_goes_to_browser(self):
        result = await crawler_io._handle_file_url(CrawlRequest(url="https://example.com/a/page.html"))
        assert result is None

    @pytest.mark.asyncio
    async def test_failure_is_tagged_as_file_processing(self, monkeypatch):
        async def failing(url, max_size_mb=100):
            return {"success": False, "error": "Failed to download file: 403"}

        monkeypatch.setattr(crawler_io.file_processor, "process_file_from_url", failing)
        result = await crawler_io._handle_file_url(CrawlRequest(url="https://example.com/a.pdf"))

        assert result.success is False
        assert result.extracted_data == {"processing_method": "markitdown"}


class TestNoBrowserFallbackForDocuments:
    def test_trigger_skipped_for_file_processing_failure(self):
        result = {"success": False, "error": "File processing failed: 403",
                  "extracted_data": {"processing_method": "markitdown"}}
        assert _should_trigger_fallback(result, generate_markdown=True) == (False, "")

    def test_trigger_still_fires_for_browser_failure(self):
        should, _ = _should_trigger_fallback({"success": False, "error": "timeout"}, generate_markdown=True)
        assert should is True

    @pytest.mark.asyncio
    async def test_fallback_processes_document_once(self, monkeypatch):
        calls = {"file": 0}

        async def fake_process(url, max_size_mb=100):
            calls["file"] += 1
            return {"success": False, "error": "PDF has no extractable text layer"}

        async def browser_must_not_run(*args, **kwargs):
            raise AssertionError("browser stages must not run for documents")

        monkeypatch.setattr(crawler_io.file_processor, "process_file_from_url", fake_process)
        monkeypatch.setattr(crawler_fallback, "crawl_url", browser_must_not_run)
        monkeypatch.setattr(crawler_fallback, "_static_fetch_content", browser_must_not_run)

        result = await crawler_fallback.crawl_url_with_fallback(
            url="https://example.com/scan.pdf", use_strategy_cache=False, save_strategy=False)

        assert calls["file"] == 1
        assert result.success is False
        assert "no extractable text" in result.error
