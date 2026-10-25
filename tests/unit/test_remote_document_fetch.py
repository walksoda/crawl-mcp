"""Remote document fetching in FileProcessor.

The type comes from the downloaded bytes, a refused download is retried in a
browser, a web page is reported as such, and slide deck pages lead to their PDF.
"""

from types import SimpleNamespace

import pytest

from crawl4ai_mcp.processors import file_processor as fp_module
from crawl4ai_mcp.processors.document_download import (
    BrowserDownload,
    DownloadRefused,
    find_document_link,
    sniff_extension,
)
from crawl4ai_mcp.processors.file_processor import FileProcessor

PDF = b"%PDF-1.7\n fake pdf body"
HTML = b"<!DOCTYPE html><html><head><title>t</title></head><body>page</body></html>"


@pytest.fixture
def fp(monkeypatch):
    """FileProcessor whose network and converter are faked per test."""
    processor = FileProcessor()
    processor.responses = {}  # url -> (bytes, content_type) or exception
    processor.browser_calls = []
    processor.browser_result = BrowserDownload(status=403, title="Just a moment...")

    async def fake_download(url, max_size_mb=100):
        value = processor.responses[url]
        if isinstance(value, Exception):
            raise value
        if len(value) == 2:  # (bytes, content_type) without a redirect
            value = (*value, url)
        return value

    async def fake_browser(url, max_bytes):
        processor.browser_calls.append(url)
        return processor.browser_result

    def fake_convert(path, data):
        return SimpleNamespace(text_content=f"converted {path.rsplit('.', 1)[-1]}", title=None, metadata={})

    monkeypatch.setattr(processor, "_download_http", fake_download)
    monkeypatch.setattr(processor, "_convert", fake_convert)
    monkeypatch.setattr(fp_module, "browser_download", fake_browser)
    return processor


class TestSniffExtension:
    @pytest.mark.parametrize("data, ext", [
        (PDF, ".pdf"),
        (b"PK\x03\x04rest", ".zip"),
        (b"{\\rtf1 text", ".rtf"),
        (HTML, ".html"),
        (b"\n  <html lang='ja'>", ".html"),
        (b"<!-- comment -->\n<!DOCTYPE html><html>", ".html"),
        (b"<?xml version='1.0'?><html xmlns='http://www.w3.org/1999/xhtml'>", ".html"),
        (b"PK\x05\x06" + b"\x00" * 18, ".zip"),
        (b"plain text", None),
    ])
    def test_signatures(self, data, ext):
        assert sniff_extension(data) == ext


class TestFindDocumentLink:
    def test_speakerdeck(self):
        page = b'<a href="https://files.speakerdeck.com/presentations/abc123/deck_name.pdf">Download</a>'
        assert find_document_link("https://speakerdeck.com/u/deck", page) == \
            "https://files.speakerdeck.com/presentations/abc123/deck_name.pdf"

    def test_docswell_relative_link(self):
        page = b'<a href="/slide/5Q2JYV/Z82YV1G8/download">PDF</a>'
        assert find_document_link("https://www.docswell.com/s/user/5Q2JYV-title", page) == \
            "https://www.docswell.com/slide/5Q2JYV/Z82YV1G8/download"

    def test_arxiv_abstract_page(self):
        page = b'<a href="/pdf/1706.03762" class="abs-button download-pdf">View PDF</a>'
        assert find_document_link("https://arxiv.org/abs/1706.03762", page) == "https://arxiv.org/pdf/1706.03762"

    def test_other_sites_are_not_guessed(self):
        page = b'<a href="https://example.org/paper.pdf">pdf</a>'
        assert find_document_link("https://example.com/page", page) is None


class TestProcessFileFromUrl:
    async def test_extensionless_url_serving_pdf(self, fp):
        fp.responses["https://www.docswell.com/slide/A/B/download"] = (PDF, "application/octet-stream")
        result = await fp.process_file_from_url("https://www.docswell.com/slide/A/B/download")
        assert result["success"] is True
        assert result["content"] == "converted pdf"

    async def test_refused_download_is_retried_in_browser(self, fp):
        url = "https://www.w3.org/doc.pdf"
        fp.responses[url] = DownloadRefused(403, url)
        fp.browser_result = BrowserDownload(data=PDF)
        result = await fp.process_file_from_url(url)
        assert result["success"] is True
        assert result["metadata"]["downloaded_with"] == "browser"
        assert fp.browser_calls == [url]

    async def test_bot_check_page_reports_download_blocked(self, fp):
        url = "https://dl.example.org/doi/pdf/10.1/x"
        fp.responses[url] = (HTML, "text/html")  # 200 with a challenge page
        result = await fp.process_file_from_url(url)
        assert result["success"] is False
        assert result["error_code"] == "download_blocked"
        assert "Just a moment..." in result["error"]
        assert result["suggestion"]
        assert fp.browser_calls == [url]

    async def test_web_page_is_not_converted(self, fp):
        url = "https://example.com/article"
        fp.responses[url] = (HTML, "text/html")
        result = await fp.process_file_from_url(url)
        assert result["error_code"] == "not_a_document"
        assert "crawl_url" in result["suggestion"]
        assert fp.browser_calls == []

    async def test_html_url_is_still_converted(self, fp):
        url = "https://example.com/page.html"
        fp.responses[url] = (HTML, "text/html")
        result = await fp.process_file_from_url(url)
        assert result["success"] is True
        assert result["content"] == "converted html"

    async def test_slide_page_follows_its_pdf(self, fp):
        page_url = "https://speakerdeck.com/user/deck"
        pdf_url = "https://files.speakerdeck.com/presentations/abc/deck.pdf"
        fp.responses[page_url] = (f'<html><a href="{pdf_url}">pdf</a></html>'.encode(), "text/html")
        fp.responses[pdf_url] = (PDF, "application/pdf")
        result = await fp.process_file_from_url(page_url)
        assert result["success"] is True
        assert result["url"] == page_url
        assert result["metadata"]["document_url"] == pdf_url


class TestDirSize:
    def test_entry_renamed_mid_scan_is_skipped(self, monkeypatch, tmp_path):
        from crawl4ai_mcp.processors import document_download as dd

        class Vanished:
            def is_file(self):
                return True

            def stat(self):
                raise FileNotFoundError("renamed")

        class Kept:
            def is_file(self):
                return True

            def stat(self):
                return SimpleNamespace(st_size=5)

        monkeypatch.setattr(dd.os, "scandir", lambda path: iter([Vanished(), Kept()]))
        assert dd._dir_size(str(tmp_path)) == 5


class TestCoreProcessFile:
    async def test_error_code_reaches_the_response(self, monkeypatch):
        from crawl4ai_mcp.core import file_processing

        async def fake(url, max_size_mb=100):
            return {"success": False, "url": url, "file_type": None, "error": "blocked",
                    "error_code": "download_blocked", "suggestion": "look elsewhere"}

        monkeypatch.setattr(file_processing.file_processor, "process_file_from_url", fake)
        response = await file_processing.process_file("https://example.com/download")
        assert response.error_code == "download_blocked"
        assert response.suggestion == "look elsewhere"


class TestReviewFindings:
    def test_markdown_quoting_html_is_not_html(self):
        data = b"# Title\n\nSome text.\n\n```html\n<html><head></head></html>\n```\n"
        assert sniff_extension(data) is None

    def test_speakerdeck_link_keeps_its_query(self):
        page = b'<a href="https://files.speakerdeck.com/presentations/ab/x.pdf?token=1&amp;e=2">'
        assert find_document_link("https://speakerdeck.com/u/d", page) == \
            "https://files.speakerdeck.com/presentations/ab/x.pdf?token=1&e=2"

    def test_lookalike_domain_is_not_followed(self):
        page = b'<a href="/slide/A1/B2/download">'
        assert find_document_link("https://notdocswell.com/s/u/x", page) is None
        assert find_document_link("https://docswell.com/s/u/x", page) == "https://docswell.com/slide/A1/B2/download"

    async def test_extensionless_rtf_is_converted(self, fp):
        url = "https://example.com/download?id=1"
        fp.responses[url] = (b"{\\rtf1\\ansi hello}", "application/octet-stream")
        result = await fp.process_file_from_url(url)
        assert result["success"] is True
        assert result["content"] == "converted rtf"

    async def test_browser_zip_carries_metadata(self, fp, monkeypatch):
        url = "https://example.com/files/archive.zip"
        fp.responses[url] = DownloadRefused(403, url)
        fp.browser_result = BrowserDownload(data=b"PK\x03\x04zip")
        monkeypatch.setattr(fp, "extract_zip_contents", lambda data: {"files": []})
        result = await fp.process_file_from_url(url)
        assert result["is_archive"] is True
        assert result["metadata"]["downloaded_with"] == "browser"

    async def test_too_large_in_browser_is_its_own_error(self, fp):
        url = "https://example.com/big.pdf"
        fp.responses[url] = DownloadRefused(403, url)
        fp.browser_result = BrowserDownload(too_large=True, error="File too large: exceeds 1MB limit")
        result = await fp.process_file_from_url(url, max_size_mb=1)
        assert result["error_code"] == "file_too_large"

    async def test_refused_deck_page_followed_from_browser_html(self, fp):
        page_url = "https://www.docswell.com/s/user/A1-title"
        pdf_url = "https://www.docswell.com/slide/A1/B2/download"
        fp.responses[page_url] = DownloadRefused(403, page_url)
        fp.responses[pdf_url] = (PDF, "application/pdf")
        fp.browser_result = BrowserDownload(status=200, title="Deck", html='<a href="/slide/A1/B2/download">')
        result = await fp.process_file_from_url(page_url)
        assert result["success"] is True
        assert result["metadata"]["document_url"] == pdf_url


class TestFinalUrl:
    async def test_type_comes_from_the_redirect_target(self, fp):
        url = "https://example.com/download?id=7"
        fp.responses[url] = (b"a,b\n1,2\n", "application/octet-stream", "https://cdn.example.com/files/report.csv")
        result = await fp.process_file_from_url(url)
        assert result["success"] is True
        assert result["content"] == "converted csv"
        assert result["metadata"]["document_url"] == "https://cdn.example.com/files/report.csv"

    async def test_linked_file_keeps_its_own_final_url(self, fp):
        page_url = "https://www.docswell.com/s/user/A1-title"
        link = "https://www.docswell.com/slide/A1/B2/download"
        fp.responses[page_url] = (b'<html><a href="/slide/A1/B2/download">pdf</a></html>', "text/html")
        fp.responses[link] = (PDF, "application/pdf", "https://cdn.docswell.com/A1.pdf")
        result = await fp.process_file_from_url(page_url)
        assert result["metadata"]["document_url"] == "https://cdn.docswell.com/A1.pdf"

    async def test_document_redirected_to_html_block_page_is_refused(self, fp):
        url = "https://example.com/report.pdf"
        fp.responses[url] = (HTML, "text/html", "https://example.com/blocked.html")
        result = await fp.process_file_from_url(url)
        assert result["error_code"] == "download_blocked"
        assert fp.browser_calls == [url]

    async def test_requested_extension_is_used_when_the_target_has_none(self, fp):
        url = "https://example.com/report.csv"
        fp.responses[url] = (b"a,b\n1,2\n", "application/octet-stream", "https://cdn.example.com/obj/8f3a")
        result = await fp.process_file_from_url(url)
        assert result["content"] == "converted csv"

    async def test_browser_file_name_tells_the_type(self, fp):
        url = "https://example.com/download?id=1"
        fp.responses[url] = DownloadRefused(403, url)
        fp.browser_result = BrowserDownload(data=b"a,b\n1,2\n", filename="report.csv")
        result = await fp.process_file_from_url(url)
        assert result["content"] == "converted csv"


def _zip(*entries):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


class TestTypeDetection:
    """The content decides first; names only for formats without a signature."""

    async def test_office_file_is_told_from_its_entries(self, fp):
        url = "https://example.com/report.xlsx"
        xlsx = _zip(("[Content_Types].xml", "x"), ("xl/workbook.xml", "x"))
        fp.responses[url] = (xlsx, "application/zip", "https://cdn.example.com/obj/1")
        result = await fp.process_file_from_url(url)
        assert result["content"] == "converted xlsx"

    async def test_misnamed_zip_holding_a_docx(self, fp):
        url = "https://example.com/files/report.zip"
        docx = _zip(("[Content_Types].xml", "x"), ("word/document.xml", "x"))
        fp.responses[url] = (docx, "application/zip")
        result = await fp.process_file_from_url(url)
        assert result["content"] == "converted docx"

    async def test_browser_download_uses_its_final_url(self, fp):
        url = "https://example.com/report.md"
        fp.responses[url] = DownloadRefused(403, url)
        fp.browser_result = BrowserDownload(data=b"a,b\n1,2\n", filename="actual.csv",
                                            url="https://example.com/actual.csv")
        result = await fp.process_file_from_url(url)
        assert result["content"] == "converted csv"

    async def test_browser_saving_a_web_page_is_blocked(self, fp):
        url = "https://example.com/report.pdf"
        fp.responses[url] = DownloadRefused(403, url)
        fp.browser_result = BrowserDownload(data=HTML, filename="report.pdf",
                                            url="https://example.com/report.pdf")
        result = await fp.process_file_from_url(url)
        assert result["error_code"] == "download_blocked"
        assert "web page instead of the file" in result["error"]

    async def test_local_markdown_starting_with_html_is_converted(self, tmp_path, monkeypatch):
        doc = tmp_path / "notes.md"
        doc.write_text("<html> is quoted here\n\n# Notes\n")
        processor = FileProcessor()
        monkeypatch.setattr(processor, "_convert",
                            lambda path, data: SimpleNamespace(text_content=path.rsplit(".", 1)[-1], title=None, metadata={}))
        result = await processor.process_file_from_url(str(doc))
        assert result["content"] == "md"

    async def test_remote_markdown_starting_with_html_is_converted(self, fp):
        url = "https://example.com/notes.md"
        fp.responses[url] = (b"<html> is quoted here\n\n# Notes\n", "text/markdown")
        result = await fp.process_file_from_url(url)
        assert result["content"] == "converted md"
        assert fp.browser_calls == []

    async def test_browser_saving_a_commented_web_page_is_blocked(self, fp):
        url = "https://example.com/report.pdf"
        fp.responses[url] = DownloadRefused(403, url)
        page = b"<!-- served by cdn --><!DOCTYPE html><html><body>check</body></html>"
        fp.browser_result = BrowserDownload(data=page, filename="report.pdf", url=url)
        result = await fp.process_file_from_url(url)
        assert result["error_code"] == "download_blocked"

    async def test_empty_zip_is_recognized(self, fp):
        url = "https://example.com/download?id=9"
        fp.responses[url] = (b"PK\x05\x06" + b"\x00" * 18, "application/octet-stream")
        fp.extract_zip_contents = lambda data: {"files": []}
        result = await fp.process_file_from_url(url)
        assert result["is_archive"] is True
