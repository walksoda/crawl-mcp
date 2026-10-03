"""Tests for file download headers and text charset handling in FileProcessor."""

import pytest

from crawl4ai_mcp.processors import file_processor as fp_module
from crawl4ai_mcp.processors.file_processor import FileProcessor


# Markdown whose first non-ASCII byte appears after the first 4 KiB. MarkItDown
# guesses the charset from that window, so it picked "ascii" and failed to decode.
_SECTION = (
    "## Section {i}\n"
    "This section describes how the gateway validates requests before they reach a backend service. "
    "It covers authentication, rate limiting, and how errors are reported back to the caller.\n\n"
    "+ Requests are signed by the client\n+ The gateway verifies the signature\n+ Failures return a structured error\n\n"
)
LATE_UTF8_MARKDOWN = (
    "## Overview\n" + "".join(_SECTION.format(i=i) for i in range(30)) + "Final note — 日本語\n"
).encode("utf-8")


class TestTextCharset:
    @pytest.mark.asyncio
    async def test_late_non_ascii_markdown_converts(self):
        result = await FileProcessor().process_file_from_data(LATE_UTF8_MARKDOWN, "doc.md")
        assert result["success"] is True, result.get("error")
        assert "日本語" in result["content"]

    @pytest.mark.asyncio
    async def test_non_utf8_text_still_converts(self):
        data = ("café " * 10).encode("latin-1")
        result = await FileProcessor().process_file_from_data(data, "note.txt")
        assert result["success"] is True, result.get("error")


class _FakeResponse:
    headers = {"content-type": "text/plain; charset=utf-8"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        yield b"hello"


class TestDownloadHeaders:
    @pytest.mark.asyncio
    async def test_download_sends_browser_user_agent(self, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

        monkeypatch.setattr(fp_module.requests, "get", fake_get)
        content, content_type = await FileProcessor().download_file("https://example.com/a.txt")

        assert content == b"hello"
        assert content_type == "text/plain"
        assert "python-requests" not in captured["headers"]["User-Agent"]
        assert captured["headers"]["User-Agent"].startswith("Mozilla/5.0")


def _blank_pdf() -> bytes:
    """Build a one-page PDF with no text (like a scanned document without OCR)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


class TestPdfWithoutText:
    @pytest.mark.asyncio
    async def test_pdf_without_text_layer_reports_clear_error(self):
        result = await FileProcessor().process_file_from_data(_blank_pdf(), "scan.pdf")
        assert result["success"] is False
        assert "no extractable text" in result["error"]
