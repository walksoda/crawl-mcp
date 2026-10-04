"""Fallback stages must not return HTTP error pages as content."""

import pytest

from crawl4ai_mcp.core import crawler_fallback
from crawl4ai_mcp.core import crawler_core
from crawl4ai_mcp.infra.content_processors import http_error_status, looks_like_error_page
from crawl4ai_mcp.models import CrawlRequest, CrawlResponse


ORIGINAL_URL = "http://127.0.0.1:8000/"
AMP_URL = "http://127.0.0.1:8000/amp/"

NOT_FOUND_MD = (
    "# Error response\nError code: 404\nMessage: File not found.\n"
    "Error code explanation: 404 - Nothing matches the given URI.\n"
)
ARTICLE_MD = "# Calibrating robot arms\n\n" + (
    "Each joint is measured against a reference fixture and corrected in software. " * 10
)


def _install(monkeypatch, amp_response, browser_response=None):
    """Browser stages return browser_response (default: failure), AMP returns amp_response."""
    calls = []

    async def fake_crawl_url(url, **kwargs):
        calls.append(url)
        if url == AMP_URL:
            return amp_response
        if browser_response is not None:
            return browser_response
        return CrawlResponse(success=False, url=url, error="Near-empty content (82 bytes)")

    async def static_fail(url, headers=None, timeout=30):
        return False, "", "Content too short"

    async def no_rss(url):
        return False, None, []

    async def no_sleep(_):
        return None

    monkeypatch.setattr(crawler_fallback, "crawl_url", fake_crawl_url)
    monkeypatch.setattr(crawler_fallback, "_static_fetch_content", static_fail)
    monkeypatch.setattr(crawler_fallback, "_try_fetch_rss_feed", no_rss)
    monkeypatch.setattr(crawler_fallback.asyncio, "sleep", no_sleep)
    return calls


async def _run():
    return await crawler_fallback.crawl_url_with_fallback(
        url=ORIGINAL_URL, use_strategy_cache=False, save_strategy=False)


@pytest.mark.asyncio
async def test_amp_404_is_not_returned_as_content(monkeypatch):
    calls = _install(monkeypatch, CrawlResponse(
        success=True, url=AMP_URL, title="Error response", markdown=NOT_FOUND_MD, status_code=404))

    result = await _run()

    assert AMP_URL in calls
    assert result.success is False
    assert "File not found" not in (result.markdown or "")
    assert result.extracted_data.get("fallback_strategy_used") != "amp_page"


@pytest.mark.asyncio
async def test_amp_error_page_without_status_is_rejected(monkeypatch):
    # Cached results may lack the status code; the page text still gives it away.
    _install(monkeypatch, CrawlResponse(
        success=True, url=AMP_URL, title="Error response", markdown=NOT_FOUND_MD))

    result = await _run()

    assert result.success is False


@pytest.mark.asyncio
async def test_real_amp_page_is_still_used(monkeypatch):
    _install(monkeypatch, CrawlResponse(
        success=True, url=AMP_URL, title="Robot arms", markdown=ARTICLE_MD, status_code=200))

    result = await _run()

    assert result.success is True
    assert result.extracted_data["fallback_strategy_used"] == "amp_page"
    assert result.extracted_data["amp_url_used"] == AMP_URL


@pytest.mark.asyncio
async def test_browser_stage_http_error_is_skipped(monkeypatch):
    # A 404/500 page with plenty of text must not satisfy stages 2-5.
    _install(
        monkeypatch,
        amp_response=CrawlResponse(success=False, url=AMP_URL, error="unreachable"),
        browser_response=CrawlResponse(
            success=True, url=ORIGINAL_URL, markdown=ARTICLE_MD, status_code=500),
    )

    result = await _run()

    assert result.success is False
    assert "HTTP 500" in result.error


def test_status_code_is_kept_but_not_serialized():
    response = CrawlResponse(success=True, url=ORIGINAL_URL, status_code=404)
    assert http_error_status(response) == 404
    assert "status_code" not in response.model_dump()
    assert http_error_status(CrawlResponse(success=True, url=ORIGINAL_URL, status_code=200)) is None
    assert http_error_status(CrawlResponse(success=True, url=ORIGINAL_URL)) is None


@pytest.mark.asyncio
async def test_single_page_result_carries_status_code():
    class FakeResult:
        cleaned_html = "<p>Not here</p>"
        markdown = "Not here"
        metadata = {"title": "404 Not Found"}
        media = {}
        screenshot = None
        status_code = 404

    response = await crawler_core._handle_single_page_result(
        FakeResult(), CrawlRequest(url=ORIGINAL_URL))
    assert response.status_code == 404


@pytest.mark.parametrize("title,markdown,expected", [
    ("404 Not Found", "The requested URL was not found on this server.", True),
    ("Error response", NOT_FOUND_MD, True),
    ("Oops", "Page not found. Try the search box.", True),
    ("Robot arms", ARTICLE_MD, False),
    # Long articles about HTTP errors are not error pages
    ("Fixing 404 Not Found errors", "How to fix 404 not found errors. " * 50, False),
])
def test_looks_like_error_page(title, markdown, expected):
    assert looks_like_error_page(
        CrawlResponse(success=True, url=AMP_URL, title=title, markdown=markdown)) is expected
