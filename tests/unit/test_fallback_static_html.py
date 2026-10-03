"""Tests for the static-HTML last-resort stage of crawl_url_with_fallback."""

import pytest

from crawl4ai_mcp.core import crawler_fallback
from crawl4ai_mcp.models import CrawlResponse


ARTICLE_HTML = """
<html><head><title>Notes</title><meta name="robots" content="index">
<script src="https://cdnjs.cloudflare.com/lib.js"></script></head>
<body><noscript>JS improves this page</noscript><article>
<h1>Calibrating robot arms</h1>
""" + "<p>Each joint is measured against a reference fixture and corrected in software.</p>\n" * 20 + """
</article></body></html>
"""

CHALLENGE_HTML = """
<html><head><title>Just a moment...</title></head>
<body><noscript>Enable JavaScript and cookies to continue</noscript></body></html>
""" + "<!-- padding -->" * 20


@pytest.fixture
def browser_always_fails(monkeypatch):
    async def fake_crawl_url(url, **kwargs):
        return CrawlResponse(success=False, url=url, error="Blocked by anti-bot protection: DataDome captcha")

    async def no_rss(url):
        return False, None, []

    async def no_sleep(_):
        return None

    monkeypatch.setattr(crawler_fallback, "crawl_url", fake_crawl_url)
    monkeypatch.setattr(crawler_fallback, "_try_fetch_rss_feed", no_rss)
    monkeypatch.setattr(crawler_fallback, "_build_amp_url", lambda url: None)
    monkeypatch.setattr(crawler_fallback.asyncio, "sleep", no_sleep)


def _static_fetch_returning(html):
    async def fake_static_fetch(url, headers=None, timeout=30):
        return True, html, ""
    return fake_static_fetch


@pytest.mark.asyncio
async def test_static_html_used_when_browser_stages_fail(monkeypatch, browser_always_fails):
    monkeypatch.setattr(crawler_fallback, "_static_fetch_content", _static_fetch_returning(ARTICLE_HTML))

    result = await crawler_fallback.crawl_url_with_fallback(
        url="https://example.com/post", use_strategy_cache=False, save_strategy=False)

    assert result.success is True
    assert "Calibrating robot arms" in result.markdown
    assert result.extracted_data["fallback_strategy_used"] == "static_html_markdown"
    assert result.extracted_data["javascript_rendered"] is False


@pytest.mark.asyncio
async def test_static_challenge_page_is_not_returned(monkeypatch, browser_always_fails):
    monkeypatch.setattr(crawler_fallback, "_static_fetch_content", _static_fetch_returning(CHALLENGE_HTML))

    result = await crawler_fallback.crawl_url_with_fallback(
        url="https://example.com/post", use_strategy_cache=False, save_strategy=False)

    assert result.success is False
    assert "static_html_markdown" in result.extracted_data["fallback_strategies_attempted"]
