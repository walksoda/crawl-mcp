"""Unit tests for batch_crawl core logic."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from crawl4ai_mcp.core.batch import batch_crawl
from crawl4ai_mcp.models import CrawlResponse


def _make_response(url: str, success: bool = True) -> CrawlResponse:
    """Create a CrawlResponse for testing."""
    return CrawlResponse(
        success=success,
        url=url,
        markdown=f"content from {url}" if success else "",
        error="" if success else "failed",
    )


class TestBatchCrawlSequential:
    """Tests for sequential execution behavior."""

    @pytest.mark.asyncio
    async def test_empty_urls_returns_empty(self):
        result = await batch_crawl([], config=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_sequential_order_preserved(self):
        """URLs are processed sequentially and results preserve input order."""
        call_order = []

        async def mock_crawl(request):
            call_order.append(request.url)
            return _make_response(request.url)

        urls = ["https://a.com", "https://b.com", "https://c.com"]
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            results = await batch_crawl(urls)

        assert call_order == urls
        assert [r.url for r in results] == urls
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_sequential_not_concurrent(self):
        """Verify crawls do not overlap (sequential, not concurrent)."""
        active = {"count": 0, "max": 0}

        async def mock_crawl(request):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return _make_response(request.url)

        urls = ["https://a.com", "https://b.com", "https://c.com"]
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            await batch_crawl(urls)

        assert active["max"] == 1, f"Max concurrent was {active['max']}, expected 1"


class TestBatchCrawlTimeout:
    """Tests for per-URL timeout behavior."""

    @pytest.mark.asyncio
    async def test_single_url_timeout_does_not_block_others(self):
        """One URL timing out should not prevent remaining URLs from being crawled."""
        async def mock_crawl(request):
            if "slow" in request.url:
                await asyncio.sleep(100)  # Will be cancelled by timeout
            return _make_response(request.url)

        urls = ["https://slow.com", "https://fast.com"]
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            results = await batch_crawl(urls, base_timeout=1)

        assert len(results) == 2
        assert results[0].success is False
        assert "timeout" in results[0].error.lower()
        assert results[1].success is True

    @pytest.mark.asyncio
    async def test_per_url_timeout_uses_base_timeout(self):
        """Per-URL timeout should equal base_timeout, not scale with URL count."""
        timeout_hit_at = {}

        async def mock_crawl(request):
            start = time.monotonic()
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                timeout_hit_at[request.url] = time.monotonic() - start
                raise

        urls = ["https://a.com", "https://b.com"]
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            results = await batch_crawl(urls, base_timeout=1)

        # Both should timeout at ~1s, not 1s + N*10s
        for url, elapsed in timeout_hit_at.items():
            assert elapsed < 2.0, f"{url} timed out after {elapsed:.1f}s, expected ~1s"


class TestBatchCrawlErrorHandling:
    """Tests for error handling in batch_crawl."""

    @pytest.mark.asyncio
    async def test_exception_captured_per_url(self):
        """An exception in one URL should not prevent others from being crawled."""
        async def mock_crawl(request):
            if "bad" in request.url:
                raise RuntimeError("connection refused")
            return _make_response(request.url)

        urls = ["https://bad.com", "https://good.com"]
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            results = await batch_crawl(urls)

        assert len(results) == 2
        assert results[0].success is False
        assert "connection refused" in results[0].error
        assert results[1].success is True


class TestBatchCrawlConfig:
    """Tests for configuration handling."""

    @pytest.mark.asyncio
    async def test_max_concurrent_stripped_from_config(self):
        """max_concurrent should be removed from config passed to CrawlRequest."""
        captured_config = {}

        async def mock_crawl(request):
            captured_config["timeout"] = request.timeout
            return _make_response(request.url)

        config = {"max_concurrent": 5, "generate_markdown": True}
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            results = await batch_crawl(["https://example.com"], config=config)

        assert results[0].success is True
        # max_concurrent should not cause CrawlRequest validation error

    @pytest.mark.asyncio
    async def test_timeout_set_to_base_timeout(self):
        """Config timeout should be set to base_timeout, not scaled by URL count."""
        captured_timeouts = []

        async def mock_crawl(request):
            captured_timeouts.append(request.timeout)
            return _make_response(request.url)

        urls = ["https://a.com", "https://b.com", "https://c.com"]
        with patch("crawl4ai_mcp.core.batch._internal_crawl_url", side_effect=mock_crawl):
            await batch_crawl(urls, base_timeout=45)

        assert all(t == 45 for t in captured_timeouts), \
            f"Expected all timeouts to be 45, got {captured_timeouts}"
