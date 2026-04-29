"""Tool-level tests for screenshot persistence wiring.

Verifies that handle_screenshot_persistence is correctly integrated
into crawl_url and crawl_url_with_fallback return paths.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai_mcp.server_tools.crawl_tools import register_crawl_tools


_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode("ascii")


class FakeMcp:
    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        if args and callable(args[0]):
            fn = args[0]
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _modules_tuple(web_crawling=None):
    return (
        web_crawling or MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )


def _crawl_result_with_screenshot(**extra):
    base = {
        "success": True,
        "url": "https://example.com",
        "title": "Example",
        "markdown": "# Example",
        "content": "body",
        "screenshot": _PNG_B64,
    }
    base.update(extra)
    return base


class TestCrawlUrlScreenshot:
    def _register(self, web_crawling):
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))
        return fm

    @pytest.mark.asyncio
    async def test_screenshot_persisted_with_output_path(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(
            return_value=_crawl_result_with_screenshot()
        )
        fm = self._register(web_crawling)

        out_file = tmp_path / "page.md"
        result = await fm.tools["crawl_url"](
            url="https://example.com",
            output_path=str(out_file),
            take_screenshot=True,
        )

        assert "screenshot" not in result
        assert "screenshot_path" in result
        ss_path = Path(result["screenshot_path"])
        assert ss_path.exists()
        assert ss_path.name == "page_screenshot.png"

    @pytest.mark.asyncio
    async def test_screenshot_dropped_without_output_path(self):
        web_crawling = MagicMock()
        web_crawling.crawl_url = AsyncMock(
            return_value=_crawl_result_with_screenshot()
        )
        fm = self._register(web_crawling)

        result = await fm.tools["crawl_url"](
            url="https://example.com",
            take_screenshot=True,
        )

        assert "screenshot" not in result
        assert "screenshot_path" not in result
        warnings = result.get("warnings", [])
        assert any("output_path" in w for w in warnings)


class TestCrawlUrlWithFallbackScreenshot:
    def _register(self, web_crawling):
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules_tuple(web_crawling=web_crawling))
        return fm

    @pytest.mark.asyncio
    async def test_screenshot_persisted_with_output_path(self, tmp_path):
        web_crawling = MagicMock()
        web_crawling.crawl_url_with_fallback = AsyncMock(
            return_value=_crawl_result_with_screenshot()
        )
        fm = self._register(web_crawling)

        out_file = tmp_path / "page.md"
        result = await fm.tools["crawl_url_with_fallback"](
            url="https://example.com",
            output_path=str(out_file),
            take_screenshot=True,
        )

        assert "screenshot" not in result
        assert "screenshot_path" in result
        ss_path = Path(result["screenshot_path"])
        assert ss_path.exists()

    @pytest.mark.asyncio
    async def test_screenshot_preserved_without_output_path(self):
        """Without output_path, crawl_url_with_fallback should NOT drop screenshots."""
        web_crawling = MagicMock()
        web_crawling.crawl_url_with_fallback = AsyncMock(
            return_value=_crawl_result_with_screenshot()
        )
        fm = self._register(web_crawling)

        result = await fm.tools["crawl_url_with_fallback"](
            url="https://example.com",
            take_screenshot=True,
        )

        assert result.get("screenshot") == _PNG_B64
        assert "warnings" not in result or not any(
            "output_path" in w for w in result.get("warnings", [])
        )
