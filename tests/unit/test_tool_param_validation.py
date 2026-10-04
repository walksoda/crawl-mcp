"""Parameter validation for search genres, extract_entities types and
deep_crawl_site limits. No network access: core modules are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawl4ai_mcp.constants import SEARCH_GENRES
from crawl4ai_mcp.core import extraction_entity
from crawl4ai_mcp.core.extraction_entity import (
    ENTITY_PATTERNS,
    normalize_entity_types,
)
from crawl4ai_mcp.processors.web_search import WebSearchProcessor
from crawl4ai_mcp.server_tools.crawl_tools import register_crawl_tools
from crawl4ai_mcp.server_tools.search_tools import register_search_tools
from crawl4ai_mcp.validators import validate_search_genre


class FakeMcp:
    """Captures decorated tool functions so they can be called directly."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _modules(web_crawling=None, search=None):
    return (web_crawling or MagicMock(), search or MagicMock(),
            MagicMock(), MagicMock(), MagicMock())


EXPECTED_GENRES = {"pdf", "documents", "presentations", "spreadsheets", "japanese", "english"}


# ---------------------------------------------------------------------------
# Search genres
# ---------------------------------------------------------------------------


class TestSearchGenreSource:
    def test_genre_list(self):
        assert set(SEARCH_GENRES) == EXPECTED_GENRES

    def test_every_genre_changes_the_query(self):
        proc = WebSearchProcessor()
        for name, spec in SEARCH_GENRES.items():
            assert proc._enhance_query_with_genre("q", name) == f"q ({spec['operator']})"
        assert proc._enhance_query_with_genre("q", "PDF") == "q (filetype:pdf)"
        assert proc._enhance_query_with_genre("q", None) == "q"


class TestValidateSearchGenre:
    @pytest.mark.parametrize("genre", [None, "", "  ", "pdf", "English", " japanese "])
    def test_valid(self, genre):
        assert validate_search_genre(genre) is None

    @pytest.mark.parametrize("genre", ["academic", "news", "technical", 3, ["pdf"]])
    def test_invalid(self, genre):
        err = validate_search_genre(genre)
        assert err["success"] is False
        assert err["error_code"] == "invalid_search_genre"
        assert set(err["valid_genres"]) == EXPECTED_GENRES
        for name in EXPECTED_GENRES:
            assert name in err["error"]


class TestSearchToolsRejectUnknownGenre:
    def _tools(self, search):
        fm = FakeMcp()
        register_search_tools(fm, lambda: _modules(search=search))
        return fm.tools

    def _search_mock(self):
        search = MagicMock()
        boom = AsyncMock(side_effect=AssertionError("must not search"))
        search.search_web = boom
        search.batch_search_web = boom
        search.search_and_crawl = boom
        return search

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool,request_dict", [
        ("search_google", {"query": "x", "search_genre": "academic"}),
        ("batch_search_google", {"queries": ["x"], "search_genre": "news"}),
        ("search_and_crawl", {"search_query": "x", "search_genre": "technical"}),
    ])
    async def test_unknown_genre_errors_before_search(self, tool, request_dict):
        tools = self._tools(self._search_mock())
        result = await tools[tool](request=request_dict)
        assert result["success"] is False
        assert result["error_code"] == "invalid_search_genre"
        assert set(result["valid_genres"]) == EXPECTED_GENRES

    @pytest.mark.asyncio
    async def test_valid_genre_passes_through(self):
        search = MagicMock()
        search.search_web = AsyncMock(return_value={"success": True, "results": []})
        tools = self._tools(search)
        result = await tools["search_google"](request={"query": "x", "search_genre": "pdf"})
        assert result["success"] is True
        search.search_web.assert_awaited_once()

    def test_descriptions_list_only_real_genres(self):
        from crawl4ai_mcp.server_tools import search_tools
        for name in EXPECTED_GENRES:
            assert name in search_tools._GENRE_DESC
        for stale in ("academic", "news", "technical", "commercial", "social"):
            assert stale not in search_tools._GENRE_DESC


class TestGetSearchGenres:
    @pytest.mark.asyncio
    async def test_returns_only_effective_genres(self):
        from crawl4ai_mcp.core.search import get_search_genres
        result = await get_search_genres()
        assert result["success"] is True
        assert result["total_genres"] == len(EXPECTED_GENRES)
        assert set(result["genres"]) == EXPECTED_GENRES
        assert all(result["genres"].values())
        assert set(result["query_operators"]) == EXPECTED_GENRES
        assert "categories" not in result


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestEntityTypeNormalization:
    def test_singular_aliases(self):
        assert normalize_entity_types(
            ["email", "phone", "url", "date", "ip", "price", "credit_card", "coordinate"]
        ) == list(ENTITY_PATTERNS)

    def test_plural_case_and_duplicates(self):
        assert normalize_entity_types(["Emails", "email", " EMAIL "]) == ["emails"]

    def test_custom_pattern_keys_kept_as_given(self):
        assert normalize_entity_types(["email", "ISBN"], {"email": r"x", "ISBN": r"y"}) == ["email", "ISBN"]

    def test_unknown_types_pass_through(self):
        assert normalize_entity_types(["person", "foo"]) == ["people", "foo"]

    @pytest.mark.asyncio
    async def test_wrapper_normalizes_before_extraction(self):
        inner = AsyncMock(return_value={"success": True})
        with patch.object(extraction_entity, "_internal_extract_entities", inner):
            await extraction_entity.extract_entities("https://x.com", ["email", "phones"])
        assert inner.await_args.kwargs["entity_types"] == ["emails", "phones"]

    @pytest.mark.asyncio
    async def test_unsupported_types_reported(self):
        crawl = AsyncMock(return_value=SimpleNamespace(
            success=True, content="mail a@b.com now", markdown="", error=None,
        ))
        with patch("crawl4ai_mcp.core.crawler_core._internal_crawl_url", crawl):
            result = await extraction_entity.extract_entities(
                "https://x.com", ["email", "people"],
            )
        assert result["success"] is True
        assert result["entities"] == {"emails": ["a@b.com"]}
        assert result["unsupported_entity_types"] == ["people"]
        assert any("people" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_no_warning_when_all_supported(self):
        crawl = AsyncMock(return_value=SimpleNamespace(
            success=True, content="nothing here", markdown="", error=None,
        ))
        with patch("crawl4ai_mcp.core.crawler_core._internal_crawl_url", crawl):
            result = await extraction_entity.extract_entities("https://x.com", ["email"])
        assert "unsupported_entity_types" not in result
        assert "warnings" not in result


# ---------------------------------------------------------------------------
# deep_crawl_site limits
# ---------------------------------------------------------------------------


class TestDeepCrawlLimits:
    def _tool(self, web_crawling):
        fm = FakeMcp()
        register_crawl_tools(fm, lambda: _modules(web_crawling=web_crawling))
        return fm.tools["deep_crawl_site"]

    def _no_fetch(self):
        wc = MagicMock()
        wc.deep_crawl_site = AsyncMock(side_effect=AssertionError("must not fetch"))
        wc.crawl_url_with_fallback = AsyncMock(side_effect=AssertionError("must not fetch"))
        return wc

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs,code", [
        ({"max_depth": 0}, "invalid_depth"),
        ({"max_depth": 3}, "depth_too_large"),
        ({"max_pages": 0}, "invalid_max_pages"),
        ({"max_pages": 11}, "max_pages_too_large"),
    ])
    async def test_out_of_range_errors_before_fetch(self, kwargs, code):
        tool = self._tool(self._no_fetch())
        result = await tool(url="https://example.com", **kwargs)
        assert result["success"] is False
        assert result["error_code"] == code

    @pytest.mark.asyncio
    async def test_limits_checked_before_output_path(self):
        tool = self._tool(self._no_fetch())
        result = await tool(url="https://example.com", max_depth=5, output_path="relative/dir")
        assert result["error_code"] == "depth_too_large"

    @pytest.mark.asyncio
    async def test_boundaries_accepted(self):
        wc = MagicMock()
        wc.deep_crawl_site = AsyncMock(return_value={"success": True, "url": "https://example.com"})
        tool = self._tool(wc)
        for depth, pages in ((1, 1), (2, 10)):
            result = await tool(url="https://example.com", max_depth=depth, max_pages=pages)
            assert result["success"] is True
        assert wc.deep_crawl_site.await_count == 2
