"""Search engine clients (Google Custom Search, Brave, Exa), engine selection
and auto fallback. HTTP is replaced with a fake aiohttp session."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai_mcp.core.search import recent_days_to_int
from crawl4ai_mcp.middleware.cache import _get_search_cache_key
from crawl4ai_mcp.processors import web_search
from crawl4ai_mcp.processors.search_engines import brave, exa, google_custom
from crawl4ai_mcp.processors.search_engines.brave import BraveSearchClient
from crawl4ai_mcp.processors.search_engines.common import failure
from crawl4ai_mcp.processors.search_engines.exa import ExaSearchClient, split_site_operators
from crawl4ai_mcp.processors.search_engines.google_custom import CustomSearchAPIClient
from crawl4ai_mcp.processors.web_search import WebSearchProcessor
from crawl4ai_mcp.server_tools.search_tools import register_search_tools

ENGINE_ENV = (
    "SEARCH_ENGINE", "SEARCH_ENGINE_ORDER", "GOOGLE_SEARCH_MODE",
    "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID", "BRAVE_SEARCH_API_KEY", "EXA_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENGINE_ENV:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Fake aiohttp
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, data: Any, headers: Optional[Dict[str, str]] = None):
        self.status = status
        self._data = data
        self.headers = headers or {}

    async def json(self, content_type=None):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Returns queued (status, json) responses and records each request."""

    def __init__(self, responses: List[tuple]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args, **kwargs):  # used as the ClientSession factory
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _next(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        # A callable builds the response from the request (e.g. Brave count/offset).
        response = response(kwargs) if callable(response) else response
        return FakeResponse(*response)

    def get(self, url, **kwargs):
        return self._next("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)


def _install(monkeypatch, module, responses) -> FakeSession:
    session = FakeSession(responses)
    monkeypatch.setattr(module.aiohttp, "ClientSession", session)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    return session


# ---------------------------------------------------------------------------
# Brave
# ---------------------------------------------------------------------------


def _brave_server(total: int):
    """Answer like Brave: offset is a page index in units of count."""
    def respond(kwargs):
        count = kwargs["params"]["count"]
        start = kwargs["params"]["offset"] * count
        n = max(0, min(count, total - start))
        return 200, _brave_page(n, start, more=start + n < total)
    return respond


def _brave_page(n: int, start: int = 0, more: bool = True) -> Dict[str, Any]:
    return {
        "query": {"more_results_available": more},
        "web": {"results": [
            {"title": f"<strong>T{i}</strong>", "url": f"https://ex{i}.com/p",
             "description": f"a <strong>b</strong> &amp; c{i}", "age": "2 days ago",
             "page_age": "2026-10-01T00:00:00"}
            for i in range(start, start + n)
        ]},
    }


class TestBrave:
    def test_build_params(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        client = BraveSearchClient()
        params = client.build_params("q", 20, 1, "ja", "uk", True, 7)
        today = date.today()
        assert params["count"] == 20
        assert params["offset"] == 1
        assert params["search_lang"] == "ja"
        assert params["country"] == "GB"
        assert params["safesearch"] == "moderate"
        assert params["freshness"] == f"{(today - timedelta(days=7)).isoformat()}to{today.isoformat()}"
        assert params["result_filter"] == "web"

    def test_unknown_country_becomes_all(self):
        params = BraveSearchClient().build_params("q", 10, region="xx", safe_search=False)
        assert params["country"] == "ALL"
        assert params["safesearch"] == "off"
        assert "freshness" not in params

    @pytest.mark.asyncio
    async def test_parses_and_pages(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        session = _install(monkeypatch, brave, [_brave_server(100)] * 2)
        result = await BraveSearchClient().search("q", num_results=25)

        assert result["success"] is True
        assert result["total_results"] == 25
        # Every page keeps count=20, otherwise offset=1 would point at results 6-10.
        assert [c["params"]["offset"] for c in session.calls] == [0, 1]
        assert [c["params"]["count"] for c in session.calls] == [20, 20]
        assert [r["url"] for r in result["results"]] == [f"https://ex{i}.com/p" for i in range(25)]
        assert session.calls[0]["headers"]["X-Subscription-Token"] == "k"
        first = result["results"][0]
        assert first["title"] == "T0"
        assert first["snippet"] == "a b & c0"
        assert first["age"] == "2 days ago"
        assert first["domain"] == "ex0.com"
        assert result["results"][-1]["rank"] == 25

    @pytest.mark.asyncio
    async def test_short_page_with_more_results_continues(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        session = _install(monkeypatch, brave, [(200, _brave_page(12)), (200, _brave_page(20, 12, more=False))])
        result = await BraveSearchClient().search("q", num_results=30)
        assert len(session.calls) == 2
        assert result["total_results"] == 30

    @pytest.mark.asyncio
    async def test_duplicate_urls_dropped(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        _install(monkeypatch, brave, [(200, _brave_page(20)), (200, _brave_page(20, 10, more=False))])
        result = await BraveSearchClient().search("q", num_results=40)
        urls = [r["url"] for r in result["results"]]
        assert len(urls) == len(set(urls)) == 30
        assert [r["rank"] for r in result["results"]] == list(range(1, 31))

    @pytest.mark.asyncio
    async def test_stops_when_no_more_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        session = _install(monkeypatch, brave, [(200, _brave_page(20, more=False))])
        result = await BraveSearchClient().search("q", num_results=50)
        assert result["total_results"] == 20
        assert len(session.calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,body,kind", [
        (429, {"error": {"code": "RATE_LIMITED", "detail": "limit"}}, "rate_limited"),
        (422, {"error": {"code": "SUBSCRIPTION_TOKEN_INVALID", "detail": "bad key"}}, "auth"),
        (422, {"error": {"code": "VALIDATION", "detail": "bad q"}}, "bad_request"),
        (503, None, "server"),
    ])
    async def test_errors(self, monkeypatch, status, body, kind):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        _install(monkeypatch, brave, [(status, body)])
        result = await BraveSearchClient().search("q")
        assert result["success"] is False
        assert result["failure_kind"] == kind
        assert result["status_code"] == status

    @pytest.mark.asyncio
    async def test_per_second_limit_is_retried(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        limited = (429, {"error": {"code": "RATE_LIMITED"}}, {"X-RateLimit-Reset": "1, 2000000"})
        session = _install(monkeypatch, brave, [limited, (200, _brave_page(3, more=False))])
        result = await BraveSearchClient().search("q", num_results=3)
        assert result["success"] is True
        assert len(session.calls) == 2
        brave.asyncio.sleep.assert_awaited_with(1.0)

    @pytest.mark.asyncio
    async def test_monthly_quota_is_not_retried(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        limited = (429, {"error": {"code": "QUOTA_LIMITED"}}, {"X-RateLimit-Reset": "86400, 2000000"})
        session = _install(monkeypatch, brave, [limited])
        result = await BraveSearchClient().search("q")
        assert result["failure_kind"] == "rate_limited"
        assert len(session.calls) == 1

    @pytest.mark.asyncio
    async def test_retries_are_bounded(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        limited = (429, {"error": {"code": "RATE_LIMITED"}}, {"X-RateLimit-Reset": "1"})
        session = _install(monkeypatch, brave, [limited] * 3)
        result = await BraveSearchClient().search("q")
        assert result["failure_kind"] == "rate_limited"
        assert len(session.calls) == 3

    @pytest.mark.asyncio
    async def test_concurrent_searches_are_serialized(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        client = BraveSearchClient()
        active = 0
        peak = 0

        def respond(kwargs):
            return 200, _brave_page(1, more=False)

        class SlowSession(FakeSession):
            def get(self, url, **kwargs):
                outer = self

                class Ctx(FakeResponse):
                    async def __aenter__(self_inner):
                        nonlocal active, peak
                        active += 1
                        peak = max(peak, active)
                        await asyncio.sleep(0)
                        return self_inner

                    async def __aexit__(self_inner, *exc):
                        nonlocal active
                        active -= 1
                        return False
                outer.calls.append(kwargs)
                return Ctx(*respond(kwargs))

        monkeypatch.setattr(brave.aiohttp, "ClientSession", lambda *a, **k: SlowSession([]))
        results = await asyncio.gather(*(client.search(f"q{i}", num_results=1) for i in range(3)))
        assert all(r["success"] for r in results)
        assert peak == 1

    @pytest.mark.asyncio
    async def test_discussions(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        long_post = "<p>\n  word " * 400 + "</p>"
        cluster = {"type": "search", "results": [
            {"title": f"Thread {i}", "url": f"https://www.reddit.com/r/rust/comments/{i}/",
             "description": "short", "type": "discussion", "age": "December 27, 2025",
             "data": {"forum_name": "r/rust", "num_answers": 322, "score": "363",
                      "title": "Is Rust the future?", "question": long_post,
                      "top_comment": "You&#39;re on <b>r/rust</b>"}}
            for i in range(12)
        ]}
        session = _install(monkeypatch, brave, [(200, {"type": "search", "query": {"more_results_available": True},
                                                       "discussions": cluster})])
        result = await BraveSearchClient().search("q", num_results=5, result_type="discussions")

        params = session.calls[0]["params"]
        assert params["result_filter"] == "discussions"
        # One request only: count/offset do not page the discussions cluster.
        assert len(session.calls) == 1
        assert result["result_type"] == "discussions"
        assert result["total_results"] == 5
        first = result["results"][0]
        assert first["forum_name"] == "r/rust"
        assert first["num_answers"] == 322
        assert first["score"] == "363"
        assert first["top_comment"] == "You're on r/rust"
        assert first["question"].startswith("word word")
        assert first["question"].endswith("...")
        assert len(first["question"]) <= brave.FORUM_TEXT_LIMIT + 3

    @pytest.mark.asyncio
    async def test_discussions_absent_is_no_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        _install(monkeypatch, brave, [(200, {"type": "search", "query": {}})])
        result = await BraveSearchClient().search("q", result_type="discussions")
        assert result["failure_kind"] == "no_results"
        assert "only for some queries" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_web_results_have_no_forum_fields(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        _install(monkeypatch, brave, [(200, _brave_page(1, more=False))])
        result = await BraveSearchClient().search("q", num_results=1)
        assert "forum_name" not in result["results"][0]
        assert result["result_type"] == "web"

    @pytest.mark.asyncio
    async def test_zero_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
        _install(monkeypatch, brave, [(200, {"web": {"results": []}})])
        result = await BraveSearchClient().search("q")
        assert result["failure_kind"] == "no_results"


# ---------------------------------------------------------------------------
# Exa
# ---------------------------------------------------------------------------


class TestExa:
    def test_split_site_operators(self):
        assert split_site_operators("llm agents site:arxiv.org -site:example.com") == (
            "llm agents", ["arxiv.org"], ["example.com"])
        assert split_site_operators("website:foo") == ("website:foo", [], [])

    def test_build_body(self):
        body = ExaSearchClient().build_body("llm site:arxiv.org", 150, "uk", True, 30)
        assert body["query"] == "llm"
        assert body["includeDomains"] == ["arxiv.org"]
        assert body["numResults"] == 100
        assert body["userLocation"] == "GB"
        assert body["moderation"] is True
        assert body["type"] == "auto"
        assert body["contents"] == {"highlights": {"maxCharacters": 400}}
        assert body["startPublishedDate"].endswith("T00:00:00.000Z")

    def test_site_only_query_keeps_text(self):
        body = ExaSearchClient().build_body("site:arxiv.org", 10, None, False, None)
        assert body["query"] == "site:arxiv.org"
        assert "moderation" not in body
        assert "startPublishedDate" not in body

    @pytest.mark.asyncio
    async def test_parses_results(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "k")
        session = _install(monkeypatch, exa, [(200, {
            "results": [
                {"title": "A", "url": "https://arxiv.org/abs/1", "publishedDate": "2026-01-01T00:00:00Z",
                 "author": "X", "highlights": ["one\n two", "three"]},
                {"title": None, "url": "https://b.com", "publishedDate": None, "author": None},
            ],
            "costDollars": {"total": 0.007},
        })])
        result = await ExaSearchClient().search("q", num_results=2)

        assert session.calls[0]["method"] == "POST"
        assert session.calls[0]["headers"]["x-api-key"] == "k"
        assert result["success"] is True
        assert result["cost_dollars"] == 0.007
        a, b = result["results"]
        assert a["snippet"] == "one two ... three"
        assert a["published_date"] == "2026-01-01T00:00:00Z"
        assert a["author"] == "X"
        assert a["type"] == "academic"
        assert b["title"] == "No title"
        assert "author" not in b

    @pytest.mark.asyncio
    async def test_no_credits(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "k")
        _install(monkeypatch, exa, [(402, {"error": "No more credits", "tag": "NO_MORE_CREDITS"})])
        result = await ExaSearchClient().search("q")
        assert result["failure_kind"] == "quota"
        assert result["api_error_code"] == "NO_MORE_CREDITS"


# ---------------------------------------------------------------------------
# Google Custom Search
# ---------------------------------------------------------------------------


class TestGoogleCustom:
    def test_build_params(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "k")
        monkeypatch.setenv("GOOGLE_SEARCH_ENGINE_ID", "cx")
        params = CustomSearchAPIClient().build_params("q", 10, 11, "ja", "jp", False, 14)
        assert params["dateRestrict"] == "d14"
        assert params["start"] == 11
        assert params["safe"] == "off"
        assert params["hl"] == "ja" and params["gl"] == "jp"

    @pytest.mark.asyncio
    async def test_error_in_body(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "k")
        monkeypatch.setenv("GOOGLE_SEARCH_ENGINE_ID", "cx")
        _install(monkeypatch, google_custom, [(403, {"error": {"message": "Quota exceeded"}})])
        result = await CustomSearchAPIClient().search("q")
        assert result["success"] is False
        assert "Quota exceeded" in result["error"]
        assert result["failure_kind"] == "auth"


@pytest.mark.asyncio
@pytest.mark.parametrize("module,client_cls,env", [
    (brave, BraveSearchClient, {"BRAVE_SEARCH_API_KEY": "k"}),
    (exa, ExaSearchClient, {"EXA_API_KEY": "k"}),
    (google_custom, CustomSearchAPIClient, {"GOOGLE_SEARCH_API_KEY": "k", "GOOGLE_SEARCH_ENGINE_ID": "cx"}),
])
@pytest.mark.parametrize("body", [None, ["not", "an", "object"]])
async def test_malformed_200_is_an_error_not_zero_results(monkeypatch, module, client_cls, env, body):
    """HTML or broken JSON with HTTP 200 must let auto move to the next engine."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _install(monkeypatch, module, [(200, body)])
    result = await client_cls().search("q")
    assert result["success"] is False
    assert result["failure_kind"] == "error"


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------


class TestEngineConfig:
    @pytest.mark.parametrize("env,expected", [
        ({}, "auto"),
        ({"SEARCH_ENGINE": "Brave"}, "brave"),
        ({"SEARCH_ENGINE": "bing"}, "auto"),
        ({"GOOGLE_SEARCH_MODE": "custom_search_only"}, "google_custom"),
        ({"GOOGLE_SEARCH_MODE": "hybrid"}, "auto"),
        ({"GOOGLE_SEARCH_MODE": "googlesearch_only"}, "auto"),
        ({"SEARCH_ENGINE": "exa", "GOOGLE_SEARCH_MODE": "custom_search_only"}, "exa"),
    ])
    def test_default_engine(self, monkeypatch, env, expected):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert web_search.resolve_default_engine() == expected

    @pytest.mark.parametrize("raw,expected", [
        (None, ("google_custom", "brave", "exa")),
        ("exa, brave", ("exa", "brave")),
        ("brave,bogus,brave", ("brave",)),
        ("bogus", ("google_custom", "brave", "exa")),
    ])
    def test_auto_order(self, monkeypatch, raw, expected):
        if raw is not None:
            monkeypatch.setenv("SEARCH_ENGINE_ORDER", raw)
        assert web_search.resolve_auto_order() == expected


class StubClient:
    def __init__(self, configured=True, result=None):
        self.configured = configured
        self.result = result or {"success": True, "results": [], "total_results": 0}
        self.calls: List[tuple] = []

    def is_configured(self):
        return self.configured

    async def search(self, *args):
        self.calls.append(args)
        return dict(self.result)


def _ok(url="https://a.com/x"):
    return {"success": True, "total_results": 1, "processing_method": "stub", "results": [
        {"rank": 1, "url": url, "domain": "a.com", "title": "t", "snippet": "s", "type": "commercial"}]}


def _processor(engine="auto", order=None, **clients) -> WebSearchProcessor:
    proc = WebSearchProcessor()
    proc.default_engine = engine
    if order:
        proc.auto_order = order
    proc.clients = {
        "google_custom": clients.get("google_custom", StubClient(configured=False)),
        "brave": clients.get("brave", StubClient(configured=False)),
        "exa": clients.get("exa", StubClient(configured=False)),
    }
    return proc


class TestWebSearchProcessor:
    @pytest.mark.asyncio
    async def test_auto_falls_back_in_order(self):
        g = StubClient(result=failure("rate_limited", "429"))
        b = StubClient(result=failure("server", "503"))
        e = StubClient(result=_ok())
        result = await _processor(google_custom=g, brave=b, exa=e).search("q")

        assert result["success"] is True
        assert result["engine"] == "exa"
        assert [a["engine"] for a in result["fallback_info"]["attempts"]] == ["google_custom", "brave", "exa"]
        assert result["fallback_info"]["attempts"][0]["failure_kind"] == "rate_limited"
        assert result["search_metadata"]["search_params"]["engine"] == "exa"

    @pytest.mark.asyncio
    async def test_auto_uses_first_configured(self):
        b = StubClient(result=_ok())
        e = StubClient(result=_ok())
        result = await _processor(brave=b, exa=e).search("q")
        assert result["engine"] == "brave"
        assert "fallback_info" not in result
        assert e.calls == []

    @pytest.mark.asyncio
    async def test_custom_order(self):
        b = StubClient(result=_ok())
        e = StubClient(result=_ok())
        result = await _processor(order=("exa", "brave"), brave=b, exa=e).search("q")
        assert result["engine"] == "exa"

    @pytest.mark.asyncio
    async def test_zero_results_do_not_fall_back(self):
        g = StubClient(result=failure("no_results", "No search results found"))
        b = StubClient(result=_ok())
        result = await _processor(google_custom=g, brave=b).search("python asyncio tutorial guide")
        assert result["success"] is False
        assert result["engine"] == "google_custom"
        assert b.calls == []
        assert result["alternative_queries"]

    @pytest.mark.asyncio
    async def test_all_fail(self):
        g = StubClient(result=failure("auth", "bad key"))
        b = StubClient(result=failure("network", "down"))
        result = await _processor(google_custom=g, brave=b).search("q")
        assert result["success"] is False
        assert result["error"] == "All search engines failed. Last error: down"
        assert len(result["fallback_info"]["attempts"]) == 2

    @pytest.mark.asyncio
    async def test_nothing_configured(self):
        result = await _processor().search("q")
        assert result["error_code"] == "search_engine_not_configured"
        assert "BRAVE_SEARCH_API_KEY" in result["suggestion"]

    @pytest.mark.asyncio
    async def test_explicit_engine_not_configured(self):
        result = await _processor(brave=StubClient(result=_ok())).search("q", engine="exa")
        assert result["error_code"] == "search_engine_not_configured"
        assert result["configured_engines"] == ["brave"]

    @pytest.mark.asyncio
    async def test_explicit_engine_overrides_default(self):
        b = StubClient(result=_ok())
        e = StubClient(result=_ok())
        result = await _processor(engine="brave", brave=b, exa=e).search("q", engine="EXA")
        assert result["engine"] == "exa"
        assert b.calls == []

    @pytest.mark.asyncio
    async def test_explicit_engine_has_no_fallback(self):
        b = StubClient(result=failure("rate_limited", "429"))
        e = StubClient(result=_ok())
        result = await _processor(brave=b, exa=e).search("q", engine="brave")
        assert result["success"] is False
        assert result["error"] == "429"
        assert e.calls == []

    @pytest.mark.asyncio
    async def test_genre_operator_for_brave_not_exa(self):
        b = StubClient(result=_ok())
        e = StubClient(result=_ok())
        proc = _processor(brave=b, exa=e)
        await proc.search("q", search_genre="pdf", engine="brave")
        await proc.search("q", search_genre="japanese", engine="exa")
        assert b.calls[0][0] == "q (filetype:pdf)"
        # exa: (query, num_results, region, safe_search, recent_days)
        assert e.calls[0][0] == "q"
        assert e.calls[0][2] == "jp"

    @pytest.mark.asyncio
    async def test_exa_rejects_file_type_genre(self):
        result = await _processor(exa=StubClient(result=_ok())).search("q", search_genre="pdf", engine="exa")
        assert result["error_code"] == "unsupported_search_genre"

    @pytest.mark.asyncio
    async def test_auto_skips_exa_for_file_type_genre(self):
        e = StubClient(result=_ok())
        result = await _processor(exa=e).search("q", search_genre="spreadsheets")
        assert result["error_code"] == "search_engine_not_configured"
        assert e.calls == []

        b = StubClient(result=failure("server", "503"))
        result = await _processor(order=("exa", "brave"), brave=b, exa=e).search("q", search_genre="pdf")
        assert e.calls == []
        assert [a["engine"] for a in result["fallback_info"]["attempts"]] == ["exa", "brave"]
        assert result["fallback_info"]["attempts"][0]["failure_kind"] == "unsupported_genre"

    @pytest.mark.asyncio
    async def test_recent_days_passed_to_client(self):
        b = StubClient(result=_ok())
        await _processor(brave=b).search("q", recent_days=7, num_results=500)
        # brave: (query, num_results, language, region, safe_search, recent_days)
        assert b.calls[0][1] == 100
        assert b.calls[0][5] == 7

    @pytest.mark.asyncio
    async def test_engine_exception_falls_back(self):
        class Boom(StubClient):
            async def search(self, *args):
                raise ValueError("unexpected shape")
        e = StubClient(result=_ok())
        result = await _processor(brave=Boom(), exa=e).search("q")
        assert result["engine"] == "exa"
        assert "unexpected shape" in result["fallback_info"]["attempts"][0]["error"]

    @pytest.mark.asyncio
    async def test_discussions_auto_uses_brave_only(self):
        g = StubClient(result=_ok())
        b = StubClient(result=_ok())
        e = StubClient(result=_ok())
        result = await _processor(google_custom=g, brave=b, exa=e).search("q", result_type="discussions")
        assert result["engine"] == "brave"
        assert result["result_type"] == "discussions"
        assert result["search_metadata"]["search_params"]["result_type"] == "discussions"
        assert g.calls == [] and e.calls == []
        # brave: (query, num_results, language, region, safe_search, recent_days, result_type)
        assert b.calls[0][6] == "discussions"
        assert "fallback_info" not in result

    @pytest.mark.asyncio
    async def test_discussions_without_brave(self):
        result = await _processor(google_custom=StubClient(result=_ok())).search("q", result_type="discussions")
        assert result["error_code"] == "search_engine_not_configured"
        assert "discussions" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("engine", ["google_custom", "exa"])
    async def test_discussions_on_other_engine(self, engine):
        proc = _processor(google_custom=StubClient(result=_ok()), exa=StubClient(result=_ok()))
        result = await proc.search("q", result_type="discussions", engine=engine)
        assert result["error_code"] == "unsupported_result_type"

    @pytest.mark.asyncio
    async def test_engine_suggestion_is_kept_on_zero_results(self):
        b = StubClient(result=failure("no_results", "No discussion results found",
                                      suggestion="Brave returns discussions only for some queries"))
        result = await _processor(brave=b).search("should I learn go or rust", result_type="discussions")
        assert result["suggestion"] == "Brave returns discussions only for some queries"
        assert result["alternative_queries"]

    @pytest.mark.asyncio
    async def test_invalid_result_type(self):
        result = await _processor(brave=StubClient(result=_ok())).search("q", result_type="news")
        assert result["error_code"] == "invalid_result_type"

    @pytest.mark.asyncio
    async def test_web_result_type_by_default(self):
        b = StubClient(result=_ok())
        result = await _processor(brave=b).search("q")
        assert result["result_type"] == "web"
        assert b.calls[0][6] == "web"

    def test_google_search_processor_alias(self):
        from crawl4ai_mcp.processors import GoogleSearchProcessor
        assert GoogleSearchProcessor is WebSearchProcessor

    @pytest.mark.asyncio
    async def test_legacy_search_google_signature(self):
        b = StubClient(result=_ok())
        result = await _processor(brave=b).search_google(
            query="q", num_results=3, language="ja", region="jp", safe_search=True,
            search_genre="pdf", include_snippets=False,
        )
        assert result["success"] is True
        assert b.calls[0][:3] == ("q (filetype:pdf)", 3, "ja")

    def test_legacy_import_paths(self):
        from crawl4ai_mcp.processors.google_search import GoogleSearchProcessor
        from crawl4ai_mcp.processors.google_custom_search import CustomSearchAPIClient as C
        from crawl4ai_mcp.processors.google_search_helpers import RateLimiter, SearchRequest  # noqa: F401
        from crawl4ai_mcp.processors.google_search_analysis import GoogleSearchAnalysisMixin
        from crawl4ai_mcp.google_search_processor import GoogleSearchProcessor as Root
        assert GoogleSearchProcessor is Root is WebSearchProcessor
        assert C is CustomSearchAPIClient
        assert issubclass(WebSearchProcessor, GoogleSearchAnalysisMixin)


def test_rate_limiter_drops_old_history_for_unlimited_engines():
    from datetime import datetime
    from crawl4ai_mcp.processors.search_engines.rate_limit import RateLimiter, SearchRequest
    limiter = RateLimiter()
    old = datetime.now() - timedelta(days=1)
    limiter.requests["brave"] = [SearchRequest(old, "brave", "success")] * 50
    limiter.record_request("brave", "success")
    assert len(limiter.requests["brave"]) == 1


# ---------------------------------------------------------------------------
# core / cache / tool layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, None), (0, None), (-3, None), (7, 7), ("14", 14), ("2w", 14), ("bogus", None),
])
def test_recent_days_to_int(value, expected):
    assert recent_days_to_int(value) == expected


def test_recent_days_months_become_days():
    assert 28 <= recent_days_to_int("1mo") <= 31


@pytest.mark.asyncio
async def test_batch_keeps_engine_and_fallback_info(monkeypatch):
    from crawl4ai_mcp.core import search as core_search
    g = StubClient(result=failure("rate_limited", "429"))
    b = StubClient(result=_ok())
    proc = _processor(google_custom=g, brave=b)
    monkeypatch.setattr(core_search, "web_search_processor", proc)
    monkeypatch.setattr("crawl4ai_mcp.processors.web_search_analysis.asyncio.sleep", AsyncMock())

    result = await core_search.batch_search_web({"queries": ["a"]})
    entry = result["results"][0]
    assert entry["engine"] == "brave"
    assert [a["engine"] for a in entry["fallback_info"]["attempts"]] == ["google_custom", "brave"]

    b.result = failure("server", "503")
    result = await core_search.batch_search_web({"queries": ["a"]})
    entry = result["results"][0]
    assert entry["success"] is False
    assert entry["engine"] == "brave"
    assert len(entry["fallback_info"]["attempts"]) == 2


def test_cache_key_includes_engine():
    base = {"query": "q"}
    assert _get_search_cache_key(base) == _get_search_cache_key({**base, "engine": ""})
    assert _get_search_cache_key(base) != _get_search_cache_key({**base, "engine": "brave"})
    assert _get_search_cache_key({**base, "engine": "Brave"}) == _get_search_cache_key({**base, "engine": "brave"})
    assert _get_search_cache_key(base) == _get_search_cache_key({**base, "result_type": "web"})
    assert _get_search_cache_key(base) != _get_search_cache_key({**base, "result_type": "discussions"})


class FakeMcp:
    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _tools(search) -> Dict[str, Any]:
    fm = FakeMcp()
    register_search_tools(fm, lambda: (MagicMock(), search, MagicMock(), MagicMock(), MagicMock()))
    return fm.tools


class TestSearchToolsEngine:
    def test_new_and_alias_tools_registered(self):
        tools = _tools(MagicMock())
        for name in ("search_web", "batch_search_web", "search_google", "batch_search_google",
                     "search_and_crawl", "get_search_genres"):
            assert name in tools

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool,request_dict", [
        ("search_web", {"query": "x", "engine": "bing"}),
        ("search_google", {"query": "x", "engine": "bing"}),
        ("batch_search_web", {"queries": ["x"], "engine": 1}),
        ("search_and_crawl", {"search_query": "x", "engine": "duckduckgo"}),
    ])
    async def test_invalid_engine_errors_before_search(self, tool, request_dict):
        search = MagicMock()
        boom = AsyncMock(side_effect=AssertionError("must not search"))
        search.search_web = search.batch_search_web = search.search_and_crawl = boom
        result = await _tools(search)[tool](request=request_dict)
        assert result["error_code"] == "invalid_search_engine"
        assert result["valid_engines"] == ["auto", "google_custom", "brave", "exa"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool,request_dict", [
        ("search_web", {"query": "x", "result_type": "news"}),
        ("batch_search_web", {"queries": ["x"], "result_type": 3}),
        ("search_and_crawl", {"search_query": "x", "result_type": "forum"}),
    ])
    async def test_invalid_result_type_errors_before_search(self, tool, request_dict):
        search = MagicMock()
        boom = AsyncMock(side_effect=AssertionError("must not search"))
        search.search_web = search.batch_search_web = search.search_and_crawl = boom
        result = await _tools(search)[tool](request=request_dict)
        assert result["error_code"] == "invalid_result_type"
        assert result["valid_result_types"] == ["web", "discussions"]

    @pytest.mark.asyncio
    async def test_search_and_crawl_passes_result_type(self):
        search = MagicMock()
        search.search_and_crawl = AsyncMock(return_value={"success": True})
        await _tools(search)["search_and_crawl"](request={"search_query": "x", "result_type": "discussions"})
        assert search.search_and_crawl.await_args.kwargs["result_type"] == "discussions"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool,core_fn,request_dict", [
        ("search_google", "search_web", {"query": "x", "engine": "brave"}),
        ("batch_search_google", "batch_search_web", {"queries": ["x"], "engine": "exa"}),
    ])
    async def test_alias_calls_same_core(self, tool, core_fn, request_dict):
        search = MagicMock()
        setattr(search, core_fn, AsyncMock(return_value={"success": True, "results": []}))
        result = await _tools(search)[tool](request=request_dict)
        assert result["success"] is True
        getattr(search, core_fn).assert_awaited_once_with(request_dict)

    @pytest.mark.asyncio
    async def test_search_and_crawl_passes_engine(self):
        search = MagicMock()
        search.search_and_crawl = AsyncMock(return_value={"success": True})
        await _tools(search)["search_and_crawl"](request={"search_query": "x", "engine": "brave"})
        assert search.search_and_crawl.await_args.kwargs["engine"] == "brave"
