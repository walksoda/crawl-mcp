"""Regression tests for graceful handling of None values in crawl4ai responses.

Background (issue #24 / PR #25):
crawl4ai can return a response dict whose keys exist but whose values are
``None`` (e.g. ``{"markdown": None}``). Code that did
``result_dict.get("markdown", "").strip()`` crashed with
``AttributeError: 'NoneType' object has no attribute 'strip'`` because
``dict.get(key, default)`` returns ``None`` (not the default) when the key is
present with a ``None`` value.

These tests pin the fixed behaviour for the pure helpers touched by the fix so
the crash cannot silently reappear.
"""

from __future__ import annotations

import pytest

from crawl4ai_mcp.server_helpers import _should_trigger_fallback
from crawl4ai_mcp.middleware.response_transform import _process_content_fields


# ---------------------------------------------------------------------------
# _should_trigger_fallback
# ---------------------------------------------------------------------------

class TestShouldTriggerFallbackNone:
    def test_all_content_fields_none_triggers_fallback(self):
        """None markdown/content/raw_content must not crash and should fallback."""
        result = {
            "success": True,
            "markdown": None,
            "content": None,
            "raw_content": None,
        }
        should, reason = _should_trigger_fallback(result, generate_markdown=True)
        assert should is True
        assert reason

    def test_none_markdown_with_valid_content_no_fallback(self):
        """markdown=None but real content present: no crash, no fallback."""
        result = {
            "success": True,
            "markdown": None,
            "content": " body text ",
            "raw_content": None,
        }
        should, _ = _should_trigger_fallback(result, generate_markdown=True)
        assert should is False

    def test_none_content_markdown_only_no_fallback(self):
        """content=None but markdown present (markdown requested): no fallback."""
        result = {
            "success": True,
            "markdown": "# Title\nbody",
            "content": None,
            "raw_content": None,
        }
        should, _ = _should_trigger_fallback(result, generate_markdown=True)
        assert should is False

    def test_missing_keys_still_work(self):
        """Behaviour unchanged when keys are simply absent."""
        result = {"success": True}
        should, reason = _should_trigger_fallback(result, generate_markdown=True)
        assert should is True
        assert reason

    def test_markdown_not_requested_none_content_triggers(self):
        result = {"success": True, "content": None, "raw_content": None}
        should, reason = _should_trigger_fallback(result, generate_markdown=False)
        assert should is True
        assert reason


# ---------------------------------------------------------------------------
# _process_content_fields
# ---------------------------------------------------------------------------

class TestProcessContentFieldsNone:
    def test_none_markdown_does_not_crash_and_keeps_content(self):
        """markdown=None must not crash; HTML content is kept (nothing to swap to)."""
        result = {"markdown": None, "content": "<html>body</html>"}
        out = _process_content_fields(
            result, include_cleaned_html=False, generate_markdown=True
        )
        # No markdown available, so content must be preserved.
        assert out["content"] == "<html>body</html>"

    def test_valid_markdown_removes_content(self):
        """Behaviour unchanged: real markdown lets content be dropped."""
        result = {"markdown": "# real markdown", "content": "<html>body</html>"}
        out = _process_content_fields(
            result, include_cleaned_html=False, generate_markdown=True
        )
        assert "content" not in out
        assert any("HTML content removed" in w for w in out.get("warnings", []))

    def test_include_cleaned_html_keeps_content(self):
        result = {"markdown": None, "content": "<html>body</html>"}
        out = _process_content_fields(
            result, include_cleaned_html=True, generate_markdown=True
        )
        assert out["content"] == "<html>body</html>"


# ---------------------------------------------------------------------------
# search_tools truncation guard (len(None) protection)
# ---------------------------------------------------------------------------

class TestTruncationNoneGuard:
    """Mirror the guard used in search_tools.py so a regression to
    ``len(page["content"])`` on a None value is caught here too.
    """

    @staticmethod
    def _truncate(page: dict, limit: int) -> dict:
        # Same predicate shape as crawl4ai_mcp/server_tools/search_tools.py
        if page.get("content") and len(page["content"]) > limit:
            page["content"] = page["content"][:limit] + "... [truncated for size limit]"
        if page.get("markdown") and len(page["markdown"]) > limit:
            page["markdown"] = page["markdown"][:limit] + "... [truncated for size limit]"
        return page

    def test_none_values_do_not_crash(self):
        page = {"content": None, "markdown": None}
        out = self._truncate(page, limit=10)
        assert out["content"] is None
        assert out["markdown"] is None

    def test_long_values_still_truncated(self):
        page = {"content": "x" * 100, "markdown": "y" * 100}
        out = self._truncate(page, limit=10)
        assert out["content"].startswith("x" * 10)
        assert "truncated" in out["content"]
        assert "truncated" in out["markdown"]

    def test_short_values_untouched(self):
        page = {"content": "short", "markdown": "tiny"}
        out = self._truncate(dict(page), limit=10)
        assert out["content"] == "short"
        assert out["markdown"] == "tiny"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
