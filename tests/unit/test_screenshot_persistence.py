"""
Unit tests for handle_screenshot_persistence.

The crawl_url tool layer wraps responses with apply_token_limit (default
25k tokens). A typical base64 PNG screenshot is ~50k tokens, which by
itself exceeds the budget and historically caused the screenshot field
to be silently dropped during emergency truncation.

handle_screenshot_persistence runs before apply_token_limit and either:
- writes the screenshot to a sibling PNG file alongside output_path and
  replaces the field with `screenshot_path`, OR
- drops the screenshot entirely and surfaces a warning telling the caller
  to pass output_path to retrieve it.

Either way, the response copy never carries the base64, so
apply_token_limit's max_tokens guarantee is preserved.
"""

import base64
import os
from pathlib import Path

import pytest

from crawl4ai_mcp.middleware.file_persistence import handle_screenshot_persistence


# Minimal valid PNG (1x1 transparent) — 67 bytes raw, ~92 chars base64.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


class TestNoOutputPath:
    """When output_path is unset, screenshot is dropped with a warning."""

    def test_screenshot_dropped(self):
        result = {
            "success": True,
            "url": "https://example.com",
            "title": "Example",
            "screenshot": _PNG_B64,
        }

        out = handle_screenshot_persistence(
            result, output_path=None, overwrite=False, source_tool="crawl_url"
        )

        assert "screenshot" not in out
        assert "screenshot_path" not in out
        warnings = out.get("warnings")
        assert isinstance(warnings, list)
        assert any("output_path" in w for w in warnings), warnings

    def test_warning_appended_to_existing_warnings_list(self):
        result = {
            "success": True,
            "url": "https://example.com",
            "warnings": ["existing warning"],
            "screenshot": _PNG_B64,
        }

        out = handle_screenshot_persistence(
            result, output_path=None, overwrite=False, source_tool="crawl_url"
        )

        warnings = out.get("warnings")
        assert "existing warning" in warnings
        assert any("output_path" in w for w in warnings)
        assert len(warnings) == 2

    def test_no_screenshot_field_is_noop(self):
        result = {"success": True, "url": "https://example.com", "title": "Example"}
        out = handle_screenshot_persistence(
            result, output_path=None, overwrite=False, source_tool="crawl_url"
        )
        assert out == {"success": True, "url": "https://example.com", "title": "Example"}

    def test_empty_screenshot_is_noop(self):
        for empty in (None, ""):
            result = {"success": True, "url": "https://example.com", "screenshot": empty}
            out = handle_screenshot_persistence(
                result, output_path=None, overwrite=False, source_tool="crawl_url"
            )
            assert "screenshot_path" not in out
            assert "warnings" not in out


class TestWithOutputPath:
    """When output_path is set, screenshot is written to a sibling file."""

    def test_sibling_file_written_and_path_returned(self, tmp_path):
        main = tmp_path / "run.json"
        result = {
            "success": True,
            "url": "https://example.com",
            "title": "Example",
            "screenshot": _PNG_B64,
        }

        out = handle_screenshot_persistence(
            result,
            output_path=str(main),
            overwrite=False,
            source_tool="crawl_url",
        )

        assert "screenshot" not in out
        ss_path = out["screenshot_path"]
        assert ss_path == str(tmp_path / "run_screenshot.png")
        assert Path(ss_path).read_bytes() == _PNG_BYTES
        assert out["screenshot_bytes"] == len(_PNG_BYTES)

    def test_no_extension_on_output_path_still_writes_sibling(self, tmp_path):
        # crawl_tools allows callers to omit the extension; finalize_tool_response
        # adds it later. We must still end up with a sibling next to the
        # eventual main artifact.
        main = tmp_path / "run"
        result = {
            "success": True,
            "screenshot": _PNG_B64,
        }

        out = handle_screenshot_persistence(
            result,
            output_path=str(main),
            overwrite=False,
            source_tool="crawl_url",
        )

        ss_path = out["screenshot_path"]
        assert ss_path == str(tmp_path / "run_screenshot.png")
        assert Path(ss_path).read_bytes() == _PNG_BYTES

    def test_overwrite_false_existing_file_drops_with_warning(self, tmp_path):
        existing = tmp_path / "run_screenshot.png"
        existing.write_bytes(b"OLD")

        main = tmp_path / "run.json"
        result = {"success": True, "screenshot": _PNG_B64}

        out = handle_screenshot_persistence(
            result,
            output_path=str(main),
            overwrite=False,
            source_tool="crawl_url",
        )

        assert "screenshot" not in out
        assert "screenshot_path" not in out
        warnings = out.get("warnings", [])
        assert any("already exists" in w for w in warnings), warnings
        # Existing file is untouched.
        assert existing.read_bytes() == b"OLD"

    def test_overwrite_true_replaces_existing(self, tmp_path):
        existing = tmp_path / "run_screenshot.png"
        existing.write_bytes(b"OLD")

        main = tmp_path / "run.json"
        result = {"success": True, "screenshot": _PNG_B64}

        out = handle_screenshot_persistence(
            result,
            output_path=str(main),
            overwrite=True,
            source_tool="crawl_url",
        )

        assert out["screenshot_path"] == str(existing)
        assert existing.read_bytes() == _PNG_BYTES

    def test_invalid_base64_drops_with_warning(self, tmp_path):
        result = {
            "success": True,
            "screenshot": "not!!!valid!!!base64==",
        }

        out = handle_screenshot_persistence(
            result,
            output_path=str(tmp_path / "run.json"),
            overwrite=False,
            source_tool="crawl_url",
        )

        assert "screenshot" not in out
        assert "screenshot_path" not in out
        warnings = out.get("warnings", [])
        assert any("not valid base64" in w for w in warnings), warnings


class TestTokenLimitGuarantee:
    """The response copy returned must never carry the base64 blob."""

    def test_response_size_after_persistence(self, tmp_path):
        # Simulate a realistic ~200kB base64 screenshot.
        big_b64 = base64.b64encode(b"X" * 150_000).decode("ascii")
        result = {"success": True, "url": "https://example.com", "screenshot": big_b64}

        # Without output_path: blob is gone.
        no_path = handle_screenshot_persistence(
            dict(result), output_path=None, overwrite=False, source_tool="crawl_url"
        )
        assert "screenshot" not in no_path
        # warnings list adds a few hundred chars at most.
        assert len(str(no_path)) < 2_000

        # With output_path: blob is gone, path is short.
        with_path = handle_screenshot_persistence(
            dict(result),
            output_path=str(tmp_path / "run.json"),
            overwrite=False,
            source_tool="crawl_url",
        )
        assert "screenshot" not in with_path
        assert with_path["screenshot_path"].endswith("_screenshot.png")
        assert len(str(with_path)) < 2_000
