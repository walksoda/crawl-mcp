"""Unit tests for crawl4ai_mcp.middleware.file_persistence.

These tests stay pure: they use tmp_path fixtures and never hit the network
or the crawl4ai core modules. The goal is to fully cover the writer helpers
and finalize_tool_response so that integration tests can focus on wiring
each tool's return paths correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crawl4ai_mcp.middleware import file_persistence as fp
from crawl4ai_mcp.middleware.file_persistence import (
    KIND_MARKDOWN_BATCH_DICT,
    KIND_MARKDOWN_BATCH_LIST,
    KIND_MARKDOWN_SINGLE,
    KIND_SEARCH_JSON,
    KIND_STRUCTURED_JSON,
    KIND_YOUTUBE_COMMENTS,
    finalize_tool_response,
)
from crawl4ai_mcp.validators import validate_output_path


# ---------------------------------------------------------------------------
# validate_output_path
# ---------------------------------------------------------------------------


class TestValidateOutputPath:
    def test_none_is_noop(self):
        assert validate_output_path(None) is None

    def test_empty_string_is_noop(self):
        assert validate_output_path("") is None

    def test_non_string_rejected(self):
        err = validate_output_path(123)  # type: ignore[arg-type]
        assert err and err["error_code"] == "invalid_output_path_type"

    def test_nul_rejected(self):
        err = validate_output_path("/tmp/a\x00b.md")
        assert err and err["error_code"] == "invalid_output_path_chars"

    def test_relative_rejected(self):
        err = validate_output_path("relative/path.md")
        assert err and err["error_code"] == "output_path_not_absolute"

    def test_absolute_ok(self, tmp_path):
        assert validate_output_path(str(tmp_path / "out.md")) is None

    def test_tilde_expanded(self):
        # ~/x.md → absolute after expansion → OK.
        assert validate_output_path("~/_file_persistence_nonexistent.md") is None

    def test_existing_file_rejected_when_overwrite_false(self, tmp_path):
        p = tmp_path / "existing.md"
        p.write_text("hi")
        err = validate_output_path(str(p), overwrite=False)
        assert err and err["error_code"] == "output_path_exists"

    def test_existing_file_ok_when_overwrite_true(self, tmp_path):
        p = tmp_path / "existing.md"
        p.write_text("hi")
        assert validate_output_path(str(p), overwrite=True) is None

    def test_existing_directory_not_rejected(self, tmp_path):
        # Batch tools use directories; must not trip on dir existence.
        assert validate_output_path(str(tmp_path), overwrite=False) is None


# ---------------------------------------------------------------------------
# Path / filename helpers
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_basic(self):
        assert (
            fp._sanitize_filename_from_url("https://example.com/article")
            == "example.com_article"
        )

    def test_query_string_slugified(self):
        slug = fp._sanitize_filename_from_url("https://a.com/p?x=1&y=2")
        assert "?" not in slug and "&" not in slug and "=" not in slug

    def test_empty_url(self):
        assert fp._sanitize_filename_from_url("") == "page"

    def test_long_url_has_hash_suffix(self):
        long_url = "https://example.com/" + ("segment/" * 50)
        slug = fp._sanitize_filename_from_url(long_url, max_len=40)
        assert len(slug) <= 40
        # Hash suffix preserves uniqueness
        assert "_" in slug
        slug2 = fp._sanitize_filename_from_url(long_url + "?q=other", max_len=40)
        assert slug != slug2


class TestValidateAbsolutePath:
    def test_relative_raises(self):
        with pytest.raises(ValueError):
            fp._validate_absolute_path("relative/p.md")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fp._validate_absolute_path("")

    def test_absolute_ok(self, tmp_path):
        p = fp._validate_absolute_path(str(tmp_path / "a.md"))
        assert p.is_absolute()


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_none_values_skipped(self):
        fm = fp._frontmatter_md({"url": "https://x.com", "title": None})
        assert "url" in fm
        assert "title" not in fm

    def test_double_quotes_escaped(self):
        fm = fp._frontmatter_md({"title": 'He said "hi"'})
        # Double quotes are downgraded to single quotes inside the value.
        assert '"hi"' not in fm
        assert "'hi'" in fm

    def test_newlines_collapsed(self):
        fm = fp._frontmatter_md({"title": "line1\nline2"})
        assert "line1 line2" in fm

    def test_structure(self):
        fm = fp._frontmatter_md({"url": "https://x.com"})
        assert fm.startswith("---\n")
        assert "\n---\n" in fm


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWriteText:
    def test_writes_utf8(self, tmp_path):
        p = tmp_path / "x.md"
        n = fp._atomic_write_text(p, "日本語", overwrite=False)
        assert p.read_text(encoding="utf-8") == "日本語"
        assert n == len("日本語".encode("utf-8"))

    def test_existing_no_overwrite_raises(self, tmp_path):
        p = tmp_path / "x.md"
        p.write_text("old")
        with pytest.raises(FileExistsError):
            fp._atomic_write_text(p, "new", overwrite=False)
        assert p.read_text() == "old"

    def test_overwrite_replaces(self, tmp_path):
        p = tmp_path / "x.md"
        p.write_text("old")
        fp._atomic_write_text(p, "new", overwrite=True)
        assert p.read_text() == "new"

    def test_no_tempfile_left_behind(self, tmp_path):
        p = tmp_path / "x.md"
        fp._atomic_write_text(p, "hi", overwrite=False)
        leftover = [f for f in tmp_path.iterdir() if f.name.startswith(".")]
        assert leftover == []


# ---------------------------------------------------------------------------
# Strip nested
# ---------------------------------------------------------------------------


class TestStripNested:
    def test_simple_key(self):
        d = {"content": "x", "url": "u"}
        fp._strip_nested(d, ["content"])
        assert d == {"url": "u"}

    def test_nested_dot(self):
        d = {"extracted_data": {"comments": [1, 2], "count": 5}, "url": "u"}
        fp._strip_nested(d, ["extracted_data.comments"])
        assert d == {"extracted_data": {"count": 5}, "url": "u"}

    def test_iter_over_list(self):
        d = {"results": [{"content": "a", "url": "u1"}, {"content": "b", "url": "u2"}]}
        fp._strip_nested(d, ["results[*].content"])
        assert d["results"] == [{"url": "u1"}, {"url": "u2"}]

    def test_missing_path_no_error(self):
        d = {"url": "u"}
        fp._strip_nested(d, ["does.not.exist", "results[*].content"])
        assert d == {"url": "u"}

    def test_top_level_list(self):
        lst = [{"content": "a", "url": "u1"}, {"content": "b", "url": "u2"}]
        fp._strip_nested(lst, ["[*].content"])
        assert lst == [{"url": "u1"}, {"url": "u2"}]


class TestParseStripPath:
    def test_plain_key(self):
        assert fp._parse_strip_path("content") == [("key", "content")]

    def test_nested(self):
        assert fp._parse_strip_path("a.b.c") == [
            ("key", "a"),
            ("key", "b"),
            ("key", "c"),
        ]

    def test_iter(self):
        assert fp._parse_strip_path("results[*].content") == [
            ("key", "results"),
            ("iter", None),
            ("key", "content"),
        ]

    def test_top_level_iter(self):
        assert fp._parse_strip_path("[*].content") == [
            ("iter", None),
            ("key", "content"),
        ]

    def test_rejects_non_star_index(self):
        with pytest.raises(ValueError):
            fp._parse_strip_path("results[0].content")


# ---------------------------------------------------------------------------
# Probe batch items
# ---------------------------------------------------------------------------


class TestProbeBatchItems:
    def test_finds_results(self):
        key, items = fp._probe_batch_items({"results": [{"a": 1}]})
        assert key == "results"
        assert items == [{"a": 1}]

    def test_finds_crawled_pages(self):
        key, items = fp._probe_batch_items({"crawled_pages": [{"url": "u"}]})
        assert key == "crawled_pages"

    def test_preference_order(self):
        # "results" wins over "crawled_pages" when both present.
        key, _ = fp._probe_batch_items(
            {"results": [{"a": 1}], "crawled_pages": [{"b": 2}]}
        )
        assert key == "results"

    def test_none_when_absent(self):
        key, items = fp._probe_batch_items({"url": "u"})
        assert key is None and items == []


# ---------------------------------------------------------------------------
# finalize_tool_response — markdown single
# ---------------------------------------------------------------------------


def _make_crawl_result(**extra):
    base = {
        "success": True,
        "url": "https://example.com/article",
        "title": "Example",
        "markdown": "# Heading\n\nBody text " * 200,
        "content": "raw html " * 100,
    }
    base.update(extra)
    return base


class TestFinalizeMarkdownSingle:
    def test_noop_when_output_path_none(self):
        r = _make_crawl_result()
        original = dict(r)
        out = finalize_tool_response(
            r,
            output_path=None,
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert out == original
        assert "persisted" not in out

    def test_noop_when_success_false(self, tmp_path):
        r = {"success": False, "error": "boom"}
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "x.md"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert out == r
        assert not (tmp_path / "x.md").exists()

    def test_writes_markdown_and_strips_content(self, tmp_path):
        r = _make_crawl_result()
        out_path = tmp_path / "article.md"
        out = finalize_tool_response(
            r,
            output_path=str(out_path),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        # File written with frontmatter + body
        body = out_path.read_text(encoding="utf-8")
        assert body.startswith("---\n")
        assert "# Heading" in body
        assert 'url: "https://example.com/article"' in body
        # Slim response
        assert out["persisted"] is True
        assert "markdown" not in out
        assert "content" not in out
        assert out["url"] == "https://example.com/article"
        assert out["output_files"] == [str(out_path)]
        assert out["output_bytes"] > 0
        assert out["content_included_in_response"] is False

    def test_auto_extension_appended(self, tmp_path):
        r = _make_crawl_result()
        base = tmp_path / "article"  # no extension
        out = finalize_tool_response(
            r,
            output_path=str(base),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        expected = tmp_path / "article.md"
        assert expected.exists()
        assert out["output_files"] == [str(expected)]

    def test_mismatched_extension_warns(self, tmp_path):
        r = _make_crawl_result()
        out_path = tmp_path / "weird.txt"
        out = finalize_tool_response(
            r,
            output_path=str(out_path),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert out_path.exists()
        assert "warnings" in out
        assert any("extension" in w for w in out["warnings"])

    def test_include_content_keeps_fields(self, tmp_path):
        r = _make_crawl_result()
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "a.md"),
            include_content_in_response=True,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert "markdown" in out
        assert out["content_included_in_response"] is True

    def test_overwrite_false_existing_file(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text("existing")
        r = _make_crawl_result()
        out = finalize_tool_response(
            r,
            output_path=str(p),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert out["success"] is False
        assert out["error_code"] == "output_path_exists"
        # Original file untouched
        assert p.read_text() == "existing"

    def test_overwrite_true_replaces(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text("existing")
        r = _make_crawl_result()
        out = finalize_tool_response(
            r,
            output_path=str(p),
            include_content_in_response=False,
            overwrite=True,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert out.get("persisted") is True
        assert "existing" not in p.read_text()

    def test_parent_dir_autocreated(self, tmp_path):
        r = _make_crawl_result()
        p = tmp_path / "nested" / "dir" / "a.md"
        out = finalize_tool_response(
            r,
            output_path=str(p),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert p.exists()
        assert out.get("persisted") is True

    def test_invalid_shape_list_for_single_kind(self, tmp_path):
        out = finalize_tool_response(
            [{"success": True, "url": "u"}],
            output_path=str(tmp_path / "a.md"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        # List-shape input → error is wrapped in a single-element list so
        # tools that declare list returns don't break.
        assert isinstance(out, list)
        assert out[0]["error_code"] == "invalid_shape"


# ---------------------------------------------------------------------------
# finalize_tool_response — markdown batch dict
# ---------------------------------------------------------------------------


class TestFinalizeMarkdownBatchDict:
    def _make_deep_crawl_result(self):
        return {
            "success": True,
            "results": [
                {
                    "success": True,
                    "url": "https://a.example.com/page",
                    "title": "A",
                    "markdown": "# A body\n\n" + "content " * 100,
                },
                {
                    "success": True,
                    "url": "https://b.example.com/page",
                    "title": "B",
                    "markdown": "# B body\n\n" + "content " * 100,
                },
            ],
        }

    def test_existing_file_path_rejected(self, tmp_path):
        """An existing regular file is rejected even for batch tools — we
        never write batch output on top of a single file."""
        existing = tmp_path / "out.md"
        existing.write_text("preexisting")
        r = self._make_deep_crawl_result()
        out = finalize_tool_response(
            r,
            output_path=str(existing),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_DICT,
            source_tool="deep_crawl_site",
        )
        assert out["success"] is False
        assert out["error_code"] == "invalid_output_path"
        assert existing.read_text() == "preexisting"  # untouched

    def test_nonexistent_dot_path_is_allowed_as_directory(self, tmp_path):
        """A non-existent path with a dot in the name is treated as a
        directory (the old brittle ``Path.suffix`` check is gone)."""
        r = self._make_deep_crawl_result()
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "run.v1"),  # looks like it has an extension
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_DICT,
            source_tool="deep_crawl_site",
        )
        assert out.get("persisted") is True
        out_dir = tmp_path / "run.v1"
        assert out_dir.is_dir()
        assert (out_dir / "index.json").exists()

    def test_writes_directory_and_index(self, tmp_path):
        r = self._make_deep_crawl_result()
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "run1"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_DICT,
            source_tool="deep_crawl_site",
        )
        # Two per-item markdown files + index.json
        out_dir = tmp_path / "run1"
        md_files = sorted(p.name for p in out_dir.glob("*.md"))
        assert len(md_files) == 2
        assert (out_dir / "index.json").exists()
        index = json.loads((out_dir / "index.json").read_text())
        assert index["count"] == 2
        assert index["source_tool"] == "deep_crawl_site"
        assert index["items_key"] == "results"
        # Slim response: results[i] no longer has content/markdown
        for item in out["results"]:
            assert "markdown" not in item
            assert "content" not in item
            assert item["url"]  # metadata preserved
        assert out["persisted"] is True
        assert len(out["output_files"]) == 3  # 2 md + 1 index

    def test_search_and_crawl_includes_search_results_in_index(self, tmp_path):
        r = {
            "success": True,
            "query": "q",
            "search_results": [{"url": "https://a.example.com", "title": "A"}],
            "crawled_pages": [
                {
                    "success": True,
                    "url": "https://a.example.com",
                    "title": "A",
                    "markdown": "body",
                }
            ],
        }
        finalize_tool_response(
            r,
            output_path=str(tmp_path / "sc"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_DICT,
            source_tool="search_and_crawl",
        )
        index = json.loads((tmp_path / "sc" / "index.json").read_text())
        assert index["items_key"] == "crawled_pages"
        assert "search_results" in index

    def test_filename_collision_suffix(self, tmp_path):
        # Two URLs that slug to the same name.
        r = {
            "success": True,
            "results": [
                {"success": True, "url": "https://a.example.com/page", "title": "A", "markdown": "x"},
                {"success": True, "url": "https://a.example.com/page", "title": "A2", "markdown": "y"},
            ],
        }
        finalize_tool_response(
            r,
            output_path=str(tmp_path / "d"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_DICT,
            source_tool="deep_crawl_site",
        )
        md_files = sorted(p.name for p in (tmp_path / "d").glob("*.md"))
        assert len(md_files) == 2
        assert md_files[0] != md_files[1]


# ---------------------------------------------------------------------------
# finalize_tool_response — markdown batch list
# ---------------------------------------------------------------------------


class TestFinalizeMarkdownBatchList:
    def test_list_shape_preserved_with_output_file(self, tmp_path):
        lst = [
            {"success": True, "url": "https://a.com", "title": "A", "markdown": "mdA"},
            {"success": True, "url": "https://b.com", "title": "B", "markdown": "mdB"},
        ]
        out = finalize_tool_response(
            lst,
            output_path=str(tmp_path / "batch"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_LIST,
            source_tool="batch_crawl",
        )
        # Return shape stays a list
        assert isinstance(out, list)
        assert len(out) == 2
        for item in out:
            assert "output_file" in item
            assert Path(item["output_file"]).exists()
            # Stripped
            assert "markdown" not in item
            assert item["url"]
        # index.json exists
        assert (tmp_path / "batch" / "index.json").exists()

    def test_partial_failure_only_persists_success(self, tmp_path):
        lst = [
            {"success": True, "url": "https://a.com", "markdown": "A"},
            {"success": False, "url": "https://b.com", "error": "boom"},
        ]
        out = finalize_tool_response(
            lst,
            output_path=str(tmp_path / "batch"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_LIST,
            source_tool="batch_crawl",
        )
        assert "output_file" in out[0]
        assert "output_file" not in out[1]
        md_files = list((tmp_path / "batch").glob("*.md"))
        assert len(md_files) == 1

    def test_all_failed_noop(self, tmp_path):
        lst = [
            {"success": False, "url": "https://a.com", "error": "x"},
            {"success": False, "url": "https://b.com", "error": "y"},
        ]
        out = finalize_tool_response(
            lst,
            output_path=str(tmp_path / "batch"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_LIST,
            source_tool="batch_crawl",
        )
        assert out == lst
        assert not (tmp_path / "batch").exists()

    def test_existing_file_path_rejected(self, tmp_path):
        """batch-list: an existing regular file is rejected."""
        existing = tmp_path / "out.md"
        existing.write_text("preexisting")
        lst = [{"success": True, "url": "https://a.com", "markdown": "x"}]
        out = finalize_tool_response(
            lst,
            output_path=str(existing),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_LIST,
            source_tool="batch_crawl",
        )
        # List shape: error is wrapped in a single-element list to preserve
        # the tool's declared return type.
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0]["error_code"] == "invalid_output_path"
        assert existing.read_text() == "preexisting"


# ---------------------------------------------------------------------------
# finalize_tool_response — JSON kinds
# ---------------------------------------------------------------------------


class TestFinalizeStructuredJson:
    def test_writes_json(self, tmp_path):
        r = {
            "success": True,
            "url": "https://x.com",
            "extracted_data": {"price": "$10", "name": "Widget"},
            "table_data": [{"h": ["a", "b"], "r": [[1, 2]]}],
            "markdown": "x " * 200,
        }
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "out"),  # no extension → auto .json
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_STRUCTURED_JSON,
            source_tool="extract_structured_data",
        )
        p = tmp_path / "out.json"
        assert p.exists()
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["extracted_data"]["price"] == "$10"
        assert loaded["table_data"]
        # Slim response: table_data and markdown stripped
        assert "table_data" not in out
        assert "markdown" not in out
        assert out["extracted_data"]["price"] == "$10"  # metadata preserved

    def test_youtube_comments_strip(self, tmp_path):
        r = {
            "success": True,
            "url": "https://youtube.com/watch?v=x",
            "extracted_data": {
                "comments": [{"author": "a", "text": "t"}] * 300,
                "comment_count": 300,
            },
        }
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "comments.json"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_YOUTUBE_COMMENTS,
            source_tool="extract_youtube_comments",
        )
        # Disk has full 300 comments
        loaded = json.loads((tmp_path / "comments.json").read_text())
        assert len(loaded["extracted_data"]["comments"]) == 300
        # Response has no comments array, but keeps comment_count
        assert "comments" not in out["extracted_data"]
        assert out["extracted_data"]["comment_count"] == 300

    def test_empty_string_output_path_is_noop(self, tmp_path):
        """Regression: ``output_path=""`` must behave identically to
        ``None`` — no write, no shape change, no spurious error. Guard A
        in :func:`validate_output_path` already accepts empty string; the
        finalize hook must match."""
        r = {"success": True, "url": "https://x", "markdown": "x", "content": "c"}
        original = dict(r)
        out = finalize_tool_response(
            r,
            output_path="",
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_SINGLE,
            source_tool="crawl_url",
        )
        assert out == original
        assert "persisted" not in out

    def test_batch_dict_partial_failure_skips_persist(self, tmp_path):
        """Regression: failed items (``success=False``) must NOT be written
        to disk as empty/stub markdown. They still appear in index.json."""
        r = {
            "success": True,
            "results": [
                {
                    "success": True,
                    "url": "https://a.com/ok",
                    "title": "A",
                    "markdown": "# A body",
                },
                {
                    "success": False,
                    "url": "https://b.com/fail",
                    "error": "timeout",
                },
            ],
        }
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "mixed"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_MARKDOWN_BATCH_DICT,
            source_tool="deep_crawl_site",
        )
        out_dir = tmp_path / "mixed"
        md_files = sorted(p.name for p in out_dir.glob("*.md"))
        # Exactly one .md for the one successful item
        assert len(md_files) == 1
        index = json.loads((out_dir / "index.json").read_text())
        assert index["count"] == 1  # only successful items are counted in "count"
        # index.items contains both — the failed one with file=None
        assert len(index["items"]) == 2
        failed = [e for e in index["items"] if not e["success"]]
        assert len(failed) == 1
        assert failed[0]["file"] is None
        assert failed[0]["error"] == "timeout"

    def test_search_json_strips_results(self, tmp_path):
        r = {
            "success": True,
            "query": "python",
            "results": [{"title": "t", "url": "u", "snippet": "s"}] * 50,
        }
        out = finalize_tool_response(
            r,
            output_path=str(tmp_path / "s.json"),
            include_content_in_response=False,
            overwrite=False,
            tool_kind=KIND_SEARCH_JSON,
            source_tool="search_google",
        )
        loaded = json.loads((tmp_path / "s.json").read_text())
        assert len(loaded["results"]) == 50
        # Response strips results (user reads from file)
        assert "results" not in out
        assert out["query"] == "python"
