"""File persistence middleware for Crawl4AI MCP Server.

Provides an opt-in mechanism to persist fetched content to disk (markdown or
JSON) via an ``output_path`` parameter, bypassing the 25000-token response
limit so LLM callers can fetch large pages without consuming a huge token
budget.

Design overview
---------------

Each information-gathering tool inserts two guards around its existing logic:

1. **Guard A** (input validation phase, before the first ``await``): call
   :func:`crawl4ai_mcp.validators.validate_output_path` and short-circuit on
   an invalid path so no external fetch is wasted.

2. **Guard B** (immediately before each successful ``return``, and before any
   internal truncation/slicing): call :func:`finalize_tool_response`. If
   ``output_path`` is set and the result reports success, this helper writes
   the full un-truncated content to disk, then (unless
   ``include_content_in_response=True``) strips the bulky fields from the
   response copy and attaches persistence metadata.

The module is shape-agnostic: batch dict tools may expose their items under
``results``, ``crawled_pages``, ``batch_results``, or ``multi_url_results``
and :func:`_probe_batch_items` probes for whichever key is present. Batch
list tools (``batch_crawl`` / ``multi_url_crawl``) preserve their historic
``List[dict]`` return shape; we embed an ``output_file`` entry in each
successful item rather than wrapping the list in a dict.
"""

from __future__ import annotations

import base64
import binascii
import copy
import datetime
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from ..utils.token_utils import estimate_tokens


# ---------------------------------------------------------------------------
# Tool-kind constants
# ---------------------------------------------------------------------------

KIND_MARKDOWN_SINGLE = "markdown_single"
"""Single-URL markdown: crawl_url, crawl_url_with_fallback, process_file,
extract_youtube_transcript, get_youtube_video_info."""

KIND_MARKDOWN_BATCH_DICT = "markdown_batch_dict"
"""Batch dict result shape (``{"results": [...]}`` / ``{"crawled_pages":
[...]}``): deep_crawl_site, search_and_crawl, batch_extract_youtube_transcripts."""

KIND_MARKDOWN_BATCH_LIST = "markdown_batch_list"
"""Batch list result shape (top-level ``List[dict]``): batch_crawl,
multi_url_crawl."""

KIND_STRUCTURED_JSON = "structured_json"
"""Structured extraction output persisted as JSON: intelligent_extract,
extract_entities, extract_structured_data, enhanced_process_large_content."""

KIND_YOUTUBE_COMMENTS = "youtube_comments"
"""extract_youtube_comments — persist as JSON (has nested
``extracted_data.comments``)."""

KIND_SEARCH_JSON = "search_json"
"""search_google / batch_search_google — persist as JSON."""


_JSON_KINDS = {KIND_STRUCTURED_JSON, KIND_YOUTUBE_COMMENTS, KIND_SEARCH_JSON}
_SINGLE_KINDS = {KIND_MARKDOWN_SINGLE} | _JSON_KINDS
_BATCH_KINDS = {KIND_MARKDOWN_BATCH_DICT, KIND_MARKDOWN_BATCH_LIST}


# ---------------------------------------------------------------------------
# Nested strip table
# ---------------------------------------------------------------------------

# Dot/bracket paths of fields to remove from the response copy when
# include_content_in_response=False. The paths are processed with
# :func:`_strip_nested`. ``[*]`` iterates over a list, ``.key`` descends into
# a dict.
_STRIP_PATHS: Dict[str, List[str]] = {
    KIND_MARKDOWN_SINGLE: [
        "content",
        "markdown",
        "cleaned_html",
        "raw_content",
        "archive_contents",
    ],
    KIND_MARKDOWN_BATCH_DICT: [
        "results[*].content",
        "results[*].markdown",
        "results[*].cleaned_html",
        "crawled_pages[*].content",
        "crawled_pages[*].markdown",
        "crawled_pages[*].cleaned_html",
        "batch_results[*].content",
        "batch_results[*].markdown",
        "multi_url_results[*].content",
        "multi_url_results[*].markdown",
    ],
    KIND_MARKDOWN_BATCH_LIST: [
        "[*].content",
        "[*].markdown",
        "[*].cleaned_html",
    ],
    KIND_STRUCTURED_JSON: [
        "content",
        "markdown",
        "cleaned_html",
        "extracted_data.raw_content",
        "table_data",
        "chunks",
        "chunk_summaries",
        "merged_summary",
        "final_summary",
    ],
    KIND_YOUTUBE_COMMENTS: [
        "extracted_data.comments",
    ],
    KIND_SEARCH_JSON: [
        "results",
    ],
}


# ---------------------------------------------------------------------------
# Path and filename helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _validate_absolute_path(path: str) -> Path:
    """Expand ``~`` and return an absolute :class:`pathlib.Path`.

    Raises :class:`ValueError` for relative paths. Most input validation
    happens earlier in :func:`crawl4ai_mcp.validators.validate_output_path`;
    this helper is a defensive re-check inside the writer so bugs surface as
    typed errors rather than silent relative writes.
    """
    if not path:
        raise ValueError("output_path must be a non-empty string")
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError(f"output_path must be absolute, got: {path}")
    return p


def _sanitize_filename_from_url(url: str, max_len: int = 80) -> str:
    """Slugify a URL into a filesystem-safe filename base (no extension).

    Long inputs are truncated and suffixed with a short SHA1 hash so distinct
    URLs that share a truncated prefix do not collide silently.
    """
    if not url:
        return "page"
    parsed = urlparse(url)
    base = f"{parsed.netloc}{parsed.path}".strip("/")
    slug = _SLUG_RE.sub("_", base).strip("_") or "page"
    if len(slug) > max_len:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        slug = slug[: max_len - 9].rstrip("_") + "_" + digest
    return slug


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _frontmatter_md(meta: Dict[str, Any]) -> str:
    """Build a minimal YAML frontmatter block.

    ``None`` values are skipped. Double quotes in values are replaced with
    single quotes and newlines are collapsed to spaces so the frontmatter is
    always valid single-line YAML.
    """
    lines = ["---"]
    for k, v in meta.items():
        if v is None:
            continue
        s = str(v).replace("\n", " ").replace('"', "'")
        lines.append(f'{k}: "{s}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _pick_markdown_body(d: Dict[str, Any]) -> str:
    """Return the best available markdown-style body for ``d``.

    Preference order: ``markdown`` → ``content`` → ``text``. Empty strings
    skip to the next candidate so the caller never gets ``None``.
    """
    for key in ("markdown", "content", "text"):
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Fallback to whichever is present, even if blank.
    return d.get("markdown") or d.get("content") or ""


def _auto_extension_for_kind(kind: str) -> str:
    return ".json" if kind in _JSON_KINDS else ".md"


def _ensure_parent_dir(p: Path) -> None:
    """Create ``p``'s parent directory (or ``p`` itself if treated as a dir)."""
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, data: str, overwrite: bool) -> int:
    """Write ``data`` to ``path`` atomically, honoring ``overwrite``.

    Uses a same-directory :class:`tempfile.NamedTemporaryFile` followed by
    :func:`os.replace`, which is atomic on POSIX and replaces any existing
    file in a single step.

    Returns the number of UTF-8 bytes written.

    Raises :class:`FileExistsError` if ``path`` already exists and
    ``overwrite=False``.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))

    parent = str(path.parent)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)
        tmp_path = None  # successfully renamed
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return len(data.encode("utf-8"))


def _atomic_write_bytes(path: Path, data: bytes, overwrite: bool) -> int:
    """Write ``data`` to ``path`` atomically, honoring ``overwrite``.

    Binary counterpart to :func:`_atomic_write_text`. Used for
    base64-decoded screenshot blobs.

    Returns the number of bytes written.

    Raises :class:`FileExistsError` if ``path`` already exists and
    ``overwrite=False``.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))

    parent = str(path.parent)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return len(data)


# ---------------------------------------------------------------------------
# Single-file writers
# ---------------------------------------------------------------------------


def _write_markdown_single(
    path: Path, item: Dict[str, Any], source_tool: str, overwrite: bool
) -> Tuple[Path, int]:
    body = _pick_markdown_body(item)
    fm = _frontmatter_md(
        {
            "url": item.get("url"),
            "title": item.get("title"),
            "fetched_at": _utcnow_iso(),
            "source_tool": source_tool,
        }
    )
    payload = fm + body
    written_bytes = _atomic_write_text(path, payload, overwrite=overwrite)
    return path, written_bytes


def _write_json_single(
    path: Path, data: Any, overwrite: bool
) -> Tuple[Path, int]:
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    written_bytes = _atomic_write_text(path, payload, overwrite=overwrite)
    return path, written_bytes


# ---------------------------------------------------------------------------
# Batch item probing
# ---------------------------------------------------------------------------


_BATCH_ITEM_KEYS = (
    "results",
    "crawled_pages",
    "batch_results",
    "multi_url_results",
)


def _probe_batch_items(
    result_dict: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Locate the per-item list in a batch-dict result.

    Returns ``(key_name, items)``. When no known container key is present,
    returns ``(None, [])`` so the caller can decide whether to treat the
    whole dict as a single item.
    """
    for key in _BATCH_ITEM_KEYS:
        val = result_dict.get(key)
        if isinstance(val, list):
            return key, [it for it in val if isinstance(it, dict)]
    return None, []


# ---------------------------------------------------------------------------
# Nested strip
# ---------------------------------------------------------------------------


def _parse_strip_path(path: str) -> List[Tuple[str, Optional[str]]]:
    """Parse ``"a.b[*].c"`` → ``[('key','a'),('key','b'),('iter',None),('key','c')]``.

    This is a tiny DSL: ``.key`` descends into a dict, ``[*]`` iterates over
    a list. No other selectors are supported.
    """
    tokens: List[Tuple[str, Optional[str]]] = []
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            i += 1
            continue
        if ch == "[":
            # Expect "[*]"
            end = path.find("]", i)
            if end == -1:
                raise ValueError(f"Unterminated [ in strip path: {path}")
            inner = path[i + 1 : end]
            if inner != "*":
                raise ValueError(
                    f"Only [*] is supported in strip paths, got [{inner}]"
                )
            tokens.append(("iter", None))
            i = end + 1
            continue
        # Plain key token, read until next '.' or '['
        j = i
        while j < len(path) and path[j] not in ".[":
            j += 1
        tokens.append(("key", path[i:j]))
        i = j
    return tokens


def _strip_nested(obj: Any, paths: List[str]) -> Any:
    """Delete fields identified by dot/bracket paths from ``obj`` in-place.

    Examples
    --------
    ``_strip_nested(d, ["content", "extracted_data.comments"])`` removes
    ``d["content"]`` and ``d["extracted_data"]["comments"]``.

    ``_strip_nested(lst, ["[*].markdown"])`` removes ``markdown`` from every
    dict element of ``lst``.

    Missing intermediate keys or type mismatches are silent no-ops — the
    intent is "remove if present", not "assert shape".
    """
    for path in paths:
        tokens = _parse_strip_path(path)
        _apply_strip(obj, tokens)
    return obj


def _apply_strip(obj: Any, tokens: List[Tuple[str, Optional[str]]]) -> None:
    if not tokens:
        return
    head, rest = tokens[0], tokens[1:]
    kind, value = head
    if kind == "iter":
        if isinstance(obj, list):
            for item in obj:
                _apply_strip(item, rest)
        return
    # kind == "key"
    if not isinstance(obj, dict):
        return
    if not rest:
        obj.pop(value, None)
        return
    # Descend, but only if the child exists
    if value in obj:
        _apply_strip(obj[value], rest)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _error_response(code: str, message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error_code": code,
        "error": message,
    }


def _write_single(
    result_dict: Dict[str, Any],
    output_path: str,
    tool_kind: str,
    source_tool: str,
    overwrite: bool,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Write a single-file tool result. Returns ``(written, warnings)``."""
    p = _validate_absolute_path(output_path)
    warnings: List[str] = []
    auto_ext = _auto_extension_for_kind(tool_kind)

    if p.suffix == "":
        p = p.with_suffix(auto_ext)
    elif p.suffix != auto_ext:
        warnings.append(
            f"Explicit extension '{p.suffix}' differs from tool default "
            f"'{auto_ext}'; honoring caller's choice."
        )

    _ensure_parent_dir(p)

    if tool_kind == KIND_MARKDOWN_SINGLE:
        written = [_write_markdown_single(p, result_dict, source_tool, overwrite)]
    else:
        written = [_write_json_single(p, result_dict, overwrite)]
    return written, warnings


def _prepare_batch_directory(output_path: str, source_tool: str) -> Path:
    """Resolve and prepare the directory for a batch writer.

    Policy (replaces the old brittle ``Path.suffix`` check):
    - If the path already exists, it MUST be a directory. A file at that
      location is rejected — we never write batch output into or on top of a
      regular file.
    - If the path does not exist yet, treat it as a directory and create it
      with ``parents=True``. This allows ``/tmp/run.v1`` or any other
      ``.``-containing directory name.
    """
    d = _validate_absolute_path(output_path)
    if d.exists():
        if not d.is_dir():
            raise ValueError(
                f"Batch output_path exists but is not a directory "
                f"({source_tool}): {output_path}"
            )
    else:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _write_batch_dict(
    result_dict: Dict[str, Any],
    output_path: str,
    source_tool: str,
    overwrite: bool,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Write a batch-dict tool result as per-item .md files + index.json.

    Items reporting ``success=False`` are NOT written to disk — they still
    appear in ``index.json`` with ``file: null`` so callers can reason
    about which URLs were attempted. This matches the batch-list policy.
    """
    d = _prepare_batch_directory(output_path, source_tool)

    items_key, items = _probe_batch_items(result_dict)
    if items_key is None or not items:
        raise ValueError(
            f"No batch items found in result for {source_tool}. "
            f"Expected one of {_BATCH_ITEM_KEYS}."
        )

    written: List[Tuple[Path, int]] = []
    seen: Dict[str, int] = {}
    index_entries: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        url = item.get("url") or f"item_{idx}"
        item_success = item.get("success", True)
        if not item_success:
            # Record the failure in the index but do not create a .md file
            # — an empty/partial markdown would misleadingly look like a
            # captured page.
            index_entries.append(
                {
                    "url": url,
                    "title": item.get("title"),
                    "success": False,
                    "error": item.get("error"),
                    "file": None,
                }
            )
            continue
        slug = _sanitize_filename_from_url(url)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        name = f"{slug}.md" if n == 0 else f"{slug}_{n}.md"
        p = d / name
        file_entry = _write_markdown_single(p, item, source_tool, overwrite)
        written.append(file_entry)
        index_entries.append(
            {
                "url": url,
                "title": item.get("title"),
                "success": True,
                "file": str(p),
            }
        )

    # ``count`` here means "number of files actually written", matching the
    # batch-list writer. ``items`` keeps the full attempt list so callers can
    # see what was tried and which entries failed.
    index_doc: Dict[str, Any] = {
        "source_tool": source_tool,
        "generated_at": _utcnow_iso(),
        "items_key": items_key,
        "count": len(written),
        "items": index_entries,
    }
    # search_and_crawl carries a separate top-level search_results list;
    # preserve it in the index so the on-disk record is self-contained.
    if "search_results" in result_dict and isinstance(
        result_dict.get("search_results"), list
    ):
        index_doc["search_results"] = result_dict["search_results"]

    index_path = d / "index.json"
    written.append(_write_json_single(index_path, index_doc, overwrite))
    return written, []


def _write_batch_list(
    result_list: List[Dict[str, Any]],
    output_path: str,
    source_tool: str,
    overwrite: bool,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Write a batch-list tool result as per-item files + index.json.

    Mutates ``result_list`` in place to attach ``output_file`` on every
    successful item. The caller's returned list gets the updated entries.
    """
    d = _prepare_batch_directory(output_path, source_tool)

    written: List[Tuple[Path, int]] = []
    seen: Dict[str, int] = {}
    for idx, item in enumerate(result_list):
        if not isinstance(item, dict):
            continue
        if not item.get("success", True):
            continue
        url = item.get("url") or f"item_{idx}"
        slug = _sanitize_filename_from_url(url)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        name = f"{slug}.md" if n == 0 else f"{slug}_{n}.md"
        p = d / name
        file_entry = _write_markdown_single(p, item, source_tool, overwrite)
        written.append(file_entry)
        item["output_file"] = str(p)

    index_path = d / "index.json"
    index_doc = {
        "source_tool": source_tool,
        "generated_at": _utcnow_iso(),
        "count": len(written),
        "items": [
            {
                "url": it.get("url") if isinstance(it, dict) else None,
                "success": it.get("success", True) if isinstance(it, dict) else False,
                "output_file": it.get("output_file") if isinstance(it, dict) else None,
            }
            for it in result_list
        ],
    }
    written.append(_write_json_single(index_path, index_doc, overwrite))
    return written, []


# ---------------------------------------------------------------------------
# Warnings helper
# ---------------------------------------------------------------------------


def _append_warning(result: Dict[str, Any], message: str) -> None:
    """Append ``message`` to ``result["warnings"]``, normalizing the field."""
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = [warnings] if warnings else []
    warnings.append(message)
    result["warnings"] = warnings


# ---------------------------------------------------------------------------
# Screenshot persistence
# ---------------------------------------------------------------------------


def handle_screenshot_persistence(
    result: Dict[str, Any],
    *,
    output_path: Optional[str],
    overwrite: bool,
    source_tool: str,
) -> Dict[str, Any]:
    """Move a base64 screenshot out of the response and onto disk (or drop).

    The base64 PNG blob in ``result["screenshot"]`` can be ~50k tokens on
    its own, which would either exceed the apply_token_limit budget or
    force special-casing inside it. Instead, this helper converts the
    blob into either a sibling file plus a ``screenshot_path`` reference
    (when ``output_path`` is set) or removes the blob entirely with a
    warning explaining how to retrieve it (when ``output_path`` is not
    set). Either way, the response copy returned to the caller never
    carries the base64, so the token-limit guarantee is preserved.

    The sibling file is written next to the main ``output_path`` artifact
    with the same stem and a ``_screenshot.png`` suffix (e.g.
    ``/tmp/run.json`` → ``/tmp/run_screenshot.png``). If the caller
    omitted the extension on ``output_path``, the screenshot is still
    written next to the inferred main artifact.

    Mutates and returns ``result``. A no-op when no screenshot is
    present (None, empty string, or absent).
    """
    screenshot_b64 = result.get("screenshot")
    if not screenshot_b64:
        return result

    if not output_path:
        # No persistence target — drop the blob and surface a warning so
        # the caller knows the screenshot was generated but not returned.
        result.pop("screenshot", None)
        _append_warning(
            result,
            "Screenshot was generated but omitted from the response to "
            "stay within the MCP token limit. Re-issue the call with "
            "`output_path` set to persist the screenshot to disk; the "
            "response will then include `screenshot_path`.",
        )
        return result

    try:
        main_path = _validate_absolute_path(output_path)
    except ValueError:
        # Bad output_path — let finalize_tool_response report it via its
        # own validation path. Drop the blob defensively so we never
        # leak the base64 into the response.
        result.pop("screenshot", None)
        return result

    # Build sibling path: <stem>_screenshot.png alongside the main artifact.
    if main_path.suffix == "":
        # Caller omitted extension; finalize_tool_response will add one
        # later. The screenshot still lives next to the main file by
        # stem, so use the path as-is for stem derivation.
        screenshot_path = main_path.with_name(main_path.name + "_screenshot.png")
    else:
        screenshot_path = main_path.with_name(main_path.stem + "_screenshot.png")

    try:
        # validate=True rejects junk characters; default (False) silently
        # strips them and would let a clearly-malformed input through.
        png_bytes = base64.b64decode(screenshot_b64, validate=True)
    except (binascii.Error, ValueError):
        # Malformed base64 — drop the blob, warn, don't raise.
        result.pop("screenshot", None)
        _append_warning(
            result,
            "Screenshot field was not valid base64; omitted from response.",
        )
        return result

    try:
        _ensure_parent_dir(screenshot_path)
        bytes_written = _atomic_write_bytes(screenshot_path, png_bytes, overwrite=overwrite)
    except FileExistsError:
        result.pop("screenshot", None)
        _append_warning(
            result,
            f"Screenshot file already exists at {screenshot_path} and "
            f"overwrite=False; not overwritten. The base64 has been "
            f"omitted from the response.",
        )
        return result
    except OSError as exc:
        result.pop("screenshot", None)
        _append_warning(result, f"Screenshot write failed: {exc}")
        return result

    # Replace the blob with a path reference.
    result.pop("screenshot", None)
    result["screenshot_path"] = str(screenshot_path)
    result["screenshot_bytes"] = bytes_written
    # Tag the source tool so multi-step pipelines can attribute the file.
    if source_tool:
        result.setdefault("screenshot_source_tool", source_tool)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def finalize_tool_response(
    result: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    output_path: Optional[str],
    include_content_in_response: bool,
    overwrite: bool,
    tool_kind: str,
    source_tool: str,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Persist ``result`` to disk (if requested) and return a slim response.

    This is the single entry point wired into every info-gathering tool.
    When ``output_path`` is ``None``, the result is returned unchanged —
    persistence is strictly opt-in. When the result reports failure, no file
    is written either, so error dicts pass through untouched.

    Parameters
    ----------
    result
        The tool's successful result. For most tools this is a dict; for
        ``batch_crawl`` / ``multi_url_crawl`` it is a ``List[dict]``.
    output_path
        Caller-supplied absolute path. Single-file tools expect a file path
        (extension inferred if missing); batch tools expect a directory
        path.
    include_content_in_response
        When ``True``, skip stripping large fields from the response copy so
        the caller receives both the file *and* the full content inline.
    overwrite
        When ``True``, silently replace existing files via atomic rename.
        Otherwise raises an ``output_path_exists`` error.
    tool_kind
        One of the ``KIND_*`` constants; drives dispatch and the nested
        strip table.
    source_tool
        Name of the calling MCP tool — recorded in frontmatter and index
        files so the on-disk artifact is self-describing.
    """
    # Treat both None and "" as "no persistence requested" to stay in sync
    # with :func:`crawl4ai_mcp.validators.validate_output_path`, which passes
    # an empty string through as a no-op. Any other falsy value still trips
    # the ``_validate_absolute_path`` check below.
    if output_path is None or output_path == "":
        return result

    # Success gate: never persist failed results.
    is_list = isinstance(result, list)
    if is_list:
        # An all-empty or all-failed list should pass through unchanged.
        if not result or all(
            not (isinstance(it, dict) and it.get("success", True)) for it in result
        ):
            return result
    else:
        if not isinstance(result, dict) or not result.get("success", True):
            return result

    def _list_error(code: str, message: str) -> List[Dict[str, Any]]:
        # Preserve list shape when the caller's tool returns a list.
        return [_error_response(code, message)]

    if tool_kind not in _STRIP_PATHS:
        err = _error_response(
            "invalid_tool_kind", f"Unknown tool_kind: {tool_kind!r}"
        )
        return [err] if is_list else err

    # Token count measured against the pre-persist payload so the caller can
    # see how much would have been sent without persistence.
    try:
        original_tokens = estimate_tokens(
            json.dumps(result, ensure_ascii=False, default=str)
        )
    except (TypeError, ValueError):
        original_tokens = -1

    # Dispatch to the appropriate writer.
    try:
        if tool_kind in _SINGLE_KINDS:
            if is_list:
                # Programming error: single-kind caller passed a list.
                return _list_error(
                    "invalid_shape",
                    f"{tool_kind} expects a dict result, got list",
                )
            written, warnings = _write_single(
                result, output_path, tool_kind, source_tool, overwrite
            )
        elif tool_kind == KIND_MARKDOWN_BATCH_DICT:
            if is_list:
                return _list_error(
                    "invalid_shape",
                    f"{tool_kind} expects a dict result, got list",
                )
            written, warnings = _write_batch_dict(
                result, output_path, source_tool, overwrite
            )
        elif tool_kind == KIND_MARKDOWN_BATCH_LIST:
            if not is_list:
                return _error_response(
                    "invalid_shape",
                    f"{tool_kind} expects a list result, got dict",
                )
            written, warnings = _write_batch_list(
                result, output_path, source_tool, overwrite
            )
        else:
            err = _error_response(
                "invalid_tool_kind", f"Unknown tool_kind: {tool_kind!r}"
            )
            return [err] if is_list else err
    except FileExistsError as exc:
        msg = f"Output file exists and overwrite=False: {exc}"
        return _list_error("output_path_exists", msg) if is_list else _error_response("output_path_exists", msg)
    except ValueError as exc:
        return _list_error("invalid_output_path", str(exc)) if is_list else _error_response("invalid_output_path", str(exc))
    except OSError as exc:
        msg = f"Write failed: {exc}"
        return _list_error("output_write_failed", msg) if is_list else _error_response("output_write_failed", msg)

    total_bytes = sum(n for _, n in written)
    output_files = [str(p) for p, _ in written]

    # Build response copy. For list shape we mutate in place (each item
    # already has output_file attached) and return. For dict shape we deep
    # copy, strip, and attach metadata.
    if is_list:
        if not include_content_in_response:
            _strip_nested(result, _STRIP_PATHS[tool_kind])
        # List shape has nowhere to put top-level metadata. The index.json
        # file acts as the container for global info; per-item output_file
        # entries already give the LLM enough to read each file.
        return result

    if include_content_in_response:
        response: Dict[str, Any] = dict(result)
    else:
        response = copy.deepcopy(result)
        _strip_nested(response, _STRIP_PATHS[tool_kind])

    response.update(
        {
            "persisted": True,
            "output_files": output_files,
            "output_bytes": total_bytes,
            "original_tokens": original_tokens,
            "content_included_in_response": include_content_in_response,
        }
    )
    for w in warnings:
        _append_warning(response, w)
    return response
