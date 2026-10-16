# Changelog

## [Unreleased]

### Added
- SearXNG engine (`searxng`) for a self-hosted [SearXNG](https://docs.searxng.org/) instance, with no API key. `SEARXNG_URL` configures it (an `http://` or `https://` URL; other values are ignored with a warning), `SEARXNG_ENGINES` picks the SearXNG engines to query and `SEARXNG_TIMEOUT` sets the request timeout (default 30 seconds, at most 120). It is last in the default `SEARCH_ENGINE_ORDER`, so `auto` uses it only when it is configured and the other engines fail or are not configured. Genre operators are added to the query, `recent_days` becomes the SearXNG `time_range` that covers it (returned as `time_range`), and up to 5 result pages are read to reach `num_results`. Results carry `source_engines` and `score`, and engines behind SearXNG that failed are listed in `unresponsive_engines`. When all of them fail, the result is a failure that `auto` moves past instead of zero results. (#7)

## [0.4.0] - 2026-10-10

### Added
- `result_type` request key for `search_web`, `batch_search_web` and `search_and_crawl`: `web` (default), `discussions`, `faq`, `videos` and `news`, alone or several as a list or a comma-separated string. `discussions`, `faq` and `videos` are Brave web search sections fetched in one request, and `news` is a separate Brave News Search request. `discussions` returns forum posts with `forum_name`, `num_answers`, `score`, `question` and `top_comment`, and `faq` returns question and answer excerpts. `results` holds web results when `web` is requested, otherwise the first type; the other types go to `extra_results`. Only the `brave` engine supports them (`auto` uses Brave alone; other engines return `unsupported_result_type`), and Brave returns them for some queries only.
- Web search through the Brave Search API (`BRAVE_SEARCH_API_KEY`) and Exa (`EXA_API_KEY`), alongside the Google Custom Search JSON API (`GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`).
- `SEARCH_ENGINE` (`auto`, `google_custom`, `brave`, `exa`; default `auto`) picks the default engine, and the `engine` request key of `search_web`, `batch_search_web` and `search_and_crawl` overrides it for one call. `auto` tries the engines that have API keys in the order of `SEARCH_ENGINE_ORDER` (default `google_custom,brave,exa`) and moves to the next one on errors, not on zero results. Responses carry `engine`, and `fallback_info.attempts` when an engine was skipped.
- `search_web` and `batch_search_web` tools. `search_google` and `batch_search_google` remain as deprecated aliases with the same request keys and results.
- `get_search_genres` reports the engine configuration (`default_engine`, `auto_order`, `configured_engines`, `genre_support`).

### Changed
- `recent_days` is passed to each engine's date parameter (`dateRestrict`, `freshness`, `startPublishedDate`) instead of appending `after:YYYY-MM-DD` to the query. For Brave, 1, 7, 31 and 365 use the rolling windows `pd`, `pw`, `pm` and `py`; other values become a UTC date range.
- `search_and_crawl` sends the query as written. It previously rewrote multi-word queries to `word1 OR word2 OR ...`.
- Search results no longer fetch each result page to read its title and description; the snippet comes from the engine.
- An unknown `engine` value returns `error_code: invalid_search_engine`, and a named engine without an API key returns `error_code: search_engine_not_configured`, both before any network request. Exa has no file type filter: naming `exa` with the `pdf`, `documents`, `presentations` or `spreadsheets` genre returns `error_code: unsupported_search_genre`, and `auto` skips Exa for those genres.
- `GOOGLE_SEARCH_MODE` is read only when `SEARCH_ENGINE` is unset. `custom_search_only` means `google_custom`; `hybrid` and `googlesearch_only` mean `auto`.

### Fixed
- `search_and_crawl` with `output_path` failed with "No batch items found", and failed pages were never crawled again or truncated by `max_content_per_page`. The core returned the pages under `crawled_content` while the tool layer and persistence read `crawled_pages`. The response now has `crawled_pages`, one flat entry per page (`search_rank`, `url`, `title`, `success`, `markdown` or `content`, `error`, `search_result`), and the `crawled_content` key is removed.

### Removed
- Google results scraping with googlesearch-python, which returns no results now that Google requires JavaScript. `GOOGLESEARCH_PYTHON_RPM`, `GOOGLE_SEARCH_MAX_RETRIES` and `GOOGLE_SEARCH_FALLBACK_DELAY` are no longer read.

### Dependencies
- Removed `googlesearch-python`.

## [0.3.4] - 2026-10-04

### Added
- `get_youtube_video_info`, `extract_youtube_transcript` and `batch_extract_youtube_transcripts` accept a `timezone` parameter (IANA name, default `UTC`). The full publish timestamp is returned under `published_at` (`metadata.published_at` for transcripts), converted to that timezone, with the original YouTube offset kept in `published_at.original`.
- Fallback crawler gains a last-resort stage that converts the statically fetched HTML to markdown when every browser stage fails.
- `deep_crawl_site` responses include `crawled_pages`, one entry per page with `url`, `title`, `success` and `error`.
- Server logging can be turned on with `CRAWL4AI_MCP_LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING` or `ERROR`). Records go to stderr only. Logging stays off by default. The variable is read before `.env` is loaded, so set it in the process environment, for example the MCP client's `env` block.
- `extract_entities` accepts singular type names (`email`, `phone`, `person` and so on) and maps them to the plural names, ignoring letter case. `entity_types_requested` shows the normalized names.

### Changed
- Pages over 400,000 characters are no longer summarized with an LLM unless `auto_summarize=True`. Previously this ran implicitly, used LLM API credits, and caused `output_path` to save the summary instead of the full content. Large pages are now persisted in full and the response is truncated by the existing token limit, with a recommendation to use `output_path`.
- `crawl_url_with_fallback` responses are now capped at 25000 tokens, the same limit as `crawl_url`; previously they could exceed the MCP output limit.
- Remote `.html` / `.htm` URLs are rendered in the browser instead of being downloaded and converted with MarkItDown, so JS-rendered content, `css_selector` and `take_screenshot` now apply to them. Only document extensions and known file URL patterns are routed to file processing; local files and `process_file` are unchanged.
- Content filtering uses crawl4ai's `PruningContentFilterLXML`, which produces near-identical output and is substantially faster.
- An unknown `search_genre` in `search_google`, `batch_search_google` or `search_and_crawl` is now an error (`error_code: invalid_search_genre`, with `valid_genres`), returned before any network request. Previously the value was ignored and the search ran unfiltered. Valid genres are `pdf`, `documents`, `presentations`, `spreadsheets`, `japanese` and `english`, matched case-insensitively.
- The `get_search_genres` response shape changed. It returns `genres`, `query_operators`, `usage_example`, `configuration` and `notes`. The `categories`, `usage_examples` and `tips` keys are removed; they listed genres that never changed the query.
- `deep_crawl_site` now enforces `max_depth` 1-2 and `max_pages` 1-10. Out-of-range values, which were previously passed through, return an error (`invalid_depth`, `depth_too_large`, `invalid_max_pages`, `max_pages_too_large`) before any page is fetched.
- `extract_entities` in regex mode lists requested types that have no pattern under `unsupported_entity_types` and adds a `warnings` entry. Previously they were dropped silently. `custom_patterns` applies in regex mode only.
- `python -m crawl4ai_mcp.server` parses its arguments with argparse: `--transport {stdio,streamable-http,http,sse}`, `--host`, `--port`. `--help` lists them, and an invalid argument exits with status 2 and an error on stderr.
- `scripts/run_http_server.sh` starts the built-in streamable-http server at `/mcp`. `scripts/setup.sh` and `scripts/setup_windows.bat` also install Playwright's Chromium and print working next steps.
- The docker-compose HTTP service's healthcheck probes `/mcp` instead of the non-existent `/health` route.
- The `multi_url_crawl` `max_concurrent` description now says it has no effect. URLs are processed sequentially, and the parameter is kept for compatibility.

### Fixed
- Over the stdio transport, stdout is reserved for MCP messages. Progress text and other writes to stdout (including from C extensions and child processes) are redirected to stderr, so clients no longer receive invalid or bursty lines.
- Disable crawl4ai's legacy cache migration, which is not idempotent: on a crawl4ai version change it re-hashed stored hashes (cached results came back as hash strings) and rewrote `~/.crawl4ai/crawl4ai.db` regardless of `CRAWL4_AI_BASE_DIRECTORY`. Schema updates still run.
- Fallback crawler no longer discards normal pages as block pages. Detection now matches specific challenge phrases against visible text only and never flags pages with more than 2000 characters of text.
- Document URLs that fail file processing no longer trigger the browser fallback, and `crawl_url_with_fallback` processes remote documents once.
- File processing declares UTF-8 to MarkItDown when the file decodes as UTF-8, fixing garbled text when non-ASCII characters appear after the first 4 KiB.
- File downloads send browser-like headers, avoiding 403 responses to the default python-requests User-Agent.
- PDFs without a text layer return a clear error instead of an empty success.
- `deep_crawl_site` with `output_path` failed with "No batch items found". It now writes one `.md` per successful page plus `index.json`. Unless `include_content_in_response=true`, the merged `markdown`/`content` and the page bodies are left out of the response. Without `output_path`, `crawled_pages` carries metadata only.
- The fallback stages of `crawl_url` and `crawl_url_with_fallback` (static fast path, browser stages, AMP) no longer accept HTTP 4xx/5xx responses as success, and the AMP stage also rejects error pages. Previously a 404 page from `<url>/amp` could be returned as the content.

### Removed
- The unmaintained demo HTTP servers in `examples/` and `scripts/start_pure_http_server.sh`. Use `--transport streamable-http` or `scripts/run_http_server.sh`.

### Dependencies
- Bump crawl4ai to `>=0.9.4,<0.10`.
- Declare `anthropic>=0.40.0` as a dependency, so Anthropic summarization works in UVX installs.

## [0.3.3] - 2026-06-06

### Fixed
- Fix `AttributeError: 'NoneType' object has no attribute 'strip'` when crawl4ai returns content fields whose value is `None`. `(value or "")` is now used before `.strip()` in `middleware/response_transform.py` and `server_helpers.py` (#24, #25).
- Harden `search_and_crawl` in `server_tools/search_tools.py`: failed-page detection no longer crashes on `None` content and now accepts markdown-only pages instead of treating them as failures; content truncation guards `len()` against `None` in both the main and fallback paths.

### Internal
- Resolve `__version__` dynamically from `pyproject.toml` (with `importlib.metadata` fallback for built wheels) so the reported version no longer drifts from the released one.

## [0.3.2] - 2026-05-17

### Fixed
- Fix `enhanced_process_large_content` crash when `content` is `None` (read the `markdown` field first).
- Surface YouTube Restricted Mode as `success=True` with a structured warning instead of a cryptic error.
- `extract_youtube_comments` returns a warning (also propagated to the top-level `warnings` list) when no comments are retrieved, keeping `success=true` and `total_comments: 0` (#17).

### Changed
- Screenshots from `crawl_url` / `crawl_url_with_fallback` no longer appear as base64 in the response. With `output_path` set, the image is written to a sibling PNG file and the response carries `screenshot_path`; without it, the screenshot is dropped with a warning so the token limit holds (#20).
- Add an upper bound to the markitdown dependency (`<0.2`).

### Security
- Pin `urllib3>=2.7.0` to resolve CVE-2026-44431 and CVE-2026-44432.

### Dependencies
- Bump markitdown to 0.1.5 with the `[pdf]` extra (replaces the separate pdfminer-six pin).
- Bump crawl4ai to `>=0.8.0,<0.9` (vulnerability fix).
- Bump mammoth to `>=1.11.0` (vulnerability fix).
- Sync `requirements.txt` with `pyproject.toml`.

## [0.3.1] - 2026-04-29

### Added
- Support local file processing via `file://` URIs and absolute paths.
- Add `is_file_uri`, `is_local_path`, and `file_uri_to_local_path` validators.
- `process_file` now accepts local file paths in addition to URLs.

### Fixed
- Normalize the `CRAWL4AI_BROWSER_TYPE` env var with `strip().lower()` and use it to override the default browser list.

### Security
- Add a minimum version constraint for litellm (`>=1.83.7`) and exclude the compromised v1.82.7 / v1.82.8 releases.
- Resolve known critical vulnerabilities: CVE-2026-35029, CVE-2026-35030, GHSA-69x8-hrgq-fjj8.

## [0.3.0] - 2026-04-11

### Added
- File persistence option for information-gathering MCP tools. Supply `output_path` (plus optional `include_content_in_response` and `overwrite`) to have the tool write full fetched content to disk as markdown or JSON and return a slim metadata-only response that bypasses the 25000-token cap. Supported on: `crawl_url`, `crawl_url_with_fallback`, `deep_crawl_site`, `batch_crawl`, `multi_url_crawl`, `process_file`, `enhanced_process_large_content`, `extract_youtube_transcript`, `batch_extract_youtube_transcripts`, `get_youtube_video_info`, `extract_youtube_comments`, `intelligent_extract`, `extract_entities`, `extract_structured_data`, `search_google`, `batch_search_google`, `search_and_crawl`. Single-file tools accept a file path (auto-determined `.md`/`.json` extension when omitted); batch tools accept a directory path and emit per-URL files plus `index.json`.
- `validate_output_path` validator: absolute path required, NUL rejection, optional existing-file guard for single-file tools.
- New `crawl4ai_mcp/middleware/file_persistence.py` module with atomic writes (`tempfile` + `os.replace`) and nested-path field stripping.
- `readOnlyHint` annotations added to all MCP tools

### Changed
- `batch_crawl` switched to sequential execution with revised timeout design
- LLM summarization skipped when pagination is active
- `crawl_url`: persistence happens BEFORE `_apply_content_slicing`, so the on-disk copy holds the full unsliced content even when the caller also supplies `content_limit`/`content_offset`. The `_should_trigger_fallback` check is now evaluated against the unsliced markdown, which prevents a large `content_offset` from spuriously triggering the undetected-browser fallback.
- `batch_crawl` / `multi_url_crawl` preserve their historical `list` return shape when `output_path` is set — each successful item simply gains an `output_file` key. An aggregate `index.json` is written in the target directory.
- Request-dict tools (`search_google`, `batch_search_google`, `search_and_crawl`, `batch_extract_youtube_transcripts`) read persistence options (`output_path`, `include_content_in_response`, `overwrite`) directly from the request dict.
- Batch tools (`deep_crawl_site`, `search_and_crawl`, `batch_extract_youtube_transcripts`, `batch_crawl`, `multi_url_crawl`) now accept any non-existent `output_path` as a directory, including names containing `.` (`/tmp/run.v1`). The previous `Path.suffix`-based rejection is gone. Existing regular files are still rejected.
- Batch-dict tools (`deep_crawl_site`, `search_and_crawl`, `batch_extract_youtube_transcripts`) skip per-item persistence for items that report `success=False`; failed items still appear in `index.json` with `file: null` so callers can reason about the attempt list. This matches the existing `batch_crawl` / `multi_url_crawl` behavior.
- `finalize_tool_response` treats `output_path=""` as a no-op, matching `validate_output_path`'s empty-string acceptance.

### Fixed
- `deep_crawl_site`: convert CrawlResponse to dict before calling `.get()` (community PR #18)
- `extract_structured_data`: convert CrawlResponse to dict and request HTML (community PR #16)
- `batch_crawl`: wrap remaining CrawlResponse returns with `_convert_result_to_dict`
- Content offset propagated through CrawlRequest for pagination cache protection
- Missing `middleware/module_check.py` added to the package

### Known caveats
- Tools with `readOnlyHint` annotation keep that hint even when `output_path` is supplied and the tool writes to disk. Callers should treat `output_path` as an explicit opt-in to filesystem writes.
- `overwrite=False` is checked with `Path.exists()` immediately before the atomic rename, which protects against most accidental overwrites but is not TOCTOU-race-safe against a concurrent writer. The atomic `os.replace` call itself is still atomic at the filesystem level. For stricter single-writer guarantees, pair this feature with a lock at the caller level.
- Batch persistence is per-file atomic but not transactional across items. A failure partway through will leave a partial directory (some `.md` files present, `index.json` potentially missing).

---

## [0.2.0] - 2026-03-01

### Added
- New `extract_youtube_comments` MCP tool with pagination support
- Content pagination support via `content_offset` and `content_limit` parameters across tools
- YouTube transcript fallback to `crawl_url` when the transcript API fails, plus metadata enrichment (upload date, view count, channel name)
- Content-Type detection for URLs without file extension
- `crawl_url` recommendation for long YouTube videos

### Changed
- **Breaking:** YouTube tools return a unified `content` / `markdown` / `extracted_data` response format. The `segments` array was removed and `include_timestamps` now defaults to `false`.
- Architecture refactored: codebase modularized into `server_tools/`, `core/`, `infra/`, `middleware/`, `processors/` layers
- `batch_crawl` limit updated from 5 to 3

### Fixed
- ReDoS protection improved with multiprocessing timeout and strict size limits
- Missing `anthropic` dependency added
- `auth_token` is injected into the Authorization header, cookies are normalized to Playwright format, block pages are detected, and size limits apply to all fallback response paths
- Remove the unsupported `xpath` parameter that caused a TypeError in `crawl_url`
- Search cache key includes `safe_search` and normalized defaults to prevent incorrect cache hits; `content_limit` / `content_offset` are type-coerced and negative values rejected
- YouTube tools no longer raise a NoneType error with `include_metadata=False`

---

## [0.1.7] - 2026-01-12

### Added
- Batch tools restored as MCP tools with rate limits: `batch_crawl`, `multi_url_crawl`, `batch_search_google`, `batch_extract_youtube_transcripts`
- Ollama and Anthropic providers for LLM summarization

### Changed
- Token counting improved using tiktoken for accurate counts
- Tool descriptions reduced to lower token usage
- Low-utility MCP tools hidden to reduce token usage

### Fixed
- Token limit handling improved to preserve more content
- Partial transcript data preserved on token overflow
- `extract_media` Pydantic validation error
- Docker image compatibility with Debian trixie

### Dependencies
- Bump crawl4ai to 0.7.8 and fastmcp to `>=2.14.2,<3`
- Bump markitdown to 0.1.3, youtube-transcript-api to 1.2.3 and playwright to 1.55.0

---

## [0.1.6] - 2025-10-18

Note: Batch tools removed in 0.1.6 were restored in 0.1.7.

### Added
- **Markdown-Only Response**: New `include_cleaned_html` parameter for crawling tools
  - Default behavior: Returns markdown content only to reduce token usage (~50% reduction)
  - Set `include_cleaned_html=True` to also receive cleaned HTML content

### Changed
- **Token Limit Optimization**: Increased response token limit from 20000 to 25000 for all crawling tools
  - Affected tools: `crawl_url`, `deep_crawl_site`, `crawl_url_with_fallback`, `intelligent_extract`, `extract_entities`, `extract_structured_data`, `search_and_crawl`
- **Token Limit Fallback**: Enhanced token limit fallback with detailed warnings and actionable recommendations for reducing token usage
- **Removed Batch Tools from MCP Interface**: `batch_extract_youtube_transcripts`, `batch_search_google`, `batch_crawl`, `multi_url_crawl` are no longer exposed as MCP tools (21 to 17 tools). The functions remain available internally.
- Require Python >= 3.11

---

## [0.1.5] - 2025-09-28

### Fixed
- Compatibility with FastMCP 2.11.0
- `batch_crawl` fallback handles `CrawlResponse` objects as well as dicts

### Changed
- `batch_crawl` total timeout shortened to base timeout + 10 seconds per URL

### Dependencies
- Add `python-dateutil`

---

## [0.1.4] - 2025-09-21

### Added
- Automatic fallback for all crawling and extraction tools, with `fallback_used` flags and original error tracking in responses
- Undetected browser support, LLM table extraction and multi-URL configuration (crawl4ai 0.7.4 features)
- `recent_days` date range filtering for search
- Dockerfile and docker-compose configuration

### Changed
- Search API parameters standardized across tools

### Removed
- DXT package distribution
- Deprecated `recent` search genre

### Fixed
- MCP response token limit safeguards
- `YouTubeTranscriptResponse is not JSON serializable` error
- File processing JSON serialization issues
- `CrawlResponse` dict conversion issues

### Dependencies
- Bump crawl4ai from 0.7.2 to 0.7.4 in `requirements.txt`

---

## [0.1.2] - 2025-08-23

### Added
- Token-aware summarization (`target_tokens`) for web, file and YouTube content, with preserved metadata
- `crawl_url_with_fallback` with multiple header, user agent and mobile strategies

### Changed
- Upgrade to FastMCP 2.0.0 for cleaner STDIO communication

### Fixed
- Config loading when the server is launched (for example via uvx) from a directory other than the project root
- Restore all 21 MCP tools and the detailed parameter descriptions lost in the FastMCP 2.0 upgrade
- CSS extraction fallback for `extract_structured_data`, timeout fallback for `enhanced_process_large_content`, and response truncation for `search_and_crawl`
- Parameter validation in `batch_crawl`

---

## [0.1.1] - 2025-08-16

### Fixed
- **Relative Import Errors**: Resolved `attempted relative import beyond top-level package` errors in `batch_crawl` and `extract_structured_data` tools
- **Import Statements**: Replaced relative imports (`..models`) with absolute imports (`crawl4ai_mcp.models`) for better reliability

### Notes
- No migration required for existing users
- All MCP tools maintain backward compatibility

---

## [0.1.0] - 2025-08-06

### Added
- Full MCP server implementation with 21 tools
- Complete web crawling, YouTube, search, and file processing capabilities
- FastMCP integration with modern async architecture
