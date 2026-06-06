# Changelog

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

### Changed
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

### Changed
- `crawl_url`: persistence happens BEFORE `_apply_content_slicing`, so the on-disk copy holds the full unsliced content even when the caller also supplies `content_limit`/`content_offset`. The `_should_trigger_fallback` check is now evaluated against the unsliced markdown, which prevents a large `content_offset` from spuriously triggering the undetected-browser fallback.
- `batch_crawl` / `multi_url_crawl` preserve their historical `list` return shape when `output_path` is set — each successful item simply gains an `output_file` key. An aggregate `index.json` is written in the target directory.
- Request-dict tools (`search_google`, `batch_search_google`, `search_and_crawl`, `batch_extract_youtube_transcripts`) read persistence options (`output_path`, `include_content_in_response`, `overwrite`) directly from the request dict.
- Batch tools (`deep_crawl_site`, `search_and_crawl`, `batch_extract_youtube_transcripts`, `batch_crawl`, `multi_url_crawl`) now accept any non-existent `output_path` as a directory, including names containing `.` (`/tmp/run.v1`). The previous `Path.suffix`-based rejection is gone. Existing regular files are still rejected.
- Batch-dict tools (`deep_crawl_site`, `search_and_crawl`, `batch_extract_youtube_transcripts`) skip per-item persistence for items that report `success=False`; failed items still appear in `index.json` with `file: null` so callers can reason about the attempt list. This matches the existing `batch_crawl` / `multi_url_crawl` behavior.
- `finalize_tool_response` treats `output_path=""` as a no-op, matching `validate_output_path`'s empty-string acceptance.

### Known caveats
- Tools with `readOnlyHint` annotation keep that hint even when `output_path` is supplied and the tool writes to disk. Callers should treat `output_path` as an explicit opt-in to filesystem writes.
- `overwrite=False` is checked with `Path.exists()` immediately before the atomic rename, which protects against most accidental overwrites but is not TOCTOU-race-safe against a concurrent writer. The atomic `os.replace` call itself is still atomic at the filesystem level. For stricter single-writer guarantees, pair this feature with a lock at the caller level.
- Batch persistence is per-file atomic but not transactional across items. A failure partway through will leave a partial directory (some `.md` files present, `index.json` potentially missing).

---

## [0.2.0] - 2026-03-01

### Added
- New `extract_youtube_comments` MCP tool with pagination support
- Batch tools restored as MCP tools: `batch_crawl`, `multi_url_crawl`, `batch_search_google`, `batch_extract_youtube_transcripts`
- `readOnlyHint` annotations added to all MCP tools
- Content pagination support via `content_offset` and `content_limit` parameters across tools
- pytest-based test infrastructure for MCP tools

### Changed
- Architecture refactored: codebase modularized into `server_tools/`, `core/`, `infra/`, `middleware/`, `processors/` layers
- `batch_crawl` switched to sequential execution with revised timeout design
- LLM summarization skipped when pagination is active

### Fixed
- `deep_crawl_site`: convert CrawlResponse to dict before calling `.get()` (community PR #18)
- `extract_structured_data`: convert CrawlResponse to dict and request HTML (community PR #16)
- `batch_crawl`: wrap remaining CrawlResponse returns with `_convert_result_to_dict`
- Content offset propagated through CrawlRequest for pagination cache protection
- YouTube comment tests made resilient to CI environment and non-deterministic ordering

---

## [0.1.7] - 2026-01-12

### Added
- YouTube transcript fallback and metadata enrichment
- Content-Type detection for URLs without file extension

### Changed
- Token counting improved using tiktoken for accurate counts
- Tool descriptions reduced to lower token usage
- Low-utility MCP tools hidden to reduce token usage
- `batch_crawl` limit updated from 5 to 3

### Fixed
- ReDoS protection improved with multiprocessing timeout and strict size limits
- Token limit handling improved to preserve more content
- Partial transcript data preserved on token overflow
- Missing `anthropic` dependency added

### Updated
- Dependencies updated to latest versions

---

## [0.1.6] - 2025-10-18

Note: Batch tools removed in 0.1.6 were restored in 0.2.0.

### 🔧 Improved
- **Token Limit Optimization**: Increased response token limit from 20000 to 25000 for all crawling tools
  - Affected tools: `crawl_url`, `deep_crawl_site`, `crawl_url_with_fallback`, `intelligent_extract`, `extract_entities`, `extract_structured_data`, `search_and_crawl`
  - Better utilization of Claude Code MCP's 25000 token limit
- **Token Limit Fallback**: Enhanced token limit fallback with detailed warnings and recommendations
  - Clearer error messages when response exceeds token limits
  - Actionable recommendations for reducing token usage
  - Better user guidance for large content handling

### ✨ Added
- **Markdown-Only Response**: New `include_cleaned_html` parameter for crawling tools
  - Default behavior: Returns markdown content only to reduce token usage (~50% reduction)
  - Set `include_cleaned_html=True` to also receive cleaned HTML content
  - Improved readability and token efficiency

### 🔄 Changed
- **Removed Batch Tools from MCP Interface**: Batch-related tools no longer exposed as MCP tools
  - Removed: `batch_extract_youtube_transcripts`, `batch_search_google`, `batch_crawl`, `multi_url_crawl`
  - Total MCP tools reduced from 21 to 17
  - Functions remain available internally but not exposed via MCP
  - Batch operations provide limited value in MCP context due to sequential processing nature

### 📝 Documentation
- Updated API_REFERENCE.md (English) with all recent changes
- Updated API_REFERENCE.md (Japanese) with all recent changes
- Updated README.md tool count (21 → 17)
- Updated README_ja.md tool count (21 → 17)
- Removed batch tool references from documentation
- Added `include_cleaned_html` parameter documentation
- Added token limit information (25000 tokens)

---

## [0.1.5] - 2025-02-05

### 🔧 Fixed
- **Import Compatibility**: Fixed compatibility with latest FastMCP 2.11.0

---

## [0.1.1] - 2025-08-16

### 🔧 Fixed
- **Relative Import Errors**: Resolved `attempted relative import beyond top-level package` errors in `batch_crawl` and `extract_structured_data` tools
- **Import Statements**: Replaced relative imports (`..models`) with absolute imports (`crawl4ai_mcp.models`) for better reliability
- **Multi-Platform Support**: Applied fixes to main server with UVX and Docker deployment options

### ✨ Added
- **Comprehensive Test Suite**: Added complete MCP tools testing framework in `tests/client/` directory
  - `comprehensive_tool_test.py` - Full 21-tool testing suite
  - `quick_tool_test.py` - Fast 8-tool core functionality test
  - `category_tool_test.py` - Category-based testing (YouTube, Search, Crawl, File, Utils)
  - `test_runner.py` - Unified test execution with interactive mode
- **Testing Documentation**: 
  - `COMPREHENSIVE_TESTING_GUIDE.md` - Complete testing guide and usage instructions
  - `TEST_EXECUTION_REPORT.md` - Detailed test execution results and analysis
  - `IMPORT_FIX_REPORT.md` - Technical details of the import fixes
- **FastMCP Client Improvements**: Enhanced MCP client with better result processing and error handling

### 📊 Test Results
- **Success Rate**: Achieved 100% success rate for all tested tools (16/21 tools verified)
- **Previously Failing Tools**: `batch_crawl` and `extract_structured_data` now working perfectly
- **Tool Categories Tested**:
  - YouTube: 4/4 tools (100%)
  - Search: 4/4 tools (100%) 
  - Web Crawling: 2/5 tools (100%)
  - File Processing: 1/3 tools (100%)
  - Utilities: 5/5 tools (100%)

### 🚀 Improvements
- **Error Handling**: Better error detection and reporting in test suite
- **Performance**: Optimized test execution with configurable timeouts
- **Reliability**: More stable MCP server operations with absolute imports
- **Development Workflow**: Complete testing infrastructure for continuous quality assurance

### 📦 Package Updates
- **pyproject.toml**: Updated version to 0.1.1
- **Docker Support**: Added comprehensive Docker and docker-compose configurations
- **Dependencies**: No breaking changes, all existing dependencies maintained

### 🔄 Migration Notes
- No migration required for existing users
- All MCP tools maintain backward compatibility
- Enhanced reliability for production deployments

---

## [0.1.0] - 2025-08-15

### Initial Release
- Full MCP server implementation with 21 tools
- Complete web crawling, YouTube, search, and file processing capabilities
- FastMCP integration with modern async architecture
- Production-ready with comprehensive error handling