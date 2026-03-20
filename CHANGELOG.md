# Changelog

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