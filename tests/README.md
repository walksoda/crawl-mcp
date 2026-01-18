# Crawl4AI MCP Server Test Suite

This directory contains the pytest-based test suite for the Crawl4AI MCP Server.

## Directory Structure

```
tests/
├── conftest.py              # Central fixture configuration
├── unit/                    # Unit tests (no MCP connection needed)
│   └── test_config.py       # Configuration module tests
├── mcp/                     # MCP tool tests (requires MCP server)
│   ├── conftest.py          # MCP-specific fixtures
│   ├── test_youtube_tools.py    # YouTube transcript tools
│   ├── test_crawl_tools.py      # Web crawling tools
│   ├── test_file_tools.py       # File processing tools
│   ├── test_search_tools.py     # Google search tools
│   └── test_utility_tools.py    # Utility tools
├── client/                  # Legacy client framework (maintained)
├── integration/             # Integration tests
└── verification/            # Verification tests
```

## Quick Start

### Install Test Dependencies

```bash
pip install -e ".[test]"
# or with uv
uv pip install -e ".[test]"

# Install browser for crawling tests
playwright install chromium
```

### Run Tests

```bash
# Quick unit tests
./scripts/run_tests.sh quick

# MCP tool tests (excluding slow tests)
./scripts/run_tests.sh mcp

# All tests including slow tests
./scripts/run_tests.sh full

# With coverage
./scripts/run_tests.sh full cov

# Specific tool category
./scripts/run_tests.sh youtube
./scripts/run_tests.sh crawl
./scripts/run_tests.sh search
./scripts/run_tests.sh file
./scripts/run_tests.sh utility
```

### Direct pytest Usage

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run MCP tests excluding slow ones
pytest tests/mcp/ -v -m "not slow"

# Run specific tool tests
pytest tests/mcp/test_youtube_tools.py -v

# Run with coverage
pytest tests/ -v --cov=crawl4ai_mcp --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)

Tests that do not require MCP server connection:
- Configuration module tests
- Utility function tests
- Model validation tests

### MCP Tool Tests (`tests/mcp/`)

Tests that require MCP server connection. Organized by tool category:

| File | Tools Tested |
|------|--------------|
| `test_youtube_tools.py` | extract_youtube_transcript, batch_extract_youtube_transcripts, get_youtube_video_info, get_youtube_api_setup_guide |
| `test_crawl_tools.py` | crawl_url, crawl_url_with_fallback, deep_crawl_site, intelligent_extract, extract_entities, extract_structured_data |
| `test_file_tools.py` | process_file, get_supported_file_formats, enhanced_process_large_content |
| `test_search_tools.py` | search_google, batch_search_google, search_and_crawl, get_search_genres |
| `test_utility_tools.py` | get_llm_config_info, batch_crawl, multi_url_crawl, get_tool_selection_guide, get_system_diagnostics |

## Test Markers

Use pytest markers to filter tests:

```bash
# Run only MCP tests
pytest -m mcp

# Run only YouTube tests
pytest -m youtube

# Run only crawl tests
pytest -m crawl

# Exclude slow tests
pytest -m "not slow"

# Run slow tests only
pytest -m slow
```

Available markers:
- `mcp`: Requires MCP server connection
- `youtube`: YouTube tool tests
- `crawl`: Crawl tool tests
- `search`: Search tool tests
- `file`: File processing tests
- `utility`: Utility tool tests
- `slow`: Tests that take longer to run

## Fixtures

### Central Fixtures (`tests/conftest.py`)

- `project_root`: Project root directory path
- `venv_python`: Path to venv Python executable
- `mcp_client`: MCP client connection (session-scoped)
- `mcp_tools`: Dictionary of available MCP tools
- `sample_urls`: Sample URLs for testing
- `search_queries`: Sample search queries

### MCP Fixtures (`tests/mcp/conftest.py`)

- `youtube_test_videos`: YouTube video URLs with expected metadata
- `crawl_test_sites`: Test websites with expected results
- `search_test_queries`: Search queries with expected results
- `file_test_urls`: Test file URLs

## Helper Functions

Available in `tests/mcp/conftest.py`:

- `extract_mcp_content(result)`: Extract text content from MCP result
- `assert_tool_success(result)`: Assert tool call was successful
- `assert_content_contains(result, *strings)`: Assert content contains strings
- `assert_content_length(result, min_length)`: Assert minimum content length
- `assert_json_structure(result, *keys)`: Assert JSON contains expected keys

## Writing New Tests

### MCP Tool Test Example

```python
import pytest
from tests.mcp.conftest import assert_tool_success, assert_content_length

@pytest.mark.mcp
@pytest.mark.youtube
class TestNewYouTubeTool:
    @pytest.mark.asyncio
    async def test_basic_functionality(self, mcp_client, youtube_test_videos):
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "new_youtube_tool",
            {"url": video["url"]}
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_slow_operation(self, mcp_client):
        # Slow test marked appropriately
        pass
```

### Unit Test Example

```python
import pytest

class TestNewFeature:
    def test_basic_functionality(self):
        from crawl4ai_mcp.new_module import new_function
        result = new_function()
        assert result is not None
```

## CI/CD

Tests run automatically on GitHub Actions:
- On push to `main` and `develop` branches
- On pull requests to `main`

The workflow:
1. Runs unit tests
2. Runs MCP tests (non-slow)
3. Runs slow tests (continue on error)
4. Generates coverage report

## Troubleshooting

### MCP Connection Issues

If tests fail to connect to MCP server:

1. Ensure venv is set up correctly:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

2. Verify server can start:
   ```bash
   python -m crawl4ai_mcp.server
   ```

3. Check FastMCP is installed:
   ```bash
   python -c "import fastmcp; print(fastmcp.__version__)"
   ```

### Slow Test Timeouts

For slow tests timing out, increase the timeout:
```bash
pytest tests/mcp/ -v --timeout=300
```

### Browser Issues

For crawling tests, ensure Playwright browser is installed:
```bash
playwright install chromium
```
