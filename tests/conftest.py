"""
Central pytest configuration and fixtures for Crawl4AI MCP Server tests.

This module provides:
- Project root and venv auto-detection
- MCP client fixture using FastMCP 2.0
- Sample URLs and test data fixtures
- Custom markers registration
"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

import pytest

# Auto-detect project paths
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
VENV_PATH = PROJECT_ROOT / "venv"
VENV_PYTHON = VENV_PATH / "bin" / "python"

# Add project root to path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "mcp: MCP server connection required")
    config.addinivalue_line("markers", "youtube: YouTube tools")
    config.addinivalue_line("markers", "crawl: crawl tools")
    config.addinivalue_line("markers", "search: search tools")
    config.addinivalue_line("markers", "file: file processing tools")
    config.addinivalue_line("markers", "utility: utility tools")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def venv_python() -> Path:
    """Return the path to the venv Python executable."""
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    return Path(sys.executable)


@pytest.fixture
async def mcp_client(project_root: Path, venv_python: Path):
    """
    Create and manage an MCP client connection for each test.

    Uses FastMCP 2.0 StdioTransport to connect to the MCP server.
    """
    from fastmcp.client.transports import StdioTransport
    from fastmcp import Client

    transport = StdioTransport(
        command=str(venv_python),
        args=["-m", "crawl4ai_mcp.server"],
        cwd=str(project_root)
    )

    client = Client(transport)
    async with client:
        yield client


@pytest.fixture
async def mcp_tools(mcp_client) -> Dict[str, Any]:
    """Get list of available MCP tools."""
    tools_result = await mcp_client.list_tools()

    if hasattr(tools_result, 'tools'):
        tools = tools_result.tools
    else:
        tools = tools_result

    tool_dict = {}
    for tool in tools:
        if hasattr(tool, 'name'):
            name = tool.name
            schema = getattr(tool, 'inputSchema', {})
        elif isinstance(tool, dict):
            name = tool.get('name', 'unknown')
            schema = tool.get('inputSchema', {})
        else:
            continue
        tool_dict[name] = schema

    return tool_dict


@pytest.fixture
def sample_urls() -> Dict[str, str]:
    """Provide sample URLs for testing."""
    return {
        "simple_html": "https://example.com",
        "wikipedia": "https://en.wikipedia.org/wiki/Web_scraping",
        "github": "https://github.com/unclecode/crawl4ai",
        "youtube_short": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "youtube_long": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "pdf_file": "https://www.w3.org/WAI/WCAG21/Techniques/pdf/img/table-word.pdf",
    }


@pytest.fixture
def search_queries() -> Dict[str, str]:
    """Provide sample search queries for testing."""
    return {
        "simple": "Python web scraping",
        "technical": "FastMCP MCP server",
        "news": "AI technology 2025",
    }


# Helper functions for assertions
def assert_mcp_result_valid(result, min_length: int = 10) -> None:
    """Assert that an MCP tool result is valid and non-empty."""
    assert result is not None, "Result should not be None"

    # Handle different result types
    if hasattr(result, 'content'):
        content = result.content
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        content = result[0]
        if hasattr(content, 'text'):
            content = content.text
    else:
        content = str(result)

    content_str = str(content)
    assert len(content_str) >= min_length, f"Result content too short: {len(content_str)} chars"


def assert_mcp_result_contains(result, expected_substring: str) -> None:
    """Assert that an MCP tool result contains an expected substring."""
    if hasattr(result, 'content'):
        content = result.content
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        content = result[0]
        if hasattr(content, 'text'):
            content = content.text
    else:
        content = str(result)

    content_str = str(content)
    assert expected_substring.lower() in content_str.lower(), \
        f"Expected '{expected_substring}' not found in result"


def assert_mcp_result_no_error(result) -> None:
    """Assert that an MCP tool result does not contain error indicators."""
    if hasattr(result, 'content'):
        content = result.content
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        content = result[0]
        if hasattr(content, 'text'):
            content = content.text
    else:
        content = str(result)

    content_str = str(content).lower()
    error_indicators = ["error:", "exception:", "failed:", "traceback"]

    for indicator in error_indicators:
        # Allow "error" to appear in normal content, only flag explicit error messages
        if indicator in content_str and "error:" in content_str[:200]:
            pytest.fail(f"Result contains error indicator: {indicator}")
