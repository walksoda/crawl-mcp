"""
MCP-specific pytest fixtures for tool testing.

This module provides specialized fixtures for testing MCP tools:
- YouTube test videos and configurations
- Crawl test sites and expected results
- Search test queries
- File processing test data
"""

import pytest
from typing import Dict, Any, List


@pytest.fixture
def youtube_test_videos() -> Dict[str, Dict[str, Any]]:
    """
    Provide YouTube test video URLs with expected metadata.

    Returns dict with video categories and their expected properties.
    """
    return {
        "short_video": {
            "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "title_contains": "zoo",
            "has_transcript": True,
            "min_duration": 10,
        },
        "music_video": {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title_contains": "never gonna",
            "has_transcript": True,
            "min_duration": 100,
        },
        "educational": {
            "url": "https://www.youtube.com/watch?v=aircAruvnKk",
            "title_contains": "neural",
            "has_transcript": True,
            "min_duration": 300,
        },
    }


@pytest.fixture
def crawl_test_sites() -> Dict[str, Dict[str, Any]]:
    """
    Provide test websites with expected crawl results.

    Returns dict with site categories and their expected properties.
    """
    return {
        "simple_html": {
            "url": "https://example.com",
            "expected_title": "Example Domain",
            "expected_content_contains": ["example", "domain"],
            "min_content_length": 100,
        },
        "wikipedia": {
            "url": "https://en.wikipedia.org/wiki/Web_scraping",
            "expected_title": "Web scraping",
            "expected_content_contains": ["web scraping", "data"],
            "min_content_length": 1000,
        },
        "github_repo": {
            "url": "https://github.com/unclecode/crawl4ai",
            "expected_content_contains": ["crawl4ai", "python"],
            "min_content_length": 500,
        },
        "news_site": {
            "url": "https://news.ycombinator.com",
            "expected_content_contains": ["hacker", "news"],
            "min_content_length": 200,
        },
    }


@pytest.fixture
def search_test_queries() -> Dict[str, Dict[str, Any]]:
    """
    Provide test search queries with expected results.

    Returns dict with query types and their expected properties.
    """
    return {
        "technical": {
            "query": "Python web scraping tutorial",
            "expected_domains": ["python", "scraping"],
            "min_results": 1,
        },
        "news": {
            "query": "artificial intelligence news",
            "expected_domains": ["ai", "news"],
            "min_results": 1,
        },
        "academic": {
            "query": "machine learning research papers",
            "expected_domains": ["machine", "learning"],
            "min_results": 1,
        },
    }


@pytest.fixture
def file_test_urls() -> Dict[str, Dict[str, Any]]:
    """
    Provide test file URLs for file processing.

    Returns dict with file types and their URLs.
    """
    return {
        "pdf": {
            "url": "https://www.w3.org/WAI/WCAG21/Techniques/pdf/img/table-word.pdf",
            "expected_type": "pdf",
            "min_content_length": 50,
        },
    }


# Helper functions for MCP result validation
def extract_mcp_content(result) -> str:
    """Extract text content from MCP tool result."""
    # Handle CallToolResult with content attribute
    if hasattr(result, 'content'):
        content = result.content
    else:
        content = result

    # Handle list/tuple of content items
    if isinstance(content, (list, tuple)) and len(content) > 0:
        item = content[0]
        if hasattr(item, 'text'):
            return item.text
        return str(item)

    # Handle direct text content
    if hasattr(content, 'text'):
        return content.text

    return str(content)


def assert_tool_success(result) -> None:
    """Assert that a tool call was successful (no error in result)."""
    content = extract_mcp_content(result)
    content_lower = content.lower()

    # Check for explicit error patterns at the start of content
    error_patterns = [
        "error:",
        "failed:",
        "exception:",
        "traceback (most recent call last)",
    ]

    for pattern in error_patterns:
        if content_lower.startswith(pattern):
            pytest.fail(f"Tool returned error: {content[:200]}")


def assert_content_contains(result, *expected_strings: str) -> None:
    """Assert that result content contains all expected strings."""
    content = extract_mcp_content(result).lower()

    for expected in expected_strings:
        assert expected.lower() in content, \
            f"Expected '{expected}' not found in result content"


def assert_content_length(result, min_length: int) -> None:
    """Assert that result content meets minimum length."""
    content = extract_mcp_content(result)
    assert len(content) >= min_length, \
        f"Content length {len(content)} is less than minimum {min_length}"


def assert_json_structure(result, *expected_keys: str) -> None:
    """Assert that result contains expected JSON keys."""
    import json
    content = extract_mcp_content(result)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # If not JSON, check for key patterns in text
        content_lower = content.lower()
        for key in expected_keys:
            if key.lower() not in content_lower:
                pytest.fail(f"Expected key '{key}' not found in result")
        return

    if isinstance(data, dict):
        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in JSON result"
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        for key in expected_keys:
            assert key in data[0], f"Expected key '{key}' not found in JSON result"


def parse_mcp_json(result) -> Dict[str, Any]:
    """Parse MCP tool result as JSON and return the data dict."""
    import json
    content = extract_mcp_content(result)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def assert_content_slicing(result, expected_limit: int, expected_offset: int = 0) -> None:
    """Assert that content slicing was applied correctly."""
    data = parse_mcp_json(result)
    slicing_info = data.get("slicing_info", {})
    markdown_info = slicing_info.get("markdown", {})

    # Check if slicing info exists
    assert slicing_info, "No slicing_info found in result"

    # Check limit and offset were applied
    if markdown_info.get("present", True):
        assert markdown_info.get("limit") == expected_limit, \
            f"Expected limit {expected_limit}, got {markdown_info.get('limit')}"
        assert markdown_info.get("offset") == expected_offset, \
            f"Expected offset {expected_offset}, got {markdown_info.get('offset')}"
        # Check returned length respects limit
        returned_length = markdown_info.get("returned_length", 0)
        assert returned_length <= expected_limit, \
            f"Returned length {returned_length} exceeds limit {expected_limit}"
