"""
File Processing MCP Tools Test Suite.

Tests for file processing MCP tools:
- process_file
"""

import pytest

from tests.mcp.conftest import (
    extract_mcp_content,
    assert_tool_success,
    assert_content_contains,
    assert_content_length,
)


@pytest.mark.mcp
@pytest.mark.file
class TestProcessFile:
    """Tests for process_file tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_pdf_file(self, mcp_client, file_test_urls):
        """Test processing a PDF file."""
        file_info = file_test_urls["pdf"]

        result = await mcp_client.call_tool(
            "process_file",
            {"url": file_info["url"]}
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        # PDF processing should return some content
        assert len(content) >= file_info["min_content_length"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_file_with_metadata(self, mcp_client, file_test_urls):
        """Test processing a file with metadata extraction."""
        file_info = file_test_urls["pdf"]

        result = await mcp_client.call_tool(
            "process_file",
            {
                "url": file_info["url"],
                "include_metadata": True
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 30)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_file_with_auto_summarize(self, mcp_client, file_test_urls):
        """Test processing a file with auto summarization."""
        file_info = file_test_urls["pdf"]

        result = await mcp_client.call_tool(
            "process_file",
            {
                "url": file_info["url"],
                "auto_summarize": False
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        assert len(content) > 0
