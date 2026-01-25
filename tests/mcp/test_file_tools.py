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
    parse_mcp_json,
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

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_file_with_content_limit(self, mcp_client, file_test_urls):
        """Test processing a file with content limit."""
        file_info = file_test_urls["pdf"]

        result = await mcp_client.call_tool(
            "process_file",
            {
                "url": file_info["url"],
                "content_limit": 100
            }
        )

        assert_tool_success(result)
        # Verify slicing_info is present in response
        data = parse_mcp_json(result)
        assert "slicing_info" in data, "slicing_info should be present in response"
        slicing_info = data["slicing_info"]
        content_info = slicing_info.get("content", {})
        assert content_info.get("limit") == 100

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_file_with_content_slicing(self, mcp_client, file_test_urls):
        """Test processing a file with content offset and limit."""
        file_info = file_test_urls["pdf"]

        result = await mcp_client.call_tool(
            "process_file",
            {
                "url": file_info["url"],
                "content_offset": 50,
                "content_limit": 100
            }
        )

        assert_tool_success(result)
        # Verify slicing_info is present in response
        data = parse_mcp_json(result)
        assert "slicing_info" in data, "slicing_info should be present in response"
        slicing_info = data["slicing_info"]
        content_info = slicing_info.get("content", {})
        assert content_info.get("limit") == 100
        assert content_info.get("offset") == 50

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_file_with_pagination(self, mcp_client, file_test_urls):
        """Test pagination by fetching file content in chunks with offset."""
        file_info = file_test_urls["pdf"]

        # Get first chunk (0-100)
        first_chunk = await mcp_client.call_tool(
            "process_file",
            {
                "url": file_info["url"],
                "content_offset": 0,
                "content_limit": 100
            }
        )
        first_data = parse_mcp_json(first_chunk)

        # Get second chunk (100-200) - continuation
        second_chunk = await mcp_client.call_tool(
            "process_file",
            {
                "url": file_info["url"],
                "content_offset": 100,
                "content_limit": 100
            }
        )
        second_data = parse_mcp_json(second_chunk)

        # Verify slicing info shows correct offsets
        assert first_data.get("slicing_info", {}).get("content", {}).get("offset") == 0
        assert second_data.get("slicing_info", {}).get("content", {}).get("offset") == 100

        # If both succeeded with content, verify they are different
        first_content = first_data.get("content", "")
        second_content = second_data.get("content", "")
        if first_content and second_content:
            assert first_content != second_content, "Chunks should be different"
