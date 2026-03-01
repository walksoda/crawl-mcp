"""
YouTube MCP Tools Test Suite.

Tests for YouTube-related MCP tools:
- extract_youtube_transcript
- batch_extract_youtube_transcripts
- get_youtube_video_info
"""

import json
import pytest

from tests.mcp.conftest import (
    extract_mcp_content,
    assert_tool_success,
    assert_content_contains,
    assert_content_length,
)


def parse_mcp_result(result) -> dict:
    """Parse MCP tool result into a dictionary."""
    content = extract_mcp_content(result)
    return json.loads(content)


@pytest.mark.mcp
@pytest.mark.youtube
class TestExtractYoutubeTranscript:
    """Tests for extract_youtube_transcript tool."""

    @pytest.mark.asyncio
    async def test_basic_extraction(self, mcp_client, youtube_test_videos):
        """Test basic transcript extraction returns content/markdown format."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {"url": video["url"]}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        # Verify new response format
        assert data["success"] is True
        assert "content" in data or "markdown" in data
        assert "video_id" in data
        assert "title" in data

        # Verify extracted_data structure
        assert "extracted_data" in data
        extracted = data["extracted_data"]
        assert "video_id" in extracted
        assert "processing_method" in extracted
        assert "language_info" in extracted
        assert "transcript_stats" in extracted

        # Verify segments are not in the response
        assert "segments" not in data
        assert "transcript" not in data  # old field removed

    @pytest.mark.asyncio
    async def test_extraction_with_timestamps(self, mcp_client, youtube_test_videos):
        """Test transcript extraction with timestamps enabled."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {
                "url": video["url"],
                "include_timestamps": True
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        # Content should contain timestamp markers
        content = data.get("content", "")
        assert "[" in content and "]" in content, "Timestamps should contain bracket markers"

    @pytest.mark.asyncio
    async def test_extraction_without_timestamps(self, mcp_client, youtube_test_videos):
        """Test that default (no timestamps) produces matching content and markdown."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {"url": video["url"], "include_timestamps": False}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        # Without timestamps, content and markdown should both be clean text
        assert data.get("content") is not None
        assert data.get("markdown") is not None

    @pytest.mark.asyncio
    async def test_extraction_with_language_preference(self, mcp_client, youtube_test_videos):
        """Test transcript extraction with language preference."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {
                "url": video["url"],
                "languages": ["en"]
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data.get("extracted_data", {}).get("language_info", {}).get("source_language") is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_extraction_with_metadata(self, mcp_client, youtube_test_videos):
        """Test transcript extraction with metadata enrichment."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {
                "url": video["url"],
                "include_metadata": True,
                "enrich_metadata": True
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        # Metadata should be in extracted_data
        extracted = data.get("extracted_data", {})
        metadata = extracted.get("metadata", {})
        assert metadata is not None

    @pytest.mark.asyncio
    async def test_transcript_stats_structure(self, mcp_client, youtube_test_videos):
        """Test that transcript_stats has expected fields."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {"url": video["url"]}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        stats = data.get("extracted_data", {}).get("transcript_stats", {})
        assert "word_count" in stats
        assert "segment_count" in stats

    @pytest.mark.asyncio
    async def test_content_slicing(self, mcp_client, youtube_test_videos):
        """Test content slicing with content_offset and content_limit."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {
                "url": video["url"],
                "content_limit": 100,
                "content_offset": 0
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        # Should have slicing_info when slicing is applied
        if data.get("slicing_info"):
            slicing = data["slicing_info"]
            assert "markdown" in slicing or "content" in slicing

    @pytest.mark.asyncio
    async def test_invalid_url(self, mcp_client):
        """Test extraction with invalid URL returns error."""
        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {"url": "https://example.com/not-youtube"}
        )

        content = extract_mcp_content(result)
        data = json.loads(content)
        assert data["success"] is False
        assert "error" in data


@pytest.mark.mcp
@pytest.mark.youtube
class TestBatchExtractYoutubeTranscripts:
    """Tests for batch_extract_youtube_transcripts tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_extraction_two_videos(self, mcp_client, youtube_test_videos):
        """Test batch extraction from two videos."""
        videos = [
            youtube_test_videos["short_video"]["url"],
            youtube_test_videos["music_video"]["url"],
        ]

        result = await mcp_client.call_tool(
            "batch_extract_youtube_transcripts",
            {
                "request": {
                    "urls": videos,
                    "include_timestamps": False
                }
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        assert data["total_urls"] == 2
        assert "results" in data
        assert isinstance(data["results"], list)

        # Each result should have the new format
        for r in data["results"]:
            if r.get("success"):
                assert "content" in r or "markdown" in r
                assert "extracted_data" in r
                assert "segments" not in r

    @pytest.mark.asyncio
    async def test_batch_extraction_single_video(self, mcp_client, youtube_test_videos):
        """Test batch extraction with single video."""
        videos = [youtube_test_videos["short_video"]["url"]]

        result = await mcp_client.call_tool(
            "batch_extract_youtube_transcripts",
            {
                "request": {
                    "urls": videos
                }
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["total_urls"] == 1
        assert len(data["results"]) == 1


@pytest.mark.mcp
@pytest.mark.youtube
class TestGetYoutubeVideoInfo:
    """Tests for get_youtube_video_info tool."""

    @pytest.mark.asyncio
    async def test_basic_video_info(self, mcp_client, youtube_test_videos):
        """Test getting basic video information with metadata."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "get_youtube_video_info",
            {"video_url": video["url"]}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        assert data["success"] is True
        assert "video_id" in data
        assert "video_info" in data
        assert "title" in data
        assert "transcript_info" in data
        assert "processing_method" in data

    @pytest.mark.asyncio
    async def test_video_info_with_options(self, mcp_client, youtube_test_videos):
        """Test video info with options."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "get_youtube_video_info",
            {
                "video_url": video["url"],
                "include_timestamps": False
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["success"] is True
