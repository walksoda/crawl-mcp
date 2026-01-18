"""
YouTube MCP Tools Test Suite.

Tests for YouTube-related MCP tools:
- extract_youtube_transcript
- batch_extract_youtube_transcripts
- get_youtube_video_info
"""

import pytest

from tests.mcp.conftest import (
    extract_mcp_content,
    assert_tool_success,
    assert_content_contains,
    assert_content_length,
)


@pytest.mark.mcp
@pytest.mark.youtube
class TestExtractYoutubeTranscript:
    """Tests for extract_youtube_transcript tool."""

    @pytest.mark.asyncio
    async def test_basic_extraction(self, mcp_client, youtube_test_videos):
        """Test basic transcript extraction from a short video."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_transcript",
            {"url": video["url"]}
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

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
        content = extract_mcp_content(result)
        # Timestamps typically contain numbers
        assert any(char.isdigit() for char in content), "Timestamps should contain numbers"

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
        assert_content_length(result, 20)

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
        # Metadata typically includes title
        content = extract_mcp_content(result)
        assert len(content) > 50


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
        content = extract_mcp_content(result)
        # Should have results for both videos
        assert len(content) > 100

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
        assert_content_length(result, 50)


@pytest.mark.mcp
@pytest.mark.youtube
class TestGetYoutubeVideoInfo:
    """Tests for get_youtube_video_info tool."""

    @pytest.mark.asyncio
    async def test_basic_video_info(self, mcp_client, youtube_test_videos):
        """Test getting basic video information."""
        video = youtube_test_videos["short_video"]

        result = await mcp_client.call_tool(
            "get_youtube_video_info",
            {"video_url": video["url"]}
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        # Should contain video information
        assert len(content) > 50

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
        assert_content_length(result, 30)
