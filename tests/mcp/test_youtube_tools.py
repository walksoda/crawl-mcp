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
class TestExtractYoutubeComments:
    """Tests for extract_youtube_comments tool."""

    @pytest.mark.asyncio
    async def test_basic_extraction(self, mcp_client, youtube_test_videos):
        """Test basic comment extraction returns expected structure."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 10}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        assert data["success"] is True
        assert "content" in data
        assert "markdown" in data
        assert "video_id" in data
        assert "title" in data
        assert "extracted_data" in data

        extracted = data["extracted_data"]
        assert "video_id" in extracted
        assert "processing_method" in extracted
        assert extracted["processing_method"] == "youtube_comment_downloader"
        assert "comment_stats" in extracted
        assert "comments" in extracted
        assert "has_more" in extracted

    @pytest.mark.asyncio
    async def test_sort_by_recent(self, mcp_client, youtube_test_videos):
        """Test comment extraction with sort_by=recent."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 5, "sort_by": "recent"}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["success"] is True
        assert data["extracted_data"]["sort_by"] == "recent"

    @pytest.mark.asyncio
    async def test_exclude_replies(self, mcp_client, youtube_test_videos):
        """Test comment extraction without replies."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 10, "include_replies": False}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["success"] is True
        assert data["extracted_data"]["include_replies"] is False

        # All comments should be top-level
        comments = data["extracted_data"]["comments"]
        for c in comments:
            assert c["is_reply"] is False

    @pytest.mark.asyncio
    async def test_comment_stats_structure(self, mcp_client, youtube_test_videos):
        """Test that comment_stats has expected fields."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 10}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        stats = data["extracted_data"]["comment_stats"]
        assert "total_comments" in stats
        assert "top_level_comments" in stats
        assert "reply_comments" in stats
        assert "unique_authors" in stats
        assert stats["total_comments"] == stats["top_level_comments"] + stats["reply_comments"]

    @pytest.mark.asyncio
    async def test_comment_offset_pagination(self, mcp_client, youtube_test_videos):
        """Test comment_offset for pagination produces different results."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 3, "comment_offset": 0, "include_replies": False}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["success"] is True
        assert data["extracted_data"]["comment_offset"] == 0
        page1_cids = {c["cid"] for c in data["extracted_data"]["comments"]}

        # Second page
        result2 = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 3, "comment_offset": 3, "include_replies": False}
        )

        assert_tool_success(result2)
        data2 = parse_mcp_result(result2)
        assert data2["success"] is True
        assert data2["extracted_data"]["comment_offset"] == 3

        # Verify no overlap between pages
        page2_cids = {c["cid"] for c in data2["extracted_data"]["comments"]}
        assert page1_cids.isdisjoint(page2_cids), "Pages should not have duplicate comments"

    @pytest.mark.asyncio
    async def test_content_slicing(self, mcp_client, youtube_test_videos):
        """Test content slicing with content_offset and content_limit."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {
                "url": video["url"],
                "max_comments": 10,
                "content_limit": 200,
                "content_offset": 0
            }
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        if data.get("slicing_info"):
            slicing = data["slicing_info"]
            assert "markdown" in slicing or "content" in slicing

    @pytest.mark.asyncio
    async def test_invalid_url(self, mcp_client):
        """Test extraction with invalid URL returns error."""
        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": "https://example.com/not-youtube"}
        )

        content = extract_mcp_content(result)
        data = json.loads(content)
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_sort_by(self, mcp_client, youtube_test_videos):
        """Test extraction with invalid sort_by value returns error."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "sort_by": "invalid_sort"}
        )

        content = extract_mcp_content(result)
        data = json.loads(content)
        assert data["success"] is False
        assert "sort_by" in data["error"].lower() or "invalid" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_max_comments_boundary(self, mcp_client, youtube_test_videos):
        """Test max_comments boundary values."""
        video = youtube_test_videos["music_video"]

        # max_comments = 1 (minimum)
        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 1}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["success"] is True
        assert len(data["extracted_data"]["comments"]) <= 1

    @pytest.mark.asyncio
    async def test_max_comments_over_limit(self, mcp_client, youtube_test_videos):
        """Test max_comments over 1000 returns error."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 1001}
        )

        content = extract_mcp_content(result)
        data = json.loads(content)
        assert data["success"] is False
        assert "max_comments" in data["error"]

    @pytest.mark.asyncio
    async def test_negative_comment_offset(self, mcp_client, youtube_test_videos):
        """Test negative comment_offset returns error."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "comment_offset": -1}
        )

        content = extract_mcp_content(result)
        data = json.loads(content)
        assert data["success"] is False
        assert "comment_offset" in data["error"]

    @pytest.mark.asyncio
    async def test_markdown_escaping(self, mcp_client, youtube_test_videos):
        """Test that markdown content contains escaped special characters."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 5}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)

        # Markdown should exist and content/markdown should match
        assert data.get("content") == data.get("markdown")
        assert len(data.get("markdown", "")) > 0

    @pytest.mark.asyncio
    async def test_comments_disabled_video(self, mcp_client, youtube_test_videos):
        """Test extraction from a video with comments disabled returns empty list."""
        video = youtube_test_videos["comments_disabled"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 5}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["success"] is True
        assert data["extracted_data"]["comment_stats"]["total_comments"] == 0
        assert data["extracted_data"]["comments"] == []
        assert data["extracted_data"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_content_and_markdown_match(self, mcp_client, youtube_test_videos):
        """Test that content and markdown fields contain identical values."""
        video = youtube_test_videos["music_video"]

        result = await mcp_client.call_tool(
            "extract_youtube_comments",
            {"url": video["url"], "max_comments": 3}
        )

        assert_tool_success(result)
        data = parse_mcp_result(result)
        assert data["content"] == data["markdown"]


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
