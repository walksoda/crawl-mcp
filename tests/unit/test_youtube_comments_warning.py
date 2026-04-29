"""Unit tests for the YouTube comments warning field.

Covers the total_seen counter and warning propagation:
- downloader yields nothing -> warning present
- comment_offset exceeds available -> no warning (total_seen > 0)
- disabled/private video (exception) -> error path, no warning
"""

from unittest.mock import patch, MagicMock

import pytest

from crawl4ai_mcp.processors.youtube_processor import YouTubeProcessor


@pytest.fixture
def processor():
    return YouTubeProcessor()


def _mock_downloader(comments):
    """Return a mock YoutubeCommentDownloader whose generator yields ``comments``."""
    downloader = MagicMock()
    downloader.get_comments_from_url.return_value = iter(comments)
    return downloader


class TestTotalSeenWarning:
    """When the downloader yields zero items, the response must include a warning."""

    @pytest.mark.asyncio
    async def test_downloader_yields_nothing(self, processor):
        mock_dl = _mock_downloader([])
        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="test123", url="https://www.youtube.com/watch?v=test123",
            )

        assert result["success"] is True
        assert result["comment_stats"]["total_comments"] == 0
        assert "warning" in result
        assert "No comments were retrieved" in result["warning"]

    @pytest.mark.asyncio
    async def test_offset_exceeds_available_no_warning(self, processor):
        raw = [
            {"cid": "1", "text": "hello", "author": "a", "time": "1h", "votes": "1", "replies": 0, "heart": False, "reply": False},
            {"cid": "2", "text": "world", "author": "b", "time": "2h", "votes": "0", "replies": 0, "heart": False, "reply": False},
        ]
        mock_dl = _mock_downloader(raw)
        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="test123", url="https://www.youtube.com/watch?v=test123",
                comment_offset=100,
            )

        assert result["success"] is True
        assert result["comment_stats"]["total_comments"] == 0
        assert "warning" not in result


class TestDisabledPrivateVideo:
    """Exception paths must remain error responses without a warning field."""

    @pytest.mark.asyncio
    async def test_comments_disabled_raises_error(self, processor):
        mock_dl = MagicMock()
        mock_dl.get_comments_from_url.side_effect = RuntimeError("Comments are disabled")
        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="test123", url="https://www.youtube.com/watch?v=test123",
            )

        assert result["success"] is False
        assert "disable" in result["error"].lower()
        assert "warning" not in result


class TestWarningPropagation:
    """Warning must reach the top-level response dict via the core layer."""

    @pytest.mark.asyncio
    async def test_warning_in_top_level_response(self):
        from crawl4ai_mcp.core.youtube_comments import extract_youtube_comments

        mock_dl = _mock_downloader([])
        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await extract_youtube_comments(
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )

        assert result["success"] is True
        warnings = result.get("warnings", [])
        assert any("No comments were retrieved" in w for w in warnings)
        assert "warning" in result.get("extracted_data", {})
