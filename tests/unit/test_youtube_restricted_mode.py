"""Unit tests for YouTube Restricted Mode handling in extract_comments.

When a network-level DNS resolver rewrites www.youtube.com to
restrictmoderate.youtube.com, the rendered page lacks the
sortFilterSubMenuRenderer and youtube-comment-downloader raises
RuntimeError('Failed to set sorting'). This test pins the contract that
the failure mode is surfaced as success=True with a structured `warning`
field — same shape as the zero-yield case from PR #17 — instead of
success=False with a cryptic raw error string.
"""

from unittest.mock import MagicMock, patch

import pytest

from crawl4ai_mcp.processors.youtube_processor import YouTubeProcessor


@pytest.fixture
def processor():
    return YouTubeProcessor()


class TestRestrictedMode:
    """Restricted-Mode DNS rewrite path."""

    @pytest.mark.asyncio
    async def test_failed_to_set_sorting_returns_success_with_warning(self, processor):
        """The exact RuntimeError raised by the downloader must be classified."""
        mock_dl = MagicMock()
        mock_dl.get_comments_from_url.side_effect = RuntimeError("Failed to set sorting")

        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="dQw4w9WgXcQ",
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )

        assert result["success"] is True
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["comments"] == []
        assert result["has_more"] is False
        assert result["comment_stats"] == {
            "total_comments": 0,
            "top_level_comments": 0,
            "reply_comments": 0,
            "unique_authors": 0,
        }
        assert "warning" in result
        warning = result["warning"]
        assert "Restricted Mode" in warning
        assert "restrictmoderate.youtube.com" in warning
        assert "getent hosts www.youtube.com" in warning
        # No `error` key on the success path.
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_match_is_case_insensitive(self, processor):
        """The downloader's exact wording could shift; match on substring,
        any case."""
        mock_dl = MagicMock()
        mock_dl.get_comments_from_url.side_effect = RuntimeError("FAILED to Set Sorting")

        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="abc", url="https://www.youtube.com/watch?v=abc",
            )

        assert result["success"] is True
        assert "warning" in result


class TestRestrictedModeDoesNotMaskOtherFailures:
    """The new branch must not swallow unrelated exceptions."""

    @pytest.mark.asyncio
    async def test_disabled_video_still_returns_error(self, processor):
        """Comments-disabled is a real failure, not Restricted Mode — keep
        success=False (PR #17's existing behavior)."""
        mock_dl = MagicMock()
        mock_dl.get_comments_from_url.side_effect = RuntimeError("Comments are disabled")

        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="abc", url="https://www.youtube.com/watch?v=abc",
            )

        assert result["success"] is False
        assert "disabled" in result["error"].lower()
        assert "warning" not in result

    @pytest.mark.asyncio
    async def test_unknown_runtime_error_still_returns_error(self, processor):
        """Other RuntimeErrors fall through to the generic failure branch."""
        mock_dl = MagicMock()
        mock_dl.get_comments_from_url.side_effect = RuntimeError("Some other failure")

        with patch(
            "youtube_comment_downloader.YoutubeCommentDownloader",
            return_value=mock_dl,
        ):
            result = await processor.extract_comments(
                video_id="abc", url="https://www.youtube.com/watch?v=abc",
            )

        assert result["success"] is False
        assert "Comment extraction failed" in result["error"]
        assert "Some other failure" in result["error"]
        assert "warning" not in result
