"""get_youtube_video_info metadata and the markdown it persists.

Metadata comes from the videoDetails object YouTube embeds in the page;
the markdown heuristics are only a fallback.
"""

import json

from crawl4ai_mcp.core.youtube import render_video_info_markdown
from crawl4ai_mcp.core.youtube_helpers import _extract_youtube_metadata_from_html

VIDEO_ID = "abcdefghijk"
# Page text the old heuristics turned into a description
NAV_MARKDOWN = "戻る " + "ナビゲーション " * 10 + "JP\n"


def _page(*details):
    blobs = ",".join(f'"videoDetails":{json.dumps(d)}' for d in details)
    return f'<script>var ytInitialPlayerResponse = {{{blobs}}};</script>'


def _details(video_id=VIDEO_ID, **extra):
    return {
        "videoId": video_id,
        "title": "Real title",
        "lengthSeconds": "3725",
        "shortDescription": "Real description\nsecond line",
        "viewCount": "1234",
        "author": "Real Channel",
        **extra,
    }


class TestMetadataFromVideoDetails:
    def test_fields_come_from_video_details(self):
        meta = _extract_youtube_metadata_from_html(
            NAV_MARKDOWN, None, _page(_details()), VIDEO_ID
        )
        assert meta["description"] == "Real description\nsecond line"
        assert meta["channel_name"] == "Real Channel"
        assert meta["duration"] == "1h 2m 5s"
        assert meta["duration_seconds"] == 3725
        assert meta["view_count"] == 1234
        assert meta["title"] == "Real title"
        assert meta["extraction_source"] == "video_details"

    def test_picks_the_requested_video(self):
        html = _page(_details("zzzzzzzzzzz", author="Other"), _details())
        meta = _extract_youtube_metadata_from_html(NAV_MARKDOWN, None, html, VIDEO_ID)
        assert meta["channel_name"] == "Real Channel"

    def test_empty_description_is_not_replaced_by_page_text(self):
        html = _page(_details(shortDescription=""))
        meta = _extract_youtube_metadata_from_html(NAV_MARKDOWN, None, html, VIDEO_ID)
        assert meta["description"] == ""
        assert meta["extraction_source"] == "video_details"

    def test_missing_field_stays_null(self):
        details = _details()
        del details["author"]
        markdown = NAV_MARKDOWN + "by Some Navigation Text\n"
        meta = _extract_youtube_metadata_from_html(markdown, None, _page(details), VIDEO_ID)
        assert meta["channel_name"] is None

    def test_other_video_only_falls_back(self):
        html = _page(_details("zzzzzzzzzzz"))
        meta = _extract_youtube_metadata_from_html(NAV_MARKDOWN, None, html, VIDEO_ID)
        assert meta["extraction_source"] == "page_crawl"

    def test_skips_broken_json_before_a_valid_candidate(self):
        html = '"videoDetails":{"videoId": broken' + _page(_details())
        meta = _extract_youtube_metadata_from_html(NAV_MARKDOWN, None, html, VIDEO_ID)
        assert meta["channel_name"] == "Real Channel"

    def test_falls_back_to_page_text_without_video_details(self):
        meta = _extract_youtube_metadata_from_html(
            NAV_MARKDOWN, None, "<html></html>", VIDEO_ID
        )
        assert meta["extraction_source"] == "page_crawl"
        assert meta["duration"] is None


class TestRenderVideoInfoMarkdown:
    INFO = {
        "success": True,
        "url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "video_id": VIDEO_ID,
        "title": "Real title",
        "metadata": {
            "channel_name": "Real Channel",
            "duration": "1h 2m 5s",
            "view_count": 1234,
            "description": "Real description",
        },
        "published_at": {"datetime": "2026-10-25T09:00:00+09:00", "original": "2026-10-24T17:00:00-07:00"},
        "transcript_info": {
            "has_transcript": True,
            "available_languages": [
                {"language": "Japanese", "language_code": "ja", "is_generated": True},
            ],
        },
    }

    def test_body_holds_the_video_info(self):
        md = render_video_info_markdown(self.INFO)
        assert md.startswith("# Real title\n")
        for text in ("- Channel: Real Channel", "- Published: 2026-10-25T09:00:00+09:00",
                     "- Duration: 1h 2m 5s", "- Views: 1234", "## Description\n\nReal description",
                     "- Japanese (ja, auto-generated)"):
            assert text in md
        assert "Transcript Summary" not in md

    def test_summary_is_included_when_present(self):
        info = {**self.INFO, "transcript_info": {"transcript_summary": "Short summary"}}
        md = render_video_info_markdown(info)
        assert "## Transcript Summary\n\nShort summary" in md
        assert "No transcript available." in md

    def test_failed_language_list_is_not_reported_as_no_transcript(self):
        info = {**self.INFO, "transcript_info": {"has_transcript": False},
                "video_info": {"error": "Request timed out"}}
        md = render_video_info_markdown(info)
        assert "Transcript languages could not be listed: Request timed out" in md
        assert "No transcript available." not in md
