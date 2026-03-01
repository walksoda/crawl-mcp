"""Core YouTube comments extraction logic."""

import re
from typing import Any, Dict, Optional

from ..models import YouTubeCommentsResponse
from ..processors.youtube_processor import YouTubeProcessor


def _escape_markdown(text: str) -> str:
    """Escape Markdown special characters in user-generated content."""
    # Escape characters that could alter Markdown rendering
    return re.sub(r'([*_`\[\]|>~\\])', r'\\\1', text)

# Initialize YouTube processor
youtube_processor = YouTubeProcessor()


async def extract_youtube_comments(
    url: str,
    sort_by: str = "popular",
    max_comments: int = 300,
    include_replies: bool = True,
    comment_offset: int = 0,
) -> Dict[str, Any]:
    """Extract comments from a YouTube video.

    Returns a dict with content/markdown/extracted_data format.
    """
    # Validate sort_by
    if sort_by not in ("popular", "recent"):
        return YouTubeCommentsResponse(
            success=False, url=url,
            error=f"Invalid sort_by value: '{sort_by}'. Must be 'popular' or 'recent'."
        ).model_dump()

    # Validate max_comments
    if max_comments < 1 or max_comments > 1000:
        return YouTubeCommentsResponse(
            success=False, url=url,
            error=f"max_comments must be between 1 and 1000, got {max_comments}."
        ).model_dump()

    # Validate comment_offset
    if comment_offset < 0:
        return YouTubeCommentsResponse(
            success=False, url=url,
            error=f"comment_offset must be >= 0, got {comment_offset}."
        ).model_dump()

    # Validate YouTube URL
    if not youtube_processor.is_youtube_url(url):
        return YouTubeCommentsResponse(
            success=False, url=url,
            error="URL is not a valid YouTube video URL"
        ).model_dump()

    video_id = youtube_processor.extract_video_id(url)
    if not video_id:
        return YouTubeCommentsResponse(
            success=False, url=url,
            error="Could not extract video ID from URL"
        ).model_dump()

    try:
        result = await youtube_processor.extract_comments(
            video_id=video_id,
            url=url,
            sort_by=sort_by,
            max_comments=max_comments,
            include_replies=include_replies,
            comment_offset=comment_offset,
        )

        if not result['success']:
            return YouTubeCommentsResponse(
                success=False, url=url, video_id=video_id,
                error=result.get('error', 'Comment extraction failed')
            ).model_dump()

        comments = result['comments']
        has_more = result['has_more']
        comment_stats = result['comment_stats']

        # Build markdown with escaped user-generated content
        lines = []
        for c in comments:
            votes = _escape_markdown(str(c.get('votes', '0')))
            time_str = _escape_markdown(c.get('time', ''))
            author = _escape_markdown(c.get('author', ''))
            text = _escape_markdown(c.get('text', ''))
            if c.get('is_reply'):
                lines.append(f"> **{author}** _{time_str} | {votes} votes_")
                for text_line in text.split('\n'):
                    lines.append(f"> {text_line}")
            else:
                lines.append(f"**{author}** _{time_str} | {votes} votes_")
                lines.append(text)
            lines.append("")

        markdown_text = "\n".join(lines)

        title = f"Comments for YouTube video {video_id}"
        response = YouTubeCommentsResponse(
            success=True,
            url=url,
            video_id=video_id,
            title=title,
            content=markdown_text,
            markdown=markdown_text,
            extracted_data={
                "video_id": video_id,
                "processing_method": "youtube_comment_downloader",
                "comment_stats": comment_stats,
                "sort_by": sort_by,
                "include_replies": include_replies,
                "comment_offset": comment_offset,
                "has_more": has_more,
                "comments": comments,
            }
        )
        return response.model_dump()

    except Exception as e:
        return YouTubeCommentsResponse(
            success=False, url=url, video_id=video_id,
            error=f"YouTube comment extraction error: {str(e)}"
        ).model_dump()
