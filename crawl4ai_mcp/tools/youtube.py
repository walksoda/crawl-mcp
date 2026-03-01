"""
YouTube tools facade.

Re-exports from core/ modules for backward compatibility.
"""

from ..core.youtube_helpers import (
    _extract_youtube_metadata_from_html,
    _filter_relevant_content,
    _build_fallback_transcript,
    _crawl_youtube_page_fallback,
)
from ..core.youtube import (
    extract_youtube_transcript,
    batch_extract_youtube_transcripts,
    get_youtube_video_info,
    get_youtube_api_setup_guide,
)

__all__ = [
    '_extract_youtube_metadata_from_html', '_filter_relevant_content',
    '_build_fallback_transcript', '_crawl_youtube_page_fallback',
    'extract_youtube_transcript', 'batch_extract_youtube_transcripts',
    'get_youtube_video_info', 'get_youtube_api_setup_guide',
]
