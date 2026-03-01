"""
YouTube Processor Base Module
Utility methods for YouTube video processing and transcript access.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter


class YouTubeProcessorBase:
    """Base class with utility methods for YouTube processing"""

    def __init__(self):
        self.formatter = TextFormatter()
        self.youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]

    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube video URL"""
        try:
            for pattern in self.youtube_patterns:
                if re.search(pattern, url):
                    return True
            return False
        except Exception:
            return False

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        try:
            for pattern in self.youtube_patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            return None
        except Exception:
            return None

    def _get_transcript_list(self, video_id):
        """
        youtube-transcript-api v1.2.1の新APIを使用してtranscript_listを返す。
        """
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        return api.list(video_id)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def _get_error_suggestion(self, error_message: str) -> str:
        """Get helpful suggestion based on error type"""
        error_lower = error_message.lower()

        if "transcript not found" in error_lower or "no transcript" in error_lower:
            return "This video may not have transcripts available. Try a different video or check if captions are enabled."
        elif "transcripts disabled" in error_lower:
            return "The video owner has disabled transcripts. Try a different video."
        elif "video unavailable" in error_lower or "private" in error_lower:
            return "Video is not accessible. Check if the video exists and is publicly available."
        elif "network" in error_lower or "connection" in error_lower or "timeout" in error_lower:
            return "Network issue detected. Check your internet connection and try again."
        elif "parsing" in error_lower or "temporary" in error_lower:
            return "This appears to be a temporary issue with YouTube's servers. Try again in a few minutes."
        else:
            return "Try using a different video or check if the video has publicly available transcripts."

    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get basic video information and available transcripts"""
        try:
            transcript_list = self._get_transcript_list(video_id)

            available_languages = []
            manual_transcripts = []
            auto_transcripts = []

            for transcript in transcript_list:
                lang_info = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }

                available_languages.append(lang_info)

                if transcript.is_generated:
                    auto_transcripts.append(lang_info)
                else:
                    manual_transcripts.append(lang_info)

            return {
                'video_id': video_id,
                'has_transcripts': len(available_languages) > 0,
                'total_transcripts': len(available_languages),
                'manual_transcripts': len(manual_transcripts),
                'auto_transcripts': len(auto_transcripts),
                'available_languages': available_languages,
                'manual_languages': manual_transcripts,
                'auto_languages': auto_transcripts,
                'api_version': 'youtube-transcript-api-1.1.0+'
            }

        except Exception as e:
            error_message = str(e)

            if "no element found" in error_message.lower() or "parseerror" in error_message.lower():
                error_message = "Video transcript parsing failed - this may be a temporary YouTube API issue"
            elif "video unavailable" in error_message.lower():
                error_message = "Video is unavailable, private, or does not exist"
            elif "transcripts disabled" in error_message.lower():
                error_message = "Transcripts are disabled for this video"
            elif "http error" in error_message.lower():
                error_message = f"Network error accessing video: {error_message}"

            return {
                'video_id': video_id,
                'has_transcripts': False,
                'total_transcripts': 0,
                'manual_transcripts': 0,
                'auto_transcripts': 0,
                'available_languages': [],
                'manual_languages': [],
                'auto_languages': [],
                'error': error_message,
                'api_version': 'youtube-transcript-api-1.1.0+'
            }

    def get_available_transcript_languages(self, video_id: str) -> List[Dict[str, Any]]:
        """Get available transcript languages for a video"""
        try:
            transcript_list = self._get_transcript_list(video_id)

            available_languages = []

            for transcript in transcript_list:
                lang_info = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }
                available_languages.append(lang_info)

            return available_languages

        except Exception as e:
            return []
