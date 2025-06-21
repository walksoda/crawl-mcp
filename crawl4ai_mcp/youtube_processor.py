"""
YouTube Processing Module for Transcript Extraction
Handles YouTube video transcript retrieval using youtube-transcript-api v1.1.0+
Simple and reliable transcript extraction without complex authentication
"""

import asyncio
import re
import logging
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter


class YouTubeProcessor:
    """Process YouTube videos and extract transcripts"""
    
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
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get basic video information and available transcripts"""
        try:
            # Get transcript list to determine available languages
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
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
            
            # Handle specific errors with clearer messages
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
    
    async def extract_transcript(
        self,
        video_id: str,
        languages: Optional[List[str]] = None,
        translate_to: Optional[str] = None,
        include_timestamps: bool = True,
        preserve_formatting: bool = True
    ) -> Dict[str, Any]:
        """Extract transcript from YouTube video"""
        try:
            # Default language preferences
            if languages is None:
                languages = ['ja', 'en', 'en-US', 'en-GB']
            
            # Get transcript
            if translate_to:
                # Get any available transcript and translate
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(languages)
                translated_transcript = transcript.translate(translate_to)
                transcript_data = translated_transcript.fetch()
                source_language = transcript.language_code
                final_language = translate_to
                is_translated = True
            else:
                # Get transcript in preferred language
                transcript_data = YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=languages
                )
                # Determine which language was actually used
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    # Find the actual language used
                    source_language = 'unknown'
                    final_language = 'unknown'
                    is_translated = False
                    
                    for lang in languages:
                        try:
                            found_transcript = transcript_list.find_transcript([lang])
                            source_language = found_transcript.language_code
                            final_language = lang
                            is_translated = False
                            break
                        except:
                            continue
                    
                    # If we couldn't match, try to get any available transcript info
                    if source_language == 'unknown':
                        try:
                            for transcript in transcript_list:
                                source_language = transcript.language_code
                                final_language = transcript.language_code
                                break
                        except:
                            pass
                except Exception:
                    source_language = 'unknown'
                    final_language = 'unknown'
                    is_translated = False
            
            if not transcript_data:
                return {
                    'success': False,
                    'error': 'No transcript data found',
                    'video_id': video_id
                }
            
            # Process transcript data
            full_text = ""
            segments = []
            
            for entry in transcript_data:
                try:
                    text = entry.get('text', '')
                    start_time = entry.get('start', 0)
                    duration = entry.get('duration', 0)
                    
                    if include_timestamps:
                        timestamp = self._format_timestamp(start_time)
                        if preserve_formatting:
                            full_text += f"[{timestamp}] {text}\n"
                        else:
                            full_text += f"{text} "
                    else:
                        full_text += f"{text} "
                    
                    segments.append({
                        'text': text,
                        'start': start_time,
                        'duration': duration,
                        'end': start_time + duration
                    })
                except Exception as e:
                    # Skip malformed entries but continue processing
                    continue
            
            # Get clean text without timestamps
            try:
                clean_text = self.formatter.format_transcript(transcript_data)
            except Exception as e:
                # Fallback: create clean text manually
                clean_text = " ".join([entry.get('text', '') for entry in transcript_data if entry.get('text')])
            
            # Calculate statistics
            total_duration = max([seg['end'] for seg in segments]) if segments else 0
            word_count = len(clean_text.split())
            
            return {
                'success': True,
                'video_id': video_id,
                'source_language': source_language,
                'final_language': final_language,
                'is_translated': is_translated,
                'transcript_data': {
                    'full_text': full_text.strip(),
                    'clean_text': clean_text,
                    'segments': segments,
                    'segment_count': len(segments),
                    'word_count': word_count,
                    'duration_seconds': total_duration,
                    'duration_formatted': self._format_duration(total_duration)
                }
            }
            
        except TranscriptsDisabled:
            return {
                'success': False,
                'error': 'Transcripts are disabled for this video',
                'video_id': video_id
            }
        except NoTranscriptFound:
            return {
                'success': False,
                'error': f'No transcript found in languages: {languages}',
                'video_id': video_id,
                'requested_languages': languages
            }
        except Exception as e:
            error_message = str(e)
            
            # Handle specific errors with helpful messages
            if "no element found" in error_message.lower() or "parseerror" in error_message.lower():
                error_message = "YouTube transcript parsing failed - this may be a temporary issue with YouTube's servers"
            elif "http error" in error_message.lower():
                error_message = f"Network error accessing video: {error_message}"
            elif "video unavailable" in error_message.lower():
                error_message = "Video is unavailable, private, or has been removed"
            elif "could not retrieve" in error_message.lower():
                error_message = "Could not retrieve transcript data from YouTube"
            elif "transcripts disabled" in error_message.lower():
                error_message = "Transcripts are disabled for this video"
            
            return {
                'success': False,
                'error': f'Transcript extraction failed: {error_message}',
                'video_id': video_id,
                'api_version': 'youtube-transcript-api-1.1.0+',
                'suggestion': "Try using a different video or check if the video has publicly available transcripts"
            }
    
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
    
    async def process_youtube_url(
        self,
        url: str,
        languages: Optional[List[str]] = None,
        translate_to: Optional[str] = None,
        include_timestamps: bool = True,
        preserve_formatting: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Process YouTube URL and extract transcript"""
        
        if not self.is_youtube_url(url):
            return {
                'success': False,
                'error': 'URL is not a valid YouTube video URL',
                'url': url
            }
        
        video_id = self.extract_video_id(url)
        if not video_id:
            return {
                'success': False,
                'error': 'Could not extract video ID from URL',
                'url': url
            }
        
        try:
            # Get transcript
            transcript_result = await self.extract_transcript(
                video_id=video_id,
                languages=languages,
                translate_to=translate_to,
                include_timestamps=include_timestamps,
                preserve_formatting=preserve_formatting
            )
            
            if not transcript_result['success']:
                return transcript_result
            
            # Get video metadata if requested
            video_metadata = None
            if include_metadata:
                video_metadata = self.get_video_info(video_id)
            
            return {
                'success': True,
                'url': url,
                'video_id': video_id,
                'processing_method': 'youtube_transcript_api_v1.1.0+',
                'transcript': transcript_result['transcript_data'],
                'language_info': {
                    'source_language': transcript_result['source_language'],
                    'final_language': transcript_result['final_language'],
                    'is_translated': transcript_result['is_translated']
                },
                'metadata': video_metadata,
                'api_version': 'youtube-transcript-api-1.1.0+'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'YouTube processing failed: {str(e)}',
                'url': url,
                'video_id': video_id
            }
    
    async def batch_extract_transcripts(
        self,
        urls: List[str],
        languages: Optional[List[str]] = None,
        translate_to: Optional[str] = None,
        include_timestamps: bool = True,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract transcripts from multiple YouTube URLs"""
        
        async def process_single_url(url):
            return await self.process_youtube_url(
                url=url,
                languages=languages,
                translate_to=translate_to,
                include_timestamps=include_timestamps
            )
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(url):
            async with semaphore:
                return await process_single_url(url)
        
        # Process all URLs concurrently
        tasks = [process_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'url': urls[i],
                    'error': f'Processing failed: {str(result)}'
                })
            else:
                processed_results.append(result)
        
        return processed_results