"""
YouTube Processing Module for Transcript Extraction
Handles YouTube video transcript retrieval using youtube-transcript-api v1.1.0+
Simple and reliable transcript extraction without complex authentication
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

from .youtube_processor_base import YouTubeProcessorBase


class YouTubeProcessor(YouTubeProcessorBase):
    """Process YouTube videos and extract transcripts"""

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
            if languages is None:
                languages = ['ja', 'en', 'en-US', 'en-GB']

            transcript_list = self._get_transcript_list(video_id)

            if translate_to:
                try:
                    transcript = transcript_list.find_transcript(languages)
                    translated_transcript = transcript.translate(translate_to)
                    transcript_data = translated_transcript.fetch()
                    source_language = transcript.language_code
                    final_language = translate_to
                    is_translated = True
                except Exception as e:
                    try:
                        transcript = transcript_list.find_transcript(['en', 'ja', 'es', 'fr', 'de', 'it', 'pt', 'ru'])
                        translated_transcript = transcript.translate(translate_to)
                        transcript_data = translated_transcript.fetch()
                        source_language = transcript.language_code
                        final_language = translate_to
                        is_translated = True
                    except Exception as e2:
                        return {
                            'success': False,
                            'error': f'No transcripts available for translation to {translate_to}. Original error: {str(e)}',
                            'video_id': video_id,
                            'available_transcripts': [t.language_code for t in transcript_list],
                            'suggestion': 'Try get_youtube_video_info to see available transcript languages'
                        }
            else:
                try:
                    transcript = transcript_list.find_transcript(languages)
                    transcript_data = transcript.fetch()
                    source_language = transcript.language_code
                    final_language = transcript.language_code
                    is_translated = False
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'No transcripts found in requested languages {languages}. Error: {str(e)}',
                        'video_id': video_id,
                        'available_transcripts': [t.language_code for t in transcript_list],
                        'suggestion': 'Try get_youtube_video_info to see available transcript languages, or use batch_extract_youtube_transcripts for alternative methods'
                    }

            if not transcript_data:
                return {
                    'success': False,
                    'error': 'No transcript data found',
                    'video_id': video_id
                }

            # Process transcript data
            # Note: youtube-transcript-api v1.2.x returns FetchedTranscriptSnippet objects
            # with text, start, duration as attributes (not dict keys)
            full_text = ""
            segments = []

            for entry in transcript_data:
                try:
                    # Support both dict (old API) and object (new API v1.2.x) formats
                    if hasattr(entry, 'text'):
                        text = entry.text
                        start_time = entry.start
                        duration = entry.duration
                    else:
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
                    continue

            # Get clean text without timestamps
            try:
                clean_text = self.formatter.format_transcript(transcript_data)
            except Exception as e:
                clean_texts = []
                for entry in transcript_data:
                    if hasattr(entry, 'text'):
                        clean_texts.append(entry.text)
                    elif hasattr(entry, 'get') and entry.get('text'):
                        clean_texts.append(entry.get('text'))
                clean_text = " ".join(clean_texts)

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
            elif "transcript not found" in error_message.lower():
                error_message = "No transcript found for the requested languages"
            elif "list index out of range" in error_message.lower():
                error_message = "Video parsing error - transcript data structure unexpected"
            elif "connection" in error_message.lower() or "timeout" in error_message.lower():
                error_message = "Network connection issue - please try again later"

            return {
                'success': False,
                'error': f'Transcript extraction failed: {error_message}',
                'video_id': video_id,
                'api_version': 'youtube-transcript-api-1.1.0+',
                'suggestion': self._get_error_suggestion(error_message),
                'retry_recommended': "connection" in error_message.lower() or "timeout" in error_message.lower() or "temporary" in error_message.lower()
            }

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
            transcript_result = await self.extract_transcript(
                video_id=video_id,
                languages=languages,
                translate_to=translate_to,
                include_timestamps=include_timestamps,
                preserve_formatting=preserve_formatting
            )

            if not transcript_result['success']:
                return transcript_result

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

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(url):
            async with semaphore:
                return await process_single_url(url)

        tasks = [process_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

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

    async def extract_comments(
        self,
        video_id: str,
        url: str,
        sort_by: str = "popular",
        max_comments: int = 300,
        include_replies: bool = True,
        comment_offset: int = 0
    ) -> Dict[str, Any]:
        """Extract comments from YouTube video using youtube-comment-downloader."""
        try:
            from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT
        except ImportError:
            return {
                'success': False,
                'error': 'youtube-comment-downloader is not installed. Install with: pip install youtube-comment-downloader==0.1.78',
                'video_id': video_id
            }

        sort_map = {
            "popular": SORT_BY_POPULAR,
            "recent": SORT_BY_RECENT,
        }
        sort_constant = sort_map.get(sort_by, SORT_BY_POPULAR)

        def _download_comments():
            downloader = YoutubeCommentDownloader()
            try:
                generator = downloader.get_comments_from_url(url, sort_by=sort_constant)
            except Exception as gen_error:
                raise RuntimeError(f"Failed to initialize comment download: {gen_error}")

            comments = []
            skipped = 0
            collected = 0
            has_more = False
            total_seen = 0

            for raw_comment in generator:
                total_seen += 1
                is_reply = raw_comment.get('reply', False)

                if not include_replies and is_reply:
                    continue

                if skipped < comment_offset:
                    skipped += 1
                    continue

                if collected >= max_comments:
                    has_more = True
                    break

                comment = {
                    'cid': raw_comment.get('cid', ''),
                    'text': raw_comment.get('text', ''),
                    'author': raw_comment.get('author', ''),
                    'time': raw_comment.get('time', ''),
                    'votes': raw_comment.get('votes', '0'),
                    'replies': raw_comment.get('replies', 0),
                    'heart': raw_comment.get('heart', False),
                    'is_reply': is_reply,
                }
                comments.append(comment)
                collected += 1

            if total_seen == 0:
                # Generator yielded nothing - likely a network/access issue
                raise RuntimeError(
                    "Comment downloader returned no data. This may indicate: "
                    "comments are disabled on this video, the video is private/unavailable, "
                    "or the downloader cannot reach YouTube (e.g., network restrictions in Docker)."
                )

            return comments, has_more

        try:
            comments, has_more = await asyncio.wait_for(
                asyncio.to_thread(_download_comments),
                timeout=120
            )
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Comment extraction timed out after 120 seconds. Try reducing max_comments or using comment_offset for pagination.',
                'video_id': video_id
            }
        except Exception as e:
            error_msg = str(e)
            if "disable" in error_msg.lower() or "unavailable" in error_msg.lower():
                return {
                    'success': False,
                    'error': f'Comments are disabled or video is unavailable: {error_msg}',
                    'video_id': video_id
                }
            return {
                'success': False,
                'error': f'Comment extraction failed: {error_msg}',
                'video_id': video_id
            }

        if not comments:
            return {
                'success': True,
                'video_id': video_id,
                'comments': [],
                'has_more': False,
                'comment_stats': {
                    'total_comments': 0,
                    'top_level_comments': 0,
                    'reply_comments': 0,
                    'unique_authors': 0
                }
            }

        top_level = [c for c in comments if not c['is_reply']]
        reply_comments = [c for c in comments if c['is_reply']]
        unique_authors = len(set(c['author'] for c in comments if c['author']))

        return {
            'success': True,
            'video_id': video_id,
            'comments': comments,
            'has_more': has_more,
            'comment_stats': {
                'total_comments': len(comments),
                'top_level_comments': len(top_level),
                'reply_comments': len(reply_comments),
                'unique_authors': unique_authors
            }
        }

    async def summarize_transcript(
        self,
        transcript_text: str,
        summary_length: str = "medium",
        include_timestamps: bool = True,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        target_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Summarize a long transcript using LLMClient with enhanced metadata preservation"""
        try:
            from ..utils.llm_client import LLMClient

            video_title = ""
            video_url = ""
            video_id = ""
            channel_name = ""
            video_description = ""

            if video_metadata:
                video_title = video_metadata.get('title', '')
                video_url = video_metadata.get('url', '')
                video_id = video_metadata.get('video_id', '')
                channel_name = video_metadata.get('channel', '')
                video_description = video_metadata.get('description', '')

            metadata = {
                "video_title": video_title,
                "video_url": video_url,
                "video_id": video_id,
                "channel_name": channel_name,
                "include_timestamps": include_timestamps,
            }
            if video_description:
                metadata["description"] = video_description[:200]

            client = LLMClient()
            result = await client.summarize(
                content=transcript_text,
                title=video_title,
                url=video_url,
                summary_length=summary_length,
                content_type="video",
                llm_provider=llm_provider,
                llm_model=llm_model,
                target_tokens=target_tokens,
                metadata=metadata
            )

            if result.get("success"):
                return {
                    "success": True,
                    "summary": result.get("summary", "Summary generation failed"),
                    "video_title": video_title or result.get("title", ""),
                    "video_url": video_url or result.get("source_url", ""),
                    "video_id": video_id,
                    "channel_name": channel_name,
                    "key_topics": result.get("key_topics", []),
                    "key_timestamps": [] if not include_timestamps else result.get("key_timestamps", []),
                    "content_type": result.get("content_type", "video"),
                    "summary_length": summary_length,
                    "target_tokens": result.get("target_tokens", 800),
                    "estimated_summary_tokens": result.get("estimated_summary_tokens", 0),
                    "original_length": result.get("original_length", len(transcript_text)),
                    "compression_ratio": result.get("compression_ratio", 0),
                    "llm_provider": result.get("llm_provider", "unknown"),
                    "llm_model": result.get("llm_model", "unknown"),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Summarization failed"),
                    "summary_length": summary_length
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Summarization failed: {str(e)}",
                "summary_length": summary_length
            }
