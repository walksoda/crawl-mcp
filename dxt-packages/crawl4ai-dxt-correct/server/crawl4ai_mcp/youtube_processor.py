"""
YouTube Processing Module for Transcript Extraction
Handles YouTube video transcript retrieval using youtube-transcript-api v1.1.0+
Simple and reliable transcript extraction without complex authentication
"""

import asyncio
import re
import logging
import os
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
    
    def _get_transcript_list(self, video_id):
        """
        youtube-transcript-api v1.2.1の新APIを使用してtranscript_listを返す。
        """
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        return api.list(video_id)

    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get basic video information and available transcripts"""
        try:
            # Get transcript list to determine available languages
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
    
    def get_available_transcript_languages(self, video_id: str) -> List[Dict[str, Any]]:
        """Get available transcript languages for a video"""
        try:
            # Get transcript list to determine available languages
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
            # Return empty list if no transcripts available or error occurs
            return []
    
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
            
            # Get transcript using modern API approach with enhanced error handling
            transcript_list = self._get_transcript_list(video_id)
            
            if translate_to:
                # Get any available transcript and translate
                try:
                    transcript = transcript_list.find_transcript(languages)
                    translated_transcript = transcript.translate(translate_to)
                    transcript_data = translated_transcript.fetch()
                    source_language = transcript.language_code
                    final_language = translate_to
                    is_translated = True
                except Exception as e:
                    # Try to get any available transcript for translation
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
                # Get transcript in preferred language using modern approach
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
        """
        Summarize a long transcript using LLM with enhanced metadata preservation
        
        Args:
            transcript_text: The full transcript text to summarize
            summary_length: "short", "medium", or "long" summary
            include_timestamps: Whether to preserve key timestamps
            llm_provider: LLM provider to use
            llm_model: Specific model to use
            video_metadata: Video metadata (title, channel, description, etc.)
            target_tokens: Target token count for summary (if specified)
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            # Import config here to avoid circular imports
            try:
                from .config import get_llm_config
            except ImportError:
                from config import get_llm_config
            
            # Extract video metadata
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
            
            # Define summary lengths with token targets
            length_configs = {
                "short": {
                    "target_length": "1-2 paragraphs",
                    "detail_level": "key points only",
                    "target_tokens": target_tokens or 300
                },
                "medium": {
                    "target_length": "3-5 paragraphs", 
                    "detail_level": "main topics with key details",
                    "target_tokens": target_tokens or 800
                },
                "long": {
                    "target_length": "6-10 paragraphs",
                    "detail_level": "comprehensive overview with subtopics",
                    "target_tokens": target_tokens or 1500
                }
            }
            
            config = length_configs.get(summary_length, length_configs["medium"])
            
            # Prepare enhanced instruction for LLM with metadata
            timestamp_instruction = (
                "Include key timestamps in your summary to help readers navigate to important sections."
                if include_timestamps else
                "Do not include timestamps in your summary."
            )
            
            # Create metadata context
            metadata_context = f"""
Video Information:
- Title: {video_title}
- Channel: {channel_name}
- Video ID: {video_id}
- URL: {video_url}
{f"- Description: {video_description[:200]}..." if video_description else ""}
"""
            
            instruction = f"""
            Summarize this YouTube video transcript in {config['target_length']}.
            Focus on {config['detail_level']}.
            Target length: approximately {config['target_tokens']} tokens.
            {timestamp_instruction}
            
            {metadata_context}
            
            Structure your summary with:
            1. Brief overview including video title and channel context
            2. Main topics or sections discussed
            3. Key insights or conclusions
            4. Important details or examples mentioned
            
            Make the summary engaging and informative, preserving the tone and style of the original content.
            IMPORTANT: Preserve the video title, channel name, and URL in your response for reference.
            """
            
            # Get LLM configuration
            llm_config = get_llm_config(llm_provider, llm_model)
            
            # Use direct LLM API call for summarization
            import json
            
            # Create the prompt for summarization
            prompt = f"""
            {instruction}
            
            Please provide a JSON response with the following structure:
            {{
                "summary": "The summarized content (approximately {config['target_tokens']} tokens)",
                "video_title": "{video_title}",
                "video_url": "{video_url}",
                "video_id": "{video_id}",
                "channel_name": "{channel_name}",
                "key_topics": ["List", "of", "main", "topics"],
                "key_timestamps": ["Important timestamps if preserved"],
                "content_type": "Type of video content",
                "duration_estimate": "Estimated reading time",
                "summary_token_count": "Estimated token count of summary"
            }}
            
            Transcript to summarize:
            {transcript_text}
            """
            
            # Use the LLM config to make direct API call
            if hasattr(llm_config, 'provider'):
                provider_info = llm_config.provider.split('/')
                provider = provider_info[0] if provider_info else 'openai'
                model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
                
                if provider == 'openai':
                    import openai
                    
                    # Get API key from config
                    api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                    
                    if not api_key:
                        raise ValueError("OpenAI API key not found")
                    
                    client = openai.AsyncOpenAI(api_key=api_key)
                    
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts while preserving important metadata."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=min(4000, config['target_tokens'] * 2)  # Allow up to 2x target for flexibility
                    )
                    
                    extracted_content = response.choices[0].message.content
                else:
                    raise ValueError(f"Provider {provider} not supported in direct mode")
            else:
                raise ValueError("Invalid LLM config format")
            
            if extracted_content:
                try:
                    import json
                    # Clean up the extracted content if it's wrapped in markdown
                    content_to_parse = extracted_content
                    if content_to_parse.startswith('```json'):
                        content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                    
                    summary_data = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                    
                    # Ensure metadata is preserved
                    return {
                        "success": True,
                        "summary": summary_data.get("summary", "Summary generation failed"),
                        "video_title": summary_data.get("video_title", video_title),
                        "video_url": summary_data.get("video_url", video_url),
                        "video_id": summary_data.get("video_id", video_id),
                        "channel_name": summary_data.get("channel_name", channel_name),
                        "key_topics": summary_data.get("key_topics", []),
                        "key_timestamps": summary_data.get("key_timestamps", []) if include_timestamps else [],
                        "content_type": summary_data.get("content_type", "Unknown"),
                        "summary_length": summary_length,
                        "target_tokens": config['target_tokens'],
                        "estimated_summary_tokens": len(summary_data.get("summary", "")) // 4,  # Rough estimate
                        "original_length": len(transcript_text),
                        "compression_ratio": len(summary_data.get("summary", "")) / len(transcript_text) if transcript_text else 0,
                        "llm_provider": llm_config.get("provider") if isinstance(llm_config, dict) else "unknown",
                        "llm_model": llm_config.get("model") if isinstance(llm_config, dict) else "unknown"
                    }
                except (json.JSONDecodeError, AttributeError) as e:
                    # Fallback: treat as plain text summary
                    return {
                        "success": True,
                        "summary": str(extracted_content),
                        "video_title": video_title,
                        "video_url": video_url,
                        "video_id": video_id,
                        "channel_name": channel_name,
                        "key_topics": [],
                        "key_timestamps": [],
                        "content_type": "Unknown",
                        "summary_length": summary_length,
                        "target_tokens": config['target_tokens'],
                        "estimated_summary_tokens": len(str(extracted_content)) // 4,
                        "original_length": len(transcript_text),
                        "compression_ratio": len(str(extracted_content)) / len(transcript_text) if transcript_text else 0,
                        "llm_provider": llm_config.get("provider") if isinstance(llm_config, dict) else "unknown",
                        "llm_model": llm_config.get("model") if isinstance(llm_config, dict) else "unknown",
                        "fallback_mode": True
                    }
            else:
                return {
                    "success": False,
                    "error": "LLM extraction returned empty result",
                    "summary_length": summary_length
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Summarization failed: {str(e)}",
                "summary_length": summary_length
            }