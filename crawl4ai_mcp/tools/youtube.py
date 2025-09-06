"""
YouTube tools for Crawl4AI MCP Server.

Contains complete YouTube transcript extraction and video information tools.
"""

import asyncio
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import (
    YouTubeTranscriptResponse,
    YouTubeBatchRequest,
    YouTubeBatchResponse
)

# Import the YouTube processor
from ..youtube_processor import YouTubeProcessor

# Initialize YouTube processor
youtube_processor = YouTubeProcessor()


async def extract_youtube_transcript(
    url: Annotated[str, Field(description="YouTube video URL")],
    languages: Annotated[Optional[List[str]], Field(description="Preferred languages in order of preference (default: ['ja', 'en'])")] = ["ja", "en"],
    translate_to: Annotated[Optional[str], Field(description="Target language for translation (default: None)")] = None,
    include_timestamps: Annotated[bool, Field(description="Include timestamps in transcript (default: True)")] = True,
    preserve_formatting: Annotated[bool, Field(description="Preserve original formatting (default: True)")] = True,
    include_metadata: Annotated[bool, Field(description="Include video metadata (default: True)")] = True,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize long transcripts using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None
) -> YouTubeTranscriptResponse:
    """
    Extract YouTube video transcripts with timestamps and optional AI summarization.
    
    Works with public videos that have captions. No authentication required.
    Auto-detects available languages and falls back appropriately.
    
    Note: Automatic transcription may contain errors.
    """
    try:
        # Check if URL is valid YouTube URL
        if not youtube_processor.is_youtube_url(url):
            response = YouTubeTranscriptResponse(
                success=False,
                url=url,
                error="URL is not a valid YouTube video URL"
            )
            return response.model_dump()
        
        # Extract video ID
        video_id = youtube_processor.extract_video_id(url)
        if not video_id:
            response = YouTubeTranscriptResponse(
                success=False,
                url=url,
                error="Could not extract video ID from URL"
            )
            return response.model_dump()
        
        # Process the YouTube URL
        result = await youtube_processor.process_youtube_url(
            url=url,
            languages=languages,
            translate_to=translate_to,
            include_timestamps=include_timestamps,
            preserve_formatting=preserve_formatting,
            include_metadata=include_metadata
        )
        
        if not result['success']:
            # Enhanced error messaging for different environments
            import os
            import sys
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            
            base_error = result.get('error', 'Unknown error during transcript extraction')
            
            # Add UVX-specific guidance if applicable
            if is_uvx_env:
                enhanced_error = f"{base_error}\n\nUVX Environment Detected:\n" \
                    f"- If this worked in STDIO local setup, the issue may be UVX environment isolation\n" \
                    f"- YouTube API may behave differently in UVX vs local environments\n" \
                    f"- Try running system diagnostics: get_system_diagnostics()\n" \
                    f"- Consider switching to STDIO local setup for YouTube functionality"
            else:
                enhanced_error = f"{base_error}\n\nTroubleshooting:\n" \
                    f"- Correct method name: 'extract_youtube_transcript' (not 'get_transcript')\n" \
                    f"- Alternative methods: get_youtube_video_info, batch_extract_youtube_transcripts\n" \
                    f"- Check if video has available captions"
            
            response = YouTubeTranscriptResponse(
                success=False,
                url=url,
                video_id=video_id,
                error=enhanced_error,
                metadata={
                    'uvx_environment': is_uvx_env,
                    'correct_method_name': 'extract_youtube_transcript',
                    'alternative_methods': ['get_youtube_video_info', 'batch_extract_youtube_transcripts'],
                    'diagnostic_tool': 'get_system_diagnostics'
                }
            )
            return response.model_dump()
        
        # Get transcript data
        transcript_data = result['transcript']
        language_info = result.get('language_info', {})
        metadata = result.get('metadata', {})
        
        # Apply auto-summarization if requested and content exceeds token limit
        if auto_summarize and transcript_data.get('full_text'):
            # Rough token estimation: 1 token ≈ 4 characters
            estimated_tokens = len(transcript_data['full_text']) // 4
            
            # Only summarize if content exceeds the specified token limit
            if estimated_tokens > max_content_tokens:
                try:
                    # Prepare video metadata for enhanced summarization
                    video_metadata = {
                        'title': metadata.get('title', ''),
                        'url': url,
                        'video_id': result.get('video_id', ''),
                        'channel': metadata.get('channel', ''),
                        'description': metadata.get('description', '')
                    }
                    
                    summary_result = await youtube_processor.summarize_transcript(
                        transcript_text=transcript_data['full_text'],
                        summary_length=summary_length,
                        include_timestamps=include_timestamps,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        video_metadata=video_metadata,
                        target_tokens=max_content_tokens
                    )
                    
                    if summary_result.get('success'):
                        # Add enhanced summarization info to metadata
                        metadata.update({
                            'summarization_applied': True,
                            'original_length': len(transcript_data['full_text']),
                            'original_tokens_estimate': estimated_tokens,
                            'summary_length_setting': summary_length,
                            'target_tokens': summary_result.get('target_tokens', max_content_tokens),
                            'estimated_summary_tokens': summary_result.get('estimated_summary_tokens', 0),
                            'compression_ratio': summary_result.get('compression_ratio', 0),
                            'llm_provider': summary_result.get('llm_provider', 'unknown'),
                            'llm_model': summary_result.get('llm_model', 'unknown'),
                            'summarization_trigger': f'Content exceeded {max_content_tokens} tokens',
                            # Preserve video metadata from summary
                            'video_title_preserved': summary_result.get('video_title', ''),
                            'video_url_preserved': summary_result.get('video_url', ''),
                            'channel_name_preserved': summary_result.get('channel_name', ''),
                            'key_topics_identified': summary_result.get('key_topics', [])
                        })
                        
                        # Replace content with summary
                        transcript_data['full_text'] = summary_result['summary']
                        transcript_data['clean_text'] = summary_result['summary']
                    else:
                        # Summarization failed, add error info
                        metadata.update({
                            'summarization_attempted': True,
                            'summarization_error': summary_result.get('error', 'Unknown error'),
                            'original_content_preserved': True
                        })
                except Exception as e:
                    # Summarization failed, add error info
                    metadata.update({
                        'summarization_attempted': True,
                        'summarization_error': f'Exception during summarization: {str(e)}',
                        'original_content_preserved': True
                    })
            else:
                # Content is below threshold - preserve original content and add info
                metadata.update({
                    'auto_summarize_requested': True,
                    'original_content_preserved': True,
                    'content_below_threshold': True,
                    'tokens_estimate': estimated_tokens,
                    'max_tokens_threshold': max_content_tokens,
                    'reason': f'Content ({estimated_tokens} tokens) is below threshold ({max_content_tokens} tokens)'
                })
        
        response = YouTubeTranscriptResponse(
            success=True,
            url=url,
            video_id=video_id,
            transcript=transcript_data,
            language_info=language_info,
            metadata=metadata,
            processing_method="youtube_transcript_api"
        )
        return response.model_dump()
                
    except Exception as e:
        response = YouTubeTranscriptResponse(
            success=False,
            url=url,
            error=f"YouTube transcript processing error: {str(e)}"
        )
        return response.model_dump()


async def batch_extract_youtube_transcripts(
    request: Annotated[Dict[str, Any], Field(description="YouTubeBatchRequest with URLs and extraction parameters")]
) -> YouTubeBatchResponse:
    """
    Extract transcripts from multiple YouTube videos using youtube-transcript-api.
    
    Processes multiple YouTube URLs concurrently with controlled rate limiting.
    No authentication required for public videos with captions.
    
    Note: Automatic transcription may contain errors.
    """
    try:
        # Extract parameters from request
        urls = request.get('urls', [])
        languages = request.get('languages', ['ja', 'en'])
        max_concurrent = min(request.get('max_concurrent', 3), 5)  # Limit max concurrency
        include_timestamps = request.get('include_timestamps', True)
        translate_to = request.get('translate_to')
        preserve_formatting = request.get('preserve_formatting', True)
        include_metadata = request.get('include_metadata', True)
        
        if not urls:
            return YouTubeBatchResponse(
                success=False,
                total_urls=0,
                successful_extractions=0,
                failed_extractions=0,
                results=[],
                error="No URLs provided in request"
            )
        
        # Limit number of URLs to prevent abuse
        if len(urls) > 20:
            urls = urls[:20]
        
        # Process URLs with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_url(url: str) -> YouTubeTranscriptResponse:
            async with semaphore:
                # Add small delay to prevent rate limiting
                await asyncio.sleep(0.1)
                return await extract_youtube_transcript(
                    url=url,
                    languages=languages,
                    translate_to=translate_to,
                    include_timestamps=include_timestamps,
                    preserve_formatting=preserve_formatting,
                    include_metadata=include_metadata
                )
        
        # Process all URLs concurrently
        tasks = [process_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        successful_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(YouTubeTranscriptResponse(
                    success=False,
                    url=urls[i],
                    error=f"Processing exception: {str(result)}"
                ))
                failed_count += 1
            else:
                processed_results.append(result)
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
        
        return YouTubeBatchResponse(
            success=True,
            total_urls=len(urls),
            successful_extractions=successful_count,
            failed_extractions=failed_count,
            results=processed_results,
            metadata={
                'max_concurrent_used': max_concurrent,
                'processing_method': 'batch_youtube_transcript_api',
                'rate_limiting_applied': True
            }
        )
        
    except Exception as e:
        return YouTubeBatchResponse(
            success=False,
            total_urls=len(request.get('urls', [])),
            successful_extractions=0,
            failed_extractions=len(request.get('urls', [])),
            results=[],
            error=f"Batch processing error: {str(e)}"
        )


async def get_youtube_video_info(
    video_url: Annotated[str, Field(description="YouTube video URL")],
    summarize_transcript: Annotated[bool, Field(description="Summarize long transcripts using LLM (default: False)")] = False,
    max_tokens: Annotated[int, Field(description="Token limit before triggering summarization (default: 25000)")] = 25000,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model to use (default: auto-detected)")] = None,
    summary_length: Annotated[str, Field(description="Summary length - 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    include_timestamps: Annotated[bool, Field(description="Preserve key timestamps in summary (default: True)")] = True
) -> Dict[str, Any]:
    """
    Get YouTube video information with optional transcript summarization.
    
    Retrieves basic video information and transcript availability using youtube-transcript-api.
    No authentication required for public videos.
    
    Note: Automatic transcription may contain errors.
    """
    try:
        # Check if URL is valid YouTube URL
        if not youtube_processor.is_youtube_url(video_url):
            return {
                "success": False,
                "url": video_url,
                "error": "URL is not a valid YouTube video URL"
            }
        
        # Extract video ID
        video_id = youtube_processor.extract_video_id(video_url)
        if not video_id:
            return {
                "success": False,
                "url": video_url,
                "error": "Could not extract video ID from URL"
            }
        
        # Get basic video information
        video_info = youtube_processor.get_video_info(video_id)
        
        # Try to get transcript information
        transcript_info = {}
        try:
            # Get available transcript languages
            available_transcripts = youtube_processor.get_available_transcript_languages(video_id)
            transcript_info = {
                "has_transcript": len(available_transcripts) > 0,
                "available_languages": available_transcripts,
                "manually_created_count": len([t for t in available_transcripts if t.get('is_generated', True) == False]),
                "auto_generated_count": len([t for t in available_transcripts if t.get('is_generated', True) == True])
            }
            
            # If transcript is available and summarization is requested
            if transcript_info["has_transcript"] and summarize_transcript:
                # Extract transcript
                transcript_result = await extract_youtube_transcript(
                    url=video_url,
                    languages=["en", "ja"],  # Default languages
                    include_timestamps=include_timestamps,
                    auto_summarize=True,
                    max_content_tokens=max_tokens,
                    summary_length=summary_length,
                    llm_provider=llm_provider,
                    llm_model=llm_model
                )
                
                if transcript_result.success:
                    transcript_info.update({
                        "transcript_summary": transcript_result.transcript.get('full_text', ''),
                        "original_length": transcript_result.metadata.get('original_length', 0),
                        "summarization_applied": transcript_result.metadata.get('summarization_applied', False),
                        "processing_method": transcript_result.processing_method
                    })
                else:
                    transcript_info["transcript_error"] = transcript_result.error
                    
        except Exception as e:
            transcript_info = {
                "has_transcript": False,
                "transcript_check_error": str(e)
            }
        
        return {
            "success": True,
            "url": video_url,
            "video_id": video_id,
            "video_info": video_info,
            "transcript_info": transcript_info,
            "processing_method": "youtube_video_info_api"
        }
        
    except Exception as e:
        return {
            "success": False,
            "url": video_url,
            "error": f"Video info processing error: {str(e)}"
        }


async def get_youtube_api_setup_guide() -> Dict[str, Any]:
    """
    Get setup information for youtube-transcript-api integration.
    
    Provides information about current youtube-transcript-api setup.
    No authentication or API keys required for basic transcript extraction.
    """
    try:
        return {
            "success": True,
            "api_name": "youtube-transcript-api",
            "authentication_required": False,
            "api_key_required": False,
            "rate_limits": {
                "requests_per_minute": "No official limit, but recommended: 60 requests/minute",
                "concurrent_requests": "Recommended: 3-5 concurrent requests",
                "note": "Rate limiting is applied automatically in batch operations"
            },
            "supported_features": {
                "transcript_extraction": True,
                "multiple_languages": True,
                "auto_generated_transcripts": True,
                "manually_created_transcripts": True,
                "timestamp_support": True,
                "translation_support": True,
                "batch_processing": True,
                "video_metadata": True,
                "ai_summarization": True
            },
            "supported_youtube_formats": [
                "https://www.youtube.com/watch?v=VIDEO_ID",
                "https://youtu.be/VIDEO_ID",
                "https://youtube.com/watch?v=VIDEO_ID",
                "https://www.youtube.com/embed/VIDEO_ID",
                "https://www.youtube.com/v/VIDEO_ID"
            ],
            "language_support": {
                "extraction_languages": [
                    "English (en)", "Japanese (ja)", "Spanish (es)", "French (fr)", 
                    "German (de)", "Italian (it)", "Portuguese (pt)", "Russian (ru)",
                    "Chinese (zh)", "Korean (ko)", "Arabic (ar)", "Hindi (hi)",
                    "And many more..."
                ],
                "translation_support": True,
                "auto_language_detection": True
            },
            "usage_tips": [
                "No API key required - works directly with YouTube's public transcript data",
                "Best results with videos that have manually created captions",
                "Auto-generated captions available for many videos",
                "Use batch processing for multiple videos to improve efficiency",
                "Enable AI summarization for long transcripts to reduce token usage",
                "Respect YouTube's terms of service when using transcript data"
            ],
            "limitations": [
                "Only works with public videos that have transcripts available",
                "Some videos may not have transcripts (especially older videos)",
                "Private or unlisted videos are not accessible",
                "Age-restricted videos may have limited access",
                "Live streams may not have stable transcript access"
            ],
            "error_handling": {
                "transcript_disabled": "Video owner has disabled transcripts",
                "no_transcript_found": "No transcript available in requested language",
                "video_unavailable": "Video is private, deleted, or restricted",
                "connection_error": "Network or YouTube service issue"
            },
            "ai_summarization": {
                "supported": True,
                "trigger_threshold": "15,000 tokens (approximately 60,000 characters)",
                "summary_lengths": ["short", "medium", "long"],
                "preserves_timestamps": True,
                "llm_providers": "Auto-detected based on configuration"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Setup guide generation error: {str(e)}"
        }