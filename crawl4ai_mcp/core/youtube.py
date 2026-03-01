"""Core YouTube transcript extraction and video information logic."""

import asyncio
import os
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

from ..models import (
    YouTubeTranscriptResponse,
    YouTubeBatchRequest,
    YouTubeBatchResponse
)

from ..processors.youtube_processor import YouTubeProcessor
from .youtube_helpers import _crawl_youtube_page_fallback

# Initialize YouTube processor
youtube_processor = YouTubeProcessor()


async def extract_youtube_transcript(
    url: str,
    languages: Optional[List[str]] = None,
    translate_to: Optional[str] = None,
    include_timestamps: bool = False,
    preserve_formatting: bool = True,
    include_metadata: bool = True,
    auto_summarize: bool = False,
    max_content_tokens: int = 15000,
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    enable_crawl_fallback: bool = True,
    fallback_timeout: int = 60,
    enrich_metadata: bool = True
) -> Dict[str, Any]:
    """Extract YouTube video transcripts with timestamps and optional AI summarization.

    Returns a dict with content/markdown/extracted_data format (CrawlResponse-compatible).
    """
    if languages is None:
        languages = ["ja", "en"]

    try:
        # Check if URL is valid YouTube URL
        if not youtube_processor.is_youtube_url(url):
            return YouTubeTranscriptResponse(
                success=False, url=url,
                error="URL is not a valid YouTube video URL"
            ).model_dump()

        # Extract video ID
        video_id = youtube_processor.extract_video_id(url)
        if not video_id:
            return YouTubeTranscriptResponse(
                success=False, url=url,
                error="Could not extract video ID from URL"
            ).model_dump()

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
            import sys
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            api_error = result.get('error', 'Unknown error during transcript extraction')

            # Attempt fallback using crawl_url if enabled
            if enable_crawl_fallback:
                fallback_result = await _crawl_youtube_page_fallback(
                    url=url, video_id=video_id,
                    wait_for_js=True, timeout=fallback_timeout
                )

                if fallback_result['success']:
                    fallback_transcript = fallback_result.get('transcript', {})
                    fallback_metadata = fallback_result.get('metadata', {})
                    fallback_metadata.update({
                        'fallback_used': True,
                        'original_api_error': api_error,
                        'extraction_source': 'page_crawl',
                        'processing_note': 'Transcript API failed, content extracted from page crawl'
                    })

                    title = fallback_metadata.get('title') or f"YouTube: {video_id}"
                    return YouTubeTranscriptResponse(
                        success=True, url=url, video_id=video_id,
                        title=title,
                        content=fallback_transcript.get('full_text', ''),
                        markdown=fallback_transcript.get('clean_text', ''),
                        extracted_data={
                            "video_id": video_id,
                            "processing_method": "crawl_url_fallback",
                            "language_info": {},
                            "transcript_stats": {
                                "word_count": fallback_transcript.get('word_count'),
                                "segment_count": fallback_transcript.get('segment_count', 0),
                            },
                            "metadata": fallback_metadata
                        }
                    ).model_dump()
                else:
                    fallback_error = fallback_result.get('error', 'Unknown fallback error')
                    return YouTubeTranscriptResponse(
                        success=False, url=url, video_id=video_id,
                        error=f"Both extraction methods failed.\n\n"
                              f"API Error: {api_error}\n\n"
                              f"Fallback Error: {fallback_error}"
                    ).model_dump()

            # Fallback disabled - return original error
            enhanced_error = f"{api_error}\n\nTroubleshooting: Try enable_crawl_fallback=True"
            return YouTubeTranscriptResponse(
                success=False, url=url, video_id=video_id, error=enhanced_error
            ).model_dump()

        # Get transcript data
        transcript_data = result['transcript']
        language_info = result.get('language_info') or {}
        metadata = result.get('metadata') or {}

        # Enrich metadata using crawl_url if requested
        if enrich_metadata:
            try:
                enrichment_result = await _crawl_youtube_page_fallback(
                    url=url, video_id=video_id,
                    wait_for_js=True, timeout=fallback_timeout
                )

                if enrichment_result['success']:
                    enriched_metadata = enrichment_result.get('metadata', {})
                    enrichment_fields = ['title', 'upload_date', 'view_count', 'duration', 'like_count', 'channel_name']
                    for field in enrichment_fields:
                        if enriched_metadata.get(field) and not metadata.get(field):
                            metadata[field] = enriched_metadata[field]

                    metadata['metadata_enriched'] = True
                    metadata['enrichment_source'] = 'page_crawl'
                else:
                    metadata['metadata_enrichment_attempted'] = True
                    metadata['metadata_enrichment_error'] = enrichment_result.get('error', 'Unknown error')
            except Exception as e:
                metadata['metadata_enrichment_attempted'] = True
                metadata['metadata_enrichment_error'] = f'Exception: {str(e)}'

        # Apply auto-summarization if requested and content exceeds token limit
        if auto_summarize and transcript_data.get('full_text'):
            estimated_tokens = len(transcript_data['full_text']) // 4

            if estimated_tokens > max_content_tokens:
                try:
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
                            'video_title_preserved': summary_result.get('video_title', ''),
                            'video_url_preserved': summary_result.get('video_url', ''),
                            'channel_name_preserved': summary_result.get('channel_name', ''),
                            'key_topics_identified': summary_result.get('key_topics', [])
                        })
                        transcript_data['full_text'] = summary_result['summary']
                        transcript_data['clean_text'] = summary_result['summary']
                    else:
                        metadata.update({
                            'summarization_attempted': True,
                            'summarization_error': summary_result.get('error', 'Unknown error'),
                            'original_content_preserved': True
                        })
                except Exception as e:
                    metadata.update({
                        'summarization_attempted': True,
                        'summarization_error': f'Exception during summarization: {str(e)}',
                        'original_content_preserved': True
                    })
            else:
                metadata.update({
                    'auto_summarize_requested': True,
                    'original_content_preserved': True,
                    'content_below_threshold': True,
                    'tokens_estimate': estimated_tokens,
                    'max_tokens_threshold': max_content_tokens,
                    'reason': f'Content ({estimated_tokens} tokens) is below threshold ({max_content_tokens} tokens)'
                })

        # Determine processing method based on enrichment
        processing_method = "youtube_transcript_api"
        if metadata.get('metadata_enriched'):
            processing_method = "youtube_transcript_api_enriched"

        title = metadata.get('title') or f"YouTube: {video_id}"

        response = YouTubeTranscriptResponse(
            success=True, url=url, video_id=video_id,
            title=title,
            content=transcript_data.get('full_text', ''),
            markdown=transcript_data.get('clean_text', ''),
            extracted_data={
                "video_id": video_id,
                "processing_method": processing_method,
                "language_info": language_info,
                "transcript_stats": {
                    "word_count": transcript_data.get('word_count'),
                    "segment_count": transcript_data.get('segment_count'),
                    "duration": transcript_data.get('duration_formatted'),
                    "duration_seconds": transcript_data.get('duration_seconds')
                },
                "metadata": metadata
            }
        )
        return response.model_dump()

    except Exception as e:
        return YouTubeTranscriptResponse(
            success=False, url=url,
            error=f"YouTube transcript processing error: {str(e)}"
        ).model_dump()


async def batch_extract_youtube_transcripts(
    request: Dict[str, Any]
) -> Dict[str, Any]:
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
        max_concurrent = min(request.get('max_concurrent', 3), 5)
        include_timestamps = request.get('include_timestamps', False)
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
            ).model_dump()

        # Limit number of URLs to prevent abuse
        if len(urls) > 20:
            urls = urls[:20]

        # Process URLs with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_url(url: str) -> Dict[str, Any]:
            async with semaphore:
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
                processed_results.append({
                    "success": False,
                    "url": urls[i],
                    "error": f"Processing exception: {str(result)}"
                })
                failed_count += 1
            else:
                processed_results.append(result)
                if result.get('success'):
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
        ).model_dump()

    except Exception as e:
        return YouTubeBatchResponse(
            success=False,
            total_urls=len(request.get('urls', [])),
            successful_extractions=0,
            failed_extractions=len(request.get('urls', [])),
            results=[],
            error=f"Batch processing error: {str(e)}"
        ).model_dump()


async def get_youtube_video_info(
    video_url: str,
    summarize_transcript: bool = False,
    max_tokens: int = 25000,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    summary_length: str = "medium",
    include_timestamps: bool = True
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

        # Enrich metadata via page crawl
        metadata = {}
        try:
            enrichment_result = await _crawl_youtube_page_fallback(
                url=video_url, video_id=video_id,
                wait_for_js=True, timeout=60
            )
            if enrichment_result.get('success'):
                metadata = enrichment_result.get('metadata', {})
        except Exception:
            pass

        title = metadata.get('title') or f"YouTube: {video_id}"

        # Try to get transcript information
        transcript_info = {}
        try:
            available_transcripts = youtube_processor.get_available_transcript_languages(video_id)
            transcript_info = {
                "has_transcript": len(available_transcripts) > 0,
                "available_languages": available_transcripts,
                "manually_created_count": len([t for t in available_transcripts if t.get('is_generated', True) is False]),
                "auto_generated_count": len([t for t in available_transcripts if t.get('is_generated', True) is True])
            }

            # If transcript is available and summarization is requested
            if transcript_info["has_transcript"] and summarize_transcript:
                transcript_result = await extract_youtube_transcript(
                    url=video_url,
                    languages=["en", "ja"],
                    include_timestamps=include_timestamps,
                    auto_summarize=True,
                    max_content_tokens=max_tokens,
                    summary_length=summary_length,
                    llm_provider=llm_provider,
                    llm_model=llm_model
                )

                if transcript_result.get('success'):
                    extracted = transcript_result.get('extracted_data', {})
                    meta = extracted.get('metadata', {})
                    transcript_info.update({
                        "transcript_summary": transcript_result.get('content', ''),
                        "original_length": meta.get('original_length', 0),
                        "summarization_applied": meta.get('summarization_applied', False),
                        "processing_method": extracted.get('processing_method')
                    })
                else:
                    transcript_info["transcript_error"] = transcript_result.get('error')

        except Exception as e:
            transcript_info = {
                "has_transcript": False,
                "transcript_check_error": str(e)
            }

        return {
            "success": True,
            "url": video_url,
            "video_id": video_id,
            "title": title,
            "video_info": video_info,
            "metadata": metadata,
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
    """Get setup information for youtube-transcript-api integration."""
    return {
        "success": True,
        "api_name": "youtube-transcript-api",
        "authentication_required": False,
        "api_key_required": False,
        "rate_limits": {"requests_per_minute": "Recommended: 60/min", "concurrent_requests": "3-5"},
        "supported_features": [
            "transcript_extraction", "multiple_languages", "auto_generated_transcripts",
            "manually_created_transcripts", "timestamps", "translation",
            "batch_processing", "video_metadata", "ai_summarization"
        ],
        "supported_youtube_formats": [
            "https://www.youtube.com/watch?v=VIDEO_ID", "https://youtu.be/VIDEO_ID",
            "https://youtube.com/watch?v=VIDEO_ID", "https://www.youtube.com/embed/VIDEO_ID"
        ],
        "usage_tips": [
            "No API key required", "Best with manually created captions",
            "Use batch processing for multiple videos",
            "Enable auto_summarize for long transcripts"
        ],
        "limitations": [
            "Only public videos with transcripts", "Private/unlisted not accessible",
            "Age-restricted may have limited access"
        ],
        "ai_summarization": {
            "trigger_threshold": "15,000 tokens",
            "summary_lengths": ["short", "medium", "long"],
            "preserves_timestamps": True
        }
    }
