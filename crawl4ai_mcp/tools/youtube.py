"""
YouTube tools for Crawl4AI MCP Server.

Contains complete YouTube transcript extraction and video information tools.
"""

import asyncio
import re
import os
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


# =============================================================================
# Helper functions for YouTube page crawling fallback
# =============================================================================

def _extract_youtube_metadata_from_html(
    markdown_content: str,
    html_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse YouTube page content to extract metadata.

    Uses both markdown and raw HTML to extract video metadata including
    title, description, channel name, view count, upload date, etc.

    Note: crawl4ai's cleaned_html removes meta tags, so we extract from
    the title tag and markdown content primarily.

    Args:
        markdown_content: Markdown content from page crawl
        html_content: Optional raw HTML content for extraction

    Returns:
        Dict with extracted metadata fields
    """
    metadata = {
        'title': None,
        'description': None,
        'channel_name': None,
        'view_count': None,
        'upload_date': None,
        'duration': None,
        'like_count': None,
        'extraction_source': 'page_crawl'
    }

    # Extract from HTML title tag (this is preserved in crawl4ai cleaned_html)
    if html_content:
        # Title from <title> tag, strip " - YouTube" suffix
        title_tag_match = re.search(r'<title>([^<]+)</title>', html_content, re.IGNORECASE)
        if title_tag_match:
            title = title_tag_match.group(1).strip()
            # Remove " - YouTube" suffix if present
            if title.endswith(' - YouTube'):
                title = title[:-10].strip()
            metadata['title'] = title

    # Extract from markdown (fallback for title, primary for other data)
    lines = markdown_content.split('\n')

    # Title from first heading if not found in HTML
    if not metadata['title']:
        for line in lines[:20]:
            if line.startswith('# '):
                metadata['title'] = line[2:].strip()
                break

    # Try to find description - look for substantial text blocks
    # Skip navigation elements and look for video-related content
    content_lines = []
    for line in lines:
        line = line.strip()
        # Skip short lines, navigation elements, and UI text
        if len(line) > 50 and not any(skip in line.lower() for skip in [
            'subscribe', 'sign in', 'search', 'home', 'shorts', 'library',
            'history', 'trending', 'music', 'gaming', 'news', 'sports'
        ]):
            content_lines.append(line)

    if content_lines:
        # Use the first substantial content line as description
        metadata['description'] = content_lines[0][:1000]

    # Try to extract view count from markdown (pattern like "1,234,567 views")
    view_patterns = [
        r'([\d,]+)\s*(?:views?|回視聴)',
        r'視聴回数\s*([\d,]+)',
    ]
    for pattern in view_patterns:
        match = re.search(pattern, markdown_content, re.IGNORECASE)
        if match:
            view_str = match.group(1).replace(',', '')
            try:
                metadata['view_count'] = int(view_str)
            except ValueError:
                pass
            break

    # Try to extract channel name from markdown
    # Look for patterns commonly found in YouTube page content
    # The channel name is often followed by subscriber count or other info
    channel_patterns = [
        r'(?:by|from|チャンネル[：:])\s*([^\n]{3,50})',
    ]
    for pattern in channel_patterns:
        match = re.search(pattern, markdown_content, re.IGNORECASE)
        if match:
            channel = match.group(1).strip()
            # Clean up channel name - stop at common delimiters
            for delimiter in ['.', '•', '|', 'Subscribe', 'subscrib', '\n', '📚', '🎵']:
                if delimiter in channel:
                    channel = channel.split(delimiter)[0].strip()
            # Clean up channel name
            if channel and 3 < len(channel) < 50:
                metadata['channel_name'] = channel
            break

    # Try to extract date from markdown (various formats)
    date_patterns = [
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',  # 2024-01-15 or 2024/01/15
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4})',  # Jan 15, 2024
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # 15 Jan 2024
    ]
    for pattern in date_patterns:
        match = re.search(pattern, markdown_content, re.IGNORECASE)
        if match:
            metadata['upload_date'] = match.group(1)
            break

    return metadata


def _filter_relevant_content(markdown_content: str) -> str:
    """
    Filter markdown content to keep only relevant video information.

    Removes common YouTube navigation/UI elements and noise.

    Args:
        markdown_content: Raw markdown content from page crawl

    Returns:
        Filtered markdown content
    """
    # Remove common YouTube navigation/UI text
    noise_patterns = [
        r'Subscribe\s*\d*[KMB]?',
        r'Share\s*Save',
        r'Sign in',
        r'Search',
        r'Subscribed',
        r'\d+:\d+\s*/\s*\d+:\d+',  # Video timestamp UI
        r'Skip navigation',
        r'Home\s*Shorts',
        r'Trending',
        r'Library',
        r'History',
    ]

    content = markdown_content
    for pattern in noise_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # Remove excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def _build_fallback_transcript(
    title: str,
    description: str,
    markdown_content: str
) -> str:
    """
    Build a fallback transcript-like text from page content.

    When actual transcript is unavailable, constructs useful content
    from video title, description, and page content.

    Args:
        title: Video title
        description: Video description
        markdown_content: Page markdown content

    Returns:
        Formatted fallback text content
    """
    parts = []

    if title:
        parts.append(f"# {title}\n")

    parts.append("---")
    parts.append("Note: Transcript was unavailable via API. The following is extracted page content.\n")

    if description:
        parts.append("## Video Description")
        parts.append(description)
        parts.append("")

    # Include relevant portion of markdown content
    relevant_content = _filter_relevant_content(markdown_content)
    if relevant_content and len(relevant_content) > 100:
        parts.append("## Page Content")
        parts.append(relevant_content[:5000])  # Limit length

    return '\n'.join(parts)


async def _crawl_youtube_page_fallback(
    url: str,
    video_id: str,
    wait_for_js: bool = True,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Fallback method to extract YouTube content using web crawling.

    Uses the crawler directly (bypassing YouTube URL detection) to extract
    video metadata from the page HTML when the transcript API fails.

    Args:
        url: YouTube video URL
        video_id: Extracted video ID
        wait_for_js: Whether to wait for JavaScript rendering
        timeout: Request timeout in seconds

    Returns:
        Dict with 'success', 'transcript', 'metadata', 'error' keys
    """
    try:
        # Import crawler components
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

        # Configure crawler for YouTube page
        config = CrawlerRunConfig(
            wait_for="#content, ytd-watch-flexy, #primary",
            page_timeout=timeout * 1000,
            verbose=False,
            log_console=False,
            cache_mode=CacheMode.BYPASS  # Always fetch fresh content
        )

        browser_config = {
            "headless": True,
            "verbose": False,
            "browser_type": "chromium"  # Chromium works better with YouTube
        }

        # Suppress output to avoid JSON parsing errors
        import contextlib
        import sys
        from io import StringIO

        @contextlib.contextmanager
        def suppress_output():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        with suppress_output():
            async with AsyncWebCrawler(**browser_config) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=config),
                    timeout=timeout
                )

        if not result or not hasattr(result, 'success') or not result.success:
            error_msg = "Crawl failed"
            if hasattr(result, 'error_message'):
                error_msg = f"Crawl failed: {result.error_message}"
            return {
                'success': False,
                'error': error_msg
            }

        # Extract content
        markdown_content = result.markdown if hasattr(result, 'markdown') else ''
        html_content = result.cleaned_html if hasattr(result, 'cleaned_html') else ''

        if not markdown_content and not html_content:
            return {
                'success': False,
                'error': 'No content extracted from page'
            }

        # Extract metadata from crawled content
        metadata = _extract_youtube_metadata_from_html(
            markdown_content=markdown_content,
            html_content=html_content
        )

        # Add video_id to metadata
        metadata['video_id'] = video_id

        # Build transcript-like structure from available content
        fallback_text = _build_fallback_transcript(
            title=metadata.get('title', ''),
            description=metadata.get('description', ''),
            markdown_content=markdown_content
        )

        transcript_data = {
            'full_text': fallback_text,
            'clean_text': metadata.get('description', '') or fallback_text[:1000],
            'segments': [],  # No timestamps available from crawl
            'segment_count': 0,
            'word_count': len(fallback_text.split()),
            'source': 'page_crawl',
            'note': 'Transcript unavailable via API. Page content extracted as fallback.'
        }

        return {
            'success': True,
            'transcript': transcript_data,
            'metadata': metadata,
            'crawl_method': 'crawl_url_fallback',
            'js_rendered': wait_for_js
        }

    except asyncio.TimeoutError:
        return {
            'success': False,
            'error': f'Fallback crawl timeout after {timeout}s'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Fallback crawl exception: {str(e)}'
        }


# =============================================================================
# Main YouTube tools
# =============================================================================


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
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None,
    # Fallback parameters
    enable_crawl_fallback: Annotated[bool, Field(description="Enable crawl_url fallback when API fails (default: True)")] = True,
    fallback_timeout: Annotated[int, Field(description="Timeout for fallback crawl in seconds (default: 60)")] = 60,
    enrich_metadata: Annotated[bool, Field(description="Enrich metadata using crawl_url even on API success (default: True)")] = True
) -> YouTubeTranscriptResponse:
    """
    Extract YouTube video transcripts with timestamps and optional AI summarization.

    Works with public videos that have captions. No authentication required.
    Auto-detects available languages and falls back appropriately.

    Features:
    - Automatic fallback to page crawling when transcript API fails
    - Optional metadata enrichment (upload_date, view_count, etc.) via page crawling
    - AI summarization for long transcripts

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
            # Store API error for potential inclusion in fallback response
            import sys
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            api_error = result.get('error', 'Unknown error during transcript extraction')

            # Attempt fallback using crawl_url if enabled
            if enable_crawl_fallback:
                fallback_result = await _crawl_youtube_page_fallback(
                    url=url,
                    video_id=video_id,
                    wait_for_js=True,
                    timeout=fallback_timeout
                )

                if fallback_result['success']:
                    # Fallback succeeded - return response with crawled content
                    fallback_metadata = fallback_result.get('metadata', {})
                    fallback_metadata.update({
                        'fallback_used': True,
                        'original_api_error': api_error,
                        'extraction_source': 'page_crawl',
                        'processing_note': 'Transcript API failed, content extracted from page crawl'
                    })

                    response = YouTubeTranscriptResponse(
                        success=True,
                        url=url,
                        video_id=video_id,
                        transcript=fallback_result.get('transcript'),
                        metadata=fallback_metadata,
                        processing_method="crawl_url_fallback"
                    )
                    return response.model_dump()
                else:
                    # Both methods failed - return comprehensive error
                    fallback_error = fallback_result.get('error', 'Unknown fallback error')

                    response = YouTubeTranscriptResponse(
                        success=False,
                        url=url,
                        video_id=video_id,
                        error=f"Both extraction methods failed.\n\n"
                              f"API Error: {api_error}\n\n"
                              f"Fallback Error: {fallback_error}",
                        metadata={
                            'api_attempted': True,
                            'api_error': api_error,
                            'fallback_attempted': True,
                            'fallback_error': fallback_error,
                            'uvx_environment': is_uvx_env,
                            'recommendations': [
                                'Video may not have any available captions',
                                'Try a different video with known subtitles',
                                'Check if video is publicly accessible'
                            ]
                        }
                    )
                    return response.model_dump()

            # Fallback disabled - return original error with enhanced messaging
            if is_uvx_env:
                enhanced_error = f"{api_error}\n\nUVX Environment Detected:\n" \
                    f"- If this worked in STDIO local setup, the issue may be UVX environment isolation\n" \
                    f"- YouTube API may behave differently in UVX vs local environments\n" \
                    f"- Try running system diagnostics: get_system_diagnostics()\n" \
                    f"- Consider switching to STDIO local setup for YouTube functionality"
            else:
                enhanced_error = f"{api_error}\n\nTroubleshooting:\n" \
                    f"- Correct method name: 'extract_youtube_transcript' (not 'get_transcript')\n" \
                    f"- Alternative methods: get_youtube_video_info, batch_extract_youtube_transcripts\n" \
                    f"- Check if video has available captions\n" \
                    f"- Try enabling enable_crawl_fallback=True for page content extraction"

            response = YouTubeTranscriptResponse(
                success=False,
                url=url,
                video_id=video_id,
                error=enhanced_error,
                metadata={
                    'uvx_environment': is_uvx_env,
                    'correct_method_name': 'extract_youtube_transcript',
                    'alternative_methods': ['get_youtube_video_info', 'batch_extract_youtube_transcripts'],
                    'diagnostic_tool': 'get_system_diagnostics',
                    'fallback_disabled': True
                }
            )
            return response.model_dump()
        
        # Get transcript data
        transcript_data = result['transcript']
        language_info = result.get('language_info', {})
        metadata = result.get('metadata', {})

        # Enrich metadata using crawl_url if requested
        if enrich_metadata:
            try:
                enrichment_result = await _crawl_youtube_page_fallback(
                    url=url,
                    video_id=video_id,
                    wait_for_js=True,
                    timeout=fallback_timeout
                )

                if enrichment_result['success']:
                    enriched_metadata = enrichment_result.get('metadata', {})
                    # Add enriched fields that are not already present
                    enrichment_fields = ['upload_date', 'view_count', 'duration', 'like_count', 'channel_name']
                    for field in enrichment_fields:
                        if enriched_metadata.get(field) and not metadata.get(field):
                            metadata[field] = enriched_metadata[field]

                    metadata['metadata_enriched'] = True
                    metadata['enrichment_source'] = 'page_crawl'
                else:
                    # Enrichment failed but transcript succeeded - continue with original metadata
                    metadata['metadata_enrichment_attempted'] = True
                    metadata['metadata_enrichment_error'] = enrichment_result.get('error', 'Unknown error')
            except Exception as e:
                # Enrichment failed but transcript succeeded - continue with original metadata
                metadata['metadata_enrichment_attempted'] = True
                metadata['metadata_enrichment_error'] = f'Exception: {str(e)}'

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
        
        # Determine processing method based on enrichment
        processing_method = "youtube_transcript_api"
        if metadata.get('metadata_enriched'):
            processing_method = "youtube_transcript_api_enriched"

        response = YouTubeTranscriptResponse(
            success=True,
            url=url,
            video_id=video_id,
            transcript=transcript_data,
            language_info=language_info,
            metadata=metadata,
            processing_method=processing_method
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