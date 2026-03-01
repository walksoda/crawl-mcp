"""YouTube helper functions for page crawling fallback and metadata extraction."""

import asyncio
import re
import os
import contextlib
import sys
from typing import Any, Dict, List, Optional


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
            for delimiter in ['.', '\u2022', '|', 'Subscribe', 'subscrib', '\n', '\U0001f4da', '\U0001f3b5']:
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
