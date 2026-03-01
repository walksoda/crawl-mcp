"""
Crawler I/O helpers for URL type detection and response building.

Contains _process_response_content, _handle_youtube_url, _handle_file_url,
and _build_json_extraction_response extracted from tools/web_crawling.py.
"""

import json
from typing import Any, Dict, List, Optional

from ..models import CrawlRequest, CrawlResponse
from ..processors.file_processor import FileProcessor
from ..processors.youtube_processor import YouTubeProcessor

# Initialize processors
file_processor = FileProcessor()
youtube_processor = YouTubeProcessor()


def _process_response_content(response: CrawlResponse, include_cleaned_html: bool) -> CrawlResponse:
    """
    Process CrawlResponse to remove content field if include_cleaned_html is False.
    By default, only markdown is returned to reduce token usage and improve readability.
    """
    if not include_cleaned_html and hasattr(response, 'content'):
        response.content = None
    return response


async def _handle_youtube_url(request: CrawlRequest) -> Optional[CrawlResponse]:
    """
    Handle YouTube URL detection and transcript extraction.

    Uses core/youtube.py extract_youtube_transcript() for unified processing.
    Returns CrawlResponse if URL is YouTube, None otherwise.
    """
    if not youtube_processor.is_youtube_url(request.url):
        return None

    from .youtube import extract_youtube_transcript
    from .crawler_summarizer import _check_and_summarize_if_needed

    try:
        result = await extract_youtube_transcript(
            url=request.url,
            languages=["ja", "en"],
            include_timestamps=False,
            preserve_formatting=True,
            include_metadata=True,
            enrich_metadata=True
        )

        if result.get('success'):
            response = CrawlResponse(
                success=True,
                url=request.url,
                title=result.get('title', ''),
                content=result.get('content'),
                markdown=result.get('markdown'),
                extracted_data=result.get('extracted_data')
            )
            response = await _check_and_summarize_if_needed(response, request)
            return _process_response_content(response, request.include_cleaned_html)
        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=result.get('error', 'Unknown error')
            )
    except Exception as e:
        return CrawlResponse(
            success=False,
            url=request.url,
            error=f"YouTube processing error: {str(e)}"
        )


async def _handle_file_url(request: CrawlRequest) -> Optional[CrawlResponse]:
    """
    Handle file URL detection and processing via MarkItDown.

    Returns CrawlResponse if URL is a supported file, None otherwise.
    """
    if not file_processor.is_supported_file(request.url):
        return None

    from .crawler_summarizer import _check_and_summarize_if_needed

    try:
        file_result = await file_processor.process_file_from_url(
            request.url,
            max_size_mb=100
        )

        if file_result['success']:
            response = CrawlResponse(
                success=True,
                url=request.url,
                title=file_result.get('title'),
                content=file_result.get('content'),
                markdown=file_result.get('content'),
                extracted_data={
                    "file_type": file_result.get('file_type'),
                    "size_bytes": file_result.get('size_bytes'),
                    "is_archive": file_result.get('is_archive', False),
                    "metadata": file_result.get('metadata'),
                    "archive_contents": file_result.get('archive_contents'),
                    "processing_method": "markitdown"
                }
            )
            response = await _check_and_summarize_if_needed(response, request)
            return _process_response_content(response, request.include_cleaned_html)
        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"File processing failed: {file_result.get('error')}"
            )
    except Exception as e:
        return CrawlResponse(
            success=False,
            url=request.url,
            error=f"File processing error: {str(e)}"
        )


async def _build_json_extraction_response(
    json_data: dict,
    json_source: str,
    url: str,
    strategy_name: str,
    stage: int,
    auto_summarize: bool,
    max_content_tokens: int,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    extract_content_keys: Optional[List[str]] = None
) -> CrawlResponse:
    """
    Build CrawlResponse from extracted JSON data.

    Common helper for Stage 1 (static JSON extraction) and Stage 7 (JSON extraction fallback).
    """
    from .crawler_summarizer import _finalize_fallback_response

    content_text = ""

    # Try to extract content from priority keys if specified
    if isinstance(json_data, dict) and extract_content_keys:
        for key in extract_content_keys:
            if key in json_data:
                content_text = json.dumps(json_data[key], indent=2, ensure_ascii=False)
                break

    # Fallback to full JSON if no priority key found or not specified
    if not content_text:
        content_text = json.dumps(json_data, indent=2, ensure_ascii=False)

    response = CrawlResponse(
        success=True,
        url=url,
        markdown=f"# Extracted JSON Data ({json_source})\n\n```json\n{content_text[:50000]}\n```",
        content=content_text,
        extracted_data={
            "fallback_strategy_used": strategy_name,
            "fallback_stage": stage,
            "json_source": json_source,
            "site_type_detected": "spa_with_json"
        }
    )

    response = await _finalize_fallback_response(
        response, url, auto_summarize, max_content_tokens, llm_provider, llm_model
    )
    return response
