"""Input validation utilities for Crawl4AI MCP Server.

This module provides validation functions for various input parameters
used in MCP tools.
"""

from typing import Any, Optional, Dict, List
from urllib.parse import urlparse


def validate_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Validate a URL and return error dict if invalid, None if valid.

    Args:
        url: The URL to validate

    Returns:
        Error dictionary if invalid, None if valid
    """
    if not url or not url.strip():
        return {
            "success": False,
            "error": "URL is required and cannot be empty",
            "error_code": "invalid_url"
        }

    url_stripped = url.strip()
    if not url_stripped.startswith(('http://', 'https://')):
        return {
            "success": False,
            "error": f"Invalid URL scheme. URL must start with http:// or https://. Got: {url_stripped[:50]}",
            "error_code": "invalid_url_scheme"
        }

    return None


def validate_timeout(timeout: Any, max_timeout: int = 300) -> Optional[Dict[str, Any]]:
    """
    Validate a timeout value and return error dict if invalid, None if valid.

    Args:
        timeout: The timeout value in seconds (can be int, float, str, or None)
        max_timeout: Maximum allowed timeout (default 300 seconds)

    Returns:
        Error dictionary if invalid, None if valid
    """
    # Handle None case
    if timeout is None:
        return {
            "success": False,
            "error": "Timeout is required and cannot be None",
            "error_code": "invalid_timeout"
        }

    # Try to convert to numeric type
    try:
        # Handle string input
        if isinstance(timeout, str):
            timeout = float(timeout)
        # Convert to int for comparison (truncate float)
        timeout_value = int(timeout)
    except (ValueError, TypeError):
        return {
            "success": False,
            "error": f"Timeout must be a valid number. Got: {timeout} (type: {type(timeout).__name__})",
            "error_code": "invalid_timeout_type"
        }

    if timeout_value <= 0:
        return {
            "success": False,
            "error": f"Timeout must be a positive integer. Got: {timeout_value}",
            "error_code": "invalid_timeout"
        }

    if timeout_value > max_timeout:
        return {
            "success": False,
            "error": f"Timeout exceeds maximum allowed value of {max_timeout} seconds. Got: {timeout_value}",
            "error_code": "timeout_too_large"
        }

    return None


def validate_crawl_url_params(url: str, timeout: int) -> Optional[Dict[str, Any]]:
    """
    Validate crawl_url parameters and return error dict if invalid, None if valid.

    Args:
        url: The URL to crawl
        timeout: The timeout value in seconds

    Returns:
        Error dictionary if invalid, None if valid
    """
    # Validate URL
    url_error = validate_url(url)
    if url_error:
        return url_error

    # Validate timeout
    timeout_error = validate_timeout(timeout)
    if timeout_error:
        return timeout_error

    return None


def validate_batch_urls(
    urls: List[str],
    max_urls: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Validate a list of URLs for batch operations.

    Args:
        urls: List of URLs to validate
        max_urls: Maximum number of URLs allowed

    Returns:
        Error dictionary if invalid, None if valid
    """
    if not urls:
        return {
            "success": False,
            "error": "At least one URL is required",
            "error_code": "no_urls"
        }

    if len(urls) > max_urls:
        return {
            "success": False,
            "error": f"Maximum {max_urls} URLs allowed. Got: {len(urls)}",
            "error_code": "too_many_urls"
        }

    # Validate each URL
    for i, url in enumerate(urls):
        url_error = validate_url(url)
        if url_error:
            url_error["error"] = f"Invalid URL at index {i}: {url_error['error']}"
            return url_error

    return None


def validate_summary_length(summary_length: str) -> Optional[Dict[str, Any]]:
    """
    Validate summary length parameter.

    Args:
        summary_length: The summary length value ("short", "medium", "long")

    Returns:
        Error dictionary if invalid, None if valid
    """
    valid_lengths = ["short", "medium", "long"]
    if summary_length not in valid_lengths:
        return {
            "success": False,
            "error": f"Invalid summary_length. Must be one of: {', '.join(valid_lengths)}. Got: {summary_length}",
            "error_code": "invalid_summary_length"
        }
    return None


def validate_crawl_depth(
    max_depth: int,
    max_allowed: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Validate crawl depth parameter.

    Args:
        max_depth: The maximum depth value
        max_allowed: Maximum allowed depth (default 5)

    Returns:
        Error dictionary if invalid, None if valid
    """
    if max_depth < 1:
        return {
            "success": False,
            "error": f"max_depth must be at least 1. Got: {max_depth}",
            "error_code": "invalid_depth"
        }

    if max_depth > max_allowed:
        return {
            "success": False,
            "error": f"max_depth exceeds maximum allowed value of {max_allowed}. Got: {max_depth}",
            "error_code": "depth_too_large"
        }

    return None


def validate_max_pages(
    max_pages: int,
    max_allowed: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Validate max pages parameter.

    Args:
        max_pages: The maximum pages value
        max_allowed: Maximum allowed pages (default 100)

    Returns:
        Error dictionary if invalid, None if valid
    """
    if max_pages < 1:
        return {
            "success": False,
            "error": f"max_pages must be at least 1. Got: {max_pages}",
            "error_code": "invalid_max_pages"
        }

    if max_pages > max_allowed:
        return {
            "success": False,
            "error": f"max_pages exceeds maximum allowed value of {max_allowed}. Got: {max_pages}",
            "error_code": "max_pages_too_large"
        }

    return None


def validate_youtube_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Validate a YouTube URL.

    Args:
        url: The URL to validate

    Returns:
        Error dictionary if invalid, None if valid
    """
    # Basic URL validation first
    url_error = validate_url(url)
    if url_error:
        return url_error

    # Check if it's a YouTube URL
    parsed = urlparse(url)
    youtube_domains = [
        'youtube.com', 'www.youtube.com', 'm.youtube.com',
        'youtu.be', 'www.youtu.be'
    ]

    if parsed.netloc not in youtube_domains:
        return {
            "success": False,
            "error": f"URL must be a YouTube URL. Got: {parsed.netloc}",
            "error_code": "not_youtube_url"
        }

    return None


def validate_file_size(
    size_bytes: int,
    max_size_mb: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Validate file size.

    Args:
        size_bytes: File size in bytes
        max_size_mb: Maximum allowed size in MB (default 100)

    Returns:
        Error dictionary if invalid, None if valid
    """
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return {
            "success": False,
            "error": f"File size ({size_bytes / 1024 / 1024:.2f}MB) exceeds maximum allowed size of {max_size_mb}MB",
            "error_code": "file_too_large"
        }
    return None


def validate_search_params(
    query: str,
    num_results: int = 10,
    max_results: int = 50
) -> Optional[Dict[str, Any]]:
    """
    Validate search parameters.

    Args:
        query: The search query
        num_results: Number of results to return
        max_results: Maximum allowed results

    Returns:
        Error dictionary if invalid, None if valid
    """
    if not query or not query.strip():
        return {
            "success": False,
            "error": "Search query is required and cannot be empty",
            "error_code": "empty_query"
        }

    if num_results < 1:
        return {
            "success": False,
            "error": f"num_results must be at least 1. Got: {num_results}",
            "error_code": "invalid_num_results"
        }

    if num_results > max_results:
        return {
            "success": False,
            "error": f"num_results exceeds maximum allowed value of {max_results}. Got: {num_results}",
            "error_code": "too_many_results"
        }

    return None


def validate_content_slicing_params(
    content_limit: int,
    content_offset: int
) -> Optional[Dict[str, Any]]:
    """
    Validate content slicing parameters.

    Args:
        content_limit: Maximum characters to return (0=unlimited)
        content_offset: Start position for content (0-indexed)

    Returns:
        Error dictionary if invalid, None if valid
    """
    if content_limit < 0:
        return {
            "success": False,
            "error": f"content_limit must be non-negative. Got: {content_limit}",
            "error_code": "invalid_content_limit"
        }
    if content_offset < 0:
        return {
            "success": False,
            "error": f"content_offset must be non-negative. Got: {content_offset}",
            "error_code": "invalid_content_offset"
        }
    return None
