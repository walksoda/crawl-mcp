"""Input validation utilities for Crawl4AI MCP Server.

This module provides validation functions for various input parameters
used in MCP tools.
"""

import re
from pathlib import Path
from typing import Any, Optional, Dict, List
from urllib.parse import urlparse, unquote


def is_file_uri(url: str) -> bool:
    return url.strip().lower().startswith('file://')


def is_local_path(url: str) -> bool:
    s = url.strip()
    if s.startswith('//') or s.startswith('\\\\'):
        return False
    if s.startswith('/'):
        return True
    if re.match(r'^[A-Za-z]:[/\\]', s):
        return True
    return False


def file_uri_to_local_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.netloc and parsed.netloc.lower() != 'localhost':
        raise ValueError(
            f"UNC file URIs are not supported: {uri}. "
            "Use file:///path/to/file for local files."
        )
    path = unquote(parsed.path)
    if not path or path == '/':
        raise ValueError(f"Empty file path in URI: {uri}")
    if path.startswith('//'):
        raise ValueError(
            f"UNC file URIs are not supported: {uri}. "
            "Use file:///path/to/file for local files."
        )
    if re.match(r'^/[A-Za-z]:/', path):
        path = path[1:]
    return path


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

    if is_file_uri(url_stripped):
        try:
            file_uri_to_local_path(url_stripped)
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "unsupported_file_uri"
            }
        return None

    if is_local_path(url_stripped):
        return None

    if not url_stripped.startswith(('http://', 'https://')):
        return {
            "success": False,
            "error": (
                "Invalid URL. Use http://, https://, file://, "
                f"or an absolute file path. Got: {url_stripped[:50]}"
            ),
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


def validate_output_path(
    path: Optional[str],
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate an ``output_path`` parameter for file persistence.

    Returns an error dict to short-circuit the tool, or ``None`` if the path
    is acceptable (or not supplied at all).

    Rules:
    - ``None``/``""`` → no persistence requested, pass through.
    - Must be a ``str`` containing no NUL bytes.
    - Must be an absolute path after ``~`` expansion.
    - If the path already exists as a regular file and ``overwrite=False``,
      reject immediately so we don't run an expensive fetch only to fail at
      the write step. Directory existence is allowed (batch tools write into
      directories) and handled lower in the stack.
    """
    if path is None or path == "":
        return None
    if not isinstance(path, str):
        return {
            "success": False,
            "error_code": "invalid_output_path_type",
            "error": f"output_path must be a string, got {type(path).__name__}",
        }
    if "\x00" in path:
        return {
            "success": False,
            "error_code": "invalid_output_path_chars",
            "error": "output_path contains NUL character",
        }
    try:
        p = Path(path).expanduser()
    except Exception as e:  # pragma: no cover - Path() almost never raises
        return {
            "success": False,
            "error_code": "invalid_output_path",
            "error": f"Invalid output_path: {e}",
        }
    if not p.is_absolute():
        return {
            "success": False,
            "error_code": "output_path_not_absolute",
            "error": f"output_path must be absolute. Got: {path}",
        }
    if p.exists() and p.is_file() and not overwrite:
        return {
            "success": False,
            "error_code": "output_path_exists",
            "error": f"Output file exists and overwrite=False: {path}",
        }
    return None
