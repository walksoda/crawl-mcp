"""Content processing utilities for Crawl4AI MCP Server.

This module provides utilities for validating, converting, and processing
crawl content.
"""

import html
import re
from typing import Any, Dict, List, Tuple, Optional

from ..constants import BLOCK_INDICATORS, BLOCK_PAGE_MAX_TEXT_LENGTH


def convert_media_to_list(media: Any) -> List[Dict[str, Any]]:
    """
    Convert crawl4ai's media dict format to flat list format.

    crawl4ai returns media as: {'images': [...], 'videos': [...], 'audios': [...]}
    CrawlResponse expects: List[Dict[str, Any]]

    Args:
        media: Media data from crawl4ai (dict or list)

    Returns:
        Flattened list of media items with media_type field
    """
    if not media:
        return []

    # If already a list, return as-is
    if isinstance(media, list):
        return media

    # If it's a dict with media type keys, flatten it
    if isinstance(media, dict):
        result = []
        for media_type in ['images', 'videos', 'audios', 'tables']:
            items = media.get(media_type, [])
            if items:
                for item in items:
                    if isinstance(item, dict):
                        item_copy = dict(item)
                        item_copy['media_type'] = media_type.rstrip('s')  # 'images' -> 'image'
                        result.append(item_copy)
        return result

    return []


def has_meaningful_content(result, min_length: int = 100) -> Tuple[bool, str]:
    """
    Check if crawl result has meaningful content.

    Checks markdown, content, and raw_content fields.
    Returns (True, content_source) if any field contains content exceeding min_length.
    Returns (False, "") if no meaningful content found.

    Args:
        result: CrawlResponse object or dict with crawl result
        min_length: Minimum content length to consider meaningful

    Returns:
        Tuple of (has_content: bool, content_source: str)
        content_source indicates which field had content: "markdown", "content", or "raw_content"
    """
    # Check markdown first (most common and useful for MCP clients)
    markdown = getattr(result, 'markdown', None) if hasattr(result, 'markdown') else (result.get('markdown') if isinstance(result, dict) else None)
    if markdown and len(str(markdown).strip()) > min_length:
        return True, "markdown"

    # Check content (cleaned HTML)
    content = getattr(result, 'content', None) if hasattr(result, 'content') else (result.get('content') if isinstance(result, dict) else None)
    if content and len(str(content).strip()) > min_length:
        return True, "content"

    # Check raw_content (original HTML)
    raw_content = getattr(result, 'raw_content', None) if hasattr(result, 'raw_content') else (result.get('raw_content') if isinstance(result, dict) else None)
    if raw_content and len(str(raw_content).strip()) > min_length:
        return True, "raw_content"

    return False, ""


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _visible_text(content: str) -> str:
    """Strip scripts, styles, comments and tags so only human-visible text remains."""
    text = _SCRIPT_STYLE_RE.sub(" ", content)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", html.unescape(text)).strip()


def is_block_page(content: str) -> bool:
    """
    Check if content appears to be a block/challenge page.

    Only the visible text is inspected (markup such as ``<meta name="robots">``
    or CDN script URLs is ignored), and long pages are never treated as block
    pages because real challenge pages carry very little text.

    Args:
        content: HTML, markdown or plain text

    Returns:
        True if the page looks like a block page, False otherwise
    """
    if not content:
        return False
    text = _visible_text(content)
    if not text or len(text) > BLOCK_PAGE_MAX_TEXT_LENGTH:
        return False
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in BLOCK_INDICATORS)


def is_block_result(result) -> bool:
    """
    Check if a crawl result is a block page.

    Uses a single representative field (markdown, then cleaned HTML, then raw
    HTML) so the length check is not inflated by concatenating duplicates.
    """
    for field in ("markdown", "content", "raw_content"):
        value = result.get(field) if isinstance(result, dict) else getattr(result, field, None)
        if value and str(value).strip():
            return is_block_page(str(value))
    return False


def http_error_status(result) -> Optional[int]:
    """
    Return the HTTP status of a crawl result if it is an error (>= 400).

    crawl4ai reports success for pages that loaded with a 4xx/5xx status,
    so the status has to be checked separately. Returns None when the status
    is unknown or below 400.
    """
    status = result.get("status_code") if isinstance(result, dict) else getattr(result, "status_code", None)
    if isinstance(status, int) and status >= 400:
        return status
    return None


# Titles and opening lines of typical HTTP error pages (web servers, CDNs,
# CMS "page not found" templates). Only short pages are matched.
_ERROR_PAGE_RE = re.compile(
    r"\b(?:40[0134]|410|50[0234])\b[^\n]{0,40}\b(?:not found|forbidden|gone|error|unavailable|bad request|unauthorized)\b"
    r"|\b(?:page|file|resource) not found\b"
    r"|\berror response\b"
    r"|\berror code:? *[45]\d\d\b"
    r"|\bnot found\b[^\n]{0,40}\b(?:40[34]|410)\b",
    re.IGNORECASE,
)
ERROR_PAGE_MAX_TEXT_LENGTH = 1000


def looks_like_error_page(result) -> bool:
    """
    Check whether a crawl result looks like an HTTP error page.

    Used where the status code may be missing (cached results) and the URL is
    a guess, such as the AMP fallback. Only short pages are considered, so
    articles that mention "404" are not rejected.
    """
    def _field(name):
        return result.get(name) if isinstance(result, dict) else getattr(result, name, None)

    text = ""
    for name in ("markdown", "content", "raw_content"):
        value = _field(name)
        if value and str(value).strip():
            text = _visible_text(str(value))
            break
    if len(text) > ERROR_PAGE_MAX_TEXT_LENGTH:
        return False
    title = _field("title") or ""
    return bool(_ERROR_PAGE_RE.search(title) or _ERROR_PAGE_RE.search(text))


def get_content_length(result) -> int:
    """
    Get the total content length from a crawl result.

    Args:
        result: CrawlResponse object or dict

    Returns:
        Total character count of content
    """
    total = 0

    markdown = getattr(result, 'markdown', None) if hasattr(result, 'markdown') else (result.get('markdown') if isinstance(result, dict) else None)
    if markdown:
        total += len(str(markdown))

    content = getattr(result, 'content', None) if hasattr(result, 'content') else (result.get('content') if isinstance(result, dict) else None)
    if content:
        total += len(str(content))

    return total


def extract_text_content(result) -> str:
    """
    Extract text content from a crawl result.

    Prioritizes markdown over raw content.

    Args:
        result: CrawlResponse object or dict

    Returns:
        Extracted text content
    """
    # Try markdown first
    markdown = getattr(result, 'markdown', None) if hasattr(result, 'markdown') else (result.get('markdown') if isinstance(result, dict) else None)
    if markdown and str(markdown).strip():
        return str(markdown).strip()

    # Fall back to content
    content = getattr(result, 'content', None) if hasattr(result, 'content') else (result.get('content') if isinstance(result, dict) else None)
    if content and str(content).strip():
        return str(content).strip()

    # Last resort: raw_content
    raw_content = getattr(result, 'raw_content', None) if hasattr(result, 'raw_content') else (result.get('raw_content') if isinstance(result, dict) else None)
    if raw_content and str(raw_content).strip():
        return str(raw_content).strip()

    return ""


def truncate_content(content: str, max_chars: int, add_ellipsis: bool = True) -> str:
    """
    Truncate content to a maximum length.

    Args:
        content: Content to truncate
        max_chars: Maximum characters
        add_ellipsis: Whether to add "..." at the end

    Returns:
        Truncated content
    """
    if not content or len(content) <= max_chars:
        return content

    truncated = content[:max_chars]
    if add_ellipsis:
        truncated = truncated.rstrip() + "..."

    return truncated


def normalize_url(url: str) -> str:
    """
    Normalize a URL by adding scheme if missing.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL with scheme
    """
    if not url:
        return url

    url = url.strip()
    if not url.startswith(('http://', 'https://', '//')):
        url = 'https://' + url

    return url


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain name
    """
    from urllib.parse import urlparse

    try:
        if not url.startswith(('http://', 'https://', '//')):
            url = 'https://' + url
        parsed = urlparse(url)
        host = parsed.hostname
        if host:
            return host.lower()
    except Exception:
        pass

    return "unknown"


def clean_html_content(html: str) -> str:
    """
    Clean HTML content by removing scripts, styles, and normalizing whitespace.

    Args:
        html: HTML content to clean

    Returns:
        Cleaned HTML
    """
    if not html:
        return html

    import re

    # Remove script tags and content
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove style tags and content
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Normalize whitespace
    html = re.sub(r'\s+', ' ', html)

    return html.strip()


def extract_links(result) -> List[Dict[str, str]]:
    """
    Extract links from a crawl result.

    Args:
        result: CrawlResponse object or dict

    Returns:
        List of link dicts with 'url' and 'text' keys
    """
    links = getattr(result, 'links', None) if hasattr(result, 'links') else (result.get('links') if isinstance(result, dict) else None)

    if not links:
        return []

    # Handle different formats
    if isinstance(links, list):
        normalized = []
        for link in links:
            if isinstance(link, dict):
                normalized.append({
                    'url': link.get('href', link.get('url', '')),
                    'text': link.get('text', link.get('title', ''))
                })
            elif isinstance(link, str):
                normalized.append({'url': link, 'text': ''})
        return normalized

    return []


def estimate_reading_time(content: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for content in minutes.

    Args:
        content: Text content
        words_per_minute: Reading speed (default 200 wpm)

    Returns:
        Estimated reading time in minutes (minimum 1)
    """
    if not content:
        return 0

    # Count words (rough estimate)
    word_count = len(content.split())
    minutes = max(1, word_count // words_per_minute)

    return minutes


# Backward compatibility aliases
_convert_media_to_list = convert_media_to_list
_has_meaningful_content = has_meaningful_content
_is_block_page = is_block_page
