"""
Shared helpers for the search engine clients.

Every client returns the same shape:
- success: {'success': True, 'results': [...], 'processing_method': ...}
- failure: {'success': False, 'error': ..., 'failure_kind': ...}

failure_kind is one of FAILURE_KINDS. The processor falls back to the next
engine in auto mode for every kind except 'no_results'.
"""

import html
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

FAILURE_KINDS = (
    'no_results',     # the engine answered with zero results
    'not_configured',  # API key missing
    'unsupported_genre',  # the engine cannot apply search_genre (auto mode skips it)
    'auth',           # key rejected
    'quota',          # credits or daily quota exhausted
    'rate_limited',   # HTTP 429
    'bad_request',    # the engine rejected the parameters
    'server',         # 5xx
    'network',        # connection or timeout
    'error',          # anything else
)

_TAG_RE = re.compile(r'<[^>]+>')


def strip_html(text: Optional[str]) -> str:
    """Drop tags (e.g. Brave's <strong>) and unescape entities."""
    if not text:
        return ''
    return ' '.join(html.unescape(_TAG_RE.sub('', text)).split())


def classify_url(url: str) -> str:
    """Classify URL by type based on domain and path."""
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc
        path = parsed.path

        if any(s in domain for s in ['youtube.com', 'youtu.be']):
            return 'video'
        elif any(s in domain for s in ['twitter.com', 'x.com', 'facebook.com', 'linkedin.com', 'instagram.com']):
            return 'social_media'
        elif any(s in domain for s in ['reddit.com', 'quora.com', 'stackoverflow.com']):
            return 'forum'
        elif any(s in domain for s in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com', 'wsj.com', 'guardian.com']):
            return 'news'
        elif any(s in domain for s in ['.edu', '.ac.', 'scholar.google', 'arxiv.org', 'researchgate']):
            return 'academic'
        elif any(s in domain for s in ['.gov', '.mil', '.org']):
            return 'official'
        elif any(s in domain for s in ['amazon.com', 'ebay.com', 'etsy.com', 'shopify']):
            return 'ecommerce'
        elif any(s in domain for s in ['github.com', 'docs.', 'documentation', 'wiki']):
            return 'documentation'
        elif any(s in path for s in ['.pdf', '.doc', '.ppt', '.xls']):
            return 'document'
        elif '.com' in domain:
            return 'commercial'
        return 'general'
    except Exception:
        return 'unknown'


def make_result(rank: int, url: str, title: Optional[str], snippet: Optional[str], **extra: Any) -> Dict[str, Any]:
    """Build one result entry in the common shape. None-valued extras are dropped."""
    result = {
        'rank': rank,
        'url': url,
        'domain': urlparse(url).netloc,
        'title': title or 'No title',
        'snippet': snippet or 'No description available',
        'type': classify_url(url),
    }
    result.update({k: v for k, v in extra.items() if v is not None})
    return result


def failure(kind: str, error: str, **extra: Any) -> Dict[str, Any]:
    """Build a failure dict with a failure_kind the processor can act on."""
    out = {'success': False, 'error': error, 'failure_kind': kind}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def failure_kind_for_status(status: int) -> str:
    """Map an HTTP status code to a failure_kind."""
    if status == 429:
        return 'rate_limited'
    if status in (401, 403):
        return 'auth'
    if status == 402:
        return 'quota'
    if status >= 500:
        return 'server'
    if status in (400, 404, 422):
        return 'bad_request'
    return 'error'


def normalize_region(region: Optional[str]) -> Optional[str]:
    """Two-letter upper-case country code; 'uk' becomes 'GB'."""
    if not region:
        return None
    code = str(region).strip().upper()
    return 'GB' if code == 'UK' else code
