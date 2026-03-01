"""Search cache middleware for Crawl4AI MCP Server.

Provides LRU + TTL caching for search results to avoid
redundant API calls.
"""

import hashlib
import time as _time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from .pipeline import Middleware, PipelineContext

# Cache settings
_SEARCH_CACHE_MAX_SIZE = 5
_SEARCH_CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class _CacheEntry:
    """Cache entry with data and timestamp."""
    data: dict
    timestamp: float


# Search result cache (LRU + TTL)
_search_result_cache: OrderedDict[str, _CacheEntry] = OrderedDict()


class SearchCacheMiddleware(Middleware):
    """LRU + TTL cache for search results.

    On before(): checks cache and returns cached result if available.
    On after(): stores the result in cache.
    """

    async def before(self, ctx: PipelineContext) -> Optional[dict]:
        request = ctx.params.get('request', {})
        if not request:
            return None

        cache_key = _get_search_cache_key(request)
        ctx.metadata['cache_key'] = cache_key

        cached = _get_cached_search_result(cache_key)
        if cached is not None:
            # Add cache_hit indicator
            result = cached.copy()
            result['cache_hit'] = True
            return result

        return None

    async def after(self, ctx: PipelineContext) -> None:
        if not isinstance(ctx.result, dict):
            return
        if ctx.result.get('cache_hit'):
            return  # Already from cache

        cache_key = ctx.metadata.get('cache_key')
        if cache_key and ctx.result.get('success', True):
            _cache_search_result(cache_key, ctx.result)


def _get_search_cache_key(request: dict) -> str:
    """Generate cache key for search query including all relevant parameters.

    Normalizes defaults to match actual search behavior and avoid cache misses.
    """
    # Normalize num_results (clamp to valid range 1-100, default 10)
    num_results = request.get('num_results', 10)
    if num_results is None:
        num_results = 10
    num_results = max(1, min(100, int(num_results)))

    # Normalize language (default 'en')
    language = request.get('language') or 'en'

    # Normalize region (default 'us')
    region = request.get('region') or 'us'

    # Normalize recent_days (None/0/'' all mean no filter)
    recent_days = request.get('recent_days')
    recent_days_str = str(recent_days) if recent_days else ''

    # Normalize safe_search (default True)
    safe_search = request.get('safe_search', True)
    if safe_search is None:
        safe_search = True

    # Normalize search_genre (None/'' both mean no genre filter)
    search_genre = request.get('search_genre') or ''

    # Build cache key with all parameters that affect search results
    key_parts = [
        request.get('query', ''),
        str(num_results),
        search_genre,
        language,
        region,
        recent_days_str,
        str(safe_search),
    ]
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _cleanup_expired_cache() -> None:
    """Remove expired cache entries."""
    now = _time.time()
    expired_keys = [
        key for key, entry in _search_result_cache.items()
        if now - entry.timestamp > _SEARCH_CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        del _search_result_cache[key]


def _get_cached_search_result(cache_key: str) -> Optional[dict]:
    """Get search result from cache (LRU update, TTL check)."""
    _cleanup_expired_cache()  # Clean up expired entries first

    if cache_key in _search_result_cache:
        entry = _search_result_cache[cache_key]
        # TTL check
        if _time.time() - entry.timestamp <= _SEARCH_CACHE_TTL_SECONDS:
            _search_result_cache.move_to_end(cache_key)
            return entry.data
        else:
            # Expired, remove it
            del _search_result_cache[cache_key]
    return None


def _cache_search_result(cache_key: str, result: dict) -> None:
    """Store search result in cache."""
    _cleanup_expired_cache()  # Clean up before storing

    if cache_key in _search_result_cache:
        _search_result_cache.move_to_end(cache_key)
        _search_result_cache[cache_key] = _CacheEntry(data=result, timestamp=_time.time())
    else:
        if len(_search_result_cache) >= _SEARCH_CACHE_MAX_SIZE:
            _search_result_cache.popitem(last=False)  # Remove oldest entry
        _search_result_cache[cache_key] = _CacheEntry(data=result, timestamp=_time.time())


def clear_search_cache() -> None:
    """Clear the search result cache. Useful for testing."""
    _search_result_cache.clear()
