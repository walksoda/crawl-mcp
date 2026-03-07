"""Content cache freshness policy for Crawl4AI MCP Server.

Tracks when URLs were last fetched via network (BYPASS) and automatically
switches between ENABLED/BYPASS cache modes based on TTL expiration.

Storage: ~/.crawl4ai_content_freshness.json
Format: { "<url>": <fetched_at_unix_timestamp>, ... }
"""

import json
import os
import threading
import time
from typing import Optional

from ..constants import DEFAULT_CRAWL_CACHE_TTL_SECONDS


class ContentCachePolicy:
    """Manages content freshness tracking and cache mode resolution.

    Thread-safe within a single process. Not process-safe.
    Uses atomic file writes. Expired entries are garbage-collected
    on each save operation.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ):
        if storage_path is None:
            storage_path = os.path.expanduser("~/.crawl4ai_content_freshness.json")
        self.storage_path = storage_path

        if ttl_seconds is None:
            env_val = os.environ.get("CRAWL4AI_CACHE_TTL_SECONDS")
            if env_val is not None:
                try:
                    ttl_seconds = int(env_val)
                except (ValueError, TypeError):
                    ttl_seconds = DEFAULT_CRAWL_CACHE_TTL_SECONDS
            else:
                ttl_seconds = DEFAULT_CRAWL_CACHE_TTL_SECONDS
        self.ttl_seconds = max(0, ttl_seconds)

        self._lock = threading.Lock()
        self._entries: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        """Load freshness data from disk. Silently reset on corruption."""
        if not os.path.exists(self.storage_path):
            self._entries = {}
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self._entries = {}
                return
            # Keep only valid entries
            self._entries = {
                k: v for k, v in data.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
        except (json.JSONDecodeError, IOError, OSError):
            self._entries = {}

    def _save(self) -> None:
        """Save freshness data with atomic write and GC of expired entries."""
        now = time.time()
        # GC: remove expired entries if TTL is active
        if self.ttl_seconds > 0:
            self._entries = {
                url: ts for url, ts in self._entries.items()
                if now < ts + self.ttl_seconds
            }

        temp_path = self.storage_path + ".tmp"
        try:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

            fd = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._entries, f, ensure_ascii=False)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            os.replace(temp_path, self.storage_path)
        except (IOError, OSError, TypeError, ValueError):
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def resolve_cache_mode(
        self,
        url: str,
        content_offset: int,
        requested_mode: str,
        enable_caching: bool,
    ) -> str:
        """Determine the effective cache mode for a crawl request.

        Priority:
        1. Explicit disabled/bypass from user -> pass through
        2. content_offset > 0 -> enabled (pagination protection)
        3. TTL disabled (0) -> enabled (legacy behavior)
        4. Fresh entry exists -> enabled
        5. Expired or missing -> bypass (trigger re-fetch)
        """
        if not enable_caching or requested_mode == "disabled":
            return "disabled"
        if requested_mode == "bypass":
            return "bypass"
        # content_offset > 0 means "read the next slice of the same page".
        # Force cache-hit so the second slice comes from the same fetch as
        # the first; bypassing here would risk returning inconsistent content.
        if content_offset > 0:
            return "enabled"
        if self.ttl_seconds <= 0:
            return "enabled"

        with self._lock:
            fetched_at = self._entries.get(url)
        if fetched_at is not None and time.time() < fetched_at + self.ttl_seconds:
            return "enabled"
        return "bypass"

    def record_fresh_fetch(self, url: str) -> None:
        """Record that a URL was freshly fetched from the network.

        Call this only after a successful BYPASS crawl.
        """
        with self._lock:
            self._entries[url] = time.time()
            self._save()


_content_cache_policy: Optional[ContentCachePolicy] = None
_init_lock = threading.Lock()


def get_content_cache_policy() -> ContentCachePolicy:
    """Get the global ContentCachePolicy singleton."""
    global _content_cache_policy
    if _content_cache_policy is None:
        with _init_lock:
            if _content_cache_policy is None:
                _content_cache_policy = ContentCachePolicy()
    return _content_cache_policy
