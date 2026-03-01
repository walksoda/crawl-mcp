"""Strategy cache for Crawl4AI MCP Server.

Caches successful crawling strategies per domain to optimize
future crawl attempts by starting from known-working strategies.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse


class StrategyCache:
    """
    Caches successful crawling strategies per domain.

    Tracks which fallback stages work best for each domain,
    allowing future crawls to start from the optimal strategy
    instead of always starting from Stage 1.

    Features:
    - Records successful strategies with response times
    - Tracks failure patterns to avoid repeated failures
    - TTL-based expiration (default 7 days)
    - Adaptive strategy selection based on failure history

    Storage:
    - JSON file at ~/.crawl4ai_strategy_cache.json
    - Secure permissions (0600)
    """

    # Strategy names mapping for stage numbers
    STAGE_NAMES = {
        1: "static_fast_path",
        2: "normal_headless",
        3: "chromium_stealth",
        4: "user_behavior",
        5: "mobile_agent",
        6: "amp_rss",
        7: "json_extraction"
    }

    def __init__(self, storage_path: str = None, default_ttl_days: int = 7):
        """
        Initialize the strategy cache.

        Args:
            storage_path: Path to store cache data. Defaults to ~/.crawl4ai_strategy_cache.json
            default_ttl_days: Default TTL in days for cache entries (min 1).
        """
        self.storage_path = storage_path or os.path.expanduser("~/.crawl4ai_strategy_cache.json")
        self.default_ttl_days = max(1, default_ttl_days) if isinstance(default_ttl_days, (int, float)) else 7
        self._cache: Dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from storage file with validation."""
        if not os.path.exists(self.storage_path):
            self._cache = {}
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self._cache = {}
                return

            # Validate each entry
            validated = {}
            for domain, entry in data.items():
                if not isinstance(domain, str) or not isinstance(entry, dict):
                    continue
                # Required fields
                if "best_stage" not in entry or "best_strategy" not in entry:
                    continue
                # Type validation
                if not isinstance(entry.get("best_stage"), int):
                    continue
                if not isinstance(entry.get("best_strategy"), str):
                    continue
                # Validate expires_at
                expires_at = entry.get("expires_at")
                if expires_at is not None and not isinstance(expires_at, (int, float)):
                    entry["expires_at"] = None
                # Validate and normalize failed_stages to List[int]
                failed_stages = entry.get("failed_stages", [])
                if not isinstance(failed_stages, list):
                    entry["failed_stages"] = []
                else:
                    entry["failed_stages"] = [s for s in failed_stages if isinstance(s, int) and 1 <= s <= 7]
                # Validate success_count
                if not isinstance(entry.get("success_count"), int):
                    entry["success_count"] = 0
                # Validate avg_response_time
                if not isinstance(entry.get("avg_response_time"), (int, float)):
                    entry["avg_response_time"] = 0.0
                validated[domain] = entry

            self._cache = validated

        except (json.JSONDecodeError, IOError, OSError):
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to storage file with secure permissions."""
        temp_path = self.storage_path + ".tmp"
        try:
            # Remove temp file if exists
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

            # Create with O_EXCL for security
            fd = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self._cache, f, indent=2, ensure_ascii=False)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            os.replace(temp_path, self.storage_path)

        except (IOError, TypeError, ValueError, OSError) as e:
            print(f"Warning: Could not save strategy cache to {self.storage_path}: {e}")
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _get_domain_key(self, url: str) -> str:
        """Extract domain key from URL, stripping credentials and path/query.

        Returns a unique key for each URL to avoid cache pollution.
        If domain extraction fails, returns a hash-based key to ensure uniqueness.
        """
        import hashlib

        if not url:
            return f"empty_url_{hashlib.md5(b'').hexdigest()[:8]}"

        try:
            # Add scheme if missing to ensure proper parsing
            normalized_url = url
            if not url.startswith(('http://', 'https://', '//')):
                normalized_url = 'https://' + url
            parsed = urlparse(normalized_url)
            # Strip user:pass from netloc and extract hostname only
            host = parsed.hostname
            if not host:
                # Fallback: split netloc manually
                netloc = parsed.netloc or url.split('/')[0]
                host = netloc.split('@')[-1].split(':')[0]
            # Never return path, query, or fragment - only domain
            if host:
                return host.lower()
        except Exception:
            pass

        # Last resort: try to extract domain-like pattern
        match = re.search(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+)', url)
        if match:
            return match.group(1).lower()

        # If all else fails, use hash of original URL to avoid cache pollution
        # Different invalid URLs will get different keys
        url_hash = hashlib.md5(url.encode('utf-8', errors='replace')).hexdigest()[:12]
        return f"unparseable_{url_hash}"

    def get_best_strategy(self, url: str) -> Optional[dict]:
        """
        Get the best known strategy for a domain.

        Args:
            url: The URL to get strategy for.

        Returns:
            Dict with strategy info or None if no cached strategy.
        """
        domain = self._get_domain_key(url)
        entry = self._cache.get(domain)

        if not entry:
            return None

        # Check TTL
        expires_at = entry.get("expires_at")
        if expires_at is not None and expires_at < time.time():
            del self._cache[domain]
            self._save_cache()
            return None

        return {
            "start_stage": entry.get("best_stage", 1),
            "strategy_name": entry.get("best_strategy", "unknown"),
            "skip_stages": entry.get("failed_stages", []),
            "success_count": entry.get("success_count", 0),
            "avg_response_time": entry.get("avg_response_time", 0.0)
        }

    def record_success(
        self,
        url: str,
        stage: int,
        strategy_name: str,
        response_time: float = None,
        ttl_days: int = None
    ) -> None:
        """
        Record a successful crawl strategy for a domain.

        Args:
            url: The URL that was crawled.
            stage: The fallback stage number that succeeded (1-7).
            strategy_name: Name of the successful strategy.
            response_time: Time taken for the crawl in seconds.
            ttl_days: TTL in days for this entry (min 1).
        """
        domain = self._get_domain_key(url)

        # Validate TTL
        if ttl_days is None:
            ttl_days = self.default_ttl_days
        elif not isinstance(ttl_days, (int, float)) or ttl_days < 1:
            ttl_days = max(1, ttl_days) if isinstance(ttl_days, (int, float)) else self.default_ttl_days

        # Get existing entry or create new
        entry = self._cache.get(domain, {
            "domain": domain,
            "best_stage": stage,
            "best_strategy": strategy_name,
            "success_count": 0,
            "last_success": 0,
            "avg_response_time": 0.0,
            "failed_stages": [],
            "expires_at": 0
        })

        # Update entry
        old_count = entry.get("success_count", 0)
        old_avg = entry.get("avg_response_time", 0.0)

        entry["success_count"] = old_count + 1

        # Update best stage if this one is better (lower = faster)
        current_best = entry.get("best_stage", 7)
        if stage <= current_best:
            entry["best_stage"] = stage
            entry["best_strategy"] = strategy_name

        entry["last_success"] = time.time()
        entry["expires_at"] = time.time() + (ttl_days * 86400)

        # Update rolling average response time
        if response_time is not None and response_time > 0:
            if old_count > 0 and old_avg > 0:
                entry["avg_response_time"] = (old_avg * old_count + response_time) / (old_count + 1)
            else:
                entry["avg_response_time"] = response_time

        # Remove this stage from failed_stages if it was there
        failed = entry.get("failed_stages", [])
        if stage in failed:
            failed.remove(stage)
            entry["failed_stages"] = failed

        self._cache[domain] = entry
        self._save_cache()

    def record_failure(
        self,
        url: str,
        stage: int,
        strategy_name: str,
        error: str = None
    ) -> None:
        """
        Record a failed strategy attempt for a domain.

        Uses a failure count threshold to avoid marking stages as failed
        due to temporary issues.

        Args:
            url: The URL that failed.
            stage: The fallback stage number that failed (1-7).
            strategy_name: Name of the failed strategy.
            error: Optional error message.
        """
        FAILURE_THRESHOLD = 3  # Require 3 consecutive failures
        DECAY_HOURS = 24  # Reset failure count after 24 hours

        domain = self._get_domain_key(url)

        # Get existing entry or create minimal one
        entry = self._cache.get(domain, {
            "domain": domain,
            "best_stage": 7,
            "best_strategy": "unknown",
            "success_count": 0,
            "failed_stages": [],
            "failure_counts": {},
            "last_failure_time": {},
            "expires_at": time.time() + (self.default_ttl_days * 86400)
        })

        # Initialize failure tracking if not present
        if "failure_counts" not in entry:
            entry["failure_counts"] = {}
        if "last_failure_time" not in entry:
            entry["last_failure_time"] = {}

        now = time.time()
        stage_key = str(stage)

        # Check if previous failure has decayed
        last_failure = entry["last_failure_time"].get(stage_key, 0)
        if now - last_failure > DECAY_HOURS * 3600:
            entry["failure_counts"][stage_key] = 0

        # Increment failure count
        current_count = entry["failure_counts"].get(stage_key, 0) + 1
        entry["failure_counts"][stage_key] = current_count
        entry["last_failure_time"][stage_key] = now

        # Only add to failed_stages if threshold reached
        failed = entry.get("failed_stages", [])
        if not isinstance(failed, list):
            failed = []

        if current_count >= FAILURE_THRESHOLD and stage not in failed:
            failed.append(stage)
            failed.sort()
            entry["failed_stages"] = failed
            print(f"Strategy cache: Stage {stage} marked as failed for {domain} after {current_count} failures")

        # Update best_stage if current best has failed
        if entry.get("best_stage", 1) in failed:
            for s in range(1, 8):
                if s not in failed:
                    entry["best_stage"] = s
                    entry["best_strategy"] = self.STAGE_NAMES.get(s, f"stage_{s}")
                    break

        self._cache[domain] = entry
        self._save_cache()

    def get_recommended_stages(self, url: str) -> List[int]:
        """
        Get recommended stage order for a URL based on cache.

        Args:
            url: The URL to get recommendations for.

        Returns:
            List of stage numbers in recommended order.
        """
        strategy = self.get_best_strategy(url)

        if not strategy:
            return [1, 2, 3, 4, 5, 6, 7]

        start_stage = strategy.get("start_stage", 1)
        skip_stages = strategy.get("skip_stages", [])

        recommended = []

        # Start with best known stage
        if start_stage not in skip_stages:
            recommended.append(start_stage)

        # Add remaining stages in order (except skipped)
        for stage in range(1, 8):
            if stage not in recommended and stage not in skip_stages:
                recommended.append(stage)

        # Add skipped stages at the end (as last resort)
        for stage in skip_stages:
            if stage not in recommended:
                recommended.append(stage)

        return recommended

    def clear_cache(self, url: str = None) -> None:
        """
        Clear cache for a specific domain or all domains.

        Args:
            url: If provided, clear only this domain. Otherwise clear all.
        """
        if url:
            domain = self._get_domain_key(url)
            if domain in self._cache:
                del self._cache[domain]
        else:
            self._cache = {}
        self._save_cache()

    def list_cached_domains(self) -> List[dict]:
        """
        List all cached domain strategies.

        Returns:
            List of dicts with domain strategy info.
        """
        result = []
        now = time.time()

        for domain, entry in self._cache.items():
            if not isinstance(entry, dict):
                continue

            expires_at = entry.get("expires_at")
            is_expired = expires_at is not None and expires_at < now

            result.append({
                "domain": domain,
                "best_stage": entry.get("best_stage"),
                "best_strategy": entry.get("best_strategy"),
                "success_count": entry.get("success_count", 0),
                "failed_stages": entry.get("failed_stages", []),
                "avg_response_time": entry.get("avg_response_time"),
                "is_expired": is_expired
            })

        return result


# Global instance for module-level access
_strategy_cache: Optional[StrategyCache] = None


def get_strategy_cache() -> StrategyCache:
    """Get the global StrategyCache instance."""
    global _strategy_cache
    if _strategy_cache is None:
        _strategy_cache = StrategyCache()
    return _strategy_cache
