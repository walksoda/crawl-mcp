"""Strategy cache and fingerprint profiles for Crawl4AI MCP Server.

This module manages crawling strategy caching and browser fingerprint
generation for anti-detection.
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


class FingerprintProfile:
    """
    Generates consistent browser fingerprint profiles for anti-detection.

    Creates matching sets of:
    - User-Agent strings
    - HTTP headers (sec-ch-ua*, Accept-Language, etc.)
    - JavaScript fingerprint evasion scripts
    - Timezone and locale settings
    """

    # Pre-defined browser profiles with consistent configurations
    BROWSER_PROFILES = {
        "chrome_windows": {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "platform": "Win32",
            "vendor": "Google Inc.",
            "app_version": "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "sec_ch_ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec_ch_ua_mobile": "?0",
            "sec_ch_ua_platform": '"Windows"',
            "languages": ["en-US", "en"],
            "timezone": "America/New_York",
            "webgl_vendor": "Google Inc. (NVIDIA)",
            "webgl_renderer": "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)",
        },
        "chrome_mac": {
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "platform": "MacIntel",
            "vendor": "Google Inc.",
            "app_version": "5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "sec_ch_ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec_ch_ua_mobile": "?0",
            "sec_ch_ua_platform": '"macOS"',
            "languages": ["en-US", "en"],
            "timezone": "America/Los_Angeles",
            "webgl_vendor": "Google Inc. (Apple)",
            "webgl_renderer": "ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)",
        },
        "firefox_windows": {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "platform": "Win32",
            "vendor": "",
            "app_version": "5.0 (Windows)",
            "sec_ch_ua": None,
            "sec_ch_ua_mobile": None,
            "sec_ch_ua_platform": None,
            "languages": ["en-US", "en"],
            "timezone": "America/New_York",
            "webgl_vendor": "Mozilla",
            "webgl_renderer": "Mozilla",
        },
        "safari_mac": {
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "platform": "MacIntel",
            "vendor": "Apple Computer, Inc.",
            "app_version": "5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "sec_ch_ua": None,
            "sec_ch_ua_mobile": None,
            "sec_ch_ua_platform": None,
            "languages": ["en-US", "en"],
            "timezone": "America/Los_Angeles",
            "webgl_vendor": "Apple Inc.",
            "webgl_renderer": "Apple GPU",
        },
        "chrome_mobile": {
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1",
            "platform": "iPhone",
            "vendor": "Google Inc.",
            "app_version": "5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1",
            "sec_ch_ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec_ch_ua_mobile": "?1",
            "sec_ch_ua_platform": '"iOS"',
            "languages": ["en-US", "en"],
            "timezone": "America/New_York",
            "webgl_vendor": "Apple Inc.",
            "webgl_renderer": "Apple GPU",
        },
        "safari_mobile": {
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "platform": "iPhone",
            "vendor": "Apple Computer, Inc.",
            "app_version": "5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "sec_ch_ua": None,
            "sec_ch_ua_mobile": None,
            "sec_ch_ua_platform": None,
            "languages": ["en-US", "en"],
            "timezone": "America/New_York",
            "webgl_vendor": "Apple Inc.",
            "webgl_renderer": "Apple GPU",
        },
    }

    def __init__(self, profile_name: str = None):
        """
        Initialize with a specific profile or random selection.

        Args:
            profile_name: One of the BROWSER_PROFILES keys, or None for random.
        """
        import random
        if profile_name and profile_name in self.BROWSER_PROFILES:
            self.profile_name = profile_name
        else:
            self.profile_name = random.choice(list(self.BROWSER_PROFILES.keys()))
        self.profile = self.BROWSER_PROFILES[self.profile_name]

    def get_user_agent(self) -> str:
        """Get the User-Agent string for this profile."""
        return self.profile["user_agent"]

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers consistent with this profile."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": ",".join(self.profile["languages"]) + ";q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        if self.profile.get("sec_ch_ua"):
            headers["Sec-CH-UA"] = self.profile["sec_ch_ua"]
        if self.profile.get("sec_ch_ua_mobile"):
            headers["Sec-CH-UA-Mobile"] = self.profile["sec_ch_ua_mobile"]
        if self.profile.get("sec_ch_ua_platform"):
            headers["Sec-CH-UA-Platform"] = self.profile["sec_ch_ua_platform"]

        return headers

    def get_stealth_js(self) -> str:
        """Generate JavaScript to evade fingerprint detection."""
        profile = self.profile
        is_mobile = 'mobile' in self.profile_name.lower() or 'iPhone' in profile["platform"]

        return f'''
(function() {{
    'use strict';

    // Remove webdriver flag
    Object.defineProperty(navigator, 'webdriver', {{
        get: () => undefined,
        configurable: true
    }});
    delete Navigator.prototype.webdriver;

    // Spoof navigator properties
    const navigatorProps = {{
        platform: '{profile["platform"]}',
        vendor: '{profile["vendor"]}',
        appVersion: '{profile["app_version"]}',
        languages: {json.dumps(profile["languages"])},
        language: '{profile["languages"][0]}',
        hardwareConcurrency: 8,
        deviceMemory: 8,
        maxTouchPoints: {'10' if is_mobile else '0'}
    }};

    for (const [prop, value] of Object.entries(navigatorProps)) {{
        try {{
            Object.defineProperty(navigator, prop, {{
                get: () => value,
                configurable: true
            }});
        }} catch (e) {{}}
    }}

    // Spoof WebGL fingerprint
    const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {{
        if (parameter === 37445) return '{profile["webgl_vendor"]}';
        if (parameter === 37446) return '{profile["webgl_renderer"]}';
        return originalGetParameter.call(this, parameter);
    }};

    if (typeof WebGL2RenderingContext !== 'undefined') {{
        const originalGetParameter2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) return '{profile["webgl_vendor"]}';
            if (parameter === 37446) return '{profile["webgl_renderer"]}';
            return originalGetParameter2.call(this, parameter);
        }};
    }}

    // Spoof Timezone
    const targetTimezone = '{profile["timezone"]}';
    const originalDateTimeFormat = Intl.DateTimeFormat;
    Intl.DateTimeFormat = function(locales, options) {{
        options = options || {{}};
        if (!options.timeZone) options.timeZone = targetTimezone;
        return new originalDateTimeFormat(locales, options);
    }};
    Intl.DateTimeFormat.prototype = originalDateTimeFormat.prototype;
    Intl.DateTimeFormat.supportedLocalesOf = originalDateTimeFormat.supportedLocalesOf;

    // Remove automation indicators
    const automationProps = [
        '__playwright', '__puppeteer_evaluation_script__',
        '__selenium_evaluate', '__webdriver_evaluate',
        '__driver_evaluate', '__webdriver_script_fn',
        '__webdriver_unwrapped', '_Selenium_IDE_Recorder',
        '_selenium', 'calledSelenium',
        '$chrome_asyncScriptInfo', '$cdc_asdjflasutopfhvcZLmcfl_'
    ];
    for (const prop of automationProps) {{
        try {{ delete window[prop]; delete document[prop]; }} catch (e) {{}}
    }}

    // Fix Chrome runtime
    if (!window.chrome) window.chrome = {{}};
    if (!window.chrome.runtime) window.chrome.runtime = {{}};
}})();
'''

    def get_timezone(self) -> str:
        """Get the timezone for this profile."""
        return self.profile["timezone"]

    def get_locale(self) -> str:
        """Get the primary locale for this profile."""
        return self.profile["languages"][0]

    @classmethod
    def get_random_profile(cls) -> "FingerprintProfile":
        """Get a random fingerprint profile."""
        import random
        profile_name = random.choice(list(cls.BROWSER_PROFILES.keys()))
        return cls(profile_name)

    @classmethod
    def get_desktop_profile(cls) -> "FingerprintProfile":
        """Get a random desktop browser profile."""
        import random
        desktop_profiles = [k for k in cls.BROWSER_PROFILES.keys() if "mobile" not in k.lower()]
        return cls(random.choice(desktop_profiles))

    @classmethod
    def get_mobile_profile(cls) -> "FingerprintProfile":
        """Get a mobile browser profile (iOS Safari)."""
        return cls("safari_mobile")


# Global instance for module-level access
_strategy_cache: Optional[StrategyCache] = None


def get_strategy_cache() -> StrategyCache:
    """Get the global StrategyCache instance."""
    global _strategy_cache
    if _strategy_cache is None:
        _strategy_cache = StrategyCache()
    return _strategy_cache


def get_fingerprint_config(profile_name: str = None) -> dict:
    """
    Get fingerprint configuration for a profile.

    Args:
        profile_name: Profile name or None for random.

    Returns:
        Dict with user_agent, headers, stealth_js, timezone, locale.
    """
    profile = FingerprintProfile(profile_name)
    return {
        "user_agent": profile.get_user_agent(),
        "headers": profile.get_headers(),
        "stealth_js": profile.get_stealth_js(),
        "timezone": profile.get_timezone(),
        "locale": profile.get_locale(),
        "profile_name": profile.profile_name
    }
