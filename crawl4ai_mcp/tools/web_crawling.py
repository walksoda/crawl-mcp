"""
Web crawling tools for Crawl4AI MCP Server.

Contains complete web crawling functionality and content extraction tools.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import (
    CrawlRequest,
    CrawlResponse,
    StructuredExtractionRequest
)

# Import required crawl4ai components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
    BM25ContentFilter,
    PruningContentFilter,
    LLMContentFilter,
    CacheMode,
)
from crawl4ai.chunking_strategy import (
    TopicSegmentationChunking,
    OverlappingWindowChunking,
    RegexChunking,
    SlidingWindowChunking
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

# Import processors and utilities
from ..file_processor import FileProcessor
from ..youtube_processor import YouTubeProcessor
from ..suppress_output import suppress_stdout_stderr

# Initialize processors
file_processor = FileProcessor()
youtube_processor = YouTubeProcessor()


def _convert_media_to_list(media: Any) -> List[Dict[str, Any]]:
    """
    Convert crawl4ai's media dict format to flat list format.

    crawl4ai returns media as: {'images': [...], 'videos': [...], 'audios': [...]}
    CrawlResponse expects: List[Dict[str, Any]]
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


def _has_meaningful_content(result, min_length: int = 100) -> tuple[bool, str]:
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


# Constants for fallback success validation
FALLBACK_MIN_CONTENT_LENGTH = 200
BLOCK_INDICATORS = [
    "access denied", "403 forbidden", "captcha",
    "please enable javascript", "bot detected",
    "unusual traffic", "rate limit", "you have been blocked",
    "security check", "verify you are human", "request blocked"
]


def _is_block_page(content: str) -> bool:
    """
    Check if content appears to be a block/error page.

    Args:
        content: Text content to check (should be lowercased or will be lowercased)

    Returns:
        True if block indicators are found, False otherwise
    """
    if not content:
        return False
    content_lower = content.lower() if not content.islower() else content
    return any(indicator in content_lower for indicator in BLOCK_INDICATORS)


def _normalize_cookies_to_playwright_format(
    cookies: Dict[str, str],
    url: str
) -> List[Dict[str, Any]]:
    """
    Convert Dict[str, str] cookies to Playwright format.

    Uses 'url' field for host-only cookies (recommended by Playwright).
    This ensures cookies are only sent to the exact host, not subdomains,
    and correctly handles IPv6 addresses and localhost.

    Args:
        cookies: Dictionary of cookie name-value pairs
        url: URL for cookie scope

    Returns:
        List of cookie dictionaries in Playwright format
    """
    # Build cookies with url field for host-only behavior (Playwright recommended)
    # Using 'url' instead of 'domain' ensures:
    # - Cookies are host-only (not sent to subdomains)
    # - IPv6 addresses work correctly
    # - localhost and IP addresses are handled properly
    result = []
    for name, value in cookies.items():
        cookie = {
            "name": name,
            "value": str(value),  # Ensure string type
            "url": url,  # Host-only cookie scope
            "path": "/",
        }
        result.append(cookie)

    return result


# Phase 3: Multi-stage fallback helper functions

async def _static_fetch_content(url: str, headers: dict = None, timeout: int = 30) -> tuple[bool, str, str]:
    """
    Stage 1: Fast static HTTP fetch without browser overhead.

    Uses httpx for direct HTTP requests with readability extraction.

    Returns:
        Tuple of (success: bool, content: str, error: str)
    """
    import httpx
    from urllib.parse import urlparse

    try:
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if headers:
            default_headers.update(headers)

        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url, headers=default_headers)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return False, "", f"Non-HTML content type: {content_type}"

            html_content = response.text
            if len(html_content.strip()) < 100:
                return False, "", "Content too short"

            return True, html_content, ""

    except httpx.HTTPStatusError as e:
        return False, "", f"HTTP error {e.response.status_code}"
    except httpx.RequestError as e:
        return False, "", f"Request error: {str(e)}"
    except Exception as e:
        return False, "", f"Static fetch error: {str(e)}"


def _extract_spa_json_data(html_content: str) -> tuple[bool, dict, str]:
    """
    Stage 6: Extract JSON data from SPA frameworks.

    Extracts data from:
    - __NEXT_DATA__ (Next.js)
    - window.__INITIAL_STATE__ (various frameworks)
    - window.__NUXT__ (Nuxt.js)
    - window.__APP_STATE__ (various frameworks)

    Uses balanced brace matching to handle nested JSON correctly.

    Returns:
        Tuple of (success: bool, data: dict, source: str)
    """
    import re

    def extract_balanced_json(text: str, start_pos: int) -> str:
        """Extract JSON object with balanced braces starting from start_pos."""
        if start_pos >= len(text) or text[start_pos] != '{':
            return ""

        depth = 0
        in_string = False
        escape_next = False
        end_pos = start_pos

        for i in range(start_pos, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

        if depth != 0:
            return ""

        return text[start_pos:end_pos]

    # Extraction patterns: (regex to find start position, source name)
    patterns = [
        # Next.js - extract content from script tag
        (r'<script[^>]*id="__NEXT_DATA__"[^>]*>', "next_data"),
        # Nuxt.js
        (r'window\.__NUXT__\s*=\s*', "nuxt_data"),
        # Generic initial state patterns
        (r'window\.__INITIAL_STATE__\s*=\s*', "initial_state"),
        (r'window\.__APP_STATE__\s*=\s*', "app_state"),
        (r'window\.__PRELOADED_STATE__\s*=\s*', "preloaded_state"),
    ]

    for pattern, source in patterns:
        try:
            match = re.search(pattern, html_content)
            if match:
                # Find the start of JSON object after the pattern
                search_start = match.end()
                # Skip any whitespace
                while search_start < len(html_content) and html_content[search_start] in ' \t\n\r':
                    search_start += 1

                if search_start < len(html_content) and html_content[search_start] == '{':
                    json_str = extract_balanced_json(html_content, search_start)
                    if json_str:
                        data = json.loads(json_str)
                        return True, data, source
        except (json.JSONDecodeError, AttributeError, IndexError):
            continue

    return False, {}, ""


def _detect_spa_framework(html_content: str) -> tuple[str, str]:
    """
    Stage 4: Detect SPA framework for optimized crawling.

    Returns:
        Tuple of (framework_name: str, suggested_selector: str)
    """
    indicators = [
        # Next.js
        ("__NEXT_DATA__", "next.js", "#__next, [data-nextjs-page]"),
        ("_next/static", "next.js", "#__next"),
        # React
        ("data-reactroot", "react", "[data-reactroot], #root, #app"),
        ("__REACT_DEVTOOLS", "react", "#root, #app"),
        # Vue.js
        ("data-v-", "vue.js", "#app, [data-v-app]"),
        ("__VUE__", "vue.js", "#app"),
        # Angular
        ("ng-version", "angular", "app-root, [ng-version]"),
        ("_ngcontent", "angular", "app-root"),
        # Nuxt.js
        ("__NUXT__", "nuxt.js", "#__nuxt, #__layout"),
        # Svelte
        ("__sveltekit", "sveltekit", "#svelte, body > div"),
    ]

    for indicator, framework, selector in indicators:
        if indicator in html_content:
            return framework, selector

    return "", ""


def _build_amp_url(url: str) -> str:
    """
    Stage 6: Build AMP version URL.

    Attempts to construct AMP URL for the given page.
    Preserves query string and fragment from the original URL.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    path = parsed.path

    # Common AMP URL patterns
    if not path.endswith('/amp') and not path.endswith('/amp/'):
        if path.endswith('/'):
            amp_path = path + 'amp/'
        else:
            amp_path = path + '/amp'

        # Preserve query string and fragment from original URL
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            amp_path,
            parsed.params,  # URL parameters (rarely used but preserve)
            parsed.query,   # Query string (?key=value)
            parsed.fragment # Fragment (#section)
        ))

    return ""


async def _try_fetch_rss_feed(url: str) -> tuple[bool, str, list]:
    """
    Stage 5: Try to find and fetch RSS/Atom feed for the page.

    Returns:
        Tuple of (success: bool, feed_url: str, items: list)
    """
    import httpx
    import re
    from urllib.parse import urljoin

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            # First fetch the page to find feed links
            response = await client.get(url)
            html = response.text

            # Look for RSS/Atom feed links
            feed_patterns = [
                r'<link[^>]*type=["\']application/rss\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                r'<link[^>]*type=["\']application/atom\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                r'<link[^>]*href=["\']([^"\']+)["\'][^>]*type=["\']application/rss\+xml["\']',
                r'<link[^>]*href=["\']([^"\']+)["\'][^>]*type=["\']application/atom\+xml["\']',
            ]

            feed_url = None
            for pattern in feed_patterns:
                match = re.search(pattern, html)
                if match:
                    feed_url = urljoin(url, match.group(1))
                    break

            if not feed_url:
                # Try common feed URL patterns
                from urllib.parse import urlparse
                parsed = urlparse(url)
                common_feeds = [
                    f"{parsed.scheme}://{parsed.netloc}/feed",
                    f"{parsed.scheme}://{parsed.netloc}/rss",
                    f"{parsed.scheme}://{parsed.netloc}/atom.xml",
                    f"{parsed.scheme}://{parsed.netloc}/feed.xml",
                    f"{parsed.scheme}://{parsed.netloc}/rss.xml",
                ]

                for potential_feed in common_feeds:
                    try:
                        feed_response = await client.get(potential_feed)
                        if feed_response.status_code == 200:
                            content_type = feed_response.headers.get("content-type", "").lower()
                            if "xml" in content_type or "rss" in content_type or "atom" in content_type:
                                feed_url = potential_feed
                                break
                    except:
                        continue

            if feed_url:
                # Fetch and parse the feed
                feed_response = await client.get(feed_url)
                feed_content = feed_response.text

                # Basic XML parsing for items
                items = []
                item_pattern = r'<item>(.*?)</item>|<entry>(.*?)</entry>'
                for match in re.finditer(item_pattern, feed_content, re.DOTALL):
                    item_xml = match.group(1) or match.group(2)
                    title_match = re.search(r'<title[^>]*>([^<]+)</title>', item_xml)
                    link_match = re.search(r'<link[^>]*>([^<]+)</link>|<link[^>]*href=["\']([^"\']+)["\']', item_xml)
                    desc_match = re.search(r'<description[^>]*>([^<]+)</description>|<summary[^>]*>([^<]+)</summary>', item_xml, re.DOTALL)

                    item = {}
                    if title_match:
                        item['title'] = title_match.group(1).strip()
                    if link_match:
                        item['link'] = (link_match.group(1) or link_match.group(2)).strip()
                    if desc_match:
                        item['description'] = (desc_match.group(1) or desc_match.group(2)).strip()

                    if item:
                        items.append(item)

                return True, feed_url, items

    except Exception as e:
        pass

    return False, "", []


# ========================================
# Phase 6: Session Management
# ========================================

class SessionManager:
    """
    Manages browser session persistence for web crawling.

    Stores and retrieves session data (cookies, localStorage) per domain,
    enabling authenticated crawling without re-login on each request.

    Session data is stored in a JSON file (~/.crawl4ai_sessions.json) with
    secure file permissions (0600).

    Current Limitations:
    - Only user-provided cookies are saved (crawl4ai doesn't expose response cookies)
    - localStorage is stored but not applied (crawl_url doesn't support storage_state)
    - Full Playwright storage_state integration requires architectural changes

    Security Notes:
    - Session file is protected with owner-only permissions
    - URL credentials (user:pass@host) are stripped from domain keys
    - Consider additional encryption for sensitive environments
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize the session manager.

        Args:
            storage_path: Path to the session storage JSON file.
                         If None, uses a default path in the user's home directory.
        """
        import os
        from pathlib import Path

        if storage_path is None:
            # Default to ~/.crawl4ai_sessions.json
            home = Path.home()
            storage_path = str(home / ".crawl4ai_sessions.json")

        self.storage_path = storage_path
        self._sessions: Dict[str, dict] = {}
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions from the storage file with validation."""
        import os

        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate data structure - must be a dict
                    if not isinstance(data, dict):
                        print(f"Warning: Invalid session data format, expected dict")
                        self._sessions = {}
                        return

                    # Validate each domain entry
                    validated = {}
                    for domain, session in data.items():
                        if not isinstance(domain, str) or not isinstance(session, dict):
                            continue  # Skip invalid entries
                        # Ensure required fields exist with valid types
                        if 'storage_state' not in session:
                            continue
                        if not isinstance(session.get('storage_state'), dict):
                            continue
                        # Validate expires_at if present
                        expires_at = session.get('expires_at')
                        if expires_at is not None and not isinstance(expires_at, (int, float)):
                            session['expires_at'] = None  # Remove invalid expiry
                        validated[domain] = session

                    self._sessions = validated
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load sessions from {self.storage_path}: {e}")
                self._sessions = {}

    def _save_sessions(self) -> None:
        """Save sessions to the storage file with secure permissions."""
        import os
        import stat

        temp_path = self.storage_path + ".tmp"
        try:
            # Remove temp file if it exists to prevent symlink attacks
            # O_EXCL will fail if file exists, so we need clean state
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

            # Create temp file with O_EXCL to prevent race conditions and symlink attacks
            # O_EXCL ensures atomic creation - fails if file already exists
            fd = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self._sessions, f, indent=2, ensure_ascii=False)
            except Exception:
                # fd is closed by fdopen even on error, but close explicitly if fdopen fails
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            # Atomic rename
            os.replace(temp_path, self.storage_path)

        except (IOError, TypeError, ValueError, OSError) as e:
            print(f"Warning: Could not save sessions to {self.storage_path}: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _get_domain_key(self, url: str) -> str:
        """Extract domain from URL for session lookup, stripping credentials."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Strip user:pass@ from netloc for security
        netloc = parsed.netloc.lower()
        if '@' in netloc:
            netloc = netloc.split('@')[-1]
        return netloc

    def get_session(self, url: str) -> Optional[dict]:
        """
        Get stored session data for a URL's domain.

        Args:
            url: The URL to get session data for.

        Returns:
            Session data dict with 'cookies' and 'origins' keys, or None.
        """
        domain = self._get_domain_key(url)
        session = self._sessions.get(domain)

        if session and isinstance(session, dict):
            # Check if session has expired
            import time
            expires_at = session.get('expires_at')
            if isinstance(expires_at, (int, float)) and expires_at < time.time():
                # Session expired, remove it
                del self._sessions[domain]
                self._save_sessions()
                return None

            storage_state = session.get('storage_state')
            if isinstance(storage_state, dict):
                return storage_state

        return None

    def save_session(
        self,
        url: str,
        cookies: List[dict] = None,
        local_storage: Dict[str, str] = None,
        ttl_hours: int = 24
    ) -> None:
        """
        Save session data for a URL's domain.

        Args:
            url: The URL to save session data for.
            cookies: List of cookie dicts with name, value, domain, etc.
            local_storage: Dict of localStorage key-value pairs.
            ttl_hours: Time-to-live in hours for the session (default 24, min 1).
        """
        import time
        from urllib.parse import urlparse

        # Validate TTL
        if ttl_hours < 1:
            ttl_hours = 1  # Minimum 1 hour

        domain = self._get_domain_key(url)
        parsed = urlparse(url)
        # Strip credentials from origin for security
        netloc = parsed.netloc
        if '@' in netloc:
            netloc = netloc.split('@')[-1]
        origin = f"{parsed.scheme}://{netloc}"

        # Build storage_state in Playwright format
        storage_state = {
            "cookies": cookies or [],
            "origins": []
        }

        if local_storage:
            storage_state["origins"].append({
                "origin": origin,
                "localStorage": [
                    {"name": k, "value": v} for k, v in local_storage.items()
                ]
            })

        self._sessions[domain] = {
            "storage_state": storage_state,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl_hours * 3600),
            "domain": domain,
            "origin": origin
        }

        self._save_sessions()

    def save_storage_state(self, url: str, storage_state: dict, ttl_hours: int = 24) -> None:
        """
        Save a complete storage_state dict for a URL's domain.

        Args:
            url: The URL to save session data for.
            storage_state: Complete storage_state dict from Playwright.
            ttl_hours: Time-to-live in hours for the session (min 1).
        """
        import time

        # Validate TTL - clamp to minimum of 1 hour (consistent with save_session)
        if not isinstance(ttl_hours, (int, float)) or ttl_hours < 1:
            ttl_hours = max(1, ttl_hours) if isinstance(ttl_hours, (int, float)) else 1

        domain = self._get_domain_key(url)

        self._sessions[domain] = {
            "storage_state": storage_state,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl_hours * 3600),
            "domain": domain
        }

        self._save_sessions()

    def clear_session(self, url: str) -> bool:
        """
        Clear session data for a URL's domain.

        Args:
            url: The URL to clear session data for.

        Returns:
            True if session was cleared, False if no session existed.
        """
        domain = self._get_domain_key(url)
        if domain in self._sessions:
            del self._sessions[domain]
            self._save_sessions()
            return True
        return False

    def clear_all_sessions(self) -> int:
        """
        Clear all stored sessions.

        Returns:
            Number of sessions cleared.
        """
        count = len(self._sessions)
        self._sessions = {}
        self._save_sessions()
        return count

    def list_sessions(self) -> List[dict]:
        """
        List all stored sessions with metadata.

        Returns:
            List of session info dicts with domain, created_at, expires_at.
        """
        import time
        current_time = time.time()
        result = []
        for domain, data in self._sessions.items():
            if not isinstance(data, dict):
                continue  # Skip invalid entries

            expires_at = data.get("expires_at")
            is_expired = False
            if isinstance(expires_at, (int, float)):
                is_expired = expires_at < current_time

            storage_state = data.get("storage_state", {})
            cookie_count = 0
            if isinstance(storage_state, dict):
                cookies = storage_state.get("cookies", [])
                if isinstance(cookies, list):
                    cookie_count = len(cookies)

            result.append({
                "domain": domain,
                "created_at": data.get("created_at"),
                "expires_at": expires_at,
                "is_expired": is_expired,
                "cookie_count": cookie_count
            })
        return result

    def get_browser_config_params(self, url: str) -> dict:
        """
        Get BrowserConfig parameters for session persistence.

        Args:
            url: The URL to get config for.

        Returns:
            Dict with storage_state and other browser config params.
        """
        session = self.get_session(url)
        if session:
            return {
                "storage_state": session,
                "use_persistent_context": True
            }
        return {}


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

    Strategy Entry Structure:
    {
        "domain": "example.com",
        "best_stage": 3,
        "best_strategy": "chromium_stealth",
        "success_count": 5,
        "last_success": 1234567890.0,
        "avg_response_time": 2.5,
        "failed_stages": [1, 2],  # Stages that consistently fail
        "expires_at": 1234567890.0
    }
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
        import os
        self.storage_path = storage_path or os.path.expanduser("~/.crawl4ai_strategy_cache.json")
        self.default_ttl_days = max(1, default_ttl_days) if isinstance(default_ttl_days, (int, float)) else 7
        self._cache: Dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from storage file with validation."""
        import os

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
                    # Filter to valid integers only
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
        import os

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
        """Extract domain key from URL, stripping credentials and path/query."""
        from urllib.parse import urlparse
        try:
            # Add scheme if missing to ensure proper parsing
            if not url.startswith(('http://', 'https://', '//')):
                url = 'https://' + url
            parsed = urlparse(url)
            # Strip user:pass from netloc and extract hostname only
            host = parsed.hostname
            if not host:
                # Fallback: split netloc manually
                netloc = parsed.netloc or url.split('/')[0]
                host = netloc.split('@')[-1].split(':')[0]
            # Never return path, query, or fragment - only domain
            return host.lower() if host else "unknown"
        except Exception:
            # Last resort: try to extract domain-like pattern
            import re
            match = re.search(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+)', url)
            return match.group(1).lower() if match else "unknown"

    def get_best_strategy(self, url: str) -> Optional[dict]:
        """
        Get the best known strategy for a domain.

        Args:
            url: The URL to get strategy for.

        Returns:
            Dict with strategy info or None if no cached strategy.
            {
                "start_stage": int,  # Stage to start from
                "strategy_name": str,
                "skip_stages": list[int],  # Stages to skip (known failures)
                "success_count": int,
                "avg_response_time": float
            }
        """
        import time

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
        import time

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
        due to temporary issues. Only after FAILURE_THRESHOLD consecutive
        failures is a stage added to failed_stages.

        Args:
            url: The URL that failed.
            stage: The fallback stage number that failed (1-7).
            strategy_name: Name of the failed strategy.
            error: Optional error message.
        """
        import time

        FAILURE_THRESHOLD = 3  # Require 3 consecutive failures to mark as "failed"
        DECAY_HOURS = 24  # Reset failure count after 24 hours

        domain = self._get_domain_key(url)

        # Get existing entry or create minimal one
        entry = self._cache.get(domain, {
            "domain": domain,
            "best_stage": 7,  # Default to last resort
            "best_strategy": "unknown",
            "success_count": 0,
            "failed_stages": [],
            "failure_counts": {},  # Track per-stage failure counts
            "last_failure_time": {},  # Track when failures occurred
            "expires_at": time.time() + (self.default_ttl_days * 86400)
        })

        # Initialize failure tracking if not present
        if "failure_counts" not in entry:
            entry["failure_counts"] = {}
        if "last_failure_time" not in entry:
            entry["last_failure_time"] = {}

        now = time.time()
        stage_key = str(stage)  # Use string keys for JSON compatibility

        # Check if previous failure has decayed
        last_failure = entry["last_failure_time"].get(stage_key, 0)
        if now - last_failure > DECAY_HOURS * 3600:
            # Reset failure count after decay period
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
            # Find the lowest non-failed stage
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

        Returns stages in optimal order: best_stage first,
        then remaining stages (excluding known failures at end).

        Args:
            url: The URL to get recommendations for.

        Returns:
            List of stage numbers in recommended order.
        """
        strategy = self.get_best_strategy(url)

        if not strategy:
            # No cache - use default order
            return [1, 2, 3, 4, 5, 6, 7]

        start_stage = strategy.get("start_stage", 1)
        skip_stages = strategy.get("skip_stages", [])

        # Build recommended order
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
        import time
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


# Global strategy cache instance
_strategy_cache: Optional[StrategyCache] = None


def get_strategy_cache() -> StrategyCache:
    """Get or create the global strategy cache instance."""
    global _strategy_cache
    if _strategy_cache is None:
        _strategy_cache = StrategyCache()
    return _strategy_cache


# ========================================
# Phase 8: Fingerprint Evasion
# ========================================

class FingerprintProfile:
    """
    Generates consistent browser fingerprint profiles for anti-detection.

    Creates matching sets of:
    - User-Agent strings
    - HTTP headers (sec-ch-ua*, Accept-Language, etc.)
    - JavaScript fingerprint evasion scripts
    - Timezone and locale settings

    The key is consistency: all components must match to avoid detection.
    For example, a Chrome User-Agent must have matching sec-ch-ua headers.
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
            "sec_ch_ua": None,  # Firefox doesn't send sec-ch-ua
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
            "sec_ch_ua": None,  # Safari doesn't send sec-ch-ua
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
            # iOS Safari - used for Stage 5 mobile agent
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "platform": "iPhone",
            "vendor": "Apple Computer, Inc.",
            "app_version": "5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "sec_ch_ua": None,  # Safari doesn't send sec-ch-ua
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
        """
        Get HTTP headers consistent with this profile.

        Returns headers that match the User-Agent to avoid detection.
        """
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

        # Add sec-ch-ua headers only for Chrome-based browsers
        if self.profile.get("sec_ch_ua"):
            headers["Sec-CH-UA"] = self.profile["sec_ch_ua"]
        if self.profile.get("sec_ch_ua_mobile"):
            headers["Sec-CH-UA-Mobile"] = self.profile["sec_ch_ua_mobile"]
        if self.profile.get("sec_ch_ua_platform"):
            headers["Sec-CH-UA-Platform"] = self.profile["sec_ch_ua_platform"]

        return headers

    def get_stealth_js(self) -> str:
        """
        Generate JavaScript to evade fingerprint detection.

        This script should be injected before page load to:
        1. Remove navigator.webdriver flag
        2. Spoof navigator properties
        3. Spoof WebGL fingerprint
        4. Spoof Canvas fingerprint
        5. Spoof AudioContext fingerprint
        """
        profile = self.profile

        return f'''
// ========================================
// Phase 8: Fingerprint Evasion Script
// ========================================

(function() {{
    'use strict';

    // 1. Remove webdriver flag
    Object.defineProperty(navigator, 'webdriver', {{
        get: () => undefined,
        configurable: true
    }});

    // Delete webdriver from navigator prototype
    delete Navigator.prototype.webdriver;

    // 2. Spoof navigator properties
    const navigatorProps = {{
        platform: '{profile["platform"]}',
        vendor: '{profile["vendor"]}',
        appVersion: '{profile["app_version"]}',
        languages: {json.dumps(profile["languages"])},
        language: '{profile["languages"][0]}',
        hardwareConcurrency: 8,
        deviceMemory: 8,
        maxTouchPoints: {'10' if 'mobile' in self.profile_name.lower() or 'iPhone' in profile["platform"] else '0'}
    }};

    for (const [prop, value] of Object.entries(navigatorProps)) {{
        try {{
            Object.defineProperty(navigator, prop, {{
                get: () => value,
                configurable: true
            }});
        }} catch (e) {{}}
    }}

    // 3. Spoof WebGL fingerprint
    const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {{
        // UNMASKED_VENDOR_WEBGL
        if (parameter === 37445) {{
            return '{profile["webgl_vendor"]}';
        }}
        // UNMASKED_RENDERER_WEBGL
        if (parameter === 37446) {{
            return '{profile["webgl_renderer"]}';
        }}
        return originalGetParameter.call(this, parameter);
    }};

    // Also for WebGL2
    if (typeof WebGL2RenderingContext !== 'undefined') {{
        const originalGetParameter2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) {{
                return '{profile["webgl_vendor"]}';
            }}
            if (parameter === 37446) {{
                return '{profile["webgl_renderer"]}';
            }}
            return originalGetParameter2.call(this, parameter);
        }};
    }}

    // 4. Spoof Canvas fingerprint (add consistent noise based on domain seed)
    // Use seeded PRNG for consistent fingerprint - reset seed on each call
    const canvasSeed = Array.from(window.location.hostname).reduce((a, c) => a + c.charCodeAt(0), 0);
    function createSeededRng(seed) {{
        let state = seed;
        return function() {{
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            return state / 0x7fffffff;
        }};
    }}
    const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function(type) {{
        if (this.width > 0 && this.height > 0) {{
            const ctx = this.getContext('2d');
            if (ctx) {{
                // Reset RNG seed on each call for consistent fingerprint
                const seededRandom = createSeededRng(canvasSeed);
                // Add invisible noise to canvas using seeded RNG for consistency
                const imageData = ctx.getImageData(0, 0, Math.min(this.width, 10), Math.min(this.height, 10));
                for (let i = 0; i < imageData.data.length; i += 4) {{
                    // Subtle modification that doesn't affect visual appearance
                    // Uses seeded RNG so same domain always gets same noise pattern
                    imageData.data[i] = imageData.data[i] ^ (seededRandom() > 0.99 ? 1 : 0);
                }}
                ctx.putImageData(imageData, 0, 0);
            }}
        }}
        return originalToDataURL.apply(this, arguments);
    }};

    // 5. Spoof AudioContext fingerprint
    if (typeof AudioContext !== 'undefined') {{
        const originalCreateOscillator = AudioContext.prototype.createOscillator;
        AudioContext.prototype.createOscillator = function() {{
            const oscillator = originalCreateOscillator.call(this);
            // Add tiny frequency variation
            const originalFrequency = oscillator.frequency;
            Object.defineProperty(oscillator, 'frequency', {{
                get: function() {{
                    return originalFrequency;
                }},
                configurable: true
            }});
            return oscillator;
        }};
    }}

    // 6. Spoof Permissions API
    if (navigator.permissions) {{
        const originalQuery = navigator.permissions.query;
        navigator.permissions.query = function(parameters) {{
            if (parameters.name === 'notifications') {{
                return Promise.resolve({{ state: 'prompt', onchange: null }});
            }}
            return originalQuery.call(this, parameters);
        }};
    }}

    // 7. Spoof Plugin array (Chrome typically has plugins)
    Object.defineProperty(navigator, 'plugins', {{
        get: () => {{
            const plugins = {{
                length: {'5' if 'chrome' in self.profile_name.lower() else '0'},
                item: function(i) {{ return this[i] || null; }},
                namedItem: function(name) {{ return null; }},
                refresh: function() {{}}
            }};
            {'// Chrome plugins' if 'chrome' in self.profile_name.lower() else ''}
            return plugins;
        }},
        configurable: true
    }});

    // 8. Spoof Timezone
    const targetTimezone = '{profile["timezone"]}';
    const originalDateTimeFormat = Intl.DateTimeFormat;
    Intl.DateTimeFormat = function(locales, options) {{
        options = options || {{}};
        if (!options.timeZone) {{
            options.timeZone = targetTimezone;
        }}
        return new originalDateTimeFormat(locales, options);
    }};
    Intl.DateTimeFormat.prototype = originalDateTimeFormat.prototype;
    Intl.DateTimeFormat.supportedLocalesOf = originalDateTimeFormat.supportedLocalesOf;

    // 9. Remove automation indicators
    // Remove Playwright/Puppeteer traces
    const automationProps = [
        '__playwright',
        '__puppeteer_evaluation_script__',
        '__selenium_evaluate',
        '__webdriver_evaluate',
        '__driver_evaluate',
        '__webdriver_script_fn',
        '__webdriver_unwrapped',
        '__lastWatirAlert',
        '__lastWatirConfirm',
        '__lastWatirPrompt',
        '_Selenium_IDE_Recorder',
        '_selenium',
        'calledSelenium',
        '$chrome_asyncScriptInfo',
        '$cdc_asdjflasutopfhvcZLmcfl_',
        '__cdc_asdjflasutopfhvcZLmcfl_'
    ];

    for (const prop of automationProps) {{
        try {{
            delete window[prop];
            delete document[prop];
        }} catch (e) {{}}
    }}

    // 10. Fix Chrome runtime
    if (!window.chrome) {{
        window.chrome = {{}};
    }}
    if (!window.chrome.runtime) {{
        window.chrome.runtime = {{}};
    }}

    console.log('Fingerprint evasion script loaded');
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


def get_fingerprint_config(profile_name: str = None) -> Dict[str, any]:
    """
    Get a complete fingerprint evasion configuration.

    Args:
        profile_name: Specific profile name or None for random desktop.

    Returns:
        Dict with user_agent, headers, stealth_js, timezone, locale
    """
    if profile_name:
        fp = FingerprintProfile(profile_name)
    else:
        fp = FingerprintProfile.get_desktop_profile()

    return {
        "profile_name": fp.profile_name,
        "user_agent": fp.get_user_agent(),
        "headers": fp.get_headers(),
        "stealth_js": fp.get_stealth_js(),
        "timezone": fp.get_timezone(),
        "locale": fp.get_locale()
    }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(storage_path: str = None) -> SessionManager:
    """
    Get or create the global session manager instance.

    Args:
        storage_path: Optional custom storage path for sessions.

    Returns:
        The SessionManager instance.
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(storage_path)
    return _session_manager


async def extract_cookies_from_result(result) -> List[dict]:
    """
    Extract cookies from a crawl result if available.

    Note: This requires the crawler to expose cookies in the result,
    which may not always be available depending on crawl4ai version.

    Args:
        result: CrawlResponse or similar result object.

    Returns:
        List of cookie dicts, or empty list if not available.
    """
    # Try to get cookies from various possible locations
    cookies = []

    # Check if result has cookies attribute
    if hasattr(result, 'cookies'):
        cookies = result.cookies or []
    elif hasattr(result, 'extracted_data') and result.extracted_data:
        cookies = result.extracted_data.get('cookies', [])

    return cookies


# Placeholder for summarize_web_content function
# Response size limit for MCP protocol (approximately 100k tokens)
# This prevents issues with oversized responses in Claude Desktop
MAX_RESPONSE_TOKENS = 100000  # Conservative limit to ensure safe transmission
MAX_RESPONSE_CHARS = MAX_RESPONSE_TOKENS * 4  # Rough estimate: 1 token ≈ 4 characters

async def summarize_web_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize web content using LLM with fallback to basic text truncation.
    """
    try:
        # Import config
        try:
            from ..config import get_llm_config
        except ImportError:
            from config import get_llm_config
        
        # Get LLM configuration
        llm_config = get_llm_config(llm_provider, llm_model)
        
        # Prepare summarization prompt based on summary length
        length_instructions = {
            "short": "Provide a brief 2-3 sentence summary of the main points.",
            "medium": "Provide a comprehensive summary in 1-2 paragraphs covering key points.",
            "long": "Provide a detailed summary covering all important information, insights, and context."
        }
        
        prompt = f"""
        Please summarize the following web content.
        
        Title: {title}
        URL: {url}
        
        {length_instructions.get(summary_length, length_instructions["medium"])}
        
        Focus on:
        - Main topics and key information
        - Important facts, statistics, or findings
        - Practical insights or conclusions
        - Technical details if present
        
        Content to summarize:
        {content[:50000]}  # Limit to prevent token overflow
        """
        
        # Get provider info from config
        provider_info = llm_config.provider.split('/')
        provider = provider_info[0] if provider_info else 'openai'
        model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
        
        summary_text = None
        
        if provider == 'openai':
            import openai
            api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = openai.AsyncOpenAI(api_key=api_key, base_url=llm_config.base_url)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert content summarizer. Provide clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            summary_text = response.choices[0].message.content
            
        elif provider == 'anthropic':
            import anthropic
            api_key = llm_config.api_token or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found")
            
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            summary_text = response.content[0].text

        elif provider == 'ollama':
            import aiohttp

            base_url = llm_config.base_url or 'http://localhost:11434'

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary_text = result.get('response', '')
                    else:
                        error_text = await response.text()
                        raise ValueError(f"Ollama API request failed: {response.status} - {error_text}")
        else:
            # Fallback for unsupported providers
            raise ValueError(f"LLM provider '{provider}' not supported for summarization. Supported: openai, anthropic, ollama")
        
        if summary_text:
            # Calculate compression ratio
            original_length = len(content)
            summary_length_chars = len(summary_text)
            compression_ratio = round((1 - summary_length_chars / original_length) * 100, 2)
            
            return {
                "success": True,
                "summary": summary_text,
                "original_length": original_length,
                "summary_length": summary_length_chars,
                "compressed_ratio": compression_ratio,
                "llm_provider": provider,
                "llm_model": model,
                "content_type": "web_content"
            }
        else:
            raise ValueError("LLM returned empty summary")
            
    except Exception as e:
        # Fallback to simple truncation if LLM fails
        max_chars = {
            "short": 500,
            "medium": 1500,
            "long": 3000
        }.get(summary_length, 1500)
        
        truncated = content[:max_chars]
        if len(content) > max_chars:
            truncated += "... [Content truncated due to size]"
        
        return {
            "success": False,
            "error": f"LLM summarization failed: {str(e)}. Returning truncated content.",
            "summary": truncated,
            "original_length": len(content),
            "summary_length": len(truncated),
            "compressed_ratio": round((1 - len(truncated) / len(content)) * 100, 2) if content else 0,
            "fallback_method": "truncation"
        }


def _process_response_content(response: CrawlResponse, include_cleaned_html: bool) -> CrawlResponse:
    """
    Process CrawlResponse to remove content field if include_cleaned_html is False.
    By default, only markdown is returned to reduce token usage and improve readability.
    """
    if not include_cleaned_html and hasattr(response, 'content'):
        response.content = None
    return response


# Complete implementation of internal crawl URL function
async def _internal_crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    try:
        # Check if URL is a YouTube video
        if youtube_processor.is_youtube_url(request.url):
            # ⚠️ WARNING: YouTube transcript extraction is currently experiencing issues
            # due to API specification changes. Attempting extraction but may fail.
            try:
                youtube_result = await youtube_processor.process_youtube_url(
                    url=request.url,
                    languages=["ja", "en"],  # Default language preferences
                    include_timestamps=True,
                    preserve_formatting=True,
                    include_metadata=True
                )
                
                if youtube_result['success']:
                    transcript_data = youtube_result['transcript']
                    response = CrawlResponse(
                        success=True,
                        url=request.url,
                        title=f"YouTube Video Transcript: {youtube_result['video_id']}",
                        content=transcript_data.get('full_text'),
                        markdown=transcript_data.get('clean_text'),
                        extracted_data={
                            "video_id": youtube_result['video_id'],
                            "processing_method": "youtube_transcript_api",
                            "language_info": youtube_result.get('language_info'),
                            "transcript_stats": {
                                "word_count": transcript_data.get('word_count'),
                                "segment_count": transcript_data.get('segment_count'),
                                "duration": transcript_data.get('duration_formatted')
                            },
                            "metadata": youtube_result.get('metadata')
                        }
                    )
                    response = await _check_and_summarize_if_needed(response, request)
                    return _process_response_content(response, request.include_cleaned_html)
                else:
                    # If YouTube transcript extraction fails, provide helpful error message
                    error_msg = youtube_result.get('error', 'Unknown error')
                    suggestion = youtube_result.get('suggestion', '')
                    
                    full_error = f"YouTube transcript extraction failed: {error_msg}"
                    if suggestion:
                        full_error += f"\n\nSuggestion: {suggestion}"
                    
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=full_error
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"YouTube processing error: {str(e)}"
                )
        
        # Check if URL points to a file that should be processed with MarkItDown
        if file_processor.is_supported_file(request.url):
            # Redirect to file processing for supported file formats
            try:
                file_result = await file_processor.process_file_from_url(
                    request.url,
                    max_size_mb=100  # Default size limit
                )
                
                if file_result['success']:
                    response = CrawlResponse(
                        success=True,
                        url=request.url,
                        title=file_result.get('title'),
                        content=file_result.get('content'),
                        markdown=file_result.get('content'),  # MarkItDown already provides markdown
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
        
        # Setup deep crawling strategy if max_depth is specified
        deep_crawl_strategy = None
        if request.max_depth is not None and request.max_depth > 0:
            # Create filter chain
            filters = []
            if request.url_pattern:
                filters.append(URLPatternFilter(patterns=[request.url_pattern]))
            if not request.include_external:
                # Extract domain from URL for domain filtering
                from urllib.parse import urlparse
                domain = urlparse(request.url).netloc
                filters.append(DomainFilter(allowed_domains=[domain]))
            
            filter_chain = FilterChain(filters) if filters else None
            
            # Select crawling strategy
            if request.crawl_strategy == "dfs":
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            elif request.crawl_strategy == "best_first":
                deep_crawl_strategy = BestFirstCrawlingStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            else:  # Default to BFS
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )

        # Setup advanced content filtering
        content_filter_strategy = None
        if request.content_filter == "bm25" and request.filter_query:
            content_filter_strategy = BM25ContentFilter(user_query=request.filter_query)
        elif request.content_filter == "pruning":
            content_filter_strategy = PruningContentFilter(threshold=0.5)
        elif request.content_filter == "llm" and request.filter_query:
            content_filter_strategy = LLMContentFilter(
                instructions=f"Filter content related to: {request.filter_query}"
            )

        # Setup cache mode
        cache_mode = CacheMode.ENABLED
        if not request.enable_caching or request.cache_mode == "disabled":
            cache_mode = CacheMode.DISABLED
        elif request.cache_mode == "bypass":
            cache_mode = CacheMode.BYPASS

        # Configure chunking if requested
        chunking_strategy = None
        if request.chunk_content:
            step_size = int(request.chunk_size * (1 - request.overlap_rate))
            chunking_strategy = SlidingWindowChunking(
                window_size=request.chunk_size,
                step=step_size
            )

        # Create config parameters
        config_params = {
            "css_selector": request.css_selector,
            "screenshot": request.take_screenshot,
            "wait_for": request.wait_for_selector,
            "page_timeout": request.timeout * 1000,
            "exclude_all_images": not request.extract_media,
            "verbose": False,
            "log_console": False,
            "deep_crawl_strategy": deep_crawl_strategy,
            "cache_mode": cache_mode,
            # Phase 2: Add simulate_user support
            "simulate_user": request.simulate_user,
        }

        # Phase 2: Handle wait_for_js - use wait_until and delay for JS-heavy sites
        # These parameters may not be supported in older crawl4ai versions
        js_wait_params = {}
        if request.wait_for_js:
            js_wait_params["wait_until"] = "networkidle"  # Wait for network to be idle
            js_wait_params["delay_before_return_html"] = 2.0  # Additional delay for JS rendering
            if not request.wait_for_selector:
                js_wait_params["scan_full_page"] = True  # Scan full page when no specific selector
        
        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy

        # Build CrawlerRunConfig with backward compatibility
        # Use inspect.signature to filter only supported parameters
        import inspect
        supported_params = None
        accepts_kwargs = False
        try:
            sig = inspect.signature(CrawlerRunConfig)
            supported_params = set(sig.parameters.keys())
            # Check if CrawlerRunConfig accepts **kwargs (VAR_KEYWORD)
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    accepts_kwargs = True
                    break
        except (ValueError, TypeError):
            # Fallback if signature inspection fails
            pass

        # Prepare all desired parameters
        all_params = {**config_params, **js_wait_params}
        if content_filter_strategy:
            all_params["content_filter"] = content_filter_strategy

        # Track which params were filtered out for diagnostics
        unsupported_params = []
        config_fallback_used = False

        # Only filter if signature inspection worked and no **kwargs
        if supported_params is not None and not accepts_kwargs:
            filtered_params = {}
            for key, value in all_params.items():
                if key in supported_params:
                    filtered_params[key] = value
                else:
                    unsupported_params.append(key)
            all_params = filtered_params

        # Create config with filtered parameters
        try:
            config = CrawlerRunConfig(**all_params)
        except TypeError as e:
            # If signature filtering didn't work or missed something, try minimal config
            config_fallback_used = True
            try:
                minimal_params = {
                    "css_selector": request.css_selector,
                    "screenshot": request.take_screenshot,
                    "wait_for": request.wait_for_selector,
                    "page_timeout": request.timeout * 1000,
                    "verbose": False
                }
                config = CrawlerRunConfig(**minimal_params)
                unsupported_params = [k for k in all_params.keys() if k not in minimal_params]
            except TypeError:
                # Absolute minimal fallback
                config = CrawlerRunConfig(page_timeout=request.timeout * 1000)
                unsupported_params = list(all_params.keys())

        # Setup browser configuration with lightweight WebKit preference
        browser_config = {
            "headless": True,
            "verbose": False,
            "browser_type": "webkit"  # Use lightweight WebKit by default
        }
        
        # Enable undetected browser mode if requested
        if request.use_undetected_browser:
            browser_config["enable_stealth"] = True
            # Use Chromium when stealth mode is needed for better compatibility
            browser_config["browser_type"] = "chromium"
        
        if request.user_agent:
            browser_config["user_agent"] = request.user_agent

        # Build headers dict with auth_token if provided
        # NOTE: browser_config["headers"] applies to ALL requests from the page,
        # including third-party subresources. For sensitive tokens, consider using
        # route-based header injection in Playwright to limit scope to same-origin.
        headers = dict(request.headers) if request.headers else {}
        if request.auth_token:
            # Only add if not already set (preserve user-specified Authorization, case-insensitive)
            has_auth = any(k.lower() == "authorization" for k in headers)
            if not has_auth:
                headers["Authorization"] = f"Bearer {request.auth_token}"
        if headers:
            browser_config["headers"] = headers

        # Convert and add cookies in Playwright format
        if request.cookies:
            browser_config["cookies"] = _normalize_cookies_to_playwright_format(
                request.cookies, request.url
            )

        # Suppress output to avoid JSON parsing errors
        with suppress_stdout_stderr():
            # Try WebKit first, fallback to Chromium if needed
            result = None
            browsers_to_try = ["webkit", "chromium"]
            
            for browser_type in browsers_to_try:
                try:
                    current_browser_config = browser_config.copy()
                    current_browser_config["browser_type"] = browser_type
                    
                    async with AsyncWebCrawler(**current_browser_config) as crawler:
                        # Execute custom JavaScript if provided
                        if request.execute_js:
                            config.js_code = request.execute_js
                        
                        # Run crawler with config and proper timeout
                        arun_params = {"url": request.url, "config": config}

                        # Apply timeout to crawler.arun() to prevent hanging
                        result = await asyncio.wait_for(
                            crawler.arun(**arun_params),
                            timeout=request.timeout
                        )
                        break  # Success, no need to try other browsers

                except asyncio.TimeoutError:
                    # Handle timeout specifically
                    error_msg = f"Crawl timeout after {request.timeout}s for {request.url}"
                    if browser_type == browsers_to_try[-1]:
                        raise TimeoutError(error_msg)
                    continue  # Try next browser
                except Exception as browser_error:
                    # If this is the last browser to try, raise the error
                    if browser_type == browsers_to_try[-1]:
                        raise browser_error
                    # Otherwise, try the next browser
                    continue
        
        # Handle different result types (single result vs list from deep crawling)
        if isinstance(result, list):
            # Deep crawling returns a list of results
            if not result:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="No results returned from deep crawling"
                )
            
            # Process multiple results from deep crawling
            all_content = []
            all_markdown = []
            all_media = []
            crawled_urls = []
            
            for page_result in result:
                if hasattr(page_result, 'success') and page_result.success:
                    crawled_urls.append(page_result.url if hasattr(page_result, 'url') else 'unknown')
                    if hasattr(page_result, 'cleaned_html') and page_result.cleaned_html:
                        all_content.append(f"=== {page_result.url} ===\n{page_result.cleaned_html}")
                    if hasattr(page_result, 'markdown') and page_result.markdown:
                        all_markdown.append(f"=== {page_result.url} ===\n{page_result.markdown}")
                    if hasattr(page_result, 'media') and page_result.media and request.extract_media:
                        all_media.extend(_convert_media_to_list(page_result.media))
            
            response = CrawlResponse(
                success=True,
                url=request.url,
                title=f"Deep crawl of {len(crawled_urls)} pages",
                content="\n\n".join(all_content) if all_content else "No content extracted",
                markdown="\n\n".join(all_markdown) if all_markdown else "No markdown content",
                media=all_media if request.extract_media else None,
                extracted_data={
                    "crawled_pages": len(crawled_urls),
                    "crawled_urls": crawled_urls,
                    "processing_method": "deep_crawling"
                }
            )
            response = await _check_and_summarize_if_needed(response, request)
            return _process_response_content(response, request.include_cleaned_html)
        
        elif hasattr(result, 'success') and result.success:
            # For deep crawling, result might contain multiple pages
            if deep_crawl_strategy and hasattr(result, 'crawled_pages'):
                # Combine content from all crawled pages
                all_content = []
                all_markdown = []
                all_media = []
                
                for page in result.crawled_pages:
                    if page.cleaned_html:
                        all_content.append(f"=== {page.url} ===\n{page.cleaned_html}")
                    if page.markdown:
                        all_markdown.append(f"=== {page.url} ===\n{page.markdown}")
                    if page.media and request.extract_media:
                        all_media.extend(_convert_media_to_list(page.media))
                
                # Prepare content for potential summarization
                combined_content = "\n\n".join(all_content) if all_content else result.cleaned_html
                combined_markdown = "\n\n".join(all_markdown) if all_markdown else result.markdown
                title_to_use = result.metadata.get("title", "")
                extracted_data = {"crawled_pages": len(result.crawled_pages)} if hasattr(result, 'crawled_pages') else {}
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and combined_content:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(combined_content) // 4
                    
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = combined_markdown or combined_content
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                combined_content = summary_result["summary"]
                                combined_markdown = summary_result["summary"]
                                
                                extracted_data.update({
                                    "summarization_applied": True,
                                    "original_content_length": len("\n\n".join(all_content) if all_content else result.cleaned_html),
                                    "original_tokens_estimate": estimated_tokens,
                                    "summary_length": request.summary_length,
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    "auto_summarization_trigger": f"Deep crawl content exceeded {request.max_content_tokens} tokens"
                                })
                            else:
                                # Summarization failed, add error info but keep original content
                                extracted_data.update({
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                })
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data.update({
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            })
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=combined_content,
                    markdown=combined_markdown,
                    media=all_media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            else:
                # Check if auto-summarization should be applied
                content_to_use = result.cleaned_html
                markdown_to_use = result.markdown
                extracted_data = None
                title_to_use = result.metadata.get("title", "")
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and content_to_use:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(content_to_use) // 4
                    
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = markdown_to_use or content_to_use
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                content_to_use = summary_result["summary"]
                                markdown_to_use = summary_result["summary"]  # Use same summary for both
                                
                                extracted_data = {
                                    "summarization_applied": True,
                                    "original_content_length": len(result.cleaned_html),
                                    "original_tokens_estimate": estimated_tokens,
                                    "summary_length": request.summary_length,
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    "auto_summarization_trigger": f"Content exceeded {request.max_content_tokens} tokens"
                                }
                            else:
                                # Summarization failed, add error info but keep original content
                                extracted_data = {
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                }
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data = {
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            }
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=content_to_use,
                    markdown=markdown_to_use,
                    media=_convert_media_to_list(result.media) if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            response = await _check_and_summarize_if_needed(response, request)
            return _process_response_content(response, request.include_cleaned_html)
        else:
            # Handle case where result doesn't have success attribute or failed
            error_msg = "Failed to crawl URL"
            if hasattr(result, 'error_message'):
                error_msg = f"Failed to crawl URL: {result.error_message}"
            elif hasattr(result, 'error'):
                error_msg = f"Failed to crawl URL: {result.error}"
            else:
                error_msg = f"Failed to crawl URL: Unknown error (result type: {type(result)})"
            
            return CrawlResponse(
                success=False,
                url=request.url,
                error=error_msg
            )
                
    except Exception as e:
        import sys  # Import here to ensure availability in error response
        error_message = f"Crawling error: {str(e)}"

        # Enhanced error handling for browser and UVX issues
        if "playwright" in str(e).lower() or "browser" in str(e).lower() or "executable doesn't exist" in str(e).lower():
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            
            if is_uvx_env:
                error_message += "\n\n🔧 UVX Environment Browser Setup Required:\n" \
                    f"1. Run system diagnostics: get_system_diagnostics()\n" \
                    f"2. Manual browser installation:\n" \
                    f"   - uvx --with playwright playwright install webkit\n" \
                    f"   - Or system-wide: playwright install webkit\n" \
                    f"3. Restart Claude Desktop after installation\n" \
                    f"4. If issues persist, consider STDIO local setup\n\n" \
                    f"💡 WebKit is lightweight (~180MB) vs Chromium (~281MB)"
            else:
                error_message += "\n\n🔧 Browser Setup Required:\n" \
                    f"1. Install Playwright browsers:\n" \
                    f"   playwright install webkit  # Lightweight option\n" \
                    f"   playwright install chromium  # Full compatibility\n" \
                    f"2. For system dependencies: sudo apt-get install libnss3 libnspr4 libasound2\n" \
                    f"3. Run diagnostics: get_system_diagnostics()"
            
        return CrawlResponse(
            success=False,
            url=request.url,
            error=error_message,
            extracted_data={
                'error_type': 'browser_setup_required',
                'uvx_environment': 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable),
                'diagnostic_tool': 'get_system_diagnostics',
                'installation_commands': [
                    'playwright install webkit',
                    'playwright install chromium'
                ]
            }
        )

async def _check_and_summarize_if_needed(
    response: CrawlResponse,
    request: CrawlRequest
) -> CrawlResponse:
    """
    Check if response content exceeds token limits and apply summarization if needed.
    Respects user-specified limits when provided.
    """
    # Skip if response failed or already summarized
    if not response.success or not response.content:
        return response
    
    # Check if already summarized
    if response.extracted_data and response.extracted_data.get("summarization_applied"):
        return response
    
    # Estimate total response size (content + markdown + metadata)
    total_chars = len(response.content or "") + len(response.markdown or "")
    
    # Determine effective token limit - user-specified takes precedence
    effective_limit_chars = MAX_RESPONSE_CHARS
    limit_source = "MCP_PROTOCOL_SAFETY"
    
    # If user explicitly set auto_summarize=True and max_content_tokens, use that instead
    if hasattr(request, 'auto_summarize') and request.auto_summarize:
        if hasattr(request, 'max_content_tokens') and request.max_content_tokens:
            # Convert user's token limit to character estimate
            user_limit_chars = request.max_content_tokens * 4
            # Use the smaller limit (user preference or protocol safety)
            if user_limit_chars < effective_limit_chars:
                effective_limit_chars = user_limit_chars
                limit_source = "USER_SPECIFIED"
    
    # Check if response exceeds the effective limit
    if total_chars > effective_limit_chars:
        try:
            # Use markdown if available, otherwise use content
            content_to_summarize = response.markdown or response.content
            
            # Determine summary length based on reduction needed
            reduction_ratio = effective_limit_chars / total_chars
            if reduction_ratio > 0.5:
                summary_length = "medium"
            elif reduction_ratio > 0.3:
                summary_length = "short"
            else:
                summary_length = "short"  # Aggressive reduction needed
            
            # Force summarization to meet token limits
            summary_result = await summarize_web_content(
                content=content_to_summarize,
                title=response.title or "",
                url=response.url,
                summary_length=summary_length,
                llm_provider=request.llm_provider if hasattr(request, 'llm_provider') else None,
                llm_model=request.llm_model if hasattr(request, 'llm_model') else None
            )
            
            if summary_result.get("success"):
                # Update response with summarized content
                if limit_source == "USER_SPECIFIED":
                    prefix = f"⚠️ Content exceeded user-specified limit ({total_chars:,} chars > {effective_limit_chars:,} chars). Auto-summarized:\n\n"
                else:
                    prefix = f"⚠️ Content exceeded MCP token limit ({total_chars:,} chars > {effective_limit_chars:,} chars). Auto-summarized:\n\n"
                
                response.content = f"{prefix}{summary_result['summary']}"
                response.markdown = response.content
                
                # Update extracted_data
                if response.extracted_data is None:
                    response.extracted_data = {}
                
                response.extracted_data.update({
                    "auto_summarization_reason": limit_source,
                    "original_size_chars": total_chars,
                    "effective_limit_chars": effective_limit_chars,
                    "user_specified_limit": limit_source == "USER_SPECIFIED",
                    "user_max_content_tokens": request.max_content_tokens if hasattr(request, 'max_content_tokens') else None,
                    "summarization_applied": True,
                    "summary_length": summary_length,
                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                    "llm_model": summary_result.get("llm_model", "unknown"),
                })
            else:
                # Summarization failed, truncate content
                truncate_at = effective_limit_chars - 500  # Leave room for message
                prefix = f"⚠️ Content exceeded limit ({total_chars:,} chars > {effective_limit_chars:,} chars).\n\nSummarization failed: {summary_result.get('error', 'Unknown error')}\n\nTruncated content:\n\n"
                response.content = f"{prefix}{response.content[:truncate_at]}... [Content truncated]"
                response.markdown = response.content
                
                if response.extracted_data is None:
                    response.extracted_data = {}
                
                response.extracted_data.update({
                    "auto_truncation_reason": limit_source,
                    "original_size_chars": total_chars,
                    "effective_limit_chars": effective_limit_chars,
                    "user_specified_limit": limit_source == "USER_SPECIFIED",
                    "truncation_applied": True,
                    "summarization_attempted": True,
                    "summarization_error": summary_result.get("error", "Unknown error")
                })
                
        except Exception as e:
            # Final fallback: aggressive truncation
            truncate_at = effective_limit_chars - 500
            prefix = f"⚠️ Content exceeded limit ({total_chars:,} chars > {effective_limit_chars:,} chars).\n\nEmergency truncation applied due to error: {str(e)}\n\n"
            response.content = f"{prefix}{response.content[:truncate_at]}... [Content truncated]"
            response.markdown = response.content
            
            if response.extracted_data is None:
                response.extracted_data = {}
                
            response.extracted_data.update({
                "emergency_truncation_reason": limit_source,
                "original_size_chars": total_chars,
                "effective_limit_chars": effective_limit_chars,
                "user_specified_limit": limit_source == "USER_SPECIFIED",
                "truncation_error": str(e)
            })
    
    return response


async def _finalize_fallback_response(
    response: CrawlResponse,
    request_url: str,
    auto_summarize: bool = False,
    max_content_tokens: int = 15000,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> CrawlResponse:
    """
    Apply size limit and summarization to fallback responses.

    This ensures all fallback return points go through the same
    size checking and summarization as regular crawl responses.
    """
    # Create a dummy CrawlRequest with the necessary parameters
    dummy_request = CrawlRequest(
        url=request_url,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    return await _check_and_summarize_if_needed(response, dummy_request)


# Other internal functions for specialized extraction
async def _internal_intelligent_extract(
    url: str,
    extraction_goal: str,
    content_filter: str = "bm25",
    filter_query: Optional[str] = None,
    chunk_content: bool = False,
    use_llm: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal intelligent extract implementation.
    Uses LLM and content filtering for targeted extraction.
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(
            url=url,
            content_filter=content_filter,
            filter_query=filter_query,
            chunk_content=chunk_content,
            generate_markdown=True
        )
        
        crawl_result = await _internal_crawl_url(request)
        
        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }
        
        # If LLM processing is disabled, return the crawled content
        if not use_llm:
            return {
                "url": url,
                "success": True,
                "extracted_content": crawl_result.content,
                "extraction_goal": extraction_goal,
                "processing_method": "basic_crawl_only"
            }
        
        # Implement LLM-based intelligent extraction
        try:
            # Import config
            try:
                from ..config import get_llm_config
            except ImportError:
                from config import get_llm_config
            
            # Get LLM configuration
            llm_config = get_llm_config(llm_provider, llm_model)
            
            # Prepare extraction prompt
            extraction_prompt = f"""
            You are an expert content analyst. Your task is to extract specific information from web content based on the extraction goal.
            
            EXTRACTION GOAL: {extraction_goal}
            
            INSTRUCTIONS:
            - Focus specifically on information relevant to the extraction goal
            - Extract concrete data, statistics, quotes, and specific details
            - Maintain accuracy and preserve exact information from the source
            - Organize findings in a structured, easy-to-understand format
            - If the content doesn't contain relevant information, clearly state that
            
            {f"ADDITIONAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}
            
            Please provide a JSON response with the following structure:
            {{
                "extracted_data": "The specific information extracted according to the goal",
                "key_findings": ["List", "of", "main", "findings"],
                "relevant_quotes": ["Important", "quotes", "from", "source"],
                "statistics_data": ["Numerical", "data", "and", "statistics"],
                "sources_references": ["References", "to", "specific", "sections"],
                "extraction_confidence": "High/Medium/Low - confidence in extraction quality",
                "missing_information": ["Information", "sought", "but", "not", "found"]
            }}
            
            CONTENT TO ANALYZE:
            {crawl_result.content[:50000]}  # Limit content to prevent token overflow
            """
            
            # Make LLM API call
            provider_info = llm_config.provider.split('/')
            provider = provider_info[0] if provider_info else 'openai'
            model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
            
            extracted_content = None
            
            if provider == 'openai':
                import openai
                
                api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found")
                
                client = openai.AsyncOpenAI(api_key=api_key, base_url=llm_config.base_url)
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert content analyst specializing in precise information extraction."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=4000
                )
                
                extracted_content = response.choices[0].message.content
                
            elif provider == 'anthropic':
                import anthropic
                
                api_key = llm_config.api_token or os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("Anthropic API key not found")
                
                client = anthropic.AsyncAnthropic(api_key=api_key)
                
                response = await client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": extraction_prompt}
                    ]
                )
                
                extracted_content = response.content[0].text
                
            else:
                # Fallback for unsupported providers
                return {
                    "url": url,
                    "success": False,
                    "error": f"LLM provider '{provider}' not supported for intelligent extraction"
                }
            
            # Parse JSON response
            if extracted_content:
                try:
                    import json
                    # Clean up the extracted content if it's wrapped in markdown
                    content_to_parse = extracted_content
                    if content_to_parse.startswith('```json'):
                        content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                    
                    extraction_data = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                    
                    return {
                        "url": url,
                        "success": True,
                        "extraction_goal": extraction_goal,
                        "extracted_data": extraction_data.get("extracted_data", ""),
                        "key_findings": extraction_data.get("key_findings", []),
                        "relevant_quotes": extraction_data.get("relevant_quotes", []),
                        "statistics_data": extraction_data.get("statistics_data", []),
                        "sources_references": extraction_data.get("sources_references", []),
                        "extraction_confidence": extraction_data.get("extraction_confidence", "Medium"),
                        "missing_information": extraction_data.get("missing_information", []),
                        "processing_method": "llm_intelligent_extraction",
                        "llm_provider": provider,
                        "llm_model": model,
                        "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                        "custom_instructions_used": bool(custom_instructions)
                    }
                    
                except (json.JSONDecodeError, AttributeError) as e:
                    # Fallback: treat as plain text extraction
                    return {
                        "url": url,
                        "success": True,
                        "extraction_goal": extraction_goal,
                        "extracted_data": str(extracted_content),
                        "key_findings": [],
                        "relevant_quotes": [],
                        "statistics_data": [],
                        "sources_references": [],
                        "extraction_confidence": "Medium",
                        "missing_information": [],
                        "processing_method": "llm_intelligent_extraction_fallback",
                        "llm_provider": provider,
                        "llm_model": model,
                        "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                        "custom_instructions_used": bool(custom_instructions),
                        "json_parse_error": str(e)
                    }
            else:
                return {
                    "url": url,
                    "success": False,
                    "error": "LLM extraction returned empty result"
                }
                
        except Exception as llm_error:
            # LLM processing failed, return crawled content with error info
            return {
                "url": url,
                "success": True,  # Still return success since we have crawled content
                "extraction_goal": extraction_goal,
                "extracted_data": crawl_result.content,
                "key_findings": [],
                "relevant_quotes": [],
                "statistics_data": [],
                "sources_references": [],
                "extraction_confidence": "Low",
                "missing_information": [],
                "processing_method": "crawl_fallback_due_to_llm_error",
                "llm_error": str(llm_error),
                "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                "custom_instructions_used": bool(custom_instructions)
            }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Intelligent extraction error: {str(e)}"
        }


async def _internal_extract_entities(
    url: str,
    entity_types: List[str],
    custom_patterns: Optional[Dict[str, str]] = None,
    include_context: bool = True,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Internal extract entities implementation using regex patterns.
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(url=url, generate_markdown=True)
        crawl_result = await _internal_crawl_url(request)
        
        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }
        
        content = crawl_result.content or ""
        entities = {}
        
        # Define regex patterns for common entity types
        patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phones": r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            "urls": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            "dates": r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b',
            "ips": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "prices": r'[$£€¥]?\s?\d+(?:[.,]\d{2,3})*(?:[.,]\d{2})?',
            "credit_cards": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "coordinates": r'[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?)'
        }
        
        # Add custom patterns if provided
        if custom_patterns:
            patterns.update(custom_patterns)
        
        # Extract entities for each requested type
        import re
        for entity_type in entity_types:
            if entity_type in patterns:
                matches = re.findall(patterns[entity_type], content, re.IGNORECASE)
                if matches:
                    # Deduplicate if requested
                    if deduplicate:
                        matches = list(set(matches))
                    entities[entity_type] = matches
        
        return {
            "url": url,
            "success": True,
            "entities": entities,
            "entity_types_requested": entity_types,
            "processing_method": "regex_extraction",
            "content_length": len(content),
            "total_entities_found": sum(len(v) for v in entities.values())
        }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Entity extraction error: {str(e)}"
        }


async def _internal_llm_extract_entities(
    url: str,
    entity_types: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    include_context: bool = True,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Internal LLM extract entities implementation using AI-powered named entity recognition.
    
    Supports both standard entity types (emails, phones, etc.) and advanced NER
    (people, organizations, locations, custom entities).
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(url=url, generate_markdown=True)
        crawl_result = await _internal_crawl_url(request)
        
        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }
        
        content = crawl_result.content or ""
        if not content.strip():
            return {
                "url": url,
                "success": True,
                "entities": {},
                "entity_types_requested": entity_types,
                "processing_method": "llm_extraction",
                "content_length": 0,
                "total_entities_found": 0,
                "note": "No content found to extract entities from"
            }
        
        # Get LLM configuration
        try:
            from ..config import get_llm_config
        except ImportError:
            from config import get_llm_config
        
        llm_config = get_llm_config(provider, model)
        
        # Define entity types and their descriptions
        entity_descriptions = {
            "emails": "Email addresses (e.g., user@example.com)",
            "phones": "Phone numbers in various formats",
            "urls": "Web URLs and links", 
            "dates": "Dates in various formats",
            "ips": "IP addresses",
            "prices": "Prices and monetary amounts",
            "credit_cards": "Credit card numbers",
            "coordinates": "Geographic coordinates (latitude, longitude)",
            "social_media": "Social media handles and profiles",
            "people": "Names of people, individuals, persons",
            "organizations": "Company names, institutions, organizations",
            "locations": "Places, cities, countries, geographic locations",
            "products": "Product names, brands, models",
            "events": "Events, conferences, meetings, occasions"
        }
        
        # Build entity types description for prompt
        requested_entities = []
        for entity_type in entity_types:
            description = entity_descriptions.get(entity_type, f"Custom entity type: {entity_type}")
            requested_entities.append(f"- {entity_type}: {description}")
        
        entity_types_text = "\n".join(requested_entities)
        
        # Prepare extraction prompt
        extraction_prompt = f"""
You are an expert entity extraction specialist. Extract all instances of the specified entity types from the given web content.

ENTITY TYPES TO EXTRACT:
{entity_types_text}

EXTRACTION INSTRUCTIONS:
- Extract ALL instances of each specified entity type from the content
- Maintain exact accuracy - extract entities exactly as they appear in the source
- For each entity type, provide a list of unique entities found
- If context is requested, include a brief surrounding text snippet for each entity
- Remove duplicates within each entity type
- If no entities of a specific type are found, return an empty list for that type
- Return results in valid JSON format

{f"ADDITIONAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}

Please provide a JSON response with the following structure:
{{
    "entities": {{
        "entity_type_1": [
            {{
                "value": "extracted_entity_text",
                "context": "surrounding text context (if requested)",
                "confidence": "High/Medium/Low"
            }}
        ],
        "entity_type_2": [...]
    }},
    "extraction_summary": {{
        "total_entities_found": number,
        "entity_types_found": ["list", "of", "types", "with", "results"],
        "entity_types_empty": ["list", "of", "types", "with", "no", "results"],
        "extraction_confidence": "High/Medium/Low"
    }}
}}

WEB CONTENT TO ANALYZE:
{content[:40000]}  # Limit content to prevent token overflow
"""
        
        # Make LLM API call
        provider_info = llm_config.provider.split('/')
        provider_name = provider_info[0] if provider_info else 'openai'
        model_name = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
        
        extracted_content = None
        
        if provider_name == 'openai':
            import openai
            
            api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = openai.AsyncOpenAI(api_key=api_key, base_url=llm_config.base_url)
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert entity extraction specialist focused on accuracy and comprehensive extraction."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4000
            )
            
            extracted_content = response.choices[0].message.content
            
        elif provider_name == 'anthropic':
            import anthropic
            
            api_key = llm_config.api_token or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found")
            
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
            response = await client.messages.create(
                model=model_name,
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": extraction_prompt}
                ]
            )
            
            extracted_content = response.content[0].text
            
        elif provider_name == 'ollama':
            import aiohttp
            
            base_url = llm_config.base_url or 'http://localhost:11434'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": extraction_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        extracted_content = result.get('response', '')
                    else:
                        raise ValueError(f"Ollama API request failed: {response.status}")
                        
        else:
            return {
                "url": url,
                "success": False,
                "error": f"LLM provider '{provider_name}' not supported for entity extraction"
            }
        
        # Parse JSON response
        if extracted_content:
            try:
                import json
                # Clean up the extracted content if it's wrapped in markdown
                content_to_parse = extracted_content
                if content_to_parse.startswith('```json'):
                    content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                
                extraction_result = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                
                # Process entities to match expected format
                processed_entities = {}
                for entity_type, entities_list in extraction_result.get("entities", {}).items():
                    if entity_type in entity_types:
                        if include_context and isinstance(entities_list, list) and entities_list:
                            # Keep full entity objects with context if requested
                            processed_entities[entity_type] = entities_list
                        else:
                            # Extract just the values if no context requested
                            if isinstance(entities_list, list):
                                values = []
                                for entity in entities_list:
                                    if isinstance(entity, dict):
                                        values.append(entity.get('value', str(entity)))
                                    else:
                                        values.append(str(entity))
                                processed_entities[entity_type] = list(set(values)) if deduplicate else values
                            else:
                                processed_entities[entity_type] = entities_list
                
                summary = extraction_result.get("extraction_summary", {})
                
                return {
                    "url": url,
                    "success": True,
                    "entities": processed_entities,
                    "entity_types_requested": entity_types,
                    "processing_method": "llm_extraction",
                    "llm_provider": provider_name,
                    "llm_model": model_name,
                    "content_length": len(content),
                    "total_entities_found": summary.get("total_entities_found", sum(len(v) for v in processed_entities.values())),
                    "extraction_confidence": summary.get("extraction_confidence", "Medium"),
                    "entity_types_found": summary.get("entity_types_found", list(processed_entities.keys())),
                    "entity_types_empty": summary.get("entity_types_empty", [et for et in entity_types if et not in processed_entities]),
                    "include_context": include_context,
                    "deduplicated": deduplicate
                }
                
            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback: treat as plain text
                return {
                    "url": url,
                    "success": True,
                    "entities": {"raw_extraction": [str(extracted_content)]},
                    "entity_types_requested": entity_types,
                    "processing_method": "llm_extraction_fallback",
                    "llm_provider": provider_name,
                    "llm_model": model_name,
                    "content_length": len(content),
                    "total_entities_found": 1,
                    "extraction_confidence": "Low",
                    "json_parse_error": str(e),
                    "note": f"JSON parsing failed, returned raw LLM output: {str(e)}"
                }
        else:
            return {
                "url": url,
                "success": False,
                "error": "LLM entity extraction returned empty result"
            }
            
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"LLM entity extraction error: {str(e)}"
        }


async def _internal_extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Internal extract structured data implementation.
    Supports both CSS selector and LLM-based extraction methods.
    """
    try:
        # First crawl the URL to get the content
        crawl_request = CrawlRequest(
            url=request.url,
            generate_markdown=True
        )
        
        crawl_result = await _internal_crawl_url(crawl_request)
        
        if not crawl_result.success:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Failed to crawl URL for structured extraction: {crawl_result.error}"
            )
        
        extracted_data = {}
        
        if request.extraction_type == "css" and request.css_selectors:
            # CSS selector-based extraction
            try:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(crawl_result.content, 'html.parser')
                
                for field_name, css_selector in request.css_selectors.items():
                    elements = soup.select(css_selector)
                    if elements:
                        if len(elements) == 1:
                            # Single element
                            extracted_data[field_name] = elements[0].get_text(strip=True)
                        else:
                            # Multiple elements
                            extracted_data[field_name] = [elem.get_text(strip=True) for elem in elements]
                    else:
                        extracted_data[field_name] = None
                
                return CrawlResponse(
                    success=True,
                    url=request.url,
                    title=crawl_result.title,
                    content=crawl_result.content,
                    markdown=crawl_result.markdown,
                    extracted_data={
                        "structured_data": extracted_data,
                        "extraction_method": "css_selectors",
                        "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                        "extracted_fields": list(extracted_data.keys())
                    }
                )
                
            except ImportError:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="BeautifulSoup4 not installed. Install with: pip install beautifulsoup4"
                )
                
        elif request.extraction_type == "llm":
            # LLM-based extraction
            try:
                # Import config
                try:
                    from ..config import get_llm_config
                except ImportError:
                    from config import get_llm_config
                
                # Get LLM configuration
                llm_config = get_llm_config(request.llm_provider, request.llm_model)
                
                # Prepare schema description
                schema_description = ""
                if request.extraction_schema:
                    schema_items = []
                    for field, description in request.extraction_schema.items():
                        schema_items.append(f"- {field}: {description}")
                    schema_description = "\n".join(schema_items)
                
                # Prepare extraction prompt
                structured_prompt = f"""
                You are an expert data extraction specialist. Extract structured data from the given web content according to the specified schema.

                SCHEMA FIELDS TO EXTRACT:
                {schema_description}

                EXTRACTION INSTRUCTIONS:
                - Extract information for each field in the schema
                - Maintain accuracy and preserve exact information from the source
                - If a field's information is not found, set it to null
                - Return data in valid JSON format matching the schema structure
                - Focus on extracting concrete, factual information
                
                {f"ADDITIONAL INSTRUCTIONS: {request.instruction}" if request.instruction else ""}

                Please provide a JSON response with the following structure:
                {{
                    "structured_data": {{
                        // Fields matching the requested schema
                    }},
                    "extraction_confidence": "High/Medium/Low",
                    "found_fields": ["list", "of", "successfully", "extracted", "fields"],
                    "missing_fields": ["list", "of", "fields", "not", "found"],
                    "additional_context": "Any relevant context or notes about the extraction"
                }}

                WEB CONTENT TO ANALYZE:
                {crawl_result.content[:40000]}  # Limit content to prevent token overflow
                """
                
                # Make LLM API call
                provider_info = llm_config.provider.split('/')
                provider = provider_info[0] if provider_info else 'openai'
                model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
                
                extracted_content = None
                
                if provider == 'openai':
                    import openai
                    
                    api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                    if not api_key:
                        raise ValueError("OpenAI API key not found")
                    
                    client = openai.AsyncOpenAI(api_key=api_key, base_url=llm_config.base_url)
                    
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert data extraction specialist focused on accuracy and structured output."},
                            {"role": "user", "content": structured_prompt}
                        ],
                        temperature=0.1,  # Low temperature for consistent extraction
                        max_tokens=4000
                    )
                    
                    extracted_content = response.choices[0].message.content
                    
                elif provider == 'anthropic':
                    import anthropic
                    
                    api_key = llm_config.api_token or os.environ.get('ANTHROPIC_API_KEY')
                    if not api_key:
                        raise ValueError("Anthropic API key not found")
                    
                    client = anthropic.AsyncAnthropic(api_key=api_key)
                    
                    response = await client.messages.create(
                        model=model,
                        max_tokens=4000,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": structured_prompt}
                        ]
                    )
                    
                    extracted_content = response.content[0].text
                    
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=f"LLM provider '{provider}' not supported for structured extraction"
                    )
                
                # Parse JSON response
                if extracted_content:
                    try:
                        import json
                        # Clean up the extracted content if it's wrapped in markdown
                        content_to_parse = extracted_content
                        if content_to_parse.startswith('```json'):
                            content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                        
                        extraction_result = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                        
                        return CrawlResponse(
                            success=True,
                            url=request.url,
                            title=crawl_result.title,
                            content=crawl_result.content,
                            markdown=crawl_result.markdown,
                            extracted_data={
                                "structured_data": extraction_result.get("structured_data", {}),
                                "extraction_method": "llm_based",
                                "extraction_confidence": extraction_result.get("extraction_confidence", "Medium"),
                                "found_fields": extraction_result.get("found_fields", []),
                                "missing_fields": extraction_result.get("missing_fields", []),
                                "additional_context": extraction_result.get("additional_context", ""),
                                "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "llm_provider": provider,
                                "llm_model": model,
                                "custom_instruction_used": bool(request.instruction)
                            }
                        )
                        
                    except (json.JSONDecodeError, AttributeError) as e:
                        # Fallback: treat as plain text
                        return CrawlResponse(
                            success=True,
                            url=request.url,
                            title=crawl_result.title,
                            content=crawl_result.content,
                            markdown=crawl_result.markdown,
                            extracted_data={
                                "structured_data": {"raw_extraction": str(extracted_content)},
                                "extraction_method": "llm_based_fallback",
                                "extraction_confidence": "Low",
                                "found_fields": [],
                                "missing_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "additional_context": f"JSON parsing failed: {str(e)}",
                                "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "llm_provider": provider,
                                "llm_model": model,
                                "json_parse_error": str(e)
                            }
                        )
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error="LLM structured extraction returned empty result"
                    )
                    
            except Exception as llm_error:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"LLM structured extraction failed: {str(llm_error)}"
                )
        
        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Unsupported extraction type: {request.extraction_type}. Supported types: 'css', 'llm'"
            )
            
    except Exception as e:
        return CrawlResponse(
            success=False,
            url=request.url,
            error=f"Structured data extraction error: {str(e)}"
        )


# MCP Tool implementations
async def crawl_url(
    url: Annotated[str, Field(description="Target URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    include_cleaned_html: Annotated[bool, Field(description="Include cleaned HTML in content field (default: False)")] = False,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Maximum crawling depth, None for single page (default: None)")] = None,
    max_pages: Annotated[Optional[int], Field(description="Maximum number of pages to crawl (default: 10)")] = 10,
    include_external: Annotated[bool, Field(description="Whether to follow external domain links (default: False)")] = False,
    crawl_strategy: Annotated[str, Field(description="Crawling strategy: 'bfs', 'dfs', or 'best_first' (default: 'bfs')")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL pattern filter e.g. '*docs*' (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum score for URLs to be crawled (default: 0.3)")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="Content filter: 'bm25', 'pruning', 'llm' (default: None)")] = None,
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 content filtering (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Whether to chunk large content (default: False)")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunking strategy: 'topic', 'regex', 'sentence' (default: 'topic')")] = "topic",
    chunk_size: Annotated[int, Field(description="Maximum chunk size in tokens (default: 1000)")] = 1000,
    overlap_rate: Annotated[float, Field(description="Overlap rate between chunks 0.0-1.0 (default: 0.1)")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="Custom user agent string (default: None)")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="Custom HTTP headers (default: None)")] = None,
    enable_caching: Annotated[bool, Field(description="Whether to enable caching (default: True)")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode: 'enabled', 'disabled', 'bypass' (default: 'enabled')")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JavaScript code to execute (default: None)")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate human-like browsing behavior (default: False)")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Use undetected browser mode to bypass bot detection (default: False)")] = False,
    auth_token: Annotated[Optional[str], Field(description="Authentication token (default: None)")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Custom cookies (default: None)")] = None,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None
) -> CrawlResponse:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.

    Core web crawling tool with comprehensive configuration options.
    Essential for SPAs: set wait_for_js=true for JavaScript-heavy sites.

    By default, returns markdown content only. Set include_cleaned_html=True to also
    receive the cleaned HTML content field for scenarios requiring both formats.
    """
    # Create CrawlRequest object from individual parameters
    request = CrawlRequest(
        url=url,
        css_selector=css_selector,
        xpath=xpath,
        extract_media=extract_media,
        take_screenshot=take_screenshot,
        generate_markdown=generate_markdown,
        include_cleaned_html=include_cleaned_html,
        wait_for_selector=wait_for_selector,
        timeout=timeout,
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        crawl_strategy=crawl_strategy,
        url_pattern=url_pattern,
        score_threshold=score_threshold,
        content_filter=content_filter,
        filter_query=filter_query,
        chunk_content=chunk_content,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap_rate=overlap_rate,
        user_agent=user_agent,
        headers=headers,
        enable_caching=enable_caching,
        cache_mode=cache_mode,
        execute_js=execute_js,
        wait_for_js=wait_for_js,
        simulate_user=simulate_user,
        use_undetected_browser=use_undetected_browser,
        auth_token=auth_token,
        cookies=cookies,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        summary_length=summary_length,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    return await _internal_crawl_url(request)


async def deep_crawl_site(
    url: Annotated[str, Field(description="Starting URL")],
    max_depth: Annotated[int, Field(description="Link depth to follow")] = 2,
    max_pages: Annotated[int, Field(description="Max pages (max: 10)")] = 5,
    crawl_strategy: Annotated[str, Field(description="'bfs'|'dfs'|'best_first'")] = "bfs",
    include_external: Annotated[bool, Field(description="Follow external links")] = False,
    url_pattern: Annotated[Optional[str], Field(description="URL wildcard filter")] = None,
    score_threshold: Annotated[float, Field(description="Min relevance 0-1")] = 0.0,
    extract_media: Annotated[bool, Field(description="Extract media")] = False,
    base_timeout: Annotated[int, Field(description="Timeout per page (sec)")] = 60
) -> Dict[str, Any]:
    """Crawl multiple pages from a site with configurable depth."""
    # Use crawl_url with deep crawling enabled
    request = CrawlRequest(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        crawl_strategy=crawl_strategy,
        include_external=include_external,
        url_pattern=url_pattern,
        score_threshold=score_threshold,
        extract_media=extract_media,
        timeout=base_timeout,
        generate_markdown=True
    )
    
    result = await _internal_crawl_url(request)
    
    # Convert CrawlResponse to dict for consistency with legacy API
    return {
        "success": result.success,
        "url": result.url,
        "title": result.title,
        "content": result.content,
        "markdown": result.markdown,
        "media": result.media,
        "extracted_data": result.extracted_data,
        "error": result.error,
        "processing_method": "deep_crawling"
    }


async def intelligent_extract(
    url: Annotated[str, Field(description="URL to extract from")],
    extraction_goal: Annotated[str, Field(description="What data to extract")],
    content_filter: Annotated[str, Field(description="'bm25'|'pruning'|'llm'")] = "bm25",
    filter_query: Annotated[Optional[str], Field(description="BM25 keywords")] = None,
    chunk_content: Annotated[bool, Field(description="Split large content")] = False,
    use_llm: Annotated[bool, Field(description="Use LLM")] = True,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
    llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
    custom_instructions: Annotated[Optional[str], Field(description="Extra LLM guidance")] = None
) -> Dict[str, Any]:
    """Extract specific data from web pages using LLM."""
    return await _internal_intelligent_extract(
        url=url,
        extraction_goal=extraction_goal,
        content_filter=content_filter,
        filter_query=filter_query,
        chunk_content=chunk_content,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
        custom_instructions=custom_instructions
    )


async def extract_entities(
    url: Annotated[str, Field(description="URL to extract from")],
    entity_types: Annotated[List[str], Field(description="Types: emails|phones|urls|dates|ips|prices|social_media")],
    custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns")] = None,
    include_context: Annotated[bool, Field(description="Include surrounding text")] = True,
    deduplicate: Annotated[bool, Field(description="Remove duplicates")] = True,
    use_llm: Annotated[bool, Field(description="Use LLM for NER")] = False,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
    llm_model: Annotated[Optional[str], Field(description="LLM model")] = None
) -> Dict[str, Any]:
    """Extract entities (emails, phones, etc.) from web pages."""
    if use_llm:
        # Use LLM-based extraction for all entity types when requested
        return await _internal_llm_extract_entities(
            url=url,
            entity_types=entity_types,
            provider=llm_provider,
            model=llm_model,
            custom_instructions=None,  # Could be added as parameter in future
            include_context=include_context,
            deduplicate=deduplicate
        )
    else:
        # Use regex-based extraction
        return await _internal_extract_entities(
            url=url,
            entity_types=entity_types,
            custom_patterns=custom_patterns,
            include_context=include_context,
            deduplicate=deduplicate
        )


async def extract_structured_data(
    request: Annotated[StructuredExtractionRequest, Field(description="Extraction request with URL and schema")]
) -> CrawlResponse:
    """Extract structured data using CSS selectors or LLM."""
    return await _internal_extract_structured_data(request)


async def crawl_url_with_fallback(
    url: Annotated[str, Field(description="URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector")] = None,
    extract_media: Annotated[bool, Field(description="Extract media")] = False,
    take_screenshot: Annotated[bool, Field(description="Take screenshot")] = False,
    generate_markdown: Annotated[bool, Field(description="Generate markdown")] = True,
    include_cleaned_html: Annotated[bool, Field(description="Include HTML")] = False,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for element")] = None,
    timeout: Annotated[int, Field(description="Timeout (sec)")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Crawl depth")] = None,
    max_pages: Annotated[Optional[int], Field(description="Max pages")] = 10,
    include_external: Annotated[bool, Field(description="Follow external")] = False,
    crawl_strategy: Annotated[str, Field(description="'bfs'|'dfs'|'best_first'")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL filter")] = None,
    score_threshold: Annotated[float, Field(description="Min score")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="'bm25'|'pruning'|'llm'")] = None,
    filter_query: Annotated[Optional[str], Field(description="Filter keywords")] = None,
    chunk_content: Annotated[bool, Field(description="Chunk content")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunk strategy")] = "topic",
    chunk_size: Annotated[int, Field(description="Chunk tokens")] = 1000,
    overlap_rate: Annotated[float, Field(description="Chunk overlap")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="User agent")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="HTTP headers")] = None,
    enable_caching: Annotated[bool, Field(description="Enable cache")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JS to execute")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JS")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate user")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Bypass bot detection")] = False,
    auth_token: Annotated[Optional[str], Field(description="Auth token")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Cookies")] = None,
    auto_summarize: Annotated[bool, Field(description="Auto summarize")] = False,
    max_content_tokens: Annotated[int, Field(description="Max tokens")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
    llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
    # Phase 6: Session management
    use_session: Annotated[bool, Field(description="Use stored session")] = False,
    save_session: Annotated[bool, Field(description="Save session on success")] = False,
    session_ttl_hours: Annotated[int, Field(description="Session TTL in hours")] = 24,
    # Phase 7: Strategy caching
    use_strategy_cache: Annotated[bool, Field(description="Use cached strategy for domain")] = True,
    save_strategy: Annotated[bool, Field(description="Save successful strategy")] = True,
    strategy_ttl_days: Annotated[int, Field(description="Strategy cache TTL in days")] = 7,
    # Phase 8: Fingerprint evasion
    use_stealth_mode: Annotated[bool, Field(description="Enable fingerprint evasion")] = False,
    fingerprint_profile: Annotated[Optional[str], Field(description="Browser profile: chrome_windows|chrome_mac|firefox_windows|safari_mac|chrome_mobile|safari_mobile")] = None
) -> CrawlResponse:
    """Crawl with multiple fallback strategies for anti-bot sites.

    Phase 3: Multi-stage fallback with 7 stages:
    1. Static fast path - HTTP fetch without browser
    2. Normal headless - Minimal browser overhead
    3. Chromium + stealth - JS wait with realistic behavior
    4. User behavior - Scroll, click simulation
    5. Mobile agent - iOS Safari user agent
    6. AMP/RSS - Alternative sources (AMP pages, RSS feeds)
    7. JSON extraction - Extract from __NEXT_DATA__ etc.

    Phase 6: Session persistence
    - use_session: Load stored cookies/localStorage for the domain
    - save_session: Save session data on successful crawl
    - session_ttl_hours: Session expiration time (default 24 hours)

    Phase 7: Strategy caching
    - use_strategy_cache: Start from the best known strategy for domain
    - save_strategy: Record successful strategy for future use
    - strategy_ttl_days: Cache expiration time (default 7 days)

    Phase 8: Fingerprint evasion
    - use_stealth_mode: Enable browser fingerprint evasion scripts
    - fingerprint_profile: Specific browser profile to emulate
    - save_strategy: Record successful strategy for future use
    - strategy_ttl_days: Cache expiration time (default 7 days)
    """
    import asyncio
    import random
    from urllib.parse import urlparse

    # Detect site characteristics for optimized strategy selection
    domain = urlparse(url).netloc.lower()
    is_hn = 'ycombinator.com' in domain
    is_reddit = 'reddit.com' in domain
    is_social_media = any(site in domain for site in ['twitter.com', 'facebook.com', 'linkedin.com'])
    is_news_site = any(site in domain for site in ['cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com'])

    # ========================================
    # Phase 6: Session Management
    # ========================================
    session_manager = get_session_manager() if (use_session or save_session) else None
    stored_session = None
    session_cookies = None

    if use_session and session_manager:
        stored_session = session_manager.get_session(url)
        if stored_session:
            print(f"Session loaded for domain: {domain}")
            # Extract cookies from stored session
            session_cookies = stored_session.get("cookies", [])
            # Convert to dict format if needed for merging with user cookies
            if session_cookies and not cookies:
                # Use session cookies directly
                cookies = {c["name"]: c["value"] for c in session_cookies if "name" in c and "value" in c}
            elif session_cookies and cookies:
                # Merge: user cookies take precedence
                merged = {c["name"]: c["value"] for c in session_cookies if "name" in c and "value" in c}
                merged.update(cookies)
                cookies = merged

    def _save_session_on_success(result: CrawlResponse) -> CrawlResponse:
        """Save session data after successful crawl if save_session is enabled."""
        if save_session and session_manager and result.success:
            # Try to extract cookies from result or use provided cookies
            cookies_to_save = []

            # If we have user-provided cookies, convert to Playwright format
            if cookies:
                parsed = urlparse(url)
                for name, value in cookies.items():
                    cookies_to_save.append({
                        "name": name,
                        "value": value,
                        "domain": parsed.netloc,
                        "path": "/",
                        "httpOnly": False,
                        "secure": parsed.scheme == "https",
                        "sameSite": "Lax"
                    })

            if cookies_to_save:
                session_manager.save_session(
                    url=url,
                    cookies=cookies_to_save,
                    ttl_hours=session_ttl_hours
                )
                print(f"Session saved for domain: {domain}")

                # Add session info to extracted_data
                if result.extracted_data is None:
                    result.extracted_data = {}
                result.extracted_data["session_saved"] = True
                result.extracted_data["session_domain"] = domain

        return result

    # ========================================
    # Phase 7: Strategy Caching
    # ========================================
    strategy_cache = get_strategy_cache() if (use_strategy_cache or save_strategy) else None
    cached_strategy = None
    recommended_stages = [1, 2, 3, 4, 5, 6, 7]  # Default order

    if use_strategy_cache and strategy_cache:
        cached_strategy = strategy_cache.get_best_strategy(url)
        if cached_strategy:
            recommended_stages = strategy_cache.get_recommended_stages(url)
            print(f"Strategy cache hit for {domain}: best_stage={cached_strategy.get('start_stage')}, "
                  f"skip={cached_strategy.get('skip_stages', [])}, "
                  f"success_count={cached_strategy.get('success_count', 0)}")

    def _record_strategy_success(stage: int, strategy_name: str, response_time: float = None):
        """Record successful strategy to cache."""
        if save_strategy and strategy_cache:
            strategy_cache.record_success(
                url=url,
                stage=stage,
                strategy_name=strategy_name,
                response_time=response_time,
                ttl_days=strategy_ttl_days
            )
            print(f"Strategy recorded: stage={stage}, strategy={strategy_name}")

    def _record_strategy_failure(stage: int, strategy_name: str, error: str = None):
        """Record failed strategy to cache."""
        if save_strategy and strategy_cache:
            strategy_cache.record_failure(
                url=url,
                stage=stage,
                strategy_name=strategy_name,
                error=error
            )

    # ========================================
    # Phase 8: Fingerprint Evasion
    # ========================================
    stealth_config = None
    stealth_js = None
    stealth_user_agent = None
    stealth_headers = None

    if use_stealth_mode:
        stealth_config = get_fingerprint_config(fingerprint_profile)
        stealth_js = stealth_config["stealth_js"]
        stealth_user_agent = stealth_config["user_agent"]
        stealth_headers = stealth_config["headers"]
        print(f"Stealth mode enabled: profile={stealth_config['profile_name']}")

        # IMPORTANT: For consistency, stealth mode always uses profile's UA and headers
        # User-provided values are ignored to maintain fingerprint integrity
        # (UA, headers, and JS must all match the same profile)
        if user_agent and user_agent != stealth_user_agent:
            print(f"  Warning: user_agent ignored in stealth mode for consistency")
        user_agent = stealth_user_agent

        if headers:
            # Only merge non-fingerprint headers (e.g., auth headers)
            # Fingerprint-related headers from profile take precedence
            user_non_fingerprint = {k: v for k, v in headers.items()
                                    if not k.lower().startswith(('sec-ch-', 'accept', 'user-agent'))}
            stealth_headers = {**stealth_headers, **user_non_fingerprint}
        headers = stealth_headers

    # Common user agents for different strategies
    realistic_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]

    realistic_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }

    # Site-specific CSS selector optimization
    if is_hn and not css_selector:
        css_selector = ".fatitem, .athing, .comtr"

    # ========================================
    # Stage 1: Static fast path (no browser)
    # ========================================
    # Inject auth_token into headers for static fetch (if provided)
    # Only add if not already set (preserve user-specified Authorization, case-insensitive)
    static_fetch_headers = dict(headers) if headers else {}
    if auth_token:
        has_auth = any(k.lower() == "authorization" for k in static_fetch_headers)
        if not has_auth:
            static_fetch_headers["Authorization"] = f"Bearer {auth_token}"

    print(f"Stage 1/7: Attempting static HTTP fetch for {url}")
    static_success, static_html, static_error = await _static_fetch_content(
        url, headers=static_fetch_headers, timeout=min(timeout, 15)
    )

    if static_success and static_html:
        # Step 1: Try JSON extraction FIRST (works for Next.js, Nuxt, etc. even in SPA mode)
        json_success, json_data, json_source = _extract_spa_json_data(static_html)
        if json_success and json_data:
            # Check for block page indicators in the raw HTML
            if _is_block_page(static_html):
                print("  Block page detected in static HTML, skipping JSON extraction")
            else:
                # Record success for Phase 7
                _record_strategy_success(1, "static_json_extraction")
                # Return JSON extracted data with size limit check
                json_response = CrawlResponse(
                    success=True,
                    url=url,
                    markdown=f"# Extracted JSON Data ({json_source})\n\n```json\n{json.dumps(json_data, indent=2, ensure_ascii=False)[:50000]}\n```",
                    content=json.dumps(json_data, ensure_ascii=False),
                    extracted_data={
                        "fallback_strategy_used": "static_json_extraction",
                        "fallback_stage": 1,
                        "json_source": json_source,
                        "site_type_detected": "spa_with_json"
                    }
                )
                json_response = await _finalize_fallback_response(
                    json_response, url, auto_summarize, max_content_tokens, llm_provider, llm_model
                )
                return _save_session_on_success(json_response)

        # Step 2: Check for SPA indicators
        spa_framework, spa_selector = _detect_spa_framework(static_html)

        if spa_framework:
            # SPA detected - proceed to browser-based stages
            print(f"  SPA detected ({spa_framework}), proceeding to browser-based stages")
            if not wait_for_selector and spa_selector:
                wait_for_selector = spa_selector
        else:
            # Step 3: Not SPA - try HTML extraction if content is substantial
            if len(static_html) > 5000 and '<noscript>' not in static_html.lower():
                # Use crawl4ai for HTML to markdown conversion only
                try:
                    result = await crawl_url(
                        url=url,
                        css_selector=css_selector,
                        generate_markdown=generate_markdown,
                        include_cleaned_html=include_cleaned_html,
                        timeout=timeout,
                        wait_for_js=False,
                        cache_mode="enabled"
                    )
                    has_content, content_source = _has_meaningful_content(result, min_length=100)
                    if has_content:
                        # Check for block page indicators
                        content_text = " ".join([
                            result.markdown or "",
                            result.content or "",
                            getattr(result, 'raw_content', None) or ""
                        ])
                        if _is_block_page(content_text):
                            print("  Block page detected in static fast path, skipping")
                        else:
                            # Record success for Phase 7
                            _record_strategy_success(1, "static_fast_path")
                            if result.extracted_data is None:
                                result.extracted_data = {}
                            result.extracted_data.update({
                                "fallback_strategy_used": "static_fast_path",
                                "fallback_stage": 1,
                                "content_source": content_source
                            })
                            return _save_session_on_success(result)
                except Exception:
                    pass

    # ========================================
    # Build browser-based fallback strategies
    # ========================================
    strategies = []

    # Stage 2: Normal headless (minimal browser overhead)
    # Phase 8: In stealth mode, use profile UA/headers/JS for consistency
    stage2_ua = user_agent if use_stealth_mode else (user_agent or realistic_user_agents[0])
    stage2_headers = headers if use_stealth_mode else (headers or {})
    stage2_execute_js = stealth_js if use_stealth_mode else None
    strategies.append({
        "name": "normal_headless",
        "stage": 2,
        "params": {
            "user_agent": stage2_ua,
            "headers": stage2_headers,
            "wait_for_js": False,
            "simulate_user": False,
            "timeout": min(timeout, 20),
            "css_selector": css_selector,
            "wait_for_selector": None,
            "cache_mode": "enabled",
            "execute_js": stage2_execute_js
        }
    })

    # Stage 3: Chromium + stealth (JS wait with realistic behavior)
    # Phase 8: Add stealth JS if enabled and use consistent UA/headers
    stage3_execute_js = stealth_js if use_stealth_mode else None
    stage3_ua = user_agent if use_stealth_mode else (user_agent or random.choice(realistic_user_agents))
    stage3_headers = headers if use_stealth_mode else {**realistic_headers, **(headers or {})}
    strategies.append({
        "name": "chromium_stealth",
        "stage": 3,
        "params": {
            "user_agent": stage3_ua,
            "headers": stage3_headers,
            "wait_for_js": True,
            "simulate_user": False,
            "timeout": max(timeout, 30),
            "css_selector": css_selector,
            "wait_for_selector": wait_for_selector or (".fatitem" if is_hn else None),
            "cache_mode": "bypass",
            "execute_js": stage3_execute_js
        }
    })

    # Stage 4: User behavior simulation (scroll, click)
    # Phase 8: In stealth mode, keep same UA/headers as Stage 3 for consistency
    stage4_behavior_js = """
        // Simulate human-like interaction
        window.scrollTo(0, document.body.scrollHeight / 3);
        await new Promise(r => setTimeout(r, 500));
        window.scrollTo(0, document.body.scrollHeight / 2);
        await new Promise(r => setTimeout(r, 500));
        window.scrollTo(0, 0);
    """ if is_hn or is_social_media else (execute_js or "")
    stage4_execute_js = (stealth_js + "\n" + stage4_behavior_js) if use_stealth_mode and stealth_js else stage4_behavior_js
    # In stealth mode, use same UA as Stage 3; otherwise pick different UA
    stage4_ua = user_agent if use_stealth_mode else random.choice([ua for ua in realistic_user_agents if ua != strategies[-1]["params"]["user_agent"]])
    stage4_headers = headers if use_stealth_mode else {**realistic_headers, **(headers or {})}
    strategies.append({
        "name": "user_behavior",
        "stage": 4,
        "params": {
            "user_agent": stage4_ua,
            "headers": stage4_headers,
            "wait_for_js": True,
            "simulate_user": True,
            "timeout": timeout + 30,
            "css_selector": css_selector,
            "execute_js": stage4_execute_js if stage4_execute_js.strip() else None,
            "cache_mode": "disabled"
        }
    })

    # Stage 5: Mobile user agent
    # Phase 8: Use safari_mobile profile for consistency (UA is iOS Safari)
    mobile_stealth_js = None
    mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    mobile_headers = {**realistic_headers}
    # Safari doesn't send Sec-CH-UA headers, so don't include them
    if use_stealth_mode:
        mobile_config = get_fingerprint_config("safari_mobile")  # Use matching Safari profile
        mobile_stealth_js = mobile_config["stealth_js"]
        mobile_ua = mobile_config["user_agent"]
        mobile_headers = mobile_config["headers"]
    strategies.append({
        "name": "mobile_agent",
        "stage": 5,
        "params": {
            "user_agent": mobile_ua,
            "headers": mobile_headers,
            "wait_for_js": True,
            "simulate_user": False,
            "timeout": timeout,
            "css_selector": css_selector,
            "cache_mode": "bypass",
            "execute_js": mobile_stealth_js
        }
    })

    last_error = None
    total_stages = 7  # Stages: 1=static, 2-5=browser, 6=AMP/RSS, 7=JSON

    # ========================================
    # Phase 7: Apply strategy cache to execution order
    # ========================================
    if use_strategy_cache and cached_strategy:
        skip_stages = set(cached_strategy.get("skip_stages", []))
        start_stage = cached_strategy.get("start_stage", 1)

        # Sort strategies based on recommended order
        # Prioritize the cached best stage, skip known failed stages
        def strategy_sort_key(s):
            stage = s.get("stage", 99)
            if stage in skip_stages:
                return 1000 + stage  # Move failed stages to end
            if stage == start_stage:
                return 0  # Prioritize best known stage
            # Maintain relative order for others based on recommended_stages
            try:
                return recommended_stages.index(stage)
            except ValueError:
                return 500 + stage

        strategies.sort(key=strategy_sort_key)

        # Log the adjusted order
        adjusted_order = [s.get("stage") for s in strategies]
        print(f"Strategy order adjusted by cache: {adjusted_order} (skip: {list(skip_stages)})")

    for i, strategy in enumerate(strategies):
        try:
            # Add random delay between attempts (1-5 seconds)
            if i > 0:
                delay = random.uniform(1, 5)
                await asyncio.sleep(delay)

            stage_num = strategy.get("stage", i + 2)  # Stage 1 was static fetch
            print(f"Stage {stage_num}/{total_stages}: Attempting {strategy['name']}")

            # Start timing for response_time tracking
            import time as time_module
            stage_start_time = time_module.time()

            # Prepare strategy-specific parameters
            strategy_params = {
                "url": url,
                "css_selector": strategy["params"].get("css_selector", css_selector),
                "xpath": xpath,
                "extract_media": extract_media,
                "take_screenshot": take_screenshot,
                "generate_markdown": generate_markdown,
                "include_cleaned_html": include_cleaned_html,
                "wait_for_selector": strategy["params"].get("wait_for_selector", wait_for_selector),
                "timeout": strategy["params"].get("timeout", timeout),
                "max_depth": None,  # Force single page to avoid deep crawling issues
                "max_pages": max_pages,
                "include_external": include_external,
                "crawl_strategy": crawl_strategy,
                "url_pattern": url_pattern,
                "score_threshold": score_threshold,
                "content_filter": content_filter,
                "filter_query": filter_query,
                "chunk_content": chunk_content,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "overlap_rate": overlap_rate,
                "user_agent": strategy["params"].get("user_agent", user_agent),
                "headers": strategy["params"].get("headers", headers),
                "enable_caching": enable_caching,
                "cache_mode": strategy["params"].get("cache_mode", cache_mode),
                "execute_js": strategy["params"].get("execute_js", execute_js),
                "wait_for_js": strategy["params"].get("wait_for_js", wait_for_js),
                "simulate_user": strategy["params"].get("simulate_user", simulate_user),
                "use_undetected_browser": use_undetected_browser,
                "auth_token": auth_token,
                "cookies": cookies,
                "auto_summarize": auto_summarize,
                "max_content_tokens": max_content_tokens,
                "summary_length": summary_length,
                "llm_provider": llm_provider,
                "llm_model": llm_model
            }
            
            # Attempt crawl with current strategy
            result = await crawl_url(**strategy_params)

            # Check explicit failure first - preserve actual error message
            if not result.success:
                actual_error = getattr(result, 'error', None) or "Unknown error"
                last_error = f"Strategy {strategy['name']}: {actual_error}"
                # Record failure for Phase 7
                _record_strategy_failure(stage_num, strategy["name"], actual_error)
                continue

            # Check if we got meaningful content (markdown, content, or raw_content)
            has_content, content_source = _has_meaningful_content(
                result, min_length=FALLBACK_MIN_CONTENT_LENGTH
            )

            # Additional check for block pages (check all fields, not just first non-empty)
            if has_content:
                raw_content = getattr(result, 'raw_content', None) or ""
                # Concatenate all fields to ensure block indicators aren't missed
                content_text = " ".join([
                    result.markdown or "",
                    result.content or "",
                    raw_content
                ]).lower()
                if _is_block_page(content_text):
                    has_content = False
                    print(f"  Block page detected, skipping strategy {strategy['name']}")
                    _record_strategy_failure(stage_num, strategy["name"], "block_page_detected")

            if has_content:
                # Calculate response time and record success for Phase 7
                response_time = time_module.time() - stage_start_time
                _record_strategy_success(stage_num, strategy["name"], response_time)
                # Add strategy info to extracted_data
                if result.extracted_data is None:
                    result.extracted_data = {}
                result.extracted_data.update({
                    "fallback_strategy_used": strategy["name"],
                    "fallback_stage": strategy.get("stage", i + 2),
                    "total_stages": total_stages,
                    "site_type_detected": "hackernews" if is_hn else "reddit" if is_reddit else "social_media" if is_social_media else "news" if is_news_site else "general",
                    "content_source": content_source
                })
                return _save_session_on_success(result)

            last_error = f"Strategy {strategy['name']}: No meaningful content in markdown/content/raw_content"
            # Record failure for Phase 7 (no content)
            _record_strategy_failure(stage_num, strategy["name"], "no_meaningful_content")

        except Exception as e:
            last_error = f"Strategy {strategy['name']}: {str(e)}"
            print(f"Strategy {strategy['name']} failed: {e}")
            # Record failure for Phase 7
            _record_strategy_failure(strategy.get("stage", i + 2), strategy["name"], str(e))
            continue

    # ========================================
    # Stage 5 Alternative: AMP/RSS fallback
    # ========================================
    print(f"Stage 6/{total_stages}: Attempting AMP/RSS fallback")

    # Try AMP version
    amp_url = _build_amp_url(url)
    if amp_url:
        try:
            amp_result = await crawl_url(
                url=amp_url,
                css_selector=css_selector,
                generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html,
                timeout=min(timeout, 20),
                wait_for_js=False
            )
            has_content, content_source = _has_meaningful_content(amp_result, min_length=100)
            if has_content:
                # Check for block page indicators
                amp_content_text = " ".join([
                    amp_result.markdown or "",
                    amp_result.content or "",
                    getattr(amp_result, 'raw_content', None) or ""
                ])
                if _is_block_page(amp_content_text):
                    print("  Block page detected in AMP result, skipping")
                else:
                    # Record success for Phase 7
                    _record_strategy_success(6, "amp_page")
                    if amp_result.extracted_data is None:
                        amp_result.extracted_data = {}
                    amp_result.extracted_data.update({
                        "fallback_strategy_used": "amp_page",
                        "fallback_stage": 6,
                        "original_url": url,
                        "amp_url_used": amp_url,
                        "content_source": content_source
                    })
                    return _save_session_on_success(amp_result)
        except Exception as e:
            print(f"AMP fallback failed: {e}")
            _record_strategy_failure(6, "amp_page", str(e))

    # Try RSS/Atom feed
    try:
        rss_success, feed_url, feed_items = await _try_fetch_rss_feed(url)
        if rss_success and feed_items:
            # Format RSS items as markdown
            markdown_content = f"# RSS Feed Content\n\nFeed URL: {feed_url}\n\n"
            for item in feed_items[:20]:  # Limit to 20 items
                if item.get('title'):
                    markdown_content += f"## {item['title']}\n"
                if item.get('link'):
                    markdown_content += f"[Link]({item['link']})\n\n"
                if item.get('description'):
                    markdown_content += f"{item['description']}\n\n"
                markdown_content += "---\n\n"

            # Check for block page indicators in RSS content
            if _is_block_page(markdown_content):
                print("  Block page detected in RSS content, skipping")
            else:
                # Record success for Phase 7
                _record_strategy_success(6, "rss_feed")
                rss_response = CrawlResponse(
                    success=True,
                    url=url,
                    markdown=markdown_content,
                    content=json.dumps(feed_items, ensure_ascii=False),
                    extracted_data={
                        "fallback_strategy_used": "rss_feed",
                        "fallback_stage": 6,
                        "feed_url": feed_url,
                        "item_count": len(feed_items),
                        "content_source": "rss"
                    }
                )
                rss_response = await _finalize_fallback_response(
                    rss_response, url, auto_summarize, max_content_tokens, llm_provider, llm_model
                )
                return _save_session_on_success(rss_response)
    except Exception as e:
        print(f"RSS fallback failed: {e}")
        _record_strategy_failure(6, "rss_feed", str(e))

    # ========================================
    # Stage 7: JSON extraction from cached static HTML
    # ========================================
    print(f"Stage 7/{total_stages}: Attempting JSON extraction from static HTML")

    if static_success and static_html:
        json_success, json_data, json_source = _extract_spa_json_data(static_html)
        if json_success and json_data:
            # Try to extract useful content from JSON
            content_text = ""
            if isinstance(json_data, dict):
                # Common patterns for content extraction
                for key in ['props', 'pageProps', 'data', 'content', 'article', 'post']:
                    if key in json_data:
                        content_text = json.dumps(json_data[key], indent=2, ensure_ascii=False)
                        break
                if not content_text:
                    content_text = json.dumps(json_data, indent=2, ensure_ascii=False)

            # Record success for Phase 7
            _record_strategy_success(7, "json_extraction")
            stage7_response = CrawlResponse(
                success=True,
                url=url,
                markdown=f"# Extracted JSON Data ({json_source})\n\n```json\n{content_text[:50000]}\n```",
                content=content_text,
                extracted_data={
                    "fallback_strategy_used": "json_extraction",
                    "fallback_stage": 7,
                    "json_source": json_source,
                    "site_type_detected": "spa_with_json"
                }
            )
            stage7_response = await _finalize_fallback_response(
                stage7_response, url, auto_summarize, max_content_tokens, llm_provider, llm_model
            )
            return _save_session_on_success(stage7_response)

    # Record failure for Stage 7 if static HTML exists but JSON extraction failed
    if static_success and static_html:
        _record_strategy_failure(7, "json_extraction", "no_json_data")

    # All strategies failed, return error with details
    all_strategies = ["static_fast_path"] + [s["name"] for s in strategies] + ["amp_page", "rss_feed", "json_extraction"]
    return CrawlResponse(
        success=False,
        url=url,
        error=f"All {total_stages} fallback stages failed. Last error: {last_error}. "
              f"This site may have strong anti-bot protection. "
              f"Stages attempted: {', '.join(all_strategies)}",
        extracted_data={
            "fallback_strategies_attempted": all_strategies,
            "total_stages": total_stages,
            "site_type_detected": "hackernews" if is_hn else "reddit" if is_reddit else "social_media" if is_social_media else "news" if is_news_site else "general",
            "static_fetch_result": "success" if static_success else f"failed: {static_error}",
            "recommendations": [
                "Try accessing the site manually to check if it's available",
                "Consider using the site's API if available",
                "Try accessing during off-peak hours",
                "Use a VPN if the site is geo-blocked",
                "Check if the site has a mobile or AMP version"
            ]
        }
    )


