"""Fallback strategies for Crawl4AI MCP Server.

This module provides fallback crawling strategies for handling
difficult-to-crawl pages and SPA frameworks.
"""

import json
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urljoin, urlunparse


def normalize_cookies_to_playwright_format(
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


async def static_fetch_content(url: str, headers: dict = None, timeout: int = 30) -> Tuple[bool, str, str]:
    """
    Stage 1: Fast static HTTP fetch without browser overhead.

    Uses httpx for direct HTTP requests with readability extraction.

    Args:
        url: URL to fetch
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success: bool, content: str, error: str)
    """
    import httpx

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

    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            error_msg = f"HTTP error {e.response.status_code}"
        return False, "", f"Static fetch error: {error_msg}"


def extract_spa_json_data(html_content: str) -> Tuple[bool, dict, str]:
    """
    Stage 6: Extract JSON data from SPA frameworks.

    Extracts data from:
    - __NEXT_DATA__ (Next.js)
    - window.__INITIAL_STATE__ (various frameworks)
    - window.__NUXT__ (Nuxt.js)
    - window.__APP_STATE__ (various frameworks)

    Uses balanced brace matching to handle nested JSON correctly.

    Args:
        html_content: HTML content to extract from

    Returns:
        Tuple of (success: bool, data: dict, source: str)
    """

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


def detect_spa_framework(html_content: str) -> Tuple[str, str]:
    """
    Stage 4: Detect SPA framework for optimized crawling.

    Args:
        html_content: HTML content to analyze

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


def build_amp_url(url: str) -> str:
    """
    Stage 6: Build AMP version URL.

    Attempts to construct AMP URL for the given page.
    Preserves query string and fragment from the original URL.

    Args:
        url: Original URL

    Returns:
        AMP URL or empty string if already AMP
    """
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


async def try_fetch_rss_feed(url: str) -> Tuple[bool, str, list]:
    """
    Stage 5: Try to find and fetch RSS/Atom feed for the page.

    Args:
        url: Page URL to find feed for

    Returns:
        Tuple of (success: bool, feed_url: str, items: list)
    """
    import httpx

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

    except Exception:
        pass

    return False, "", []


def get_fallback_stage_info() -> Dict[int, Dict[str, str]]:
    """
    Get information about each fallback stage.

    Returns:
        Dictionary mapping stage numbers to stage info
    """
    return {
        1: {
            "name": "static_fast_path",
            "description": "Fast static HTTP fetch without browser overhead",
            "use_case": "Simple HTML pages without JavaScript requirements"
        },
        2: {
            "name": "normal_headless",
            "description": "Standard headless browser crawl",
            "use_case": "Pages requiring basic JavaScript execution"
        },
        3: {
            "name": "chromium_stealth",
            "description": "Stealth mode with fingerprint evasion",
            "use_case": "Pages with basic bot detection"
        },
        4: {
            "name": "user_behavior",
            "description": "Simulated user behavior (scrolling, delays)",
            "use_case": "Pages requiring interaction or lazy loading"
        },
        5: {
            "name": "mobile_agent",
            "description": "Mobile browser emulation",
            "use_case": "Sites with mobile-friendly versions"
        },
        6: {
            "name": "amp_rss",
            "description": "AMP version or RSS feed extraction",
            "use_case": "News sites and content platforms"
        },
        7: {
            "name": "json_extraction",
            "description": "SPA JSON data extraction",
            "use_case": "Single-page applications with embedded data"
        }
    }


# Backward compatibility aliases
_normalize_cookies_to_playwright_format = normalize_cookies_to_playwright_format
_static_fetch_content = static_fetch_content
_extract_spa_json_data = extract_spa_json_data
_detect_spa_framework = detect_spa_framework
_build_amp_url = build_amp_url
_try_fetch_rss_feed = try_fetch_rss_feed
