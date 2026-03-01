"""Browser fingerprint profiles for Crawl4AI MCP Server.

Generates consistent browser fingerprint profiles for anti-detection,
including user agents, HTTP headers, and stealth JavaScript.
"""

import json
from typing import Dict, List, Optional


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
