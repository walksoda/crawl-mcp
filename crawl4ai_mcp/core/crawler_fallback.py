"""
Crawler fallback strategies for anti-bot sites.

Contains crawl_url_with_fallback with 7-stage fallback:
1. Static fast path
2. Normal headless
3. Chromium + stealth
4. User behavior simulation
5. Mobile user agent
6. AMP/RSS fallback
7. JSON extraction
"""

import asyncio
import json
import random
import time as time_module
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..models import CrawlResponse
from ..infra.session import get_session_manager
from ..infra.strategy_cache import get_strategy_cache
from ..infra.fingerprint import get_fingerprint_config
from ..infra.fallback_strategies import (
    static_fetch_content as _static_fetch_content,
    detect_spa_framework as _detect_spa_framework,
    extract_spa_json_data as _extract_spa_json_data,
    build_amp_url as _build_amp_url,
    try_fetch_rss_feed as _try_fetch_rss_feed,
)
from ..infra.content_processors import (
    is_block_page as _is_block_page,
    has_meaningful_content as _has_meaningful_content,
)
from ..constants import FALLBACK_MIN_CONTENT_LENGTH
from .crawl_url import crawl_url
from .crawler_io import _build_json_extraction_response
from .crawler_summarizer import _finalize_fallback_response

REALISTIC_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

REALISTIC_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1", "Connection": "keep-alive", "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none", "Cache-Control": "max-age=0",
}


def _save_session_on_success(
    result: CrawlResponse, url: str, save_session: bool,
    session_manager, cookies: Optional[Dict[str, str]], session_ttl_hours: int,
) -> CrawlResponse:
    """Save session data after successful crawl if save_session is enabled."""
    if not (save_session and session_manager and result.success):
        return result
    cookies_to_save, parsed = [], urlparse(url)
    domain = parsed.netloc.lower()
    _cookie_defaults = {"domain": parsed.netloc, "path": "/", "httpOnly": False,
                        "secure": parsed.scheme == "https", "sameSite": "Lax"}
    if result.extracted_data and isinstance(result.extracted_data, dict):
        for cookie in (result.extracted_data.get("cookies") or []):
            if isinstance(cookie, dict) and "name" in cookie and "value" in cookie:
                cookies_to_save.append({
                    "name": cookie["name"], "value": cookie["value"],
                    "domain": cookie.get("domain", parsed.netloc),
                    "path": cookie.get("path", "/"),
                    "httpOnly": cookie.get("httpOnly", False),
                    "secure": cookie.get("secure", parsed.scheme == "https"),
                    "sameSite": cookie.get("sameSite", "Lax"),
                })
    if cookies:
        existing_names = {c["name"] for c in cookies_to_save}
        for name, value in cookies.items():
            if name not in existing_names:
                cookies_to_save.append({"name": name, "value": value, **_cookie_defaults})
            else:
                for c in cookies_to_save:
                    if c["name"] == name:
                        c["value"] = value
                        break
    if cookies_to_save:
        session_manager.save_session(url=url, cookies=cookies_to_save, ttl_hours=session_ttl_hours)
        print(f"Session saved for domain: {domain}")
        if result.extracted_data is None:
            result.extracted_data = {}
        result.extracted_data["session_saved"] = True
        result.extracted_data["session_domain"] = domain
        result.extracted_data["session_cookies_count"] = len(cookies_to_save)
    return result


def _record_strategy_success(
    strategy_cache, save_strategy: bool, url: str,
    stage: int, strategy_name: str,
    response_time: float = None, strategy_ttl_days: int = 7,
) -> None:
    """Record successful strategy to cache."""
    if save_strategy and strategy_cache:
        strategy_cache.record_success(
            url=url, stage=stage, strategy_name=strategy_name,
            response_time=response_time, ttl_days=strategy_ttl_days,
        )
        print(f"Strategy recorded: stage={stage}, strategy={strategy_name}")


def _record_strategy_failure(
    strategy_cache, save_strategy: bool, url: str,
    stage: int, strategy_name: str, error: str = None,
) -> None:
    """Record failed strategy to cache."""
    if save_strategy and strategy_cache:
        strategy_cache.record_failure(url=url, stage=stage, strategy_name=strategy_name, error=error)


def _build_browser_strategies(
    *, url: str, css_selector: Optional[str], wait_for_selector: Optional[str],
    timeout: int, user_agent: Optional[str], headers: Optional[Dict[str, str]],
    execute_js: Optional[str], use_stealth_mode: bool, stealth_js: Optional[str],
    is_hn: bool, is_social_media: bool,
) -> List[dict]:
    """Build browser-based fallback strategy list (stages 2-5)."""
    strategies = []
    # Stage 2: Normal headless
    strategies.append({"name": "normal_headless", "stage": 2, "params": {
        "user_agent": user_agent if use_stealth_mode else (user_agent or REALISTIC_USER_AGENTS[0]),
        "headers": headers if use_stealth_mode else (headers or {}),
        "wait_for_js": False, "simulate_user": False, "timeout": min(timeout, 20),
        "css_selector": css_selector, "wait_for_selector": None,
        "cache_mode": "enabled", "execute_js": stealth_js if use_stealth_mode else None,
    }})
    # Stage 3: Chromium + stealth
    s3_ua = user_agent if use_stealth_mode else (user_agent or random.choice(REALISTIC_USER_AGENTS))
    strategies.append({"name": "chromium_stealth", "stage": 3, "params": {
        "user_agent": s3_ua,
        "headers": headers if use_stealth_mode else {**REALISTIC_HEADERS, **(headers or {})},
        "wait_for_js": True, "simulate_user": False, "timeout": max(timeout, 30),
        "css_selector": css_selector,
        "wait_for_selector": wait_for_selector or (".fatitem" if is_hn else None),
        "cache_mode": "bypass", "execute_js": stealth_js if use_stealth_mode else None,
    }})
    # Stage 4: User behavior simulation
    behavior_js = (
        "window.scrollTo(0, document.body.scrollHeight / 3);\n"
        "await new Promise(r => setTimeout(r, 500));\n"
        "window.scrollTo(0, document.body.scrollHeight / 2);\n"
        "await new Promise(r => setTimeout(r, 500));\n"
        "window.scrollTo(0, 0);"
    ) if is_hn or is_social_media else (execute_js or "")
    s4_js = (stealth_js + "\n" + behavior_js) if use_stealth_mode and stealth_js else behavior_js
    s4_ua = user_agent if use_stealth_mode else random.choice(
        [ua for ua in REALISTIC_USER_AGENTS if ua != strategies[-1]["params"]["user_agent"]])
    strategies.append({"name": "user_behavior", "stage": 4, "params": {
        "user_agent": s4_ua,
        "headers": headers if use_stealth_mode else {**REALISTIC_HEADERS, **(headers or {})},
        "wait_for_js": True, "simulate_user": True, "timeout": timeout + 30,
        "css_selector": css_selector,
        "execute_js": s4_js if s4_js.strip() else None, "cache_mode": "disabled",
    }})
    # Stage 5: Mobile user agent
    mobile_js, mobile_ua = None, "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    mobile_hdrs = {**REALISTIC_HEADERS}
    if use_stealth_mode:
        mc = get_fingerprint_config("safari_mobile")
        mobile_js, mobile_ua, mobile_hdrs = mc["stealth_js"], mc["user_agent"], mc["headers"]
    strategies.append({"name": "mobile_agent", "stage": 5, "params": {
        "user_agent": mobile_ua, "headers": mobile_hdrs, "wait_for_js": True,
        "simulate_user": False, "timeout": timeout, "css_selector": css_selector,
        "cache_mode": "bypass", "execute_js": mobile_js,
    }})
    return strategies


def _detect_site_type(domain: str) -> str:
    """Return site type string from domain."""
    if "ycombinator.com" in domain: return "hackernews"
    if "reddit.com" in domain: return "reddit"
    if any(s in domain for s in ["twitter.com", "facebook.com", "linkedin.com"]): return "social_media"
    if any(s in domain for s in ["cnn.com", "bbc.com", "nytimes.com", "theguardian.com"]): return "news"
    return "general"


async def crawl_url_with_fallback(
    url: str, css_selector: Optional[str] = None,
    extract_media: bool = False, take_screenshot: bool = False,
    generate_markdown: bool = True, include_cleaned_html: bool = False,
    wait_for_selector: Optional[str] = None, timeout: int = 60,
    max_depth: Optional[int] = None, max_pages: Optional[int] = 10,
    include_external: bool = False, crawl_strategy: str = "bfs",
    url_pattern: Optional[str] = None, score_threshold: float = 0.3,
    content_filter: Optional[str] = None, filter_query: Optional[str] = None,
    chunk_content: bool = False, chunk_strategy: str = "topic",
    chunk_size: int = 1000, overlap_rate: float = 0.1,
    user_agent: Optional[str] = None, headers: Optional[Dict[str, str]] = None,
    enable_caching: bool = True, cache_mode: str = "enabled",
    execute_js: Optional[str] = None, wait_for_js: bool = False,
    simulate_user: bool = False, use_undetected_browser: bool = False,
    auth_token: Optional[str] = None, cookies: Optional[Dict[str, str]] = None,
    auto_summarize: bool = False, max_content_tokens: int = 15000,
    summary_length: str = "medium", llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    use_session: bool = False, save_session: bool = False, session_ttl_hours: int = 24,
    use_strategy_cache: bool = True, save_strategy: bool = True, strategy_ttl_days: int = 7,
    use_stealth_mode: bool = False, fingerprint_profile: Optional[str] = None,
) -> CrawlResponse:
    """Crawl with multiple fallback strategies for anti-bot sites.

    7-stage fallback with session persistence, strategy caching, and fingerprint evasion.
    """
    domain = urlparse(url).netloc.lower()
    is_hn = "ycombinator.com" in domain
    is_social_media = any(s in domain for s in ["twitter.com", "facebook.com", "linkedin.com"])
    site_type = _detect_site_type(domain)

    # Phase 6: Session Management
    session_manager = get_session_manager() if (use_session or save_session) else None
    if use_session and session_manager:
        stored_session = session_manager.get_session(url)
        if stored_session:
            print(f"Session loaded for domain: {domain}")
            sc = stored_session.get("cookies", [])
            if sc and not cookies:
                cookies = {c["name"]: c["value"] for c in sc if "name" in c and "value" in c}
            elif sc and cookies:
                merged = {c["name"]: c["value"] for c in sc if "name" in c and "value" in c}
                merged.update(cookies)
                cookies = merged

    def _save_session(r: CrawlResponse) -> CrawlResponse:
        return _save_session_on_success(r, url, save_session, session_manager, cookies, session_ttl_hours)

    # Phase 7: Strategy Caching
    strategy_cache = get_strategy_cache() if (use_strategy_cache or save_strategy) else None
    cached_strategy, recommended_stages = None, [1, 2, 3, 4, 5, 6, 7]
    if use_strategy_cache and strategy_cache:
        cached_strategy = strategy_cache.get_best_strategy(url)
        if cached_strategy:
            recommended_stages = strategy_cache.get_recommended_stages(url)
            print(f"Strategy cache hit for {domain}: best_stage={cached_strategy.get('start_stage')}, "
                  f"skip={cached_strategy.get('skip_stages', [])}, "
                  f"success_count={cached_strategy.get('success_count', 0)}")

    def _rec_success(stage: int, name: str, rt: float = None):
        _record_strategy_success(strategy_cache, save_strategy, url, stage, name, rt, strategy_ttl_days)

    def _rec_failure(stage: int, name: str, err: str = None):
        _record_strategy_failure(strategy_cache, save_strategy, url, stage, name, err)

    # Phase 8: Fingerprint Evasion
    stealth_js = None
    if use_stealth_mode:
        stealth_config = get_fingerprint_config(fingerprint_profile)
        if not stealth_config or not isinstance(stealth_config, dict):
            stealth_config = get_fingerprint_config("chrome_win")
        stealth_js = stealth_config.get("stealth_js", "")
        stealth_user_agent = stealth_config.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        stealth_headers = stealth_config.get("headers", {})
        print(f"Stealth mode enabled: profile={stealth_config.get('profile_name', 'fallback')}")
        if user_agent and user_agent != stealth_user_agent:
            print("  Warning: user_agent ignored in stealth mode for consistency")
        user_agent = stealth_user_agent
        if headers:
            user_non_fp = {k: v for k, v in headers.items() if not k.lower().startswith(("sec-ch-", "accept", "user-agent"))}
            stealth_headers = {**stealth_headers, **user_non_fp}
        headers = stealth_headers

    if is_hn and not css_selector:
        css_selector = ".fatitem, .athing, .comtr"

    # Stage 1: Static fast path (no browser)
    static_fetch_headers = dict(headers) if headers else {}
    if auth_token and not any(k.lower() == "authorization" for k in static_fetch_headers):
        static_fetch_headers["Authorization"] = f"Bearer {auth_token}"

    print(f"Stage 1/7: Attempting static HTTP fetch for {url}")
    static_success, static_html, static_error = await _static_fetch_content(
        url, headers=static_fetch_headers, timeout=min(timeout, 15))

    if static_success and static_html:
        json_success, json_data, json_source = _extract_spa_json_data(static_html)
        if json_success and json_data:
            if _is_block_page(static_html):
                print("  Block page detected in static HTML, skipping JSON extraction")
            else:
                _rec_success(1, "static_json_extraction")
                json_response = await _build_json_extraction_response(
                    json_data=json_data, json_source=json_source, url=url,
                    strategy_name="static_json_extraction", stage=1,
                    auto_summarize=auto_summarize, max_content_tokens=max_content_tokens,
                    llm_provider=llm_provider, llm_model=llm_model)
                return _save_session(json_response)

        spa_framework, spa_selector = _detect_spa_framework(static_html)
        if spa_framework:
            print(f"  SPA detected ({spa_framework}), proceeding to browser-based stages")
            if not wait_for_selector and spa_selector:
                wait_for_selector = spa_selector
        elif len(static_html) > 5000 and "<noscript>" not in static_html.lower():
            try:
                result = await crawl_url(
                    url=url, css_selector=css_selector, generate_markdown=generate_markdown,
                    include_cleaned_html=include_cleaned_html, timeout=timeout,
                    wait_for_js=False, cache_mode="enabled")
                has_content, content_source = _has_meaningful_content(result, min_length=100)
                if has_content:
                    ct = " ".join([result.markdown or "", result.content or "", getattr(result, "raw_content", None) or ""])
                    if _is_block_page(ct):
                        print("  Block page detected in static fast path, skipping")
                    else:
                        _rec_success(1, "static_fast_path")
                        if result.extracted_data is None:
                            result.extracted_data = {}
                        result.extracted_data.update({"fallback_strategy_used": "static_fast_path", "fallback_stage": 1, "content_source": content_source})
                        return _save_session(result)
            except Exception:
                pass

    # Build browser-based fallback strategies (stages 2-5)
    strategies = _build_browser_strategies(
        url=url, css_selector=css_selector, wait_for_selector=wait_for_selector,
        timeout=timeout, user_agent=user_agent, headers=headers,
        execute_js=execute_js, use_stealth_mode=use_stealth_mode,
        stealth_js=stealth_js, is_hn=is_hn, is_social_media=is_social_media)

    last_error, total_stages = None, 7
    # Phase 7: Apply strategy cache to execution order
    if use_strategy_cache and cached_strategy:
        skip_stages = set(cached_strategy.get("skip_stages", []))
        start_stage = cached_strategy.get("start_stage", 1)

        def _sort_key(s):
            st = s.get("stage", 99)
            if st in skip_stages: return 1000 + st
            if st == start_stage: return 0
            try: return recommended_stages.index(st)
            except ValueError: return 500 + st
        strategies.sort(key=_sort_key)
        print(f"Strategy order adjusted by cache: {[s.get('stage') for s in strategies]} (skip: {list(skip_stages)})")

    # Shared params passed through to crawl_url for each strategy
    shared = dict(
        extract_media=extract_media, take_screenshot=take_screenshot,
        generate_markdown=generate_markdown, include_cleaned_html=include_cleaned_html,
        max_depth=None, max_pages=max_pages, include_external=include_external,
        crawl_strategy=crawl_strategy, url_pattern=url_pattern, score_threshold=score_threshold,
        content_filter=content_filter, filter_query=filter_query,
        chunk_content=chunk_content, chunk_strategy=chunk_strategy,
        chunk_size=chunk_size, overlap_rate=overlap_rate,
        enable_caching=enable_caching, use_undetected_browser=use_undetected_browser,
        auth_token=auth_token, cookies=cookies, auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens, summary_length=summary_length,
        llm_provider=llm_provider, llm_model=llm_model)

    for i, strategy in enumerate(strategies):
        try:
            if i > 0:
                await asyncio.sleep(random.uniform(1, 5))
            stage_num = strategy.get("stage", i + 2)
            print(f"Stage {stage_num}/{total_stages}: Attempting {strategy['name']}")
            stage_start_time = time_module.time()

            sp = strategy["params"]
            result = await crawl_url(
                url=url, css_selector=sp.get("css_selector", css_selector),
                wait_for_selector=sp.get("wait_for_selector", wait_for_selector),
                timeout=sp.get("timeout", timeout),
                user_agent=sp.get("user_agent", user_agent),
                headers=sp.get("headers", headers),
                cache_mode=sp.get("cache_mode", cache_mode),
                execute_js=sp.get("execute_js", execute_js),
                wait_for_js=sp.get("wait_for_js", wait_for_js),
                simulate_user=sp.get("simulate_user", simulate_user),
                **shared)

            if not result.success:
                actual_error = getattr(result, "error", None) or "Unknown error"
                last_error = f"Strategy {strategy['name']}: {actual_error}"
                _rec_failure(stage_num, strategy["name"], actual_error)
                continue

            has_content, content_source = _has_meaningful_content(result, min_length=FALLBACK_MIN_CONTENT_LENGTH)
            if has_content:
                ct = " ".join([result.markdown or "", result.content or "", getattr(result, "raw_content", None) or ""]).lower()
                if _is_block_page(ct):
                    has_content = False
                    print(f"  Block page detected, skipping strategy {strategy['name']}")
                    _rec_failure(stage_num, strategy["name"], "block_page_detected")

            if has_content:
                response_time = time_module.time() - stage_start_time
                _rec_success(stage_num, strategy["name"], response_time)
                if result.extracted_data is None:
                    result.extracted_data = {}
                result.extracted_data.update({
                    "fallback_strategy_used": strategy["name"],
                    "fallback_stage": strategy.get("stage", i + 2),
                    "total_stages": total_stages, "site_type_detected": site_type,
                    "content_source": content_source,
                })
                return _save_session(result)

            last_error = f"Strategy {strategy['name']}: No meaningful content in markdown/content/raw_content"
            _rec_failure(stage_num, strategy["name"], "no_meaningful_content")
        except Exception as e:
            last_error = f"Strategy {strategy['name']}: {str(e)}"
            print(f"Strategy {strategy['name']} failed: {e}")
            _rec_failure(strategy.get("stage", i + 2), strategy["name"], str(e))
            continue

    # Stage 6: AMP/RSS fallback
    print(f"Stage 6/{total_stages}: Attempting AMP/RSS fallback")
    amp_url = _build_amp_url(url)
    if amp_url:
        try:
            amp_result = await crawl_url(
                url=amp_url, css_selector=css_selector, generate_markdown=generate_markdown,
                include_cleaned_html=include_cleaned_html, timeout=min(timeout, 20), wait_for_js=False)
            has_content, content_source = _has_meaningful_content(amp_result, min_length=100)
            if has_content:
                act = " ".join([amp_result.markdown or "", amp_result.content or "", getattr(amp_result, "raw_content", None) or ""])
                if _is_block_page(act):
                    print("  Block page detected in AMP result, skipping")
                else:
                    _rec_success(6, "amp_page")
                    if amp_result.extracted_data is None:
                        amp_result.extracted_data = {}
                    amp_result.extracted_data.update({
                        "fallback_strategy_used": "amp_page", "fallback_stage": 6,
                        "original_url": url, "amp_url_used": amp_url, "content_source": content_source})
                    return _save_session(amp_result)
        except Exception as e:
            print(f"AMP fallback failed: {e}")
            _rec_failure(6, "amp_page", str(e))

    try:
        rss_success, feed_url, feed_items = await _try_fetch_rss_feed(url)
        if rss_success and feed_items:
            md = f"# RSS Feed Content\n\nFeed URL: {feed_url}\n\n"
            for item in feed_items[:20]:
                if item.get("title"):
                    md += f"## {item['title']}\n"
                if item.get("link"):
                    md += f"[Link]({item['link']})\n\n"
                if item.get("description"):
                    md += f"{item['description']}\n\n"
                md += "---\n\n"
            if _is_block_page(md):
                print("  Block page detected in RSS content, skipping")
            else:
                _rec_success(6, "rss_feed")
                rss_response = CrawlResponse(
                    success=True, url=url, markdown=md,
                    content=json.dumps(feed_items, ensure_ascii=False),
                    extracted_data={"fallback_strategy_used": "rss_feed", "fallback_stage": 6,
                                    "feed_url": feed_url, "item_count": len(feed_items), "content_source": "rss"})
                rss_response = await _finalize_fallback_response(
                    rss_response, url, auto_summarize, max_content_tokens, llm_provider, llm_model)
                return _save_session(rss_response)
    except Exception as e:
        print(f"RSS fallback failed: {e}")
        _rec_failure(6, "rss_feed", str(e))

    # Stage 7: JSON extraction from cached static HTML
    print(f"Stage 7/{total_stages}: Attempting JSON extraction from static HTML")
    if static_success and static_html:
        json_success, json_data, json_source = _extract_spa_json_data(static_html)
        if json_success and json_data:
            _rec_success(7, "json_extraction")
            stage7_response = await _build_json_extraction_response(
                json_data=json_data, json_source=json_source, url=url,
                strategy_name="json_extraction", stage=7,
                auto_summarize=auto_summarize, max_content_tokens=max_content_tokens,
                llm_provider=llm_provider, llm_model=llm_model,
                extract_content_keys=["props", "pageProps", "data", "content", "article", "post"])
            return _save_session(stage7_response)

    if static_success and static_html:
        _rec_failure(7, "json_extraction", "no_json_data")
    # All strategies failed
    all_strats = ["static_fast_path"] + [s["name"] for s in strategies] + ["amp_page", "rss_feed", "json_extraction"]
    return CrawlResponse(
        success=False, url=url,
        error=(f"All {total_stages} fallback stages failed. Last error: {last_error}. "
               f"This site may have strong anti-bot protection. Stages attempted: {', '.join(all_strats)}"),
        extracted_data={
            "fallback_strategies_attempted": all_strats, "total_stages": total_stages,
            "site_type_detected": site_type,
            "static_fetch_result": "success" if static_success else f"failed: {static_error}",
            "recommendations": ["Try accessing the site manually to check if it's available",
                "Consider using the site's API if available", "Try accessing during off-peak hours",
                "Use a VPN if the site is geo-blocked", "Check if the site has a mobile or AMP version"],
        })
