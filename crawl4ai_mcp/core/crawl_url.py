"""
Public crawl_url and deep_crawl_site wrappers.

These create CrawlRequest objects from individual parameters and delegate
to _internal_crawl_url.
"""

from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

from ..models import CrawlRequest, CrawlResponse
from .crawler_core import _internal_crawl_url


async def crawl_url(
    url: str,
    css_selector: Optional[str] = None,
    extract_media: bool = False,
    take_screenshot: bool = False,
    generate_markdown: bool = True,
    include_cleaned_html: bool = False,
    wait_for_selector: Optional[str] = None,
    timeout: int = 60,
    max_depth: Optional[int] = None,
    max_pages: Optional[int] = 10,
    include_external: bool = False,
    crawl_strategy: str = "bfs",
    url_pattern: Optional[str] = None,
    score_threshold: float = 0.3,
    content_filter: Optional[str] = None,
    filter_query: Optional[str] = None,
    chunk_content: bool = False,
    chunk_strategy: str = "topic",
    chunk_size: int = 1000,
    overlap_rate: float = 0.1,
    user_agent: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    enable_caching: bool = True,
    cache_mode: str = "enabled",
    execute_js: Optional[str] = None,
    wait_for_js: bool = False,
    simulate_user: bool = False,
    use_undetected_browser: bool = False,
    auth_token: Optional[str] = None,
    cookies: Optional[Dict[str, str]] = None,
    auto_summarize: bool = False,
    max_content_tokens: int = 15000,
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> CrawlResponse:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.
    """
    request = CrawlRequest(
        url=url, css_selector=css_selector, extract_media=extract_media,
        take_screenshot=take_screenshot, generate_markdown=generate_markdown,
        include_cleaned_html=include_cleaned_html, wait_for_selector=wait_for_selector,
        timeout=timeout, max_depth=max_depth, max_pages=max_pages,
        include_external=include_external, crawl_strategy=crawl_strategy,
        url_pattern=url_pattern, score_threshold=score_threshold,
        content_filter=content_filter, filter_query=filter_query,
        chunk_content=chunk_content, chunk_strategy=chunk_strategy,
        chunk_size=chunk_size, overlap_rate=overlap_rate,
        user_agent=user_agent, headers=headers,
        enable_caching=enable_caching, cache_mode=cache_mode,
        execute_js=execute_js, wait_for_js=wait_for_js,
        simulate_user=simulate_user, use_undetected_browser=use_undetected_browser,
        auth_token=auth_token, cookies=cookies,
        auto_summarize=auto_summarize, max_content_tokens=max_content_tokens,
        summary_length=summary_length, llm_provider=llm_provider, llm_model=llm_model
    )
    return await _internal_crawl_url(request)


async def deep_crawl_site(
    url: str,
    max_depth: int = 2,
    max_pages: int = 5,
    crawl_strategy: str = "bfs",
    include_external: bool = False,
    url_pattern: Optional[str] = None,
    score_threshold: float = 0.0,
    extract_media: bool = False,
    base_timeout: int = 60
) -> Dict[str, Any]:
    """Crawl multiple pages from a site with configurable depth."""
    request = CrawlRequest(
        url=url, max_depth=max_depth, max_pages=max_pages,
        crawl_strategy=crawl_strategy, include_external=include_external,
        url_pattern=url_pattern, score_threshold=score_threshold,
        extract_media=extract_media, timeout=base_timeout, generate_markdown=True
    )

    result = await _internal_crawl_url(request)

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
