"""Batch crawling operations for processing multiple URLs concurrently."""

import asyncio
from typing import Any, Dict, List, Optional

from ..models import CrawlResponse, CrawlRequest
from .crawler_core import _internal_crawl_url


async def batch_crawl(
    urls: List[str],
    config: Optional[Dict[str, Any]] = None,
    base_timeout: int = 30
) -> List[CrawlResponse]:
    """
    Crawl multiple URLs in batch.

    Process multiple URLs concurrently for efficiency. Timeout auto-scales based on URL count.

    Parameters:
    - urls: List of URLs to crawl (required)
    - config: Optional configuration parameters (default: None)
    - base_timeout: Base timeout in seconds, adjusted based on URL count (default: 30)

    Example:
    {"urls": ["https://example.com/page1", "https://example.com/page2"], "config": {"generate_markdown": true}}

    Returns List of CrawlResponse objects for each URL.
    """
    if not urls:
        return []

    # Prepare default configuration
    default_config = {
        "generate_markdown": True,
        "extract_media": False,
        "wait_for_js": False,
        "auto_summarize": False
    }

    if config:
        default_config.update(config)

    # Calculate dynamic timeout based on URL count
    # Base timeout + additional time per URL (10s per additional URL after the first)
    dynamic_timeout = base_timeout + max(0, (len(urls) - 1) * 10)
    default_config["timeout"] = dynamic_timeout

    # Limit concurrent processing to be respectful to servers and prevent hangs
    max_concurrent = min(len(urls), 2)  # Reduced to 2 concurrent requests for stability
    semaphore = asyncio.Semaphore(max_concurrent)

    async def crawl_single_url(url: str) -> CrawlResponse:
        async with semaphore:
            try:
                # Create crawl request with merged configuration
                request = CrawlRequest(url=url, **default_config)
                # Add individual URL timeout to prevent hangs
                result = await asyncio.wait_for(
                    _internal_crawl_url(request),
                    timeout=dynamic_timeout
                )
                return result
            except asyncio.TimeoutError:
                # Handle individual URL timeouts
                return CrawlResponse(
                    success=False,
                    url=url,
                    error=f"Individual URL timeout after {dynamic_timeout}s: {url}"
                )
            except Exception as e:
                # Return error response for failed URLs
                return CrawlResponse(
                    success=False,
                    url=url,
                    error=f"Batch crawl error for {url}: {str(e)}"
                )

    # Process all URLs concurrently with semaphore control and global timeout
    tasks = [crawl_single_url(url) for url in urls]
    try:
        # Add overall batch timeout to prevent infinite hangs
        batch_timeout = dynamic_timeout * 2  # Allow extra time for batch processing
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=batch_timeout
        )
    except asyncio.TimeoutError:
        # If batch times out, return timeout errors for all URLs
        results = [
            CrawlResponse(
                success=False,
                url=url,
                error=f"Batch timeout after {batch_timeout}s"
            ) for url in urls
        ]

    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(CrawlResponse(
                success=False,
                url=urls[i],
                error=f"Batch processing exception: {str(result)}"
            ))
        else:
            processed_results.append(result)

    return processed_results
