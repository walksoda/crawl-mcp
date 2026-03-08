"""Batch crawling operations for processing multiple URLs sequentially."""

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

    Process URLs sequentially for stability (avoids headless browser resource contention).

    Parameters:
    - urls: List of URLs to crawl (required)
    - config: Optional configuration parameters (default: None)
    - base_timeout: Per-URL timeout in seconds (default: 30)

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
    default_config.pop("max_concurrent", None)  # Not used in sequential mode
    default_config["timeout"] = base_timeout  # Per-URL timeout (not scaled by URL count)

    results = []
    for url in urls:
        try:
            request = CrawlRequest(url=url, **default_config)
            result = await asyncio.wait_for(
                _internal_crawl_url(request),
                timeout=base_timeout
            )
            results.append(result)
        except asyncio.TimeoutError:
            results.append(CrawlResponse(
                success=False,
                url=url,
                error=f"URL timeout after {base_timeout}s: {url}"
            ))
        except Exception as e:
            results.append(CrawlResponse(
                success=False,
                url=url,
                error=f"Batch crawl error for {url}: {str(e)}"
            ))

    return results
