"""
Web crawling tools facade.

Re-exports from core/ modules for backward compatibility.
"""

from ..core.crawler_summarizer import summarize_web_content, MAX_RESPONSE_CHARS
from ..core.crawler_core import _internal_crawl_url
from ..core.crawl_url import crawl_url, deep_crawl_site
from ..core.crawler_fallback import crawl_url_with_fallback
from ..core.extraction_intelligent import _internal_intelligent_extract, intelligent_extract
from ..core.extraction_entity import (
    _regex_worker,
    _safe_regex_findall,
    _internal_extract_entities,
    _internal_llm_extract_entities,
    extract_entities,
)
from ..core.extraction_structured import _internal_extract_structured_data, extract_structured_data
from ..core.crawler_summarizer import _check_and_summarize_if_needed, _finalize_fallback_response
from ..core.crawler_io import (
    _process_response_content,
    _handle_youtube_url,
    _handle_file_url,
    _build_json_extraction_response,
)

# Re-export models that were previously imported here
from ..models import CrawlRequest, CrawlResponse

__all__ = [
    'summarize_web_content', 'MAX_RESPONSE_CHARS',
    '_internal_crawl_url', 'crawl_url', 'deep_crawl_site',
    'crawl_url_with_fallback',
    '_internal_intelligent_extract', 'intelligent_extract',
    '_regex_worker', '_safe_regex_findall',
    '_internal_extract_entities', '_internal_llm_extract_entities', 'extract_entities',
    '_internal_extract_structured_data', 'extract_structured_data',
    '_check_and_summarize_if_needed', '_finalize_fallback_response',
    '_process_response_content', '_handle_youtube_url', '_handle_file_url',
    '_build_json_extraction_response',
    'CrawlRequest', 'CrawlResponse',
]
