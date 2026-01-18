"""
Tools module for Crawl4AI MCP Server.

Contains all MCP tool implementations organized by functionality.
"""

# YouTube tools
from .youtube import (
    extract_youtube_transcript,
    batch_extract_youtube_transcripts,
    get_youtube_video_info,
    get_youtube_api_setup_guide
)

# File processing tools
from .file_processing import (
    process_file,
    get_supported_file_formats,
    enhanced_process_large_content
)

# Web crawling tools
from .web_crawling import (
    crawl_url,
    deep_crawl_site,
    crawl_url_with_fallback,
    intelligent_extract,
    extract_entities,
    extract_structured_data
)

# Session management (refactored module)
from .session_manager import (
    SessionManager,
    get_session_manager,
    extract_cookies_from_result,
)

# Strategy cache (refactored module)
from .strategy_cache import (
    StrategyCache,
    FingerprintProfile,
    get_strategy_cache,
    get_fingerprint_config,
)

# Content processors (refactored module)
from .content_processors import (
    convert_media_to_list,
    has_meaningful_content,
    is_block_page,
    get_content_length,
    extract_text_content,
    truncate_content,
    normalize_url,
    extract_domain,
    clean_html_content,
    extract_links,
    estimate_reading_time,
)

# Fallback strategies (refactored module)
from .fallback_strategies import (
    normalize_cookies_to_playwright_format,
    static_fetch_content,
    extract_spa_json_data,
    detect_spa_framework,
    build_amp_url,
    try_fetch_rss_feed,
    get_fallback_stage_info,
)

# Search tools
from .search import (
    search_google,
    batch_search_google, 
    search_and_crawl,
    get_search_genres
)

# Utility tools
from .utilities import (
    get_llm_config_info,
    batch_crawl,
    get_tool_selection_guide
)

__all__ = [
    # YouTube tools
    'extract_youtube_transcript',
    'batch_extract_youtube_transcripts',
    'get_youtube_video_info',
    'get_youtube_api_setup_guide',

    # File processing tools
    'process_file',
    'get_supported_file_formats',
    'enhanced_process_large_content',

    # Web crawling tools
    'crawl_url',
    'deep_crawl_site',
    'crawl_url_with_fallback',
    'intelligent_extract',
    'extract_entities',
    'extract_structured_data',

    # Session management (refactored)
    'SessionManager',
    'get_session_manager',
    'extract_cookies_from_result',

    # Strategy cache (refactored)
    'StrategyCache',
    'FingerprintProfile',
    'get_strategy_cache',
    'get_fingerprint_config',

    # Content processors (refactored)
    'convert_media_to_list',
    'has_meaningful_content',
    'is_block_page',
    'get_content_length',
    'extract_text_content',
    'truncate_content',
    'normalize_url',
    'extract_domain',
    'clean_html_content',
    'extract_links',
    'estimate_reading_time',

    # Fallback strategies (refactored)
    'normalize_cookies_to_playwright_format',
    'static_fetch_content',
    'extract_spa_json_data',
    'detect_spa_framework',
    'build_amp_url',
    'try_fetch_rss_feed',
    'get_fallback_stage_info',

    # Search tools
    'search_google',
    'batch_search_google',
    'search_and_crawl',
    'get_search_genres',

    # Utility tools
    'get_llm_config_info',
    'batch_crawl',
    'get_tool_selection_guide',
]