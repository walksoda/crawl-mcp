"""Utility modules for Crawl4AI MCP Server.

This package provides common utilities used across the codebase:
- time_parser: Time string parsing and formatting
- token_utils: Token estimation and content size management
- llm_client: Unified LLM client for summarization
- llm_extraction: Unified LLM client for extraction tasks
"""

from .time_parser import parse_time_period
from .token_utils import (
    estimate_tokens,
    estimate_tokens_fallback,
    truncate_content,
    apply_token_limit,
)
from .llm_client import (
    LLMClient,
    get_llm_config_safe,
    summarize_content,
)
from .llm_extraction import (
    LLMExtractionClient,
    extract_with_llm,
)

__all__ = [
    # time_parser
    "parse_time_period",
    # token_utils
    "estimate_tokens",
    "estimate_tokens_fallback",
    "truncate_content",
    "apply_token_limit",
    # llm_client
    "LLMClient",
    "get_llm_config_safe",
    "summarize_content",
    # llm_extraction
    "LLMExtractionClient",
    "extract_with_llm",
]
