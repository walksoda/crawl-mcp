"""Constants for Crawl4AI MCP Server.

This module centralizes all constants used throughout the codebase to ensure
consistency and make configuration changes easier.
"""

from typing import Dict, Any

# =============================================================================
# Token Limits
# =============================================================================

# Maximum tokens for MCP response to prevent Claude Code errors
MAX_RESPONSE_TOKENS = 100000

# Default maximum content tokens before auto-summarization
MAX_CONTENT_TOKENS_DEFAULT = 15000

# Token estimation ratios (characters per token)
TOKEN_ESTIMATE_CHARS_PER_TOKEN_EN = 4  # English: ~4 chars per token
TOKEN_ESTIMATE_CHARS_PER_TOKEN_JA = 2  # Japanese: ~2 chars per token

# Threshold for detecting Japanese-heavy text (ratio of Japanese characters)
JAPANESE_DETECTION_THRESHOLD = 0.3

# =============================================================================
# Summary Length Configurations
# =============================================================================

SUMMARY_LENGTH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "short": {
        "target_length": "2-3 paragraphs",
        "detail_level": "key points and main findings only",
        "target_tokens": 400,
    },
    "medium": {
        "target_length": "4-6 paragraphs",
        "detail_level": "comprehensive overview with important details",
        "target_tokens": 1000,
    },
    "long": {
        "target_length": "8-12 paragraphs",
        "detail_level": "detailed analysis with examples and context",
        "target_tokens": 2000,
    },
}

# =============================================================================
# Fallback Constants
# =============================================================================

# Minimum content length to consider meaningful
FALLBACK_MIN_CONTENT_LENGTH = 200

# Block page indicators (lowercase patterns)
BLOCK_INDICATORS = [
    "access denied",
    "403 forbidden",
    "captcha",
    "please verify",
    "security check",
    "blocked",
    "rate limit",
    "robot",
    "cloudflare",
    "ddos protection",
]

# =============================================================================
# Content Field Configurations for Token Limiting
# =============================================================================

# Priority order for content truncation (field_name, max_tokens)
CONTENT_FIELD_PRIORITIES = [
    ("content", 8000),           # Main content - keep substantial portion
    ("markdown", 6000),          # Markdown version
    ("raw_content", 2000),       # Raw text content
    ("text", 2000),              # Extracted text
    ("results", 4000),           # Search/crawl results
    ("chunks", 2000),            # Content chunks
    ("extracted_data", 1500),    # Structured data
    ("summary", 1500),           # Summary content
    ("final_summary", 1000),     # Final summary
    ("metadata", 500),           # Metadata - keep small portion
    ("entities", 1000),          # Extracted entities
    ("table_data", 1500),        # Table data
]

# Essential fields to always preserve in emergency truncation
ESSENTIAL_FIELDS = [
    "success",
    "url",
    "error",
    "title",
    "file_type",
    "processing_method",
    "query",
    "search_query",
    "video_id",
    "language_info",
]

# =============================================================================
# LLM Provider Defaults
# =============================================================================

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4o"

# Supported LLM providers
SUPPORTED_LLM_PROVIDERS = ["openai", "anthropic", "ollama", "aoai"]

# Default temperature for summarization
SUMMARIZATION_TEMPERATURE = 0.3

# Default timeout for LLM API calls (seconds)
LLM_API_TIMEOUT = 120

# Content truncation limits for LLM prompts
MAX_CONTENT_FOR_LLM = 50000  # Maximum characters to send to LLM

# =============================================================================
# File Processing Constants
# =============================================================================

# Maximum file size in MB
MAX_FILE_SIZE_MB = 100

# Supported file extensions by category
SUPPORTED_FILE_EXTENSIONS = {
    "pdf": [".pdf"],
    "word": [".docx", ".doc"],
    "excel": [".xlsx", ".xls"],
    "powerpoint": [".pptx", ".ppt"],
    "archive": [".zip"],
}

# =============================================================================
# YouTube Constants
# =============================================================================

# Default languages for transcript extraction (priority order)
DEFAULT_YOUTUBE_LANGUAGES = ["ja", "en"]

# Maximum URLs for batch transcript extraction
MAX_BATCH_YOUTUBE_URLS = 3
