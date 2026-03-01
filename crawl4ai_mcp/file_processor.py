"""
File Processing Module - Backward compatibility facade.
Implementation moved to crawl4ai_mcp.processors.file_processor
"""

from .processors.file_processor import *  # noqa: F401, F403
from .processors.file_processor import CONTENT_TYPE_TO_EXT, FileProcessor  # noqa: F401
