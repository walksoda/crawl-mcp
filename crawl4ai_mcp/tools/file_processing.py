"""
File processing tools facade.

Re-exports from core/ modules for backward compatibility.
"""

from ..core.file_chunking import (
    CustomSentenceChunking,
    BM25SimilarityFilter,
    AdaptiveChunking,
    AdaptiveFiltering,
)
from ..core.file_processing import (
    summarize_content,
    process_file,
    get_supported_file_formats,
)
from ..core.file_large_content import enhanced_process_large_content

__all__ = [
    'CustomSentenceChunking', 'BM25SimilarityFilter',
    'AdaptiveChunking', 'AdaptiveFiltering',
    'summarize_content', 'process_file', 'get_supported_file_formats',
    'enhanced_process_large_content',
]
