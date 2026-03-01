"""Chunking and filtering utility classes for large content processing."""

from typing import Any, Dict, List, Optional

from crawl4ai import BM25ContentFilter, PruningContentFilter, LLMContentFilter
from crawl4ai.chunking_strategy import (
    TopicSegmentationChunking,
    OverlappingWindowChunking,
    RegexChunking,
)


class CustomSentenceChunking:
    """Custom sentence-based chunking implementation"""

    def __init__(self, max_sentences_per_chunk: int = 5):
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def chunk(self, text: str) -> List[str]:
        """Split text into sentence-based chunks"""
        # Simple sentence splitting - improved implementation
        import re
        # Split on sentence endings, keeping the periods
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class BM25SimilarityFilter:
    """BM25-based similarity filter for content chunks"""

    def __init__(self, query: str, similarity_threshold: float = 0.5, max_chunks: int = 10):
        self.query = query.lower().split() if query else []
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks

    def _calculate_bm25_score(self, chunk: str) -> float:
        """Calculate simple BM25-like score"""
        if not self.query:
            return 1.0

        chunk_words = chunk.lower().split()
        if not chunk_words:
            return 0.0

        # Simple scoring based on query term frequency
        score = 0.0
        for term in self.query:
            term_freq = chunk_words.count(term)
            if term_freq > 0:
                # Simple BM25-like formula
                score += (term_freq * 2.2) / (term_freq + 1.2)

        return score / len(self.query) if self.query else 0.0

    def filter_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Filter chunks based on BM25-like similarity to query"""
        if not chunks:
            return []

        # Score all chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            score = self._calculate_bm25_score(chunk)
            if score >= self.similarity_threshold:
                scored_chunks.append({
                    'chunk_id': i,
                    'content': chunk,
                    'score': score,
                    'length': len(chunk)
                })

        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:self.max_chunks]


class AdaptiveChunking:
    """Adaptive chunking strategy selector"""

    def get_optimal_strategy(self, content: str, url: str = "",
                           max_chunk_tokens: int = 8000, chunk_overlap: int = 500):
        """Select optimal chunking strategy based on content analysis"""
        content_length = len(content)

        # Analyze content characteristics
        if content_length > 100000:  # Very large content
            return TopicSegmentationChunking(num_keywords=8), "topic"
        elif content_length > 50000:  # Large content
            return TopicSegmentationChunking(num_keywords=5), "topic"
        elif ".pdf" in url.lower() or "academic" in content.lower():
            # Academic/structured content benefits from topic segmentation
            return TopicSegmentationChunking(num_keywords=3), "topic"
        else:
            # Default to overlapping windows
            window_size = min(max_chunk_tokens, content_length // 10)
            return OverlappingWindowChunking(window_size=window_size, overlap=chunk_overlap), "overlap"


class AdaptiveFiltering:
    """Adaptive filtering strategy selector"""

    def get_optimal_filter(self, content: str, filter_query: str = "", url: str = ""):
        """Select optimal filtering strategy based on content analysis"""
        if filter_query and len(filter_query.strip()) > 0:
            # Use BM25 when query is provided
            return BM25ContentFilter(user_query=filter_query, bm25_threshold=1.0, language='english'), "bm25"
        elif len(content) > 200000:  # Very large content
            # Use pruning for very large content
            return PruningContentFilter(threshold=0.5, threshold_type="percentile"), "pruning"
        else:
            # Use basic pruning as default
            return PruningContentFilter(), "pruning"
