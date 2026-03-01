"""Enhanced large content processing with chunking and filtering."""

import time
from typing import Any, Dict, List, Optional

from ..models import CrawlRequest, LargeContentRequest, LargeContentResponse
from .crawler_core import _internal_crawl_url
from .file_chunking import (
    CustomSentenceChunking,
    BM25SimilarityFilter,
    AdaptiveChunking,
    AdaptiveFiltering,
)
from .file_processing import summarize_content
from crawl4ai.chunking_strategy import OverlappingWindowChunking, RegexChunking


async def enhanced_process_large_content(
    url: str,
    chunking_strategy: str = "sentence",
    filtering_strategy: str = "bm25",
    filter_query: Optional[str] = None,
    max_chunk_tokens: int = 2000,
    chunk_overlap: int = 200,
    similarity_threshold: float = 0.5,
    extract_top_chunks: int = 5,
    summarize_chunks: bool = False,
    merge_strategy: str = "linear",
    final_summary_length: str = "short"
) -> LargeContentResponse:
    """Process large content with chunking and BM25 filtering."""
    start_time = time.time()

    try:
        # Validate input parameters
        valid_chunking = ["topic", "sentence", "overlap", "regex"]
        valid_filtering = ["bm25", "pruning", "llm"]
        valid_summary_lengths = ["short", "medium", "long"]
        valid_merge_strategies = ["hierarchical", "linear"]

        if chunking_strategy not in valid_chunking:
            return LargeContentResponse(
                success=False,
                url=url,
                original_content_length=0,
                filtered_content_length=0,
                total_chunks=0,
                relevant_chunks=0,
                processing_method="enhanced_large_content",
                chunking_strategy_used=chunking_strategy,
                filtering_strategy_used=filtering_strategy,
                chunks=[],
                metadata={},
                processing_stats={},
                error=f"Invalid chunking strategy. Must be one of: {valid_chunking}"
            )

        if filtering_strategy not in valid_filtering:
            return LargeContentResponse(
                success=False,
                url=url,
                original_content_length=0,
                filtered_content_length=0,
                total_chunks=0,
                relevant_chunks=0,
                processing_method="enhanced_large_content",
                chunking_strategy_used=chunking_strategy,
                filtering_strategy_used=filtering_strategy,
                chunks=[],
                metadata={},
                processing_stats={},
                error=f"Invalid filtering strategy. Must be one of: {valid_filtering}"
            )

        if merge_strategy not in valid_merge_strategies:
            merge_strategy = "hierarchical"  # Default fallback

        # Step 1: First, crawl the URL to get the content
        try:
            # Create a crawl request for the URL
            crawl_request = CrawlRequest(
                url=url,
                generate_markdown=True,
                auto_summarize=False,  # We'll handle summarization ourselves
                extract_media=False,
                content_filter=None,  # We'll apply filtering after chunking
                wait_for_js=False,
                timeout=120
            )

            # Get the content using internal crawl function
            crawl_result = await _internal_crawl_url(crawl_request)

            if not crawl_result.success:
                return LargeContentResponse(
                    success=False,
                    url=url,
                    original_content_length=0,
                    filtered_content_length=0,
                    total_chunks=0,
                    relevant_chunks=0,
                    processing_method="enhanced_large_content",
                    chunking_strategy_used=chunking_strategy,
                    filtering_strategy_used=filtering_strategy,
                    chunks=[],
                    metadata={},
                    processing_stats={"processing_time": time.time() - start_time},
                    error=f"Failed to crawl URL: {crawl_result.error}"
                )

            original_content = crawl_result.markdown or crawl_result.cleaned_html or ""
            if not original_content:
                return LargeContentResponse(
                    success=False,
                    url=url,
                    original_content_length=0,
                    filtered_content_length=0,
                    total_chunks=0,
                    relevant_chunks=0,
                    processing_method="enhanced_large_content",
                    chunking_strategy_used=chunking_strategy,
                    filtering_strategy_used=filtering_strategy,
                    chunks=[],
                    metadata={},
                    processing_stats={"processing_time": time.time() - start_time},
                    error="No content extracted from URL"
                )

        except Exception as e:
            return LargeContentResponse(
                success=False,
                url=url,
                original_content_length=0,
                filtered_content_length=0,
                total_chunks=0,
                relevant_chunks=0,
                processing_method="enhanced_large_content",
                chunking_strategy_used=chunking_strategy,
                filtering_strategy_used=filtering_strategy,
                chunks=[],
                metadata={},
                processing_stats={"processing_time": time.time() - start_time},
                error=f"Error during content retrieval: {str(e)}"
            )

        original_length = len(original_content)

        # Step 2: Choose and apply chunking strategy
        adaptive_chunking = AdaptiveChunking()

        if chunking_strategy == "sentence":
            # Use custom sentence chunking
            sentence_chunker = CustomSentenceChunking(max_sentences_per_chunk=10)
            chunks = sentence_chunker.chunk(original_content)
            actual_chunking_strategy = "sentence"
        elif chunking_strategy == "topic":
            # Use adaptive chunking which will choose topic segmentation
            strategy, actual_chunking_strategy = adaptive_chunking.get_optimal_strategy(
                original_content, url, max_chunk_tokens, chunk_overlap
            )
            # Apply chunking using Crawl4AI's chunking strategy
            chunks = strategy.chunk(original_content)
        elif chunking_strategy == "overlap":
            # Force overlapping window chunking
            strategy = OverlappingWindowChunking(
                window_size=max_chunk_tokens,
                overlap=chunk_overlap
            )
            chunks = strategy.chunk(original_content)
            actual_chunking_strategy = "overlap"
        elif chunking_strategy == "regex":
            # Use regex-based chunking (split on double newlines)
            strategy = RegexChunking(patterns=[r'\n\n+'])
            chunks = strategy.chunk(original_content)
            actual_chunking_strategy = "regex"
        else:
            # Default to adaptive
            strategy, actual_chunking_strategy = adaptive_chunking.get_optimal_strategy(
                original_content, url, max_chunk_tokens, chunk_overlap
            )
            chunks = strategy.chunk(original_content)

        total_chunks = len(chunks)

        # Step 3: Apply filtering to chunks
        if filtering_strategy == "bm25" and filter_query:
            # Use BM25-style filtering
            bm25_filter = BM25SimilarityFilter(
                query=filter_query,
                similarity_threshold=similarity_threshold,
                max_chunks=extract_top_chunks
            )
            filtered_chunks = bm25_filter.filter_chunks(chunks)
            actual_filtering_strategy = "bm25"
        else:
            # Default: return top chunks by length (simple heuristic)
            filtered_chunks = []
            chunk_lengths = [(i, len(chunk)) for i, chunk in enumerate(chunks)]
            chunk_lengths.sort(key=lambda x: x[1], reverse=True)

            for i, (chunk_idx, length) in enumerate(chunk_lengths[:extract_top_chunks]):
                filtered_chunks.append({
                    'chunk_id': chunk_idx,
                    'content': chunks[chunk_idx],
                    'score': 1.0 - (i * 0.1),  # Decreasing score
                    'length': length
                })
            actual_filtering_strategy = "length_based"

        relevant_chunks_count = len(filtered_chunks)
        filtered_content_length = sum(chunk['length'] for chunk in filtered_chunks)

        # Step 4: Summarize chunks if requested
        processed_chunks = []

        if summarize_chunks and filtered_chunks:
            for chunk_info in filtered_chunks:
                try:
                    # Summarize each chunk
                    chunk_summary = await summarize_content(
                        content=chunk_info['content'],
                        title=f"Chunk {chunk_info['chunk_id']}",
                        url=url,
                        summary_length="short",  # Individual chunks get short summaries
                        content_type="chunk",
                        target_tokens=300  # Short summaries for chunks
                    )

                    processed_chunks.append({
                        'chunk_id': chunk_info['chunk_id'],
                        'original_content': chunk_info['content'],
                        'original_length': chunk_info['length'],
                        'score': chunk_info['score'],
                        'summary': chunk_summary.get('summary', chunk_info['content'][:200] + '...') if chunk_summary.get('success') else chunk_info['content'][:200] + '...',
                        'key_topics': chunk_summary.get('key_topics', []) if chunk_summary.get('success') else [],
                        'summarization_success': chunk_summary.get('success', False)
                    })
                except Exception as e:
                    # If summarization fails, use truncated content
                    processed_chunks.append({
                        'chunk_id': chunk_info['chunk_id'],
                        'original_content': chunk_info['content'],
                        'original_length': chunk_info['length'],
                        'score': chunk_info['score'],
                        'summary': chunk_info['content'][:200] + '...',
                        'key_topics': [],
                        'summarization_success': False,
                        'summarization_error': str(e)
                    })
        else:
            # Just use the filtered chunks as-is
            for chunk_info in filtered_chunks:
                processed_chunks.append({
                    'chunk_id': chunk_info['chunk_id'],
                    'original_content': chunk_info['content'],
                    'original_length': chunk_info['length'],
                    'score': chunk_info['score'],
                    'summary': chunk_info['content'],
                    'key_topics': [],
                    'summarization_success': True
                })

        # Step 5: Create final summary if requested
        final_summary = None
        if processed_chunks:
            try:
                # Combine chunk summaries for final summary
                if merge_strategy == "hierarchical":
                    # Group chunks by score and summarize hierarchically
                    high_score_chunks = [c for c in processed_chunks if c['score'] > 0.7]
                    mid_score_chunks = [c for c in processed_chunks if 0.3 <= c['score'] <= 0.7]

                    combined_content = ""
                    if high_score_chunks:
                        combined_content += "## High Priority Content\n\n"
                        combined_content += "\n\n".join([c['summary'] for c in high_score_chunks])

                    if mid_score_chunks:
                        combined_content += "\n\n## Additional Content\n\n"
                        combined_content += "\n\n".join([c['summary'] for c in mid_score_chunks[:5]])  # Limit additional content
                else:
                    # Linear merge strategy
                    combined_content = "\n\n".join([c['summary'] for c in processed_chunks])

                # Create final summary
                final_summary_result = await summarize_content(
                    content=combined_content,
                    title=crawl_result.title or "Large Content Analysis",
                    url=url,
                    summary_length=final_summary_length,
                    content_type="analysis",
                    target_tokens=2000 if final_summary_length == "long" else 1000  # Appropriate token target
                )

                if final_summary_result.get('success'):
                    final_summary = final_summary_result

            except Exception as e:
                final_summary = {
                    'success': False,
                    'error': f'Final summarization failed: {str(e)}'
                }

        # Step 6: Build metadata
        processing_time = time.time() - start_time
        metadata = {
            'original_title': crawl_result.title,
            'original_url': url,
            'content_type': 'large_content_analysis',
            'processing_method': 'enhanced_large_content',
            'chunking_details': {
                'strategy_requested': chunking_strategy,
                'strategy_used': actual_chunking_strategy,
                'max_chunk_tokens': max_chunk_tokens,
                'chunk_overlap': chunk_overlap,
                'total_chunks_created': total_chunks
            },
            'filtering_details': {
                'strategy_requested': filtering_strategy,
                'strategy_used': actual_filtering_strategy,
                'filter_query': filter_query,
                'similarity_threshold': similarity_threshold,
                'chunks_requested': extract_top_chunks,
                'chunks_found': relevant_chunks_count
            },
            'summarization_details': {
                'individual_chunks_summarized': summarize_chunks,
                'merge_strategy': merge_strategy,
                'final_summary_length': final_summary_length,
                'final_summary_success': final_summary.get('success', False) if final_summary else False
            },
            'performance_metrics': {
                'processing_time_seconds': processing_time,
                'original_content_size': original_length,
                'filtered_content_size': filtered_content_length,
                'compression_ratio': filtered_content_length / original_length if original_length > 0 else 0,
                'chunks_processed_per_second': relevant_chunks_count / processing_time if processing_time > 0 else 0
            }
        }

        # Step 7: Build processing stats
        processing_stats = {
            'processing_time': processing_time,
            'content_reduction_ratio': (original_length - filtered_content_length) / original_length if original_length > 0 else 0,
            'chunks_processed': relevant_chunks_count,
            'average_chunk_size': filtered_content_length / relevant_chunks_count if relevant_chunks_count > 0 else 0,
            'summarization_success_rate': sum(1 for c in processed_chunks if c.get('summarization_success', False)) / len(processed_chunks) if processed_chunks else 0
        }

        return LargeContentResponse(
            success=True,
            url=url,
            original_content_length=original_length,
            filtered_content_length=filtered_content_length,
            total_chunks=total_chunks,
            relevant_chunks=relevant_chunks_count,
            processing_method="enhanced_large_content",
            chunking_strategy_used=actual_chunking_strategy,
            filtering_strategy_used=actual_filtering_strategy,
            chunks=processed_chunks,
            final_summary=final_summary,
            metadata=metadata,
            processing_stats=processing_stats
        )

    except Exception as e:
        return LargeContentResponse(
            success=False,
            url=url,
            original_content_length=0,
            filtered_content_length=0,
            total_chunks=0,
            relevant_chunks=0,
            processing_method="enhanced_large_content",
            chunking_strategy_used=chunking_strategy,
            filtering_strategy_used=filtering_strategy,
            chunks=[],
            metadata={},
            processing_stats={"processing_time": time.time() - start_time},
            error=f"Large content processing error: {str(e)}"
        )
