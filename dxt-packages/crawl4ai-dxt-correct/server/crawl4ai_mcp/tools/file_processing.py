"""
File processing tools for Crawl4AI MCP Server.

Contains complete file processing, document conversion, and large content processing tools.
"""

import time
import asyncio
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import (
    FileProcessRequest,
    FileProcessResponse,
    LargeContentRequest,
    LargeContentResponse
)

# Import required crawl4ai components
from crawl4ai import AsyncWebCrawler
from crawl4ai import (
    BM25ContentFilter,
    PruningContentFilter,
    LLMContentFilter,
    CacheMode,
)
from crawl4ai.chunking_strategy import (
    TopicSegmentationChunking,
    OverlappingWindowChunking,
    RegexChunking
)

# Import file processor
from ..file_processor import FileProcessor

# Import the internal crawl function for enhanced processing
from .web_crawling import _internal_crawl_url

# Initialize file processor
file_processor = FileProcessor()


# Utility classes for enhanced processing
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


# Summarization function using existing capabilities
async def summarize_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    content_type: str = "document",
    target_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """Summarize content using LLM with enhanced metadata preservation"""
    try:
        # Import config for LLM
        try:
            from ..config import get_llm_config
        except ImportError:
            return {
                "success": False,
                "error": "LLM configuration not available"
            }
        
        # Define summary configurations with enhanced token targets
        length_configs = {
            "short": {
                "target_length": "2-3 paragraphs",
                "detail_level": "key points and main findings only",
                "target_tokens": target_tokens or 400
            },
            "medium": {
                "target_length": "4-6 paragraphs", 
                "detail_level": "comprehensive overview with important details",
                "target_tokens": target_tokens or 1000
            },
            "long": {
                "target_length": "8-12 paragraphs",
                "detail_level": "detailed analysis with examples and context",
                "target_tokens": target_tokens or 2000
            }
        }
        
        config = length_configs.get(summary_length, length_configs["medium"])
        
        # Extract file information for context
        file_extension = ""
        filename = ""
        if url:
            try:
                import os
                filename = os.path.basename(url)
                file_extension = os.path.splitext(filename)[1]
            except:
                filename = url
        
        # Create enhanced summarization prompt
        file_context = f"""
File Information:
- Title: {title if title else 'Document'}
- Filename: {filename}
- File Type: {file_extension}
- Content Type: {content_type}
- Source: {url if url else 'File upload'}
"""
        
        instruction = f"""
        Summarize this {content_type} content in {config['target_length']}.
        Focus on {config['detail_level']}.
        Target length: approximately {config['target_tokens']} tokens.
        
        {file_context}
        
        Structure your summary with:
        1. Brief overview including document title and type context
        2. Main topics or sections covered
        3. Key insights, findings, or conclusions
        4. Important details, data, or examples mentioned
        
        Make the summary informative and well-structured, preserving important technical details and maintaining context.
        IMPORTANT: Preserve the document title, filename, and source information in your response for reference.
        """
        
        # Get LLM configuration
        llm_config = get_llm_config(llm_provider, llm_model)
        
        # Create the complete prompt
        full_prompt = f"""
        {instruction}
        
        Please provide a JSON response with the following structure:
        {{
            "summary": "The summarized content (approximately {config['target_tokens']} tokens)",
            "document_title": "{title if title else 'Document'}",
            "filename": "{filename}",
            "file_extension": "{file_extension}",
            "content_type": "{content_type}",
            "source_url": "{url}",
            "key_topics": ["List", "of", "main", "topics"],
            "main_insights": ["Key", "findings", "or", "insights"],
            "technical_details": ["Important", "technical", "details"],
            "reading_time_estimate": "Estimated reading time",
            "summary_token_count": "Estimated token count of summary"
        }}
        
        Content to summarize:
        {content}
        """
        
        # Use LLM for summarization
        if hasattr(llm_config, 'provider'):
            provider_info = llm_config.provider.split('/')
            provider = provider_info[0] if provider_info else 'openai'
            model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
            
            if provider == 'openai':
                import openai
                import os
                
                api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found")
                
                client = openai.AsyncOpenAI(api_key=api_key)
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are a helpful assistant that summarizes {content_type} content while preserving important metadata."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=min(4000, config['target_tokens'] * 2)  # Allow up to 2x target for flexibility
                )
                
                extracted_content = response.choices[0].message.content
            else:
                raise ValueError(f"Provider {provider} not supported in direct mode")
        else:
            raise ValueError("Invalid LLM config format")
        
        # Parse the response
        if extracted_content:
            try:
                import json
                # Clean up the extracted content if it's wrapped in markdown
                content_to_parse = extracted_content
                if content_to_parse.startswith('```json'):
                    content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                
                summary_data = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                
                # Ensure metadata is preserved
                return {
                    "success": True,
                    "summary": summary_data.get("summary", "Summary generation failed"),
                    "document_title": summary_data.get("document_title", title),
                    "filename": summary_data.get("filename", filename),
                    "file_extension": summary_data.get("file_extension", file_extension),
                    "content_type": summary_data.get("content_type", content_type),
                    "source_url": summary_data.get("source_url", url),
                    "key_topics": summary_data.get("key_topics", []),
                    "main_insights": summary_data.get("main_insights", []),
                    "technical_details": summary_data.get("technical_details", []),
                    "summary_length": summary_length,
                    "target_tokens": config['target_tokens'],
                    "estimated_summary_tokens": len(summary_data.get("summary", "")) // 4,  # Rough estimate
                    "original_length": len(content),
                    "compressed_ratio": len(summary_data.get("summary", "")) / len(content) if content else 0,
                    "llm_provider": provider,
                    "llm_model": model
                }
            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback: treat as plain text summary
                return {
                    "success": True,
                    "summary": str(extracted_content),
                    "document_title": title,
                    "filename": filename,
                    "file_extension": file_extension,
                    "content_type": content_type,
                    "source_url": url,
                    "key_topics": [],
                    "main_insights": [],
                    "technical_details": [],
                    "summary_length": summary_length,
                    "target_tokens": config['target_tokens'],
                    "estimated_summary_tokens": len(str(extracted_content)) // 4,
                    "original_length": len(content),
                    "compressed_ratio": len(str(extracted_content)) / len(content) if content else 0,
                    "llm_provider": provider,
                    "llm_model": model,
                    "fallback_mode": True
                }
        else:
            return {
                "success": False,
                "error": "LLM returned empty result"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Summarization failed: {str(e)}"
        }


# MCP Tool implementations
async def process_file(
    url: Annotated[str, Field(description="URL of the file to process (PDF, Office, ZIP)")],
    max_size_mb: Annotated[int, Field(description="Maximum file size in MB (default: 100)")] = 100,
    extract_all_from_zip: Annotated[bool, Field(description="Whether to extract all files from ZIP archives (default: True)")] = True,
    include_metadata: Annotated[bool, Field(description="Whether to include file metadata (default: True)")] = True,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None
) -> FileProcessResponse:
    """
    Convert documents to markdown text with optional AI summarization.
    
    Supports: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), ZIP archives, ePub.
    Auto-detects file format and applies appropriate conversion method.
    """
    try:
        # Check if file format is supported
        if not file_processor.is_supported_file(url):
            return FileProcessResponse(
                success=False,
                url=url,
                error=f"Unsupported file format. Supported formats: {', '.join(file_processor.supported_extensions.keys())}",
                file_type=file_processor.get_file_type(url)
            )
        
        # Process the file
        result = await file_processor.process_file_from_url(
            url,
            max_size_mb=max_size_mb
        )
        
        if result['success']:
            # Prepare content for potential summarization
            content_to_use = result.get('content', '')
            final_metadata = result.get('metadata', {}) if include_metadata else {}
            
            # Apply auto-summarization if enabled and content exceeds token limit
            if auto_summarize and content_to_use:
                # Rough token estimation: 1 token ≈ 4 characters
                estimated_tokens = len(content_to_use) // 4
                
                # Only summarize if content exceeds the specified token limit
                if estimated_tokens > max_content_tokens:
                    try:
                        # Use summarize_content function for document summarization
                        summary_result = await summarize_content(
                            content=content_to_use,
                            title=result.get('title', ''),
                            url=url,
                            summary_length=summary_length,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            content_type="document",
                            target_tokens=max_content_tokens
                        )
                        
                        if summary_result.get("success"):
                            # Replace content with summary and preserve original info
                            content_to_use = summary_result["summary"]
                            summarization_info = {
                                "summarization_applied": True,
                                "original_tokens_estimate": estimated_tokens,
                                "summary_length": summary_length,
                                "compression_ratio": summary_result.get("compressed_ratio", 0),
                                "key_topics": summary_result.get("key_topics", []),
                                "content_type": summary_result.get("content_type", "Document"),
                                "main_insights": summary_result.get("main_insights", []),
                                "technical_details": summary_result.get("technical_details", []),
                                "llm_provider": summary_result.get("llm_provider", "unknown"),
                                "llm_model": summary_result.get("llm_model", "unknown"),
                                "auto_summarization_trigger": f"Document exceeded {max_content_tokens} tokens"
                            }
                            
                            # Add summarization info to metadata
                            final_metadata['summarization'] = summarization_info
                        else:
                            # Summarization failed, add error info to metadata
                            final_metadata['summarization'] = {
                                "summarization_attempted": True,
                                "summarization_error": summary_result.get("error", "Unknown error"),
                                "original_content_preserved": True
                            }
                    except Exception as e:
                        # Summarization failed, add error info to metadata
                        final_metadata['summarization'] = {
                            "summarization_attempted": True,
                            "summarization_error": f"Exception during summarization: {str(e)}",
                            "original_content_preserved": True
                        }
                else:
                    # Content is below threshold - preserve original content and add info
                    final_metadata['summarization'] = {
                        "auto_summarize_requested": True,
                        "original_content_preserved": True,
                        "content_below_threshold": True,
                        "tokens_estimate": estimated_tokens,
                        "max_tokens_threshold": max_content_tokens,
                        "reason": f"Content ({estimated_tokens} tokens) is below threshold ({max_content_tokens} tokens)"
                    }
            
            # Handle ZIP archives specially
            if result.get('is_archive', False) and extract_all_from_zip:
                archive_contents = result.get('archive_contents', {})
                
                # Combine content from all files if requested
                if archive_contents and archive_contents.get('files'):
                    combined_content = []
                    successful_files = []
                    
                    for file_info in archive_contents['files']:
                        if file_info.get('content') and not file_info.get('error'):
                            file_header = f"\n\n## File: {file_info['name']} ({file_info['type']})\n\n"
                            combined_content.append(file_header + file_info['content'])
                            successful_files.append(file_info['name'])
                    
                    if combined_content:
                        content_to_use = '\n'.join(combined_content)
                        final_metadata['archive_processing'] = {
                            'total_files_in_archive': archive_contents.get('total_files', 0),
                            'successfully_processed': len(successful_files),
                            'processed_files': successful_files,
                            'content_combined': True
                        }
            
            response = FileProcessResponse(
                success=True,
                url=result.get('url', url),
                filename=result.get('filename'),
                file_type=result.get('file_type'),
                size_bytes=result.get('size_bytes'),
                is_archive=result.get('is_archive', False),
                content=content_to_use,
                title=result.get('title'),
                metadata=final_metadata if include_metadata else None,
                archive_contents=result.get('archive_contents') if result.get('is_archive') and extract_all_from_zip else None
            )
        else:
            response = FileProcessResponse(
                success=False,
                url=url,
                error=result.get('error'),
                file_type=result.get('file_type')
            )
            
        return response
        
    except Exception as e:
        return FileProcessResponse(
            success=False,
            url=url,
            error=f"File processing error: {str(e)}"
        )


async def get_supported_file_formats() -> Dict[str, Any]:
    """
    Get list of supported file formats for file processing.
    
    Provides comprehensive information about supported file formats and their capabilities.
    
    Parameters: None
    
    Returns dictionary with supported file formats and descriptions.
    """
    try:
        return {
            "success": True,
            "supported_formats": list(file_processor.supported_extensions.keys()),
            "format_descriptions": file_processor.supported_extensions,
            "categories": {
                "pdf": {
                    "description": "PDF Documents",
                    "extensions": [".pdf"],
                    "features": ["Text extraction", "Structure preservation", "Metadata extraction", "Multi-page support"]
                },
                "microsoft_office": {
                    "description": "Microsoft Office Documents",
                    "extensions": [".docx", ".pptx", ".xlsx", ".xls"],
                    "features": ["Content extraction", "Table processing", "Slide content", "Cell data", "Formatting preservation"]
                },
                "archives": {
                    "description": "Archive Files",
                    "extensions": [".zip"],
                    "features": ["Multi-file extraction", "Nested processing", "Format detection", "Batch processing"]
                },
                "web_and_text": {
                    "description": "Web and Text Formats",
                    "extensions": [".html", ".htm", ".txt", ".md", ".csv", ".rtf"],
                    "features": ["HTML parsing", "Text processing", "CSV structure", "Rich text", "Encoding detection"]
                },
                "ebooks": {
                    "description": "eBook Formats",
                    "extensions": [".epub"],
                    "features": ["Chapter extraction", "Metadata", "Content structure", "Table of contents"]
                }
            },
            "processing_capabilities": {
                "max_file_size": "100MB (configurable)",
                "output_format": "Markdown",
                "ai_summarization": "Available for large documents",
                "batch_processing": "ZIP archives support multiple files",
                "metadata_extraction": "File metadata and document properties"
            },
            "additional_features": [
                "Automatic file type detection",
                "MarkItDown integration for accurate conversion",
                "ZIP archive processing with individual file extraction",
                "AI-powered summarization for large documents",
                "Error handling and recovery",
                "Size limit protection",
                "Temporary file cleanup",
                "Content validation"
            ],
            "usage_examples": {
                "simple_pdf": {"url": "https://example.com/document.pdf"},
                "with_summarization": {"url": "https://example.com/large-report.pdf", "auto_summarize": True},
                "zip_archive": {"url": "https://example.com/documents.zip", "extract_all_from_zip": True},
                "size_limited": {"url": "https://example.com/document.pdf", "max_size_mb": 50}
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving format information: {str(e)}"
        }


async def enhanced_process_large_content(
    url: Annotated[str, Field(description="Target URL to process")],
    chunking_strategy: Annotated[str, Field(description="Chunking strategy: 'topic', 'sentence', 'overlap', 'regex' (default: 'topic')")] = "topic",
    filtering_strategy: Annotated[str, Field(description="Filtering strategy: 'bm25', 'pruning', 'llm' (default: 'bm25')")] = "bm25", 
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 filtering, keywords related to desired content (default: None)")] = None,
    max_chunk_tokens: Annotated[int, Field(description="Maximum tokens per chunk (default: 8000)")] = 8000,
    chunk_overlap: Annotated[int, Field(description="Token overlap between chunks (default: 500)")] = 500,
    similarity_threshold: Annotated[float, Field(description="Minimum similarity threshold for relevant chunks (default: 0.7)")] = 0.7,
    extract_top_chunks: Annotated[int, Field(description="Number of top relevant chunks to extract (default: 10)")] = 10,
    summarize_chunks: Annotated[bool, Field(description="Whether to summarize individual chunks (default: True)")] = True,
    merge_strategy: Annotated[str, Field(description="Strategy for merging chunk summaries: 'hierarchical', 'linear' (default: 'hierarchical')")] = "hierarchical",
    final_summary_length: Annotated[str, Field(description="Final summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium"
) -> LargeContentResponse:
    """
    Enhanced processing for large content using advanced chunking and filtering.
    
    Uses BM25 filtering and intelligent chunking to reduce token usage while preserving semantic boundaries.
    Supports hierarchical summarization for progressive content refinement.
    """
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
            from .web_crawling import CrawlRequest
            
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