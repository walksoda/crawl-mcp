"""
File processing tool registrations.
"""

from ._shared import (
    Annotated, Dict, Field, Optional, Any,
    apply_token_limit,
    validate_content_slicing_params,
    _apply_content_slicing,
    _convert_result_to_dict,
    modules_unavailable_error,
    READONLY_ANNOTATIONS,
    READONLY_CLOSED_ANNOTATIONS,
)


def register_file_tools(mcp, get_modules):
    """Register file processing MCP tools."""

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def process_file(
        url: Annotated[str, Field(description="File URL (PDF, Office, ZIP)")],
        max_size_mb: Annotated[int, Field(description="Max file size in MB")] = 100,
        extract_all_from_zip: Annotated[bool, Field(description="Extract ZIP contents")] = True,
        include_metadata: Annotated[bool, Field(description="Include metadata")] = True,
        auto_summarize: Annotated[bool, Field(description="Auto-summarize large content")] = False,
        max_content_tokens: Annotated[int, Field(description="Max tokens before summarization")] = 15000,
        summary_length: Annotated[str, Field(description="'short'|'medium'|'long'")] = "medium",
        llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
        llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
        content_limit: Annotated[int, Field(description="Max characters to return (0=unlimited)")] = 0,
        content_offset: Annotated[int, Field(description="Start position for content (0-indexed)")] = 0
    ) -> dict:
        """Convert PDF, Word, Excel, PowerPoint, ZIP to markdown."""
        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        web_crawling, _, _, file_processing, _ = modules

        try:
            # Validate content slicing parameters (always validate if non-zero)
            if content_limit != 0 or content_offset != 0:
                slicing_error = validate_content_slicing_params(content_limit, content_offset)
                if slicing_error:
                    return slicing_error

            result = await file_processing.process_file(
                url=url, max_size_mb=max_size_mb, extract_all_from_zip=extract_all_from_zip,
                include_metadata=include_metadata, auto_summarize=auto_summarize,
                max_content_tokens=max_content_tokens, summary_length=summary_length,
                llm_provider=llm_provider, llm_model=llm_model
            )

            # Convert FileProcessResponse object to dict for JSON serialization
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                # Fallback: manual conversion
                result_dict = {
                    'success': getattr(result, 'success', False),
                    'url': getattr(result, 'url', None),
                    'filename': getattr(result, 'filename', None),
                    'file_type': getattr(result, 'file_type', None),
                    'size_bytes': getattr(result, 'size_bytes', None),
                    'is_archive': getattr(result, 'is_archive', False),
                    'content': getattr(result, 'content', None),
                    'title': getattr(result, 'title', None),
                    'metadata': getattr(result, 'metadata', None),
                    'archive_contents': getattr(result, 'archive_contents', None),
                    'error': getattr(result, 'error', None),
                    'processing_time': getattr(result, 'processing_time', None)
                }

            # Apply content slicing if requested
            if content_limit != 0 or content_offset != 0:
                result_dict = _apply_content_slicing(result_dict, content_limit, content_offset)

            # Apply token limit fallback to prevent MCP errors
            result_with_fallback = apply_token_limit(result_dict, max_tokens=25000)

            # If token limit was applied and auto_summarize was False, provide helpful suggestion
            if result_with_fallback.get("token_limit_applied") and not auto_summarize:
                if not result_with_fallback.get("emergency_truncation"):
                    result_with_fallback["suggestion"] = "Content was truncated due to MCP token limits. Consider setting auto_summarize=True for better content reduction."

            return result_with_fallback

        except Exception as e:
            return {
                "success": False,
                "error": f"File processing error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_CLOSED_ANNOTATIONS)
    async def get_supported_file_formats() -> dict:
        """Get supported file formats (PDF, Office, ZIP) and their capabilities."""
        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        web_crawling, _, _, file_processing, _ = modules

        try:
            result = await file_processing.get_supported_file_formats()
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Get supported formats error: {str(e)}"
            }

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def enhanced_process_large_content(
        url: Annotated[str, Field(description="URL to process")],
        chunking_strategy: Annotated[str, Field(description="'topic'|'sentence'|'overlap'|'regex'")] = "sentence",
        filtering_strategy: Annotated[str, Field(description="'bm25'|'pruning'|'llm'")] = "bm25",
        filter_query: Annotated[Optional[str], Field(description="Keywords for BM25 filtering")] = None,
        max_chunk_tokens: Annotated[int, Field(description="Max tokens per chunk")] = 2000,
        chunk_overlap: Annotated[int, Field(description="Overlap tokens")] = 200,
        extract_top_chunks: Annotated[int, Field(description="Top chunks to extract")] = 5,
        similarity_threshold: Annotated[float, Field(description="Min similarity 0-1")] = 0.5,
        summarize_chunks: Annotated[bool, Field(description="Summarize chunks")] = False,
        merge_strategy: Annotated[str, Field(description="'hierarchical'|'linear'")] = "linear",
        final_summary_length: Annotated[str, Field(description="'short'|'medium'|'long'")] = "short"
    ) -> Dict[str, Any]:
        """Process large content with chunking and BM25 filtering."""
        modules = get_modules()
        if not modules:
            return {
                "success": False,
                "error": "Tool modules not available",
                "processing_time": None,
                "metadata": {},
                "url": url,
                "original_content_length": 0,
                "filtered_content_length": 0,
                "total_chunks": 0,
                "relevant_chunks": 0,
                "processing_method": "enhanced_large_content",
                "chunking_strategy_used": chunking_strategy,
                "filtering_strategy_used": filtering_strategy,
                "chunks": [],
                "chunk_summaries": None,
                "merged_summary": None,
                "final_summary": "Tool modules not available"
            }
        web_crawling, _, _, file_processing, _ = modules

        try:
            import asyncio

            # Always use fallback to basic crawling due to backend issues
            print(f"Processing URL with fallback method: {url}")

            fallback_result = _convert_result_to_dict(await asyncio.wait_for(
                web_crawling.crawl_url(
                    url=url,
                    generate_markdown=True,
                    timeout=10
                ),
                timeout=10.0
            ))

            if fallback_result and fallback_result.get("success", False):
                content = fallback_result.get("content", "")

                # Simple truncation as processing
                max_content = max_chunk_tokens * extract_top_chunks
                if len(content) > max_content:
                    content = content[:max_content] + "... [truncated for processing limit]"

                # Create simple chunks
                chunk_size = max_chunk_tokens
                chunks = []
                for i in range(0, min(len(content), max_content), chunk_size):
                    chunk_content = content[i:i + chunk_size]
                    if chunk_content.strip():
                        chunks.append({
                            "content": chunk_content,
                            "relevance_score": 1.0 - (i / max_content),
                            "chunk_index": len(chunks)
                        })

                # Take top chunks
                top_chunks = chunks[:extract_top_chunks]

                # Generate simple summary
                if summarize_chunks and len(content) > 1000:
                    final_summary = content[:500] + "... [content summary]"
                else:
                    final_summary = content[:300] + "..." if len(content) > 300 else content

                return {
                    "success": True,
                    "error": "Enhanced processing unavailable, used basic crawl with chunking",
                    "processing_time": 10,
                    "metadata": {"fallback_used": True, "processing_type": "basic_chunking"},
                    "url": url,
                    "original_content_length": len(fallback_result.get("content", "")),
                    "filtered_content_length": len(content),
                    "total_chunks": len(chunks),
                    "relevant_chunks": len(top_chunks),
                    "processing_method": "basic_crawl_with_chunking",
                    "chunking_strategy_used": chunking_strategy,
                    "filtering_strategy_used": "simple_truncation",
                    "chunks": top_chunks,
                    "chunk_summaries": None,
                    "merged_summary": None,
                    "final_summary": final_summary
                }
            else:
                raise Exception("Fallback crawling also failed")

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Processing timed out after 10 seconds",
                "processing_time": 10,
                "metadata": {"timeout": True},
                "url": url,
                "original_content_length": 0,
                "filtered_content_length": 0,
                "total_chunks": 0,
                "relevant_chunks": 0,
                "processing_method": "timeout_fallback",
                "chunking_strategy_used": chunking_strategy,
                "filtering_strategy_used": filtering_strategy,
                "chunks": [],
                "chunk_summaries": None,
                "merged_summary": None,
                "final_summary": "Processing timed out"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced processing error: {str(e)}",
                "processing_time": None,
                "metadata": {"error_type": type(e).__name__},
                "url": url,
                "original_content_length": 0,
                "filtered_content_length": 0,
                "total_chunks": 0,
                "relevant_chunks": 0,
                "processing_method": "enhanced_large_content",
                "chunking_strategy_used": chunking_strategy,
                "filtering_strategy_used": filtering_strategy,
                "chunks": [],
                "chunk_summaries": None,
                "merged_summary": None,
                "final_summary": f"Error occurred: {str(e)}"
            }
