"""File processing core functions for document conversion and format support."""

import time
from typing import Any, Dict, List, Optional

from ..processors.file_processor import FileProcessor
from ..models import FileProcessRequest, FileProcessResponse

# Initialize file processor
file_processor = FileProcessor()


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
    """Summarize content using LLMClient with enhanced metadata preservation"""
    try:
        # Import LLMClient
        from ..utils.llm_client import LLMClient

        # Extract file information for context
        file_extension = ""
        filename = ""
        if url:
            try:
                import os
                filename = os.path.basename(url)
                file_extension = os.path.splitext(filename)[1]
            except Exception:
                filename = url

        # Prepare metadata for LLMClient
        metadata = {
            "filename": filename,
            "file_extension": file_extension,
        }

        # Create LLMClient and call summarize
        client = LLMClient()
        result = await client.summarize(
            content=content,
            title=title if title else "Document",
            url=url,
            summary_length=summary_length,
            content_type=content_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
            target_tokens=target_tokens,
            metadata=metadata
        )

        # Transform result to file processing-specific format
        if result.get("success"):
            return {
                "success": True,
                "summary": result.get("summary", "Summary generation failed"),
                "document_title": title if title else result.get("title", "Document"),
                "filename": filename,
                "file_extension": file_extension,
                "content_type": result.get("content_type", content_type),
                "source_url": url or result.get("source_url", ""),
                "key_topics": result.get("key_topics", []),
                "main_insights": result.get("main_insights", []),
                "technical_details": [],  # LLMClient doesn't extract this separately
                "summary_length": summary_length,
                "target_tokens": result.get("target_tokens", 1000),
                "estimated_summary_tokens": result.get("estimated_summary_tokens", 0),
                "original_length": result.get("original_length", len(content)),
                "compressed_ratio": result.get("compression_ratio", 0),
                "llm_provider": result.get("llm_provider", "unknown"),
                "llm_model": result.get("llm_model", "unknown"),
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Summarization failed")
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Summarization failed: {str(e)}"
        }


# MCP Tool implementations
async def process_file(
    url: str,
    max_size_mb: int = 100,
    extract_all_from_zip: bool = True,
    include_metadata: bool = True,
    auto_summarize: bool = False,
    max_content_tokens: int = 15000,
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
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
                # Rough token estimation: 1 token ~ 4 characters
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
