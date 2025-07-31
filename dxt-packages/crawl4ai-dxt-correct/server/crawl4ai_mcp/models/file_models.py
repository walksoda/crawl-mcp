"""
File processing models for Crawl4AI MCP Server.

Contains request and response models for file processing operations.
"""

from typing import Any, Dict, Optional
from pydantic import Field
from .base import BaseRequest, BaseResponse


class FileProcessRequest(BaseRequest):
    """Request model for file processing operations."""
    url: str = Field(..., description="URL of the file to process (PDF, Office, ZIP)")
    max_size_mb: int = Field(100, description="Maximum file size in MB")
    extract_all_from_zip: bool = Field(True, description="Whether to extract all files from ZIP archives")
    include_metadata: bool = Field(True, description="Whether to include file metadata")


class FileProcessResponse(BaseResponse):
    """Response model for file processing operations."""
    url: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    size_bytes: Optional[int] = None
    is_archive: bool = False
    content: Optional[str] = None
    title: Optional[str] = None
    archive_contents: Optional[Dict[str, Any]] = None