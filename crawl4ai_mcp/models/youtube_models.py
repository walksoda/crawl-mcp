"""
YouTube processing models for Crawl4AI MCP Server.

Contains request and response models for YouTube transcript operations.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from .base import BaseRequest, BaseResponse


class YouTubeTranscriptRequest(BaseRequest):
    """Request model for YouTube transcript extraction."""
    url: str = Field(..., description="YouTube video URL")
    languages: Optional[List[str]] = Field(["ja", "en"], description="Preferred languages in order of preference")
    translate_to: Optional[str] = Field(None, description="Target language for translation")
    include_timestamps: bool = Field(False, description="Include timestamps in transcript")
    preserve_formatting: bool = Field(True, description="Preserve original formatting")
    include_metadata: bool = Field(True, description="Include video metadata")


class YouTubeTranscriptResponse(BaseResponse):
    """Response model for YouTube transcript extraction."""
    url: Optional[str] = None
    video_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    markdown: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None


class YouTubeCommentsResponse(BaseResponse):
    """Response model for YouTube comments extraction."""
    url: Optional[str] = None
    video_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    markdown: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None


class YouTubeBatchRequest(BaseRequest):
    """Request model for batch YouTube transcript extraction."""
    urls: List[str] = Field(..., description="List of YouTube video URLs")
    languages: Optional[List[str]] = Field(["ja", "en"], description="Preferred languages in order of preference")
    translate_to: Optional[str] = Field(None, description="Target language for translation")
    include_timestamps: bool = Field(False, description="Include timestamps in transcript")
    max_concurrent: int = Field(3, description="Maximum concurrent requests (1-10)")


class YouTubeBatchResponse(BaseResponse):
    """Response model for batch YouTube transcript extraction."""
    total_urls: int
    successful_extractions: int
    failed_extractions: int
    results: List[Dict[str, Any]]