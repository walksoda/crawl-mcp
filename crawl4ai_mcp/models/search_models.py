"""
Search-related models for Crawl4AI MCP Server.

Contains request and response models for Google search operations.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from .base import BaseRequest, BaseResponse


class GoogleSearchRequest(BaseRequest):
    """Request model for Google search operations."""
    query: str = Field(..., description="Search query")
    num_results: int = Field(10, description="Number of results to return (1-100)")
    language: str = Field("en", description="Search language (e.g., 'en', 'ja')")
    region: str = Field("us", description="Search region (e.g., 'us', 'jp')")
    search_genre: Optional[str] = Field(None, description="Search genre for content filtering (e.g., 'academic', 'news', 'technical')")
    safe_search: bool = Field(True, description="Enable safe search filtering (always enabled for safety)")


class GoogleSearchResponse(BaseResponse):
    """Response model for Google search operations."""
    query: Optional[str] = None
    total_results: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    search_metadata: Optional[Dict[str, Any]] = None


class GoogleBatchSearchRequest(BaseRequest):
    """Request model for batch Google search operations."""
    queries: List[str] = Field(..., description="List of search queries")
    num_results_per_query: int = Field(10, description="Number of results per query (1-100)")
    max_concurrent: int = Field(3, description="Maximum concurrent searches (1-5)")
    language: str = Field("en", description="Search language (e.g., 'en', 'ja')")
    region: str = Field("us", description="Search region (e.g., 'us', 'jp')")
    search_genre: Optional[str] = Field(None, description="Search genre for content filtering")


class GoogleBatchSearchResponse(BaseResponse):
    """Response model for batch Google search operations."""
    total_queries: int
    successful_searches: int
    failed_searches: int
    results: List[GoogleSearchResponse]
    analysis: Optional[Dict[str, Any]] = None