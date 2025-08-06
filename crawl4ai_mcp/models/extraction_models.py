"""
Data extraction models for Crawl4AI MCP Server.

Contains request models for structured data extraction operations.
"""

from typing import Any, Dict, Optional
from pydantic import Field
from .base import BaseRequest


class StructuredExtractionRequest(BaseRequest):
    """Request model for structured data extraction."""
    url: str = Field(..., description="URL to crawl")
    extraction_schema: Dict[str, Any] = Field(..., description="JSON schema for extraction")
    extraction_type: str = Field("css", description="Type of extraction: 'css' or 'llm'")
    css_selectors: Optional[Dict[str, str]] = Field(None, description="CSS selectors for each field")
    llm_provider: Optional[str] = Field("openai", description="LLM provider for LLM-based extraction")
    llm_model: Optional[str] = Field("gpt-3.5-turbo", description="LLM model name")
    instruction: Optional[str] = Field(None, description="Custom instruction for LLM extraction")