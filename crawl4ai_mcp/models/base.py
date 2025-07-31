"""
Base models for Crawl4AI MCP Server.

Contains common base classes and types used across all models.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    """Base class for all request models."""
    pass


class BaseResponse(BaseModel):
    """Base class for all response models."""
    success: bool = Field(..., description="Whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")