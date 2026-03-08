"""
Shared imports and helpers for server_tools modules.

Centralizes common dependencies used by all tool registration modules.
"""

from typing import Any, Dict, List, Optional, Union, Annotated
from pydantic import Field

from ..utils import apply_token_limit
from ..server_helpers import (
    _convert_result_to_dict,
    _process_content_fields,
    _should_trigger_fallback,
    _apply_content_slicing,
    _apply_search_content_slicing,
    _get_search_cache_key,
    _get_cached_search_result,
    _cache_search_result,
)
from ..validators import validate_crawl_url_params, validate_content_slicing_params

# Tool annotations constants
READONLY_ANNOTATIONS = {"readOnlyHint": True}
READONLY_CLOSED_ANNOTATIONS = {"readOnlyHint": True, "openWorldHint": False}

# Re-export for convenient access
__all__ = [
    # typing
    "Any", "Dict", "List", "Optional", "Union", "Annotated", "Field",
    # helpers
    "apply_token_limit",
    "_convert_result_to_dict",
    "_process_content_fields",
    "_should_trigger_fallback",
    "_apply_content_slicing",
    "_apply_search_content_slicing",
    "_get_search_cache_key",
    "_get_cached_search_result",
    "_cache_search_result",
    "validate_crawl_url_params",
    "validate_content_slicing_params",
    # utilities
    "modules_unavailable_error",
    # annotations
    "READONLY_ANNOTATIONS",
    "READONLY_CLOSED_ANNOTATIONS",
]


def modules_unavailable_error():
    """Return standard error dict when tool modules are not loaded."""
    return {
        "success": False,
        "error": "Tool modules not available",
        "error_code": "modules_unavailable",
    }
