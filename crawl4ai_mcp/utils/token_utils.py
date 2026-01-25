"""Token estimation and content size management utilities.

This module provides functions for estimating token counts and applying
token limits to content to prevent context overflow issues.
"""

from typing import Dict, Any, List, Tuple, Optional
import json

from ..constants import (
    TOKEN_ESTIMATE_CHARS_PER_TOKEN_EN,
    TOKEN_ESTIMATE_CHARS_PER_TOKEN_JA,
    JAPANESE_DETECTION_THRESHOLD,
    CONTENT_FIELD_PRIORITIES,
    ESSENTIAL_FIELDS,
    MAX_RESPONSE_TOKENS,
)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (GPT-4 encoding as Claude approximation).
    Falls back to character-based estimation if tiktoken is unavailable.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    try:
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-4")
        return len(encoder.encode(str(text)))
    except Exception:
        # Fallback to character-based estimation
        return estimate_tokens_fallback(text)


def estimate_tokens_fallback(text: str) -> int:
    """
    Character-based token estimation without tiktoken dependency.

    Uses different ratios for English and Japanese text based on
    the proportion of Japanese characters detected.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    text_str = str(text)
    if not text_str:
        return 0

    # Count Japanese characters (Hiragana, Katakana, CJK Unified Ideographs)
    japanese_chars = sum(1 for c in text_str if '\u3040' <= c <= '\u9fff')
    total_chars = len(text_str)

    if total_chars > 0 and japanese_chars / total_chars > JAPANESE_DETECTION_THRESHOLD:
        # Japanese-heavy text: ~2 chars per token
        return total_chars // TOKEN_ESTIMATE_CHARS_PER_TOKEN_JA
    else:
        # English-heavy text: ~4 chars per token
        return total_chars // TOKEN_ESTIMATE_CHARS_PER_TOKEN_EN


def truncate_content(content: str, max_tokens: int) -> str:
    """
    Truncate content to fit within a token limit.

    Args:
        content: The content to truncate
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated content with ellipsis if truncated
    """
    if not content:
        return content

    current_tokens = estimate_tokens(content)
    if current_tokens <= max_tokens:
        return content

    # Calculate approximate character limit based on token ratio
    chars_per_token = len(content) / current_tokens if current_tokens > 0 else 4
    estimated_chars = int(max_tokens * chars_per_token)

    truncated = content[:estimated_chars]
    actual_tokens = estimate_tokens(truncated)

    # Adjust if estimation was off
    while actual_tokens > max_tokens and len(truncated) > 100:
        truncated = truncated[:int(len(truncated) * 0.9)]
        actual_tokens = estimate_tokens(truncated)

    if len(truncated) < len(content):
        return truncated + f"... [TRUNCATED: showing {len(truncated)} of {len(content)} chars]"

    return truncated


def apply_token_limit(result: Dict[str, Any], max_tokens: int = 25000) -> Dict[str, Any]:
    """
    Apply token limit to MCP tool responses to prevent Claude Code errors.

    This function implements a multi-stage truncation strategy:
    1. Check if response is within limits
    2. Progressively truncate content fields by priority
    3. Apply emergency truncation if needed

    Args:
        result: The result dictionary to limit
        max_tokens: Maximum tokens allowed (default 20000)

    Returns:
        Token-limited result dictionary
    """
    result_copy = result.copy()

    # Check current size
    current_tokens = estimate_tokens(json.dumps(result_copy))

    if current_tokens <= max_tokens:
        return result_copy

    # Add fallback indicators with clear warning
    result_copy["token_limit_applied"] = True
    result_copy["original_response_tokens"] = current_tokens
    result_copy["truncated_to_tokens"] = max_tokens
    result_copy["warning"] = (
        f"Response truncated from {current_tokens} to ~{max_tokens} tokens "
        "due to size limits. Partial data returned below."
    )

    # Add recommendations based on content type
    recommendations = _generate_recommendations(result_copy)
    if recommendations:
        result_copy["recommendations"] = recommendations

    # Apply progressive truncation
    result_copy = _truncate_content_fields(result_copy, max_tokens)

    # Final size check and emergency truncation
    current_tokens = estimate_tokens(json.dumps(result_copy))
    if current_tokens > max_tokens:
        result_copy = _apply_emergency_truncation(
            result, result_copy, current_tokens, max_tokens
        )

    return result_copy


def _generate_recommendations(result: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on content type."""
    recommendations = []

    if "results" in result and isinstance(result["results"], list):
        original_count = result.get("results_truncated_from", len(result["results"]))
        recommendations.append(
            f"Consider reducing num_results parameter (current response had {original_count} results)"
        )

    if any(key in result for key in ["content", "markdown", "text"]):
        recommendations.append("Consider using auto_summarize=True for large content")
        recommendations.append(
            "Consider using css_selector to extract specific content sections"
        )
        recommendations.append(
            "Use content_limit and content_offset to retrieve content in chunks "
            "(e.g., content_limit=5000, content_offset=0 for first 5000 chars)"
        )

    if "search_query" in result or "query" in result:
        recommendations.append(
            "Consider narrowing your search query or using search_genre filtering"
        )

    return recommendations


def _truncate_content_fields(result: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Progressively truncate content fields by priority."""
    for field, max_field_tokens in CONTENT_FIELD_PRIORITIES:
        if field not in result or not result[field]:
            continue

        field_content = result[field]

        # Handle different field types
        if isinstance(field_content, list):
            # For lists, truncate by limiting number of items
            original_length = len(field_content)
            if original_length > 10:
                result[field] = field_content[:10]
                result[f"{field}_truncated_from"] = original_length
                result[f"{field}_truncated_info"] = f"Showing 10 of {original_length} items"

        elif isinstance(field_content, dict):
            # For dicts, truncate string values
            truncated_dict = {}
            for k, v in field_content.items():
                if isinstance(v, str) and len(v) > 500:
                    truncated_dict[k] = v[:500] + "... [TRUNCATED]"
                else:
                    truncated_dict[k] = v
            result[field] = truncated_dict

        elif isinstance(field_content, str):
            # For strings, truncate with ellipsis
            max_chars = max_field_tokens * 4
            original_length = len(field_content)
            if original_length > max_chars:
                result[field] = (
                    field_content[:max_chars] +
                    f"... [TRUNCATED: showing {max_chars} of {original_length} chars]"
                )

    return result


def _apply_emergency_truncation(
    original_result: Dict[str, Any],
    result_copy: Dict[str, Any],
    current_tokens: int,
    max_tokens: int
) -> Dict[str, Any]:
    """Apply emergency truncation when progressive truncation wasn't enough."""
    # Keep only essential fields
    essential_result = {
        key: result_copy.get(key)
        for key in ESSENTIAL_FIELDS
        if key in result_copy
    }

    essential_result.update({
        "token_limit_applied": True,
        "emergency_truncation": True,
        "original_response_tokens": original_result.get(
            "original_response_tokens", current_tokens
        ),
        "warning": (
            f"Response truncated from {current_tokens} to ~{max_tokens} tokens. "
            "Partial content returned."
        ),
        "recommendations": [
            "Use more specific parameters to reduce content size",
            "Enable auto_summarize for large content",
            "Use filtering parameters (css_selector, xpath, search_genre)",
            "Reduce num_results for search queries",
            "Use max_content_per_page to limit page content size",
        ],
        "available_fields_in_original": list(original_result.keys()),
    })

    # Calculate available tokens for content after essential fields
    essential_base_tokens = estimate_tokens(json.dumps(essential_result))
    available_content_tokens = max(
        max_tokens - essential_base_tokens - 500, 1000
    )  # Reserve 500 for safety margin

    # Try to fit content from priority sources
    content_added = _add_priority_content(
        original_result, essential_result, available_content_tokens
    )

    # Keep partial results if available
    if original_result.get("results") and isinstance(original_result["results"], list):
        essential_result["results"] = original_result["results"][:3]
        essential_result["results_truncated_info"] = (
            f"Showing 3 of {len(original_result['results'])} results"
        )

    # Special handling for YouTube transcript data
    if original_result.get("transcript") and isinstance(original_result["transcript"], list):
        _add_transcript_content(
            original_result, essential_result,
            available_content_tokens, content_added
        )

    return essential_result


def _add_priority_content(
    original_result: Dict[str, Any],
    essential_result: Dict[str, Any],
    available_tokens: int
) -> bool:
    """Try to add content from priority sources."""
    content_sources = [
        ("markdown", original_result.get("markdown")),
        ("content", original_result.get("content")),
        ("summary", original_result.get("summary")),
        ("text", original_result.get("text")),
    ]

    for field_name, field_value in content_sources:
        if not field_value:
            continue

        content_str = str(field_value)
        content_tokens = estimate_tokens(content_str)

        if content_tokens <= available_tokens:
            # Content fits completely
            essential_result[field_name] = content_str
            essential_result[f"{field_name}_info"] = (
                f"Complete {field_name} content ({content_tokens} tokens)"
            )
            return True
        else:
            # Truncate content to fit available tokens
            truncated_content = _fit_content_to_tokens(
                content_str, content_tokens, available_tokens
            )

            if truncated_content:
                percentage = int((len(truncated_content) / len(content_str)) * 100)
                essential_result[field_name] = (
                    truncated_content +
                    "\n\n[TRUNCATED - Content continues beyond token limit]"
                )
                essential_result[f"{field_name}_info"] = (
                    f"Partial {field_name} ({percentage}% of original, "
                    f"{estimate_tokens(truncated_content)}/{content_tokens} tokens)"
                )
                return True
            # Truncation failed for this field, try next source instead of breaking
            continue

    return False


def _fit_content_to_tokens(
    content: str,
    content_tokens: int,
    available_tokens: int
) -> Optional[str]:
    """Fit content to available token budget."""
    chars_per_token = len(content) / content_tokens if content_tokens > 0 else 4
    estimated_chars = int(available_tokens * chars_per_token)

    truncated_content = content[:estimated_chars]
    actual_tokens = estimate_tokens(truncated_content)

    # Adjust if estimation was off
    while actual_tokens > available_tokens and len(truncated_content) > 100:
        truncated_content = truncated_content[:int(len(truncated_content) * 0.9)]
        actual_tokens = estimate_tokens(truncated_content)

    return truncated_content if truncated_content else None


def _add_transcript_content(
    original_result: Dict[str, Any],
    essential_result: Dict[str, Any],
    available_tokens: int,
    content_added: bool
) -> None:
    """Add YouTube transcript content with intelligent truncation."""
    transcript_entries = original_result["transcript"]
    total_entries = len(transcript_entries)

    # Use remaining available tokens if content wasn't added
    transcript_available_tokens = (
        available_tokens if not content_added else max(available_tokens // 2, 1000)
    )

    # Find maximum number of entries that fit
    entries_to_include = []
    for i, entry in enumerate(transcript_entries):
        test_entries = transcript_entries[:i + 1]
        test_size = estimate_tokens(json.dumps(test_entries))
        if test_size > transcript_available_tokens:
            break
        entries_to_include = test_entries

    if entries_to_include:
        essential_result["transcript"] = entries_to_include
        included_count = len(entries_to_include)
        percentage = int((included_count / total_entries) * 100)
        essential_result["transcript_truncated_info"] = (
            f"Showing {included_count} of {total_entries} entries ({percentage}%)"
        )
        essential_result["transcript_total_entries"] = total_entries

        # Calculate time coverage if timestamps are available
        if entries_to_include and "start" in entries_to_include[-1]:
            last_timestamp = entries_to_include[-1].get("start", 0)
            essential_result["transcript_time_coverage_seconds"] = last_timestamp
            essential_result["transcript_time_coverage_formatted"] = (
                f"{int(last_timestamp // 60)}:{int(last_timestamp % 60):02d}"
            )
