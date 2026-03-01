"""
Crawler summarization and token limit enforcement.

Contains summarize_web_content, _check_and_summarize_if_needed,
and _finalize_fallback_response extracted from tools/web_crawling.py.
"""

from typing import Any, Dict, Optional

from ..models import CrawlRequest, CrawlResponse
from ..constants import MAX_RESPONSE_TOKENS

# Response size limit for MCP protocol
MAX_RESPONSE_CHARS = MAX_RESPONSE_TOKENS * 4  # Rough estimate: 1 token ≈ 4 characters


async def summarize_web_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize web content using LLM with fallback to basic text truncation.
    """
    try:
        from ..utils.llm_extraction import LLMExtractionClient

        # Create LLM client from config
        client = LLMExtractionClient.from_config(llm_provider, llm_model)

        # Prepare summarization prompt based on summary length
        length_instructions = {
            "short": "Provide a brief 2-3 sentence summary of the main points.",
            "medium": "Provide a comprehensive summary in 1-2 paragraphs covering key points.",
            "long": "Provide a detailed summary covering all important information, insights, and context."
        }

        prompt = f"""
        Please summarize the following web content.

        Title: {title}
        URL: {url}

        {length_instructions.get(summary_length, length_instructions["medium"])}

        Focus on:
        - Main topics and key information
        - Important facts, statistics, or findings
        - Practical insights or conclusions
        - Technical details if present

        Content to summarize:
        {content[:50000]}  # Limit to prevent token overflow
        """

        system_message = "You are an expert content summarizer. Provide clear, concise summaries."

        summary_text = await client.call_llm(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,
            max_tokens=2000
        )

        if summary_text:
            # Calculate compression ratio
            original_length = len(content)
            summary_length_chars = len(summary_text)
            compression_ratio = round((1 - summary_length_chars / original_length) * 100, 2) if original_length > 0 else 0

            return {
                "success": True,
                "summary": summary_text,
                "original_length": original_length,
                "summary_length": summary_length_chars,
                "compressed_ratio": compression_ratio,
                "llm_provider": client.provider,
                "llm_model": client.model,
                "content_type": "web_content"
            }
        else:
            raise ValueError("LLM returned empty summary")

    except Exception as e:
        # Fallback to simple truncation if LLM fails
        max_chars = {
            "short": 500,
            "medium": 1500,
            "long": 3000
        }.get(summary_length, 1500)

        truncated = content[:max_chars]
        if len(content) > max_chars:
            truncated += "... [Content truncated due to size]"

        return {
            "success": False,
            "error": f"LLM summarization failed: {str(e)}. Returning truncated content.",
            "summary": truncated,
            "original_length": len(content),
            "summary_length": len(truncated),
            "compressed_ratio": round((1 - len(truncated) / len(content)) * 100, 2) if content else 0,
            "fallback_method": "truncation"
        }


async def _check_and_summarize_if_needed(
    response: CrawlResponse,
    request: CrawlRequest
) -> CrawlResponse:
    """
    Check if response content exceeds token limits and apply summarization if needed.
    Respects user-specified limits when provided.
    """
    # Skip if response failed
    if not response.success:
        return response

    # Skip if neither content nor markdown exists
    if not response.content and not response.markdown:
        return response

    # Check if already summarized
    if response.extracted_data and response.extracted_data.get("summarization_applied"):
        return response

    # Estimate total response size (content + markdown + metadata)
    total_chars = len(response.content or "") + len(response.markdown or "")

    # Determine effective token limit - user-specified takes precedence
    effective_limit_chars = MAX_RESPONSE_CHARS
    limit_source = "MCP_PROTOCOL_SAFETY"

    # If user explicitly set auto_summarize=True and max_content_tokens, use that instead
    if hasattr(request, 'auto_summarize') and request.auto_summarize:
        if hasattr(request, 'max_content_tokens') and request.max_content_tokens:
            # Convert user's token limit to character estimate
            user_limit_chars = request.max_content_tokens * 4
            # Use the smaller limit (user preference or protocol safety)
            if user_limit_chars < effective_limit_chars:
                effective_limit_chars = user_limit_chars
                limit_source = "USER_SPECIFIED"

    # Check if response exceeds the effective limit
    if total_chars > effective_limit_chars:
        try:
            # Use markdown if available, otherwise use content
            content_to_summarize = response.markdown or response.content

            # Determine summary length based on reduction needed
            reduction_ratio = effective_limit_chars / total_chars
            if reduction_ratio > 0.5:
                summary_length = "medium"
            elif reduction_ratio > 0.3:
                summary_length = "short"
            else:
                summary_length = "short"  # Aggressive reduction needed

            # Force summarization to meet token limits
            summary_result = await summarize_web_content(
                content=content_to_summarize,
                title=response.title or "",
                url=response.url,
                summary_length=summary_length,
                llm_provider=request.llm_provider if hasattr(request, 'llm_provider') else None,
                llm_model=request.llm_model if hasattr(request, 'llm_model') else None
            )

            if summary_result.get("success"):
                summary_text = summary_result['summary']

                # Build prefix based on limit source
                if limit_source == "USER_SPECIFIED":
                    prefix = f"Content exceeded user-specified limit ({total_chars:,} chars > {effective_limit_chars:,} chars). Auto-summarized:\n\n"
                else:
                    prefix = f"Content exceeded MCP token limit ({total_chars:,} chars > {effective_limit_chars:,} chars). Auto-summarized:\n\n"

                # Verify summarized content fits within limits with strict control
                final_content = f"{prefix}{summary_text}"
                if len(final_content) > effective_limit_chars:
                    # Truncate if summary still exceeds limit
                    suffix = "... [Summary truncated]"
                    available_chars = effective_limit_chars - len(prefix) - len(suffix)

                    if available_chars > 0:
                        summary_text = summary_text[:available_chars] + suffix
                    else:
                        # Extreme case: prefix alone exceeds limit, use minimal prefix
                        minimal_prefix = "[Exceeded limit] "
                        available_chars = effective_limit_chars - len(minimal_prefix) - len(suffix)
                        if available_chars > 0:
                            summary_text = summary_text[:available_chars] + suffix
                            prefix = minimal_prefix
                        else:
                            # Ultimate fallback: no content fits
                            summary_text = ""
                            prefix = f"[Content exceeded {effective_limit_chars} char limit]"

                    final_content = f"{prefix}{summary_text}"

                    # Final safety check - hard truncate if still over limit
                    if len(final_content) > effective_limit_chars:
                        final_content = final_content[:effective_limit_chars - 3] + "..."

                response.content = final_content
                response.markdown = final_content

                # Update extracted_data
                if response.extracted_data is None:
                    response.extracted_data = {}

                response.extracted_data.update({
                    "auto_summarization_reason": limit_source,
                    "original_size_chars": total_chars,
                    "effective_limit_chars": effective_limit_chars,
                    "user_specified_limit": limit_source == "USER_SPECIFIED",
                    "user_max_content_tokens": request.max_content_tokens if hasattr(request, 'max_content_tokens') else None,
                    "summarization_applied": True,
                    "summary_length": summary_length,
                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                    "llm_model": summary_result.get("llm_model", "unknown"),
                    "post_summary_truncated": len(f"{prefix}{summary_result['summary']}") > effective_limit_chars,
                })
            else:
                # Summarization failed, truncate content
                content_to_truncate = response.content or response.markdown or ""
                truncate_at = max(100, effective_limit_chars - 500)
                prefix = f"Content exceeded limit ({total_chars:,} chars > {effective_limit_chars:,} chars).\n\nSummarization failed: {summary_result.get('error', 'Unknown error')}\n\nTruncated content:\n\n"
                response.content = f"{prefix}{content_to_truncate[:truncate_at]}... [Content truncated]"
                response.markdown = response.content

                if response.extracted_data is None:
                    response.extracted_data = {}

                response.extracted_data.update({
                    "auto_truncation_reason": limit_source,
                    "original_size_chars": total_chars,
                    "effective_limit_chars": effective_limit_chars,
                    "user_specified_limit": limit_source == "USER_SPECIFIED",
                    "truncation_applied": True,
                    "summarization_attempted": True,
                    "summarization_error": summary_result.get("error", "Unknown error")
                })

        except Exception as e:
            # Final fallback: aggressive truncation
            content_to_truncate = response.content or response.markdown or ""
            truncate_at = max(100, effective_limit_chars - 500)
            prefix = f"Content exceeded limit ({total_chars:,} chars > {effective_limit_chars:,} chars).\n\nEmergency truncation applied due to error: {str(e)}\n\n"
            response.content = f"{prefix}{content_to_truncate[:truncate_at]}... [Content truncated]"
            response.markdown = response.content

            if response.extracted_data is None:
                response.extracted_data = {}

            response.extracted_data.update({
                "emergency_truncation_reason": limit_source,
                "original_size_chars": total_chars,
                "effective_limit_chars": effective_limit_chars,
                "user_specified_limit": limit_source == "USER_SPECIFIED",
                "truncation_error": str(e)
            })

    return response


async def _finalize_fallback_response(
    response: CrawlResponse,
    request_url: str,
    auto_summarize: bool = False,
    max_content_tokens: int = 15000,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> CrawlResponse:
    """
    Apply size limit and summarization to fallback responses.

    This ensures all fallback return points go through the same
    size checking and summarization as regular crawl responses.
    """
    # Create a dummy CrawlRequest with the necessary parameters
    dummy_request = CrawlRequest(
        url=request_url,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    return await _check_and_summarize_if_needed(response, dummy_request)
