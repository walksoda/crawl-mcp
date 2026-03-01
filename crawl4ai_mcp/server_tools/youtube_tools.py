"""
YouTube tool registrations.
"""

from ._shared import (
    Annotated, Dict, Field, List, Optional, Union, Any,
    apply_token_limit,
    _apply_content_slicing,
    validate_content_slicing_params,
    modules_unavailable_error,
)


def register_youtube_tools(mcp, get_modules):
    """Register YouTube-related MCP tools."""

    @mcp.tool()
    async def extract_youtube_transcript(
        url: Annotated[str, Field(description="YouTube video URL")],
        languages: Annotated[Optional[Union[List[str], str]], Field(description="Language codes in preference order")] = ["ja", "en"],
        translate_to: Annotated[Optional[str], Field(description="Target language for translation")] = None,
        include_timestamps: Annotated[bool, Field(description="Include timestamps")] = False,
        preserve_formatting: Annotated[bool, Field(description="Preserve formatting")] = True,
        include_metadata: Annotated[bool, Field(description="Include video metadata")] = True,
        auto_summarize: Annotated[bool, Field(description="Auto-summarize large content")] = False,
        max_content_tokens: Annotated[int, Field(description="Max tokens before summarization")] = 15000,
        summary_length: Annotated[str, Field(description="'short'|'medium'|'long'")] = "medium",
        llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
        llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
        enable_crawl_fallback: Annotated[bool, Field(description="Enable page crawl fallback when API fails")] = True,
        fallback_timeout: Annotated[int, Field(description="Fallback crawl timeout in seconds")] = 60,
        enrich_metadata: Annotated[bool, Field(description="Enrich metadata (upload_date, view_count) via page crawl")] = True,
        content_offset: Annotated[int, Field(description="Start position for content (0-indexed)")] = 0,
        content_limit: Annotated[int, Field(description="Max characters to return (0=unlimited)")] = 0,
    ) -> dict:
        """Extract YouTube transcripts with timestamps. Works with public captioned videos. Supports fallback to page crawl."""
        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        _, _, youtube, _, _ = modules

        # Handle string-encoded array for languages parameter
        if isinstance(languages, str):
            try:
                import json
                languages = json.loads(languages)
            except (json.JSONDecodeError, ValueError):
                import re
                matches = re.findall(r'"([^"]*)"', languages)
                if matches:
                    languages = matches
                else:
                    languages = ["ja", "en"]

        # Validate content slicing params
        slicing_error = validate_content_slicing_params(content_limit, content_offset)
        if slicing_error:
            return slicing_error

        try:
            result = await youtube.extract_youtube_transcript(
                url=url, languages=languages, translate_to=translate_to,
                include_timestamps=include_timestamps, preserve_formatting=preserve_formatting,
                include_metadata=include_metadata, auto_summarize=auto_summarize,
                max_content_tokens=max_content_tokens, summary_length=summary_length,
                llm_provider=llm_provider, llm_model=llm_model,
                enable_crawl_fallback=enable_crawl_fallback, fallback_timeout=fallback_timeout,
                enrich_metadata=enrich_metadata
            )

            # Apply content slicing if requested
            if content_limit > 0 or content_offset > 0:
                result = _apply_content_slicing(result, content_limit, content_offset)

            # Apply token limit fallback to prevent MCP errors
            result_with_fallback = apply_token_limit(result, max_tokens=25000)

            # Add YouTube-specific recommendations when truncation occurs
            if result_with_fallback.get("token_limit_applied") or result_with_fallback.get("emergency_truncation"):
                youtube_recommendations = [
                    "For long YouTube videos, consider using crawl_url for different extraction options",
                    f"Example: crawl_url(url='{url}', wait_for_js=true)",
                    "Use content_limit and content_offset to retrieve content in chunks",
                ]
                existing_recs = result_with_fallback.get("recommendations", [])
                result_with_fallback["recommendations"] = youtube_recommendations + existing_recs

                if not auto_summarize:
                    result_with_fallback["suggestion"] = "Transcript was truncated due to MCP token limits. Consider using content_limit/content_offset for pagination, or set auto_summarize=True (requires OPENAI_API_KEY)."

            return result_with_fallback

        except Exception as e:
            return {
                "success": False,
                "error": f"YouTube transcript error: {str(e)}"
            }

    @mcp.tool()
    async def batch_extract_youtube_transcripts(
        request: Annotated[Dict[str, Any], Field(description="Dict with: urls (max 3), languages, include_timestamps")]
    ) -> Dict[str, Any]:
        """Extract transcripts from multiple YouTube videos. Max 3 URLs per call."""
        # URL limit check (MCP best practice: bounded toolsets)
        urls = request.get('urls', [])
        if len(urls) > 3:
            return {"success": False, "error": "Maximum 3 YouTube URLs allowed per batch. Split into multiple calls."}

        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        _, _, youtube, _, _ = modules

        try:
            result = await youtube.batch_extract_youtube_transcripts(request)

            # Apply token limit fallback to prevent MCP errors
            result_with_fallback = apply_token_limit(result, max_tokens=25000)

            # If token limit was applied, provide helpful suggestion
            if result_with_fallback.get("token_limit_applied"):
                if not result_with_fallback.get("emergency_truncation"):
                    result_with_fallback["suggestion"] = "Batch transcript data was truncated due to MCP token limits. Consider reducing the number of videos or enabling auto_summarize for individual videos."

            return result_with_fallback

        except Exception as e:
            return {
                "success": False,
                "error": f"Batch YouTube extraction error: {str(e)}"
            }

    @mcp.tool()
    async def get_youtube_video_info(
        video_url: Annotated[str, Field(description="YouTube video URL")],
        summarize_transcript: Annotated[bool, Field(description="Summarize transcript")] = False,
        max_tokens: Annotated[int, Field(description="Token limit for summarization")] = 25000,
        llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
        llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
        summary_length: Annotated[str, Field(description="'short'|'medium'|'long'")] = "medium",
        include_timestamps: Annotated[bool, Field(description="Include timestamps")] = True
    ) -> Dict[str, Any]:
        """Get YouTube video metadata and transcript availability."""
        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        _, _, youtube, _, _ = modules

        try:
            result = await youtube.get_youtube_video_info(
                video_url=video_url, summarize_transcript=summarize_transcript,
                max_tokens=max_tokens, llm_provider=llm_provider, llm_model=llm_model,
                summary_length=summary_length, include_timestamps=include_timestamps
            )

            # Apply token limit fallback to prevent MCP errors
            result_with_fallback = apply_token_limit(result, max_tokens=25000)

            # If token limit was applied and summarize_transcript was False, provide helpful suggestion
            if result_with_fallback.get("token_limit_applied") and not summarize_transcript:
                if not result_with_fallback.get("emergency_truncation"):
                    result_with_fallback["suggestion"] = "Video info was truncated due to MCP token limits. Consider setting summarize_transcript=True for long transcripts."

            return result_with_fallback

        except Exception as e:
            return {
                "success": False,
                "error": f"YouTube video info error: {str(e)}"
            }

    async def get_youtube_api_setup_guide() -> Dict[str, Any]:
        """Get youtube-transcript-api setup info. No API key required."""
        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        _, _, youtube, _, _ = modules

        try:
            result = await youtube.get_youtube_api_setup_guide()
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"YouTube API setup guide error: {str(e)}"
            }
