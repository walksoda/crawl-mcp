"""
YouTube tool registrations.
"""

from ._shared import (
    Annotated, Dict, Field, List, Optional, Union, Any,
    apply_token_limit,
    _apply_content_slicing,
    validate_content_slicing_params,
    validate_output_path,
    finalize_tool_response,
    KIND_MARKDOWN_SINGLE,
    KIND_MARKDOWN_BATCH_DICT,
    KIND_YOUTUBE_COMMENTS,
    modules_unavailable_error,
    READONLY_ANNOTATIONS,
)


def register_youtube_tools(mcp, get_modules):
    """Register YouTube-related MCP tools."""

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
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
        output_path: Annotated[Optional[str], Field(description="Absolute file path (auto .md extension) to persist the full unsliced transcript. When set, the response is slimmed to metadata+file path. content_limit/content_offset still affect the response copy but not the on-disk file.")] = None,
        include_content_in_response: Annotated[bool, Field(description="When True (with output_path set), keep the transcript in the response too. Note: the response copy is still subject to content_limit/content_offset slicing; only the on-disk file holds the full transcript. Defaults to False.")] = False,
        overwrite: Annotated[bool, Field(description="Overwrite an existing output file at output_path. Defaults to False.")] = False,
    ) -> dict:
        """Extract YouTube transcripts with timestamps. Works with public captioned videos. Supports fallback to page crawl. Use output_path to persist the full unsliced transcript to disk as markdown."""
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

        # Output path validation (Guard A)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

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

            # Guard B: persist BEFORE slicing/truncation so disk holds full transcript.
            if output_path:
                result = finalize_tool_response(
                    result,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_SINGLE,
                    source_tool="extract_youtube_transcript",
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

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def batch_extract_youtube_transcripts(
        request: Annotated[Dict[str, Any], Field(description="Dict with: urls (max 3), languages, include_timestamps. Optional persistence keys: output_path (absolute directory — per-video .md files + index.json; dot-containing dir names are fine), include_content_in_response (bool; default False — when True, per-video transcripts stay in the response as well), overwrite (bool; default False — existing files rejected). Failed items (success=False) are recorded in index.json with file=null but no .md is written.")]
    ) -> Dict[str, Any]:
        """Extract transcripts from multiple YouTube videos. Max 3 URLs per call. Supply output_path (directory) in the request to persist per-video markdown files + index.json and receive a slim response."""
        # URL limit check (MCP best practice: bounded toolsets)
        urls = request.get('urls', [])
        if len(urls) > 3:
            return {"success": False, "error": "Maximum 3 YouTube URLs allowed per batch. Split into multiple calls."}

        # Read persistence options from request dict.
        output_path = request.get('output_path')
        include_content_in_response = bool(request.get('include_content_in_response', False))
        overwrite = bool(request.get('overwrite', False))

        # Output path validation (Guard A)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        _, _, youtube, _, _ = modules

        try:
            result = await youtube.batch_extract_youtube_transcripts(request)

            # Guard B: persist BEFORE apply_token_limit so disk holds full data.
            if output_path:
                result = finalize_tool_response(
                    result,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_BATCH_DICT,
                    source_tool="batch_extract_youtube_transcripts",
                )

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

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def get_youtube_video_info(
        video_url: Annotated[str, Field(description="YouTube video URL")],
        summarize_transcript: Annotated[bool, Field(description="Summarize transcript")] = False,
        max_tokens: Annotated[int, Field(description="Token limit for summarization")] = 25000,
        llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
        llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
        summary_length: Annotated[str, Field(description="'short'|'medium'|'long'")] = "medium",
        include_timestamps: Annotated[bool, Field(description="Include timestamps")] = True,
        output_path: Annotated[Optional[str], Field(description="Absolute file path (auto .md extension) to persist the full video info + transcript as markdown. When set, the response is slimmed to metadata+file path.")] = None,
        include_content_in_response: Annotated[bool, Field(description="When True (with output_path set), also include the full transcript in the response. Defaults to False.")] = False,
        overwrite: Annotated[bool, Field(description="Overwrite an existing output file at output_path. Defaults to False.")] = False,
    ) -> Dict[str, Any]:
        """Get YouTube video metadata and transcript availability. Use output_path to persist the full transcript to disk as markdown and receive a slim response."""
        # Output path validation (Guard A)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

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

            # Guard B: persist BEFORE apply_token_limit.
            if output_path:
                result = finalize_tool_response(
                    result,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_MARKDOWN_SINGLE,
                    source_tool="get_youtube_video_info",
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

    @mcp.tool(annotations=READONLY_ANNOTATIONS)
    async def extract_youtube_comments(
        url: Annotated[str, Field(description="YouTube video URL")],
        sort_by: Annotated[str, Field(description="'popular'|'recent'")] = "popular",
        max_comments: Annotated[int, Field(description="Max comments to retrieve (1-1000)")] = 300,
        comment_offset: Annotated[int, Field(description="Number of comments to skip (for pagination)")] = 0,
        include_replies: Annotated[bool, Field(description="Include reply comments")] = True,
        content_offset: Annotated[int, Field(description="Start position for content (0-indexed)")] = 0,
        content_limit: Annotated[int, Field(description="Max characters to return (0=unlimited)")] = 0,
        output_path: Annotated[Optional[str], Field(description="Absolute file path (auto .json extension) to persist the full unsliced/untruncated comment list. When set, the complete comment array is written to disk BEFORE the internal token-limit reduction, and the response is slimmed (extracted_data.comments removed but comment_count etc. kept).")] = None,
        include_content_in_response: Annotated[bool, Field(description="When True (with output_path set), keep the comment list in the response too. Note: the response copy is still subject to content_limit/content_offset slicing and the token-limit comment-array reduction; only the on-disk file holds the full list. Defaults to False.")] = False,
        overwrite: Annotated[bool, Field(description="Overwrite an existing output file at output_path. Defaults to False.")] = False,
    ) -> dict:
        """Extract YouTube video comments. Supports pagination via comment_offset. Use output_path to persist the full unsliced comment list to disk as JSON; the response is then slimmed to metadata only."""
        modules = get_modules()
        if not modules:
            return {"success": False, "error": "Tool modules not available"}
        _, _, youtube, _, _ = modules

        # Validate content slicing params
        slicing_error = validate_content_slicing_params(content_limit, content_offset)
        if slicing_error:
            return slicing_error

        # Output path validation (Guard A)
        output_error = validate_output_path(output_path, overwrite)
        if output_error:
            return output_error

        try:
            result = await youtube.extract_youtube_comments(
                url=url, sort_by=sort_by, max_comments=max_comments,
                include_replies=include_replies, comment_offset=comment_offset,
            )

            # Guard B: persist BEFORE any internal truncation so disk holds the
            # full comment list. Slicing and the comments-array shrink below
            # only affect the response copy.
            if output_path:
                result = finalize_tool_response(
                    result,
                    output_path=output_path,
                    include_content_in_response=include_content_in_response,
                    overwrite=overwrite,
                    tool_kind=KIND_YOUTUBE_COMMENTS,
                    source_tool="extract_youtube_comments",
                )

            # Apply content slicing if requested
            if content_limit > 0 or content_offset > 0:
                result = _apply_content_slicing(result, content_limit, content_offset)

            # Pre-truncate comments array to prevent token limit bypass
            # The generic apply_token_limit only truncates top-level string/list
            # fields and does not reach nested arrays inside extracted_data
            extracted = result.get("extracted_data")
            if isinstance(extracted, dict) and "comments" in extracted:
                import json as _json
                from ..utils.token_utils import estimate_tokens
                result_tokens = estimate_tokens(_json.dumps(result))
                if result_tokens > 25000:
                    comments_list = extracted["comments"]
                    original_count = len(comments_list)
                    # Progressively reduce until under limit
                    while len(comments_list) > 1:
                        comments_list = comments_list[:len(comments_list) // 2]
                        extracted["comments"] = comments_list
                        if estimate_tokens(_json.dumps(result)) <= 25000:
                            break
                    if original_count != len(comments_list):
                        extracted["comments_truncated_from"] = original_count
                        extracted["comments_truncated_info"] = (
                            f"Showing {len(comments_list)} of {original_count} comments. "
                            "Use comment_offset for pagination."
                        )

            # Apply token limit fallback to prevent MCP errors
            result_with_fallback = apply_token_limit(result, max_tokens=25000)

            # Add recommendations when truncation occurs
            if result_with_fallback.get("token_limit_applied") or result_with_fallback.get("emergency_truncation"):
                comment_recommendations = [
                    "Use max_comments to limit the number of comments retrieved",
                    "Use comment_offset to paginate through comments",
                    f"Example: extract_youtube_comments(url='{url}', max_comments=100, comment_offset={comment_offset + max_comments})",
                    "Use content_limit and content_offset to retrieve content in chunks",
                ]
                existing_recs = result_with_fallback.get("recommendations", [])
                result_with_fallback["recommendations"] = comment_recommendations + existing_recs
                result_with_fallback["suggestion"] = "Comments were truncated due to MCP token limits. Use comment_offset for pagination or reduce max_comments."

            return result_with_fallback

        except Exception as e:
            return {
                "success": False,
                "error": f"YouTube comment extraction error: {str(e)}"
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
