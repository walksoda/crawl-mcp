"""
Intelligent content extraction using LLM.

Contains _internal_intelligent_extract and intelligent_extract wrapper
extracted from tools/web_crawling.py.
"""

import json
from typing import Any, Dict, Optional

from ..models import CrawlRequest


async def _internal_intelligent_extract(
    url: str,
    extraction_goal: str,
    content_filter: str = "bm25",
    filter_query: Optional[str] = None,
    chunk_content: bool = False,
    use_llm: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal intelligent extract implementation.
    Uses LLM and content filtering for targeted extraction.
    """
    from .crawler_core import _internal_crawl_url

    try:
        # First crawl the URL to get the content
        request = CrawlRequest(
            url=url,
            content_filter=content_filter,
            filter_query=filter_query,
            chunk_content=chunk_content,
            generate_markdown=True,
            include_cleaned_html=True
        )

        crawl_result = await _internal_crawl_url(request)

        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }

        # If LLM processing is disabled, return the crawled content
        if not use_llm:
            return {
                "url": url,
                "success": True,
                "extracted_content": crawl_result.content,
                "extraction_goal": extraction_goal,
                "processing_method": "basic_crawl_only"
            }

        # Implement LLM-based intelligent extraction
        try:
            from ..utils.llm_extraction import LLMExtractionClient

            # Create LLM client from config
            client = LLMExtractionClient.from_config(llm_provider, llm_model)

            # Prepare extraction prompt
            extraction_prompt = f"""
            You are an expert content analyst. Your task is to extract specific information from web content based on the extraction goal.

            EXTRACTION GOAL: {extraction_goal}

            INSTRUCTIONS:
            - Focus specifically on information relevant to the extraction goal
            - Extract concrete data, statistics, quotes, and specific details
            - Maintain accuracy and preserve exact information from the source
            - Organize findings in a structured, easy-to-understand format
            - If the content doesn't contain relevant information, clearly state that

            {f"ADDITIONAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}

            Please provide a JSON response with the following structure:
            {{
                "extracted_data": "The specific information extracted according to the goal",
                "key_findings": ["List", "of", "main", "findings"],
                "relevant_quotes": ["Important", "quotes", "from", "source"],
                "statistics_data": ["Numerical", "data", "and", "statistics"],
                "sources_references": ["References", "to", "specific", "sections"],
                "extraction_confidence": "High/Medium/Low - confidence in extraction quality",
                "missing_information": ["Information", "sought", "but", "not", "found"]
            }}

            CONTENT TO ANALYZE:
            {crawl_result.content[:50000]}  # Limit content to prevent token overflow
            """

            system_message = "You are an expert content analyst specializing in precise information extraction."

            extracted_content = await client.call_llm(
                prompt=extraction_prompt,
                system_message=system_message,
                temperature=0.1,
                max_tokens=4000
            )

            # Parse JSON response
            if extracted_content:
                try:
                    extraction_data = client.parse_json_response(extracted_content)

                    return {
                        "url": url,
                        "success": True,
                        "extraction_goal": extraction_goal,
                        "extracted_data": extraction_data.get("extracted_data", ""),
                        "key_findings": extraction_data.get("key_findings", []),
                        "relevant_quotes": extraction_data.get("relevant_quotes", []),
                        "statistics_data": extraction_data.get("statistics_data", []),
                        "sources_references": extraction_data.get("sources_references", []),
                        "extraction_confidence": extraction_data.get("extraction_confidence", "Medium"),
                        "missing_information": extraction_data.get("missing_information", []),
                        "processing_method": "llm_intelligent_extraction",
                        "llm_provider": client.provider,
                        "llm_model": client.model,
                        "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                        "custom_instructions_used": bool(custom_instructions)
                    }

                except (json.JSONDecodeError, AttributeError) as e:
                    # Fallback: treat as plain text extraction
                    return {
                        "url": url,
                        "success": True,
                        "extraction_goal": extraction_goal,
                        "extracted_data": str(extracted_content),
                        "key_findings": [],
                        "relevant_quotes": [],
                        "statistics_data": [],
                        "sources_references": [],
                        "extraction_confidence": "Medium",
                        "missing_information": [],
                        "processing_method": "llm_intelligent_extraction_fallback",
                        "llm_provider": client.provider,
                        "llm_model": client.model,
                        "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                        "custom_instructions_used": bool(custom_instructions),
                        "json_parse_error": str(e)
                    }
            else:
                return {
                    "url": url,
                    "success": False,
                    "error": "LLM extraction returned empty result"
                }

        except Exception as llm_error:
            # LLM processing failed, return crawled content with error info
            return {
                "url": url,
                "success": True,  # Still return success since we have crawled content
                "extraction_goal": extraction_goal,
                "extracted_data": crawl_result.content,
                "key_findings": [],
                "relevant_quotes": [],
                "statistics_data": [],
                "sources_references": [],
                "extraction_confidence": "Low",
                "missing_information": [],
                "processing_method": "crawl_fallback_due_to_llm_error",
                "llm_error": str(llm_error),
                "original_content_length": len(crawl_result.content) if crawl_result.content else 0,
                "custom_instructions_used": bool(custom_instructions)
            }

    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Intelligent extraction error: {str(e)}"
        }


async def intelligent_extract(
    url: str,
    extraction_goal: str,
    content_filter: str = "bm25",
    filter_query: Optional[str] = None,
    chunk_content: bool = False,
    use_llm: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Extract specific data from web pages using LLM."""
    return await _internal_intelligent_extract(
        url=url,
        extraction_goal=extraction_goal,
        content_filter=content_filter,
        filter_query=filter_query,
        chunk_content=chunk_content,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
        custom_instructions=custom_instructions
    )
