"""
Structured data extraction using CSS selectors and LLM.

Contains _internal_extract_structured_data and extract_structured_data wrapper
extracted from tools/web_crawling.py.
"""

import json
from typing import Any, Dict, Optional

from ..models import CrawlRequest, CrawlResponse, StructuredExtractionRequest


async def _internal_extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Internal extract structured data implementation.
    Supports both CSS selector and LLM-based extraction methods.
    """
    from .crawler_core import _internal_crawl_url

    try:
        crawl_request = CrawlRequest(
            url=request.url,
            generate_markdown=True,
            include_cleaned_html=True
        )

        crawl_result = await _internal_crawl_url(crawl_request)

        if not crawl_result.success:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Failed to crawl URL for structured extraction: {crawl_result.error}"
            )

        extracted_data = {}

        if request.extraction_type == "css" and request.css_selectors:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(crawl_result.content, 'html.parser')

                for field_name, css_selector in request.css_selectors.items():
                    elements = soup.select(css_selector)
                    if elements:
                        if len(elements) == 1:
                            extracted_data[field_name] = elements[0].get_text(strip=True)
                        else:
                            extracted_data[field_name] = [elem.get_text(strip=True) for elem in elements]
                    else:
                        extracted_data[field_name] = None

                return CrawlResponse(
                    success=True,
                    url=request.url,
                    title=crawl_result.title,
                    content=crawl_result.content,
                    markdown=crawl_result.markdown,
                    extracted_data={
                        "structured_data": extracted_data,
                        "extraction_method": "css_selectors",
                        "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                        "extracted_fields": list(extracted_data.keys())
                    }
                )

            except ImportError:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="BeautifulSoup4 not installed. Install with: pip install beautifulsoup4"
                )

        elif request.extraction_type == "llm":
            try:
                from ..utils.llm_extraction import LLMExtractionClient

                client = LLMExtractionClient.from_config(request.llm_provider, request.llm_model)

                schema_description = ""
                if request.extraction_schema:
                    schema_items = []
                    for field, description in request.extraction_schema.items():
                        schema_items.append(f"- {field}: {description}")
                    schema_description = "\n".join(schema_items)

                structured_prompt = f"""
                You are an expert data extraction specialist. Extract structured data from the given web content according to the specified schema.

                SCHEMA FIELDS TO EXTRACT:
                {schema_description}

                EXTRACTION INSTRUCTIONS:
                - Extract information for each field in the schema
                - Maintain accuracy and preserve exact information from the source
                - If a field's information is not found, set it to null
                - Return data in valid JSON format matching the schema structure
                - Focus on extracting concrete, factual information

                {f"ADDITIONAL INSTRUCTIONS: {request.instruction}" if request.instruction else ""}

                Please provide a JSON response with the following structure:
                {{
                    "structured_data": {{
                        // Fields matching the requested schema
                    }},
                    "extraction_confidence": "High/Medium/Low",
                    "found_fields": ["list", "of", "successfully", "extracted", "fields"],
                    "missing_fields": ["list", "of", "fields", "not", "found"],
                    "additional_context": "Any relevant context or notes about the extraction"
                }}

                WEB CONTENT TO ANALYZE:
                {crawl_result.content[:40000]}  # Limit content to prevent token overflow
                """

                system_message = "You are an expert data extraction specialist focused on accuracy and structured output."

                extracted_content = await client.call_llm(
                    prompt=structured_prompt,
                    system_message=system_message,
                    temperature=0.1,
                    max_tokens=4000
                )

                if extracted_content:
                    try:
                        extraction_result = client.parse_json_response(extracted_content)

                        return CrawlResponse(
                            success=True,
                            url=request.url,
                            title=crawl_result.title,
                            content=crawl_result.content,
                            markdown=crawl_result.markdown,
                            extracted_data={
                                "structured_data": extraction_result.get("structured_data", {}),
                                "extraction_method": "llm_based",
                                "extraction_confidence": extraction_result.get("extraction_confidence", "Medium"),
                                "found_fields": extraction_result.get("found_fields", []),
                                "missing_fields": extraction_result.get("missing_fields", []),
                                "additional_context": extraction_result.get("additional_context", ""),
                                "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "llm_provider": client.provider,
                                "llm_model": client.model,
                                "custom_instruction_used": bool(request.instruction)
                            }
                        )

                    except (json.JSONDecodeError, AttributeError) as e:
                        return CrawlResponse(
                            success=True,
                            url=request.url,
                            title=crawl_result.title,
                            content=crawl_result.content,
                            markdown=crawl_result.markdown,
                            extracted_data={
                                "structured_data": {"raw_extraction": str(extracted_content)},
                                "extraction_method": "llm_based_fallback",
                                "extraction_confidence": "Low",
                                "found_fields": [],
                                "missing_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "additional_context": f"JSON parsing failed: {str(e)}",
                                "schema_fields": list(request.extraction_schema.keys()) if request.extraction_schema else [],
                                "llm_provider": client.provider,
                                "llm_model": client.model,
                                "json_parse_error": str(e)
                            }
                        )
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error="LLM structured extraction returned empty result"
                    )

            except Exception as llm_error:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"LLM structured extraction failed: {str(llm_error)}"
                )

        else:
            return CrawlResponse(
                success=False,
                url=request.url,
                error=f"Unsupported extraction type: {request.extraction_type}. Supported types: 'css', 'llm'"
            )

    except Exception as e:
        return CrawlResponse(
            success=False,
            url=request.url,
            error=f"Structured data extraction error: {str(e)}"
        )


async def extract_structured_data(
    request: StructuredExtractionRequest
) -> CrawlResponse:
    """Extract structured data using CSS selectors or LLM."""
    return await _internal_extract_structured_data(request)
