"""
Extraction tool registrations: intelligent_extract, extract_entities, extract_structured_data.
"""

from ._shared import (
    Annotated, Dict, Field, List, Optional, Any,
    apply_token_limit,
    _convert_result_to_dict,
    modules_unavailable_error,
)


def register_extraction_tools(mcp, get_modules):
    """Register extraction-related MCP tools."""

    @mcp.tool()
    async def intelligent_extract(
        url: Annotated[str, Field(description="Target URL")],
        extraction_goal: Annotated[str, Field(description="Data to extract")],
        content_filter: Annotated[str, Field(description="'bm25'|'pruning'|'llm'")] = "bm25",
        filter_query: Annotated[Optional[str], Field(description="BM25 filter keywords")] = None,
        chunk_content: Annotated[bool, Field(description="Split content")] = False,
        use_llm: Annotated[bool, Field(description="Enable LLM")] = True,
        llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
        llm_model: Annotated[Optional[str], Field(description="LLM model")] = None,
        custom_instructions: Annotated[Optional[str], Field(description="LLM instructions")] = None
    ) -> Dict[str, Any]:
        """Extract specific data from web pages using LLM."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            result = await web_crawling.intelligent_extract(
                url=url, extraction_goal=extraction_goal, content_filter=content_filter,
                filter_query=filter_query, chunk_content=chunk_content, use_llm=use_llm,
                llm_provider=llm_provider, llm_model=llm_model, custom_instructions=custom_instructions
            )

            # Check if extraction was successful
            if result.get("success", True):
                return apply_token_limit(result, max_tokens=25000)

            # If intelligent extraction failed, try with fallback crawling
            try:
                fallback_crawl = await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=60
                )
                if fallback_crawl.get("success", False):
                    content = fallback_crawl.get("markdown", "") or fallback_crawl.get("content", "")
                    if content.strip():
                        fallback_response = {
                            "success": True,
                            "url": url,
                            "extraction_goal": extraction_goal,
                            "extracted_data": {
                                "raw_content": content[:2000] + ("..." if len(content) > 2000 else ""),
                                "note": "Fallback extraction - manual processing may be needed"
                            },
                            "content": fallback_crawl.get("content", ""),
                            "markdown": fallback_crawl.get("markdown", ""),
                            "fallback_used": True,
                            "original_error": result.get("error", "Intelligent extraction failed")
                        }
                        return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                result["fallback_error"] = str(fallback_error)
            return result

        except Exception as e:
            # If intelligent extraction throws an exception, try basic fallback
            try:
                fallback_crawl = await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=60
                )
                if fallback_crawl.get("success", False):
                    content = fallback_crawl.get("markdown", "") or fallback_crawl.get("content", "")
                    fallback_response = {
                        "success": True,
                        "url": url,
                        "extraction_goal": extraction_goal,
                        "extracted_data": {
                            "raw_content": content[:2000] + ("..." if len(content) > 2000 else ""),
                            "note": "Fallback extraction - manual processing may be needed"
                        },
                        "content": fallback_crawl.get("content", ""),
                        "markdown": fallback_crawl.get("markdown", ""),
                        "fallback_used": True,
                        "original_error": str(e)
                    }
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                pass
            return {
                "success": False,
                "error": f"Intelligent extraction error: {str(e)}"
            }

    @mcp.tool()
    async def extract_entities(
        url: Annotated[str, Field(description="Target URL")],
        entity_types: Annotated[List[str], Field(description="Types: email, phone, url, date, ip, price")],
        custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns")] = None,
        include_context: Annotated[bool, Field(description="Include context")] = True,
        deduplicate: Annotated[bool, Field(description="Remove duplicates")] = True,
        use_llm: Annotated[bool, Field(description="Use LLM for NER")] = False,
        llm_provider: Annotated[Optional[str], Field(description="LLM provider")] = None,
        llm_model: Annotated[Optional[str], Field(description="LLM model")] = None
    ) -> Dict[str, Any]:
        """Extract entities (emails, phones, etc.) from web pages."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            result = await web_crawling.extract_entities(
                url=url, entity_types=entity_types, custom_patterns=custom_patterns,
                include_context=include_context, deduplicate=deduplicate, use_llm=use_llm,
                llm_provider=llm_provider, llm_model=llm_model
            )

            # Check if entity extraction was successful
            if result.get("success", True):
                return apply_token_limit(result, max_tokens=25000)

            # If entity extraction failed, try with fallback crawling
            try:
                fallback_crawl = await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=60
                )
                if fallback_crawl.get("success", False):
                    content = fallback_crawl.get("content", "") or fallback_crawl.get("markdown", "")
                    # Basic regex-based entity extraction on fallback content
                    import re
                    entities = {}

                    if "emails" in entity_types:
                        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                        if emails:
                            entities["emails"] = list(set(emails)) if deduplicate else emails

                    if "phones" in entity_types:
                        phones = re.findall(r'[\+]?[1-9]?[0-9]{7,15}', content)
                        if phones:
                            entities["phones"] = list(set(phones)) if deduplicate else phones

                    if "urls" in entity_types:
                        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                        if urls:
                            entities["urls"] = list(set(urls)) if deduplicate else urls

                    fallback_response = {
                        "success": True,
                        "url": url,
                        "entities": entities,
                        "entity_types": entity_types,
                        "total_found": sum(len(v) for v in entities.values()),
                        "content": content[:500] + ("..." if len(content) > 500 else ""),
                        "fallback_used": True,
                        "note": "Basic regex extraction used - some entity types may not be fully supported",
                        "original_error": result.get("error", "Entity extraction failed")
                    }
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                result["fallback_error"] = str(fallback_error)
            return result

        except Exception as e:
            # If entity extraction throws an exception, try basic fallback
            try:
                fallback_crawl = await web_crawling.crawl_url_with_fallback(
                    url=url, generate_markdown=True, timeout=60
                )
                if fallback_crawl.get("success", False):
                    content = fallback_crawl.get("content", "") or fallback_crawl.get("markdown", "")
                    # Basic regex-based entity extraction
                    import re
                    entities = {}

                    if "emails" in entity_types:
                        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                        if emails:
                            entities["emails"] = list(set(emails)) if deduplicate else emails

                    if "phones" in entity_types:
                        phones = re.findall(r'[\+]?[1-9]?[0-9]{7,15}', content)
                        if phones:
                            entities["phones"] = list(set(phones)) if deduplicate else phones

                    if "urls" in entity_types:
                        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                        if urls:
                            entities["urls"] = list(set(urls)) if deduplicate else urls

                    fallback_response = {
                        "success": True,
                        "url": url,
                        "entities": entities,
                        "entity_types": entity_types,
                        "total_found": sum(len(v) for v in entities.values()),
                        "content": content[:500] + ("..." if len(content) > 500 else ""),
                        "fallback_used": True,
                        "note": "Basic regex extraction used - some entity types may not be fully supported",
                        "original_error": str(e)
                    }
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception as fallback_error:
                pass
            return {
                "success": False,
                "error": f"Entity extraction error: {str(e)}"
            }

    @mcp.tool()
    async def extract_structured_data(
        url: Annotated[str, Field(description="Target URL")],
        extraction_type: Annotated[str, Field(description="'css'|'llm'|'table'")] = "css",
        css_selectors: Annotated[Optional[Dict[str, str]], Field(description="CSS selector mapping")] = None,
        extraction_schema: Annotated[Optional[Dict[str, str]], Field(description="Schema definition")] = None,
        generate_markdown: Annotated[bool, Field(description="Generate markdown")] = False,
        wait_for_js: Annotated[bool, Field(description="Wait for JavaScript")] = False,
        timeout: Annotated[int, Field(description="Timeout in seconds")] = 30,
        use_llm_table_extraction: Annotated[bool, Field(description="Use LLM table extraction")] = False,
        table_chunking_strategy: Annotated[str, Field(description="'intelligent'|'fixed'|'semantic'")] = "intelligent"
    ) -> Dict[str, Any]:
        """Extract structured data using CSS selectors or LLM."""
        modules = get_modules()
        if not modules:
            return modules_unavailable_error()
        web_crawling, search, youtube, file_processing, utilities = modules

        try:
            # NEW: LLM Table Extraction mode
            if extraction_type == "table" or use_llm_table_extraction:
                try:
                    result = await web_crawling.extract_structured_data(
                        url=url,
                        extraction_type="llm_table",
                        extraction_schema=extraction_schema,
                        generate_markdown=generate_markdown,
                        wait_for_js=wait_for_js,
                        timeout=timeout,
                        chunking_strategy=table_chunking_strategy
                    )

                    if result.get("success", False):
                        result["processing_method"] = "llm_table_extraction"
                        result["features_used"] = ["intelligent_chunking", "massive_table_support"]
                        # Apply token limit fallback before returning
                        return apply_token_limit(result, max_tokens=25000)

                except Exception as table_error:
                    # Fallback to CSS extraction if table extraction fails
                    if css_selectors:
                        extraction_type = "css"
                    else:
                        return {
                            "success": False,
                            "error": f"LLM table extraction failed: {str(table_error)}",
                            "suggested_fallback": "Try with css_selectors or extraction_type='css'"
                        }

            # CSS selectors provided and extraction_type is css
            if css_selectors and extraction_type == "css":
                # Use basic crawling with CSS selector post-processing
                try:
                    # Basic crawl first
                    crawl_result = await web_crawling.crawl_url(
                        url=url,
                        generate_markdown=generate_markdown,
                        wait_for_js=wait_for_js,
                        timeout=timeout
                    )

                    # If initial crawl fails, try fallback
                    if not crawl_result.get("success", False) or not crawl_result.get("content", "").strip():
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            generate_markdown=generate_markdown,
                            wait_for_js=wait_for_js,
                            timeout=timeout,
                            use_undetected_browser=True
                        )

                        if fallback_result.get("success", False):
                            crawl_result = fallback_result
                            crawl_result["fallback_used"] = True
                        else:
                            return crawl_result

                    # Enhanced CSS selector extraction with table detection
                    from bs4 import BeautifulSoup

                    html_content = crawl_result.get("content", "")
                    soup = BeautifulSoup(html_content, 'html.parser')

                    extracted_data = {}
                    tables_found = []

                    # Enhanced table detection and extraction
                    tables = soup.find_all('table')
                    if tables and use_llm_table_extraction:
                        for i, table in enumerate(tables):
                            table_data = {
                                "table_index": i,
                                "headers": [],
                                "rows": [],
                                "extraction_method": "enhanced_css_with_table_support"
                            }

                            # Extract headers
                            headers = table.find_all(['th', 'td'])
                            if headers:
                                table_data["headers"] = [h.get_text().strip() for h in headers[:10]]  # Limit for performance

                            # Extract first few rows
                            rows = table.find_all('tr')
                            for j, row in enumerate(rows[:5]):  # Limit for performance
                                cells = row.find_all(['td', 'th'])
                                row_data = [cell.get_text().strip() for cell in cells]
                                if row_data:
                                    table_data["rows"].append(row_data)

                            tables_found.append(table_data)

                    # Standard CSS selector extraction
                    for key, selector in css_selectors.items():
                        elements = soup.select(selector)
                        if elements:
                            if len(elements) == 1:
                                extracted_data[key] = elements[0].get_text().strip()
                            else:
                                extracted_data[key] = [elem.get_text().strip() for elem in elements]
                        else:
                            extracted_data[key] = None

                    result = {
                        "success": True,
                        "url": url,
                        "extracted_data": extracted_data,
                        "processing_method": "enhanced_css_selector_extraction",
                        "content": crawl_result.get("content", ""),
                        "markdown": crawl_result.get("markdown", "")
                    }

                    if tables_found:
                        result["tables_detected"] = len(tables_found)
                        result["table_data"] = tables_found
                        result["table_extraction_enhanced"] = True

                    if crawl_result.get("fallback_used"):
                        result["fallback_used"] = True

                    # Apply token limit fallback before returning
                    return apply_token_limit(result, max_tokens=25000)

                except ImportError:
                    # If BeautifulSoup not available, try fallback crawl
                    try:
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            generate_markdown=generate_markdown,
                            wait_for_js=wait_for_js,
                            timeout=timeout,
                            use_undetected_browser=True
                        )

                        if fallback_result.get("success", False):
                            fallback_response = {
                                "success": True,
                                "url": url,
                                "extracted_data": {"raw_content": fallback_result.get("content", "")[:500] + "..."},
                                "processing_method": "fallback_crawl_only",
                                "content": fallback_result.get("content", ""),
                                "markdown": fallback_result.get("markdown", ""),
                                "fallback_used": True,
                                "note": "BeautifulSoup not available - CSS extraction skipped"
                            }

                            # Apply token limit fallback before returning
                            return apply_token_limit(fallback_response, max_tokens=25000)

                    except Exception:
                        pass

                    return {
                        "success": False,
                        "error": "BeautifulSoup not available for CSS extraction"
                    }

                except Exception as e:
                    # Try fallback on CSS extraction error
                    try:
                        fallback_result = await web_crawling.crawl_url_with_fallback(
                            url=url,
                            generate_markdown=generate_markdown,
                            wait_for_js=wait_for_js,
                            timeout=timeout,
                            use_undetected_browser=True
                        )

                        if fallback_result.get("success", False):
                            fallback_response = {
                                "success": True,
                                "url": url,
                                "extracted_data": {"raw_content": fallback_result.get("content", "")[:500] + "..."},
                                "processing_method": "fallback_crawl_after_css_error",
                                "content": fallback_result.get("content", ""),
                                "markdown": fallback_result.get("markdown", ""),
                                "fallback_used": True,
                                "original_error": str(e)
                            }

                            # Apply token limit fallback before returning
                            return apply_token_limit(fallback_response, max_tokens=25000)

                    except Exception:
                        pass

                    return {
                        "success": False,
                        "error": f"CSS extraction error: {str(e)}"
                    }

            else:
                # Fallback to basic crawling or LLM extraction
                crawl_result = await web_crawling.crawl_url(
                    url=url,
                    generate_markdown=generate_markdown,
                    wait_for_js=wait_for_js,
                    timeout=timeout
                )

                # If basic crawl fails, try fallback
                if not crawl_result.get("success", False) or not crawl_result.get("content", "").strip():
                    fallback_result = await web_crawling.crawl_url_with_fallback(
                        url=url,
                        generate_markdown=generate_markdown,
                        wait_for_js=wait_for_js,
                        timeout=timeout,
                        use_undetected_browser=True
                    )

                    if fallback_result.get("success", False):
                        crawl_result = fallback_result
                        crawl_result["fallback_used"] = True

                if crawl_result.get("success", False):
                    crawl_result["processing_method"] = "basic_crawl_fallback"
                    crawl_result["note"] = "Used basic crawling - structured extraction not configured"
                    crawl_result["extracted_data"] = {"raw_content": crawl_result.get("content", "")[:500] + "..."}

                # Apply token limit fallback before returning
                return apply_token_limit(crawl_result, max_tokens=25000)

        except Exception as e:
            # Final fallback attempt
            try:
                fallback_result = await web_crawling.crawl_url_with_fallback(
                    url=url,
                    generate_markdown=generate_markdown,
                    wait_for_js=wait_for_js,
                    timeout=timeout,
                    use_undetected_browser=True
                )

                if fallback_result.get("success", False):
                    fallback_response = {
                        "success": True,
                        "url": url,
                        "extracted_data": {"raw_content": fallback_result.get("content", "")[:500] + "..."},
                        "processing_method": "emergency_fallback",
                        "content": fallback_result.get("content", ""),
                        "markdown": fallback_result.get("markdown", ""),
                        "fallback_used": True,
                        "original_error": str(e)
                    }

                    # Apply token limit fallback before returning
                    return apply_token_limit(fallback_response, max_tokens=25000)

            except Exception:
                pass

            return {
                "success": False,
                "error": f"Structured extraction error: {str(e)}"
            }
