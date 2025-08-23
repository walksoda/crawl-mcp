"""
Web crawling tools for Crawl4AI MCP Server.

Contains complete web crawling functionality and content extraction tools.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import (
    CrawlRequest,
    CrawlResponse,
    StructuredExtractionRequest
)

# Import required crawl4ai components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
    BM25ContentFilter,
    PruningContentFilter,
    LLMContentFilter,
    CacheMode,
)
from crawl4ai.chunking_strategy import (
    TopicSegmentationChunking,
    OverlappingWindowChunking,
    RegexChunking,
    SlidingWindowChunking
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

# Import processors and utilities
from ..file_processor import FileProcessor
from ..youtube_processor import YouTubeProcessor
from ..suppress_output import suppress_stdout_stderr

# Initialize processors
file_processor = FileProcessor()
youtube_processor = YouTubeProcessor()


# Placeholder for summarize_web_content function
async def summarize_web_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Placeholder for web content summarization.
    TODO: Implement LLM-based summarization.
    """
    return {
        "success": False,
        "error": "Summarization not yet implemented in modular architecture",
        "summary": content  # Return original content as fallback
    }


# Complete implementation of internal crawl URL function
async def _internal_crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    try:
        # Check if URL is a YouTube video
        if youtube_processor.is_youtube_url(request.url):
            # ⚠️ WARNING: YouTube transcript extraction is currently experiencing issues
            # due to API specification changes. Attempting extraction but may fail.
            try:
                youtube_result = await youtube_processor.process_youtube_url(
                    url=request.url,
                    languages=["ja", "en"],  # Default language preferences
                    include_timestamps=True,
                    preserve_formatting=True,
                    include_metadata=True
                )
                
                if youtube_result['success']:
                    transcript_data = youtube_result['transcript']
                    return CrawlResponse(
                        success=True,
                        url=request.url,
                        title=f"YouTube Video Transcript: {youtube_result['video_id']}",
                        content=transcript_data.get('full_text'),
                        markdown=transcript_data.get('clean_text'),
                        extracted_data={
                            "video_id": youtube_result['video_id'],
                            "processing_method": "youtube_transcript_api",
                            "language_info": youtube_result.get('language_info'),
                            "transcript_stats": {
                                "word_count": transcript_data.get('word_count'),
                                "segment_count": transcript_data.get('segment_count'),
                                "duration": transcript_data.get('duration_formatted')
                            },
                            "metadata": youtube_result.get('metadata')
                        }
                    )
                else:
                    # If YouTube transcript extraction fails, provide helpful error message
                    error_msg = youtube_result.get('error', 'Unknown error')
                    suggestion = youtube_result.get('suggestion', '')
                    
                    full_error = f"YouTube transcript extraction failed: {error_msg}"
                    if suggestion:
                        full_error += f"\n\nSuggestion: {suggestion}"
                    
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=full_error
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"YouTube processing error: {str(e)}"
                )
        
        # Check if URL points to a file that should be processed with MarkItDown
        if file_processor.is_supported_file(request.url):
            # Redirect to file processing for supported file formats
            try:
                file_result = await file_processor.process_file_from_url(
                    request.url,
                    max_size_mb=100  # Default size limit
                )
                
                if file_result['success']:
                    return CrawlResponse(
                        success=True,
                        url=request.url,
                        title=file_result.get('title'),
                        content=file_result.get('content'),
                        markdown=file_result.get('content'),  # MarkItDown already provides markdown
                        extracted_data={
                            "file_type": file_result.get('file_type'),
                            "size_bytes": file_result.get('size_bytes'),
                            "is_archive": file_result.get('is_archive', False),
                            "metadata": file_result.get('metadata'),
                            "archive_contents": file_result.get('archive_contents'),
                            "processing_method": "markitdown"
                        }
                    )
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=f"File processing failed: {file_result.get('error')}"
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"File processing error: {str(e)}"
                )
        
        # Setup deep crawling strategy if max_depth is specified
        deep_crawl_strategy = None
        if request.max_depth is not None and request.max_depth > 0:
            # Create filter chain
            filters = []
            if request.url_pattern:
                filters.append(URLPatternFilter(patterns=[request.url_pattern]))
            if not request.include_external:
                # Extract domain from URL for domain filtering
                from urllib.parse import urlparse
                domain = urlparse(request.url).netloc
                filters.append(DomainFilter(allowed_domains=[domain]))
            
            filter_chain = FilterChain(filters) if filters else None
            
            # Select crawling strategy
            if request.crawl_strategy == "dfs":
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            elif request.crawl_strategy == "best_first":
                deep_crawl_strategy = BestFirstCrawlingStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            else:  # Default to BFS
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )

        # Setup advanced content filtering
        content_filter_strategy = None
        if request.content_filter == "bm25" and request.filter_query:
            content_filter_strategy = BM25ContentFilter(user_query=request.filter_query)
        elif request.content_filter == "pruning":
            content_filter_strategy = PruningContentFilter(threshold=0.5)
        elif request.content_filter == "llm" and request.filter_query:
            content_filter_strategy = LLMContentFilter(
                instructions=f"Filter content related to: {request.filter_query}"
            )

        # Setup cache mode
        cache_mode = CacheMode.ENABLED
        if not request.enable_caching or request.cache_mode == "disabled":
            cache_mode = CacheMode.DISABLED
        elif request.cache_mode == "bypass":
            cache_mode = CacheMode.BYPASS

        # Configure chunking if requested
        chunking_strategy = None
        if request.chunk_content:
            step_size = int(request.chunk_size * (1 - request.overlap_rate))
            chunking_strategy = SlidingWindowChunking(
                window_size=request.chunk_size,
                step=step_size
            )

        # Create config parameters
        config_params = {
            "css_selector": request.css_selector,
            "screenshot": request.take_screenshot,
            "wait_for": request.wait_for_selector,
            "page_timeout": request.timeout * 1000,
            "exclude_all_images": not request.extract_media,
            "verbose": False,
            "log_console": False,
            "deep_crawl_strategy": deep_crawl_strategy,
            "cache_mode": cache_mode
        }
        
        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy
        
        # Add content filter if supported by current crawl4ai version
        try:
            config = CrawlerRunConfig(**config_params, content_filter=content_filter_strategy)
        except TypeError:
            # Fallback for older versions without content_filter support
            config = CrawlerRunConfig(**config_params)

        # Setup browser configuration with lightweight WebKit preference
        browser_config = {
            "headless": True,
            "verbose": False,
            "browser_type": "webkit"  # Use lightweight WebKit by default
        }
        
        # Enable undetected browser mode if requested
        if request.use_undetected_browser:
            browser_config["enable_stealth"] = True
            # Use Chromium when stealth mode is needed for better compatibility
            browser_config["browser_type"] = "chromium"
        
        if request.user_agent:
            browser_config["user_agent"] = request.user_agent
        
        if request.headers:
            browser_config["headers"] = request.headers

        # Suppress output to avoid JSON parsing errors
        with suppress_stdout_stderr():
            # Try WebKit first, fallback to Chromium if needed
            result = None
            browsers_to_try = ["webkit", "chromium"]
            
            for browser_type in browsers_to_try:
                try:
                    current_browser_config = browser_config.copy()
                    current_browser_config["browser_type"] = browser_type
                    
                    async with AsyncWebCrawler(**current_browser_config) as crawler:
                        # Handle authentication
                        if request.cookies:
                            # Set cookies if provided
                            await crawler.set_cookies(request.cookies)
                        
                        # Execute custom JavaScript if provided
                        if request.execute_js:
                            config.js_code = request.execute_js
                        
                        # Run crawler with config
                        arun_params = {"url": request.url, "config": config}
                        
                        result = await crawler.arun(**arun_params)
                        break  # Success, no need to try other browsers
                        
                except Exception as browser_error:
                    # If this is the last browser to try, raise the error
                    if browser_type == browsers_to_try[-1]:
                        raise browser_error
                    # Otherwise, try the next browser
                    continue
        
        # Handle different result types (single result vs list from deep crawling)
        if isinstance(result, list):
            # Deep crawling returns a list of results
            if not result:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="No results returned from deep crawling"
                )
            
            # Process multiple results from deep crawling
            all_content = []
            all_markdown = []
            all_media = []
            crawled_urls = []
            
            for page_result in result:
                if hasattr(page_result, 'success') and page_result.success:
                    crawled_urls.append(page_result.url if hasattr(page_result, 'url') else 'unknown')
                    if hasattr(page_result, 'cleaned_html') and page_result.cleaned_html:
                        all_content.append(f"=== {page_result.url} ===\n{page_result.cleaned_html}")
                    if hasattr(page_result, 'markdown') and page_result.markdown:
                        all_markdown.append(f"=== {page_result.url} ===\n{page_result.markdown}")
                    if hasattr(page_result, 'media') and page_result.media and request.extract_media:
                        all_media.extend(page_result.media)
            
            return CrawlResponse(
                success=True,
                url=request.url,
                title=f"Deep crawl of {len(crawled_urls)} pages",
                content="\n\n".join(all_content) if all_content else "No content extracted",
                markdown="\n\n".join(all_markdown) if all_markdown else "No markdown content",
                media=all_media if request.extract_media else None,
                extracted_data={
                    "crawled_pages": len(crawled_urls),
                    "crawled_urls": crawled_urls,
                    "processing_method": "deep_crawling"
                }
            )
        
        elif hasattr(result, 'success') and result.success:
            # For deep crawling, result might contain multiple pages
            if deep_crawl_strategy and hasattr(result, 'crawled_pages'):
                # Combine content from all crawled pages
                all_content = []
                all_markdown = []
                all_media = []
                
                for page in result.crawled_pages:
                    if page.cleaned_html:
                        all_content.append(f"=== {page.url} ===\n{page.cleaned_html}")
                    if page.markdown:
                        all_markdown.append(f"=== {page.url} ===\n{page.markdown}")
                    if page.media and request.extract_media:
                        all_media.extend(page.media)
                
                # Prepare content for potential summarization
                combined_content = "\n\n".join(all_content) if all_content else result.cleaned_html
                combined_markdown = "\n\n".join(all_markdown) if all_markdown else result.markdown
                title_to_use = result.metadata.get("title", "")
                extracted_data = {"crawled_pages": len(result.crawled_pages)} if hasattr(result, 'crawled_pages') else {}
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and combined_content:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(combined_content) // 4
                    
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = combined_markdown or combined_content
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                combined_content = summary_result["summary"]
                                combined_markdown = summary_result["summary"]
                                
                                extracted_data.update({
                                    "summarization_applied": True,
                                    "original_content_length": len("\n\n".join(all_content) if all_content else result.cleaned_html),
                                    "original_tokens_estimate": estimated_tokens,
                                    "summary_length": request.summary_length,
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    "auto_summarization_trigger": f"Deep crawl content exceeded {request.max_content_tokens} tokens"
                                })
                            else:
                                # Summarization failed, add error info but keep original content
                                extracted_data.update({
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                })
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data.update({
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            })
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=combined_content,
                    markdown=combined_markdown,
                    media=all_media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            else:
                # Check if auto-summarization should be applied
                content_to_use = result.cleaned_html
                markdown_to_use = result.markdown
                extracted_data = None
                title_to_use = result.metadata.get("title", "")
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and content_to_use:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(content_to_use) // 4
                    
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = markdown_to_use or content_to_use
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                content_to_use = summary_result["summary"]
                                markdown_to_use = summary_result["summary"]  # Use same summary for both
                                
                                extracted_data = {
                                    "summarization_applied": True,
                                    "original_content_length": len(result.cleaned_html),
                                    "original_tokens_estimate": estimated_tokens,
                                    "summary_length": request.summary_length,
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    "auto_summarization_trigger": f"Content exceeded {request.max_content_tokens} tokens"
                                }
                            else:
                                # Summarization failed, add error info but keep original content
                                extracted_data = {
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                }
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data = {
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            }
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=content_to_use,
                    markdown=markdown_to_use,
                    media=result.media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            return response
        else:
            # Handle case where result doesn't have success attribute or failed
            error_msg = "Failed to crawl URL"
            if hasattr(result, 'error_message'):
                error_msg = f"Failed to crawl URL: {result.error_message}"
            elif hasattr(result, 'error'):
                error_msg = f"Failed to crawl URL: {result.error}"
            else:
                error_msg = f"Failed to crawl URL: Unknown error (result type: {type(result)})"
            
            return CrawlResponse(
                success=False,
                url=request.url,
                error=error_msg
            )
                
    except Exception as e:
        error_message = f"Crawling error: {str(e)}"
        
        # Enhanced error handling for browser and UVX issues
        if "playwright" in str(e).lower() or "browser" in str(e).lower() or "executable doesn't exist" in str(e).lower():
            import os
            import sys
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            
            if is_uvx_env:
                error_message += "\n\n🔧 UVX Environment Browser Setup Required:\n" \
                    f"1. Run system diagnostics: get_system_diagnostics()\n" \
                    f"2. Manual browser installation:\n" \
                    f"   - uvx --with playwright playwright install webkit\n" \
                    f"   - Or system-wide: playwright install webkit\n" \
                    f"3. Restart Claude Desktop after installation\n" \
                    f"4. If issues persist, consider STDIO local setup\n\n" \
                    f"💡 WebKit is lightweight (~180MB) vs Chromium (~281MB)"
            else:
                error_message += "\n\n🔧 Browser Setup Required:\n" \
                    f"1. Install Playwright browsers:\n" \
                    f"   playwright install webkit  # Lightweight option\n" \
                    f"   playwright install chromium  # Full compatibility\n" \
                    f"2. For system dependencies: sudo apt-get install libnss3 libnspr4 libasound2\n" \
                    f"3. Run diagnostics: get_system_diagnostics()"
            
        return CrawlResponse(
            success=False,
            url=request.url,
            error=error_message,
            extracted_data={
                'error_type': 'browser_setup_required',
                'uvx_environment': 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable),
                'diagnostic_tool': 'get_system_diagnostics',
                'installation_commands': [
                    'playwright install webkit',
                    'playwright install chromium'
                ]
            }
        )


# Other internal functions for specialized extraction
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
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(
            url=url,
            content_filter=content_filter,
            filter_query=filter_query,
            chunk_content=chunk_content,
            generate_markdown=True
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
            # Import config
            try:
                from ..config import get_llm_config
            except ImportError:
                from config import get_llm_config
            
            # Get LLM configuration
            llm_config = get_llm_config(llm_provider, llm_model)
            
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
            
            # Make LLM API call
            provider_info = llm_config.provider.split('/')
            provider = provider_info[0] if provider_info else 'openai'
            model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
            
            extracted_content = None
            
            if provider == 'openai':
                import openai
                
                api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found")
                
                client = openai.AsyncOpenAI(api_key=api_key, base_url=llm_config.base_url)
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert content analyst specializing in precise information extraction."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=4000
                )
                
                extracted_content = response.choices[0].message.content
                
            elif provider == 'anthropic':
                import anthropic
                
                api_key = llm_config.api_token or os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("Anthropic API key not found")
                
                client = anthropic.AsyncAnthropic(api_key=api_key)
                
                response = await client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": extraction_prompt}
                    ]
                )
                
                extracted_content = response.content[0].text
                
            else:
                # Fallback for unsupported providers
                return {
                    "url": url,
                    "success": False,
                    "error": f"LLM provider '{provider}' not supported for intelligent extraction"
                }
            
            # Parse JSON response
            if extracted_content:
                try:
                    import json
                    # Clean up the extracted content if it's wrapped in markdown
                    content_to_parse = extracted_content
                    if content_to_parse.startswith('```json'):
                        content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                    
                    extraction_data = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                    
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
                        "llm_provider": provider,
                        "llm_model": model,
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
                        "llm_provider": provider,
                        "llm_model": model,
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


async def _internal_extract_entities(
    url: str,
    entity_types: List[str],
    custom_patterns: Optional[Dict[str, str]] = None,
    include_context: bool = True,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Internal extract entities implementation using regex patterns.
    """
    try:
        # First crawl the URL to get the content
        request = CrawlRequest(url=url, generate_markdown=True)
        crawl_result = await _internal_crawl_url(request)
        
        if not crawl_result.success:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to crawl URL: {crawl_result.error}"
            }
        
        content = crawl_result.content or ""
        entities = {}
        
        # Define regex patterns for common entity types
        patterns = {
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phones": r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            "urls": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            "dates": r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b',
            "ips": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "prices": r'[$£€¥]?\s?\d+(?:[.,]\d{2,3})*(?:[.,]\d{2})?',
            "credit_cards": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "coordinates": r'[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?)'
        }
        
        # Add custom patterns if provided
        if custom_patterns:
            patterns.update(custom_patterns)
        
        # Extract entities for each requested type
        import re
        for entity_type in entity_types:
            if entity_type in patterns:
                matches = re.findall(patterns[entity_type], content, re.IGNORECASE)
                if matches:
                    # Deduplicate if requested
                    if deduplicate:
                        matches = list(set(matches))
                    entities[entity_type] = matches
        
        return {
            "url": url,
            "success": True,
            "entities": entities,
            "entity_types_requested": entity_types,
            "processing_method": "regex_extraction",
            "content_length": len(content),
            "total_entities_found": sum(len(v) for v in entities.values())
        }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": f"Entity extraction error: {str(e)}"
        }


async def _internal_llm_extract_entities(
    url: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    instruction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal LLM extract entities implementation.
    TODO: Implement LLM-based named entity recognition.
    """
    return {
        "url": url,
        "success": False,
        "error": "LLM entity extraction not yet implemented in modular architecture"
    }


async def _internal_extract_structured_data(request: StructuredExtractionRequest) -> CrawlResponse:
    """
    Internal extract structured data implementation.
    Supports both CSS selector and LLM-based extraction methods.
    """
    try:
        # First crawl the URL to get the content
        crawl_request = CrawlRequest(
            url=request.url,
            generate_markdown=True
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
            # CSS selector-based extraction
            try:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(crawl_result.content, 'html.parser')
                
                for field_name, css_selector in request.css_selectors.items():
                    elements = soup.select(css_selector)
                    if elements:
                        if len(elements) == 1:
                            # Single element
                            extracted_data[field_name] = elements[0].get_text(strip=True)
                        else:
                            # Multiple elements
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
            # LLM-based extraction
            try:
                # Import config
                try:
                    from ..config import get_llm_config
                except ImportError:
                    from config import get_llm_config
                
                # Get LLM configuration
                llm_config = get_llm_config(request.llm_provider, request.llm_model)
                
                # Prepare schema description
                schema_description = ""
                if request.extraction_schema:
                    schema_items = []
                    for field, description in request.extraction_schema.items():
                        schema_items.append(f"- {field}: {description}")
                    schema_description = "\n".join(schema_items)
                
                # Prepare extraction prompt
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
                
                # Make LLM API call
                provider_info = llm_config.provider.split('/')
                provider = provider_info[0] if provider_info else 'openai'
                model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
                
                extracted_content = None
                
                if provider == 'openai':
                    import openai
                    
                    api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                    if not api_key:
                        raise ValueError("OpenAI API key not found")
                    
                    client = openai.AsyncOpenAI(api_key=api_key, base_url=llm_config.base_url)
                    
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert data extraction specialist focused on accuracy and structured output."},
                            {"role": "user", "content": structured_prompt}
                        ],
                        temperature=0.1,  # Low temperature for consistent extraction
                        max_tokens=4000
                    )
                    
                    extracted_content = response.choices[0].message.content
                    
                elif provider == 'anthropic':
                    import anthropic
                    
                    api_key = llm_config.api_token or os.environ.get('ANTHROPIC_API_KEY')
                    if not api_key:
                        raise ValueError("Anthropic API key not found")
                    
                    client = anthropic.AsyncAnthropic(api_key=api_key)
                    
                    response = await client.messages.create(
                        model=model,
                        max_tokens=4000,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": structured_prompt}
                        ]
                    )
                    
                    extracted_content = response.content[0].text
                    
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=f"LLM provider '{provider}' not supported for structured extraction"
                    )
                
                # Parse JSON response
                if extracted_content:
                    try:
                        import json
                        # Clean up the extracted content if it's wrapped in markdown
                        content_to_parse = extracted_content
                        if content_to_parse.startswith('```json'):
                            content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                        
                        extraction_result = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                        
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
                                "llm_provider": provider,
                                "llm_model": model,
                                "custom_instruction_used": bool(request.instruction)
                            }
                        )
                        
                    except (json.JSONDecodeError, AttributeError) as e:
                        # Fallback: treat as plain text
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
                                "llm_provider": provider,
                                "llm_model": model,
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


# MCP Tool implementations
async def crawl_url(
    url: Annotated[str, Field(description="Target URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Maximum crawling depth, None for single page (default: None)")] = None,
    max_pages: Annotated[Optional[int], Field(description="Maximum number of pages to crawl (default: 10)")] = 10,
    include_external: Annotated[bool, Field(description="Whether to follow external domain links (default: False)")] = False,
    crawl_strategy: Annotated[str, Field(description="Crawling strategy: 'bfs', 'dfs', or 'best_first' (default: 'bfs')")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL pattern filter e.g. '*docs*' (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum score for URLs to be crawled (default: 0.3)")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="Content filter: 'bm25', 'pruning', 'llm' (default: None)")] = None,
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 content filtering (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Whether to chunk large content (default: False)")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunking strategy: 'topic', 'regex', 'sentence' (default: 'topic')")] = "topic",
    chunk_size: Annotated[int, Field(description="Maximum chunk size in tokens (default: 1000)")] = 1000,
    overlap_rate: Annotated[float, Field(description="Overlap rate between chunks 0.0-1.0 (default: 0.1)")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="Custom user agent string (default: None)")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="Custom HTTP headers (default: None)")] = None,
    enable_caching: Annotated[bool, Field(description="Whether to enable caching (default: True)")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode: 'enabled', 'disabled', 'bypass' (default: 'enabled')")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JavaScript code to execute (default: None)")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate human-like browsing behavior (default: False)")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Use undetected browser mode to bypass bot detection (default: False)")] = False,
    auth_token: Annotated[Optional[str], Field(description="Authentication token (default: None)")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Custom cookies (default: None)")] = None,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization, auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization, auto-detected if not specified (default: None)")] = None
) -> CrawlResponse:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.
    
    Core web crawling tool with comprehensive configuration options.
    Essential for SPAs: set wait_for_js=true for JavaScript-heavy sites.
    """
    # Create CrawlRequest object from individual parameters
    request = CrawlRequest(
        url=url,
        css_selector=css_selector,
        xpath=xpath,
        extract_media=extract_media,
        take_screenshot=take_screenshot,
        generate_markdown=generate_markdown,
        wait_for_selector=wait_for_selector,
        timeout=timeout,
        max_depth=max_depth,
        max_pages=max_pages,
        include_external=include_external,
        crawl_strategy=crawl_strategy,
        url_pattern=url_pattern,
        score_threshold=score_threshold,
        content_filter=content_filter,
        filter_query=filter_query,
        chunk_content=chunk_content,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap_rate=overlap_rate,
        user_agent=user_agent,
        headers=headers,
        enable_caching=enable_caching,
        cache_mode=cache_mode,
        execute_js=execute_js,
        wait_for_js=wait_for_js,
        simulate_user=simulate_user,
        use_undetected_browser=use_undetected_browser,
        auth_token=auth_token,
        cookies=cookies,
        auto_summarize=auto_summarize,
        max_content_tokens=max_content_tokens,
        summary_length=summary_length,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    return await _internal_crawl_url(request)


async def deep_crawl_site(
    url: Annotated[str, Field(description="Starting URL for multi-page crawling")],
    max_depth: Annotated[int, Field(description="Link levels to follow from start URL (default: 2)")] = 2,
    max_pages: Annotated[int, Field(description="Maximum pages to crawl (default: 5)")] = 5,
    crawl_strategy: Annotated[str, Field(description="Crawling approach: 'bfs', 'dfs', 'best_first' (default: 'bfs')")] = "bfs",
    include_external: Annotated[bool, Field(description="Follow external domain links (default: False)")] = False,
    url_pattern: Annotated[Optional[str], Field(description="Wildcard filter like '*docs*' or '*api*' (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum relevance score 0.0-1.0 (default: 0.0)")] = 0.0,
    extract_media: Annotated[bool, Field(description="Include images/videos (default: False)")] = False,
    base_timeout: Annotated[int, Field(description="Timeout per page in seconds (default: 60)")] = 60
) -> Dict[str, Any]:
    """
    Crawl multiple related pages from a website (maximum 5 pages for stability).
    
    Multi-page crawling with configurable depth and filtering options.
    Perfect for documentation sites and content discovery.
    """
    # Use crawl_url with deep crawling enabled
    request = CrawlRequest(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        crawl_strategy=crawl_strategy,
        include_external=include_external,
        url_pattern=url_pattern,
        score_threshold=score_threshold,
        extract_media=extract_media,
        timeout=base_timeout,
        generate_markdown=True
    )
    
    result = await _internal_crawl_url(request)
    
    # Convert CrawlResponse to dict for consistency with legacy API
    return {
        "success": result.success,
        "url": result.url,
        "title": result.title,
        "content": result.content,
        "markdown": result.markdown,
        "media": result.media,
        "extracted_data": result.extracted_data,
        "error": result.error,
        "processing_method": "deep_crawling"
    }


async def intelligent_extract(
    url: Annotated[str, Field(description="Target webpage URL")],
    extraction_goal: Annotated[str, Field(description="Specific data to extract, be precise")],
    content_filter: Annotated[str, Field(description="Pre-filter content - 'bm25', 'pruning', 'llm' (default: 'bm25')")] = "bm25",
    filter_query: Annotated[Optional[str], Field(description="Keywords for BM25 filtering to improve accuracy (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Split large content for better processing (default: False)")] = False,
    use_llm: Annotated[bool, Field(description="Enable LLM processing (default: True)")] = True,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name (default: auto-detected)")] = None,
    custom_instructions: Annotated[Optional[str], Field(description="Additional guidance for LLM (default: None)")] = None
) -> Dict[str, Any]:
    """
    AI-powered extraction of specific data from web pages using LLM semantic understanding.
    
    Uses LLM to extract specific information based on your extraction goal.
    Pre-filtering improves accuracy and reduces processing time.
    """
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


async def extract_entities(
    url: Annotated[str, Field(description="Target webpage URL")],
    entity_types: Annotated[List[str], Field(description="List of entity types to extract")],
    custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns for specialized extraction (default: None)")] = None,
    include_context: Annotated[bool, Field(description="Include surrounding text context (default: True)")] = True,
    deduplicate: Annotated[bool, Field(description="Remove duplicate entities (default: True)")] = True,
    use_llm: Annotated[bool, Field(description="Use AI for named entity recognition (default: False)")] = False,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for NER (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name (default: auto-detected)")] = None
) -> Dict[str, Any]:
    """
    Extract specific entity types from web pages using regex patterns or LLM.
    
    Supports regex: emails, phones, urls, dates, ips, social_media, prices, credit_cards, coordinates
    Supports LLM: names (people/organizations/locations) when use_llm=True
    """
    if use_llm and "names" in entity_types:
        # Use LLM-based extraction for named entities
        return await _internal_llm_extract_entities(url, llm_provider, llm_model)
    else:
        # Use regex-based extraction
        return await _internal_extract_entities(
            url=url,
            entity_types=entity_types,
            custom_patterns=custom_patterns,
            include_context=include_context,
            deduplicate=deduplicate
        )


async def extract_structured_data(
    request: Annotated[StructuredExtractionRequest, Field(description="StructuredExtractionRequest with URL, schema, and extraction parameters")]
) -> CrawlResponse:
    """
    Extract structured data from a URL using CSS selectors or LLM-based extraction.
    
    Extract data matching a predefined schema using CSS selectors or LLM processing.
    Useful for consistent data extraction from similar page structures.
    """
    return await _internal_extract_structured_data(request)


async def crawl_url_with_fallback(
    url: Annotated[str, Field(description="Target URL to crawl")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds")] = 60,
    max_depth: Annotated[Optional[int], Field(description="Maximum crawling depth (None for single page)")] = None,
    max_pages: Annotated[Optional[int], Field(description="Maximum number of pages to crawl")] = 10,
    include_external: Annotated[bool, Field(description="Whether to follow external domain links")] = False,
    crawl_strategy: Annotated[str, Field(description="Crawling strategy: 'bfs', 'dfs', or 'best_first'")] = "bfs",
    url_pattern: Annotated[Optional[str], Field(description="URL pattern filter (e.g., '*docs*')")] = None,
    score_threshold: Annotated[float, Field(description="Minimum score for URLs to be crawled")] = 0.3,
    content_filter: Annotated[Optional[str], Field(description="Content filter type: 'bm25', 'pruning', 'llm'")] = None,
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 content filtering")] = None,
    chunk_content: Annotated[bool, Field(description="Whether to chunk large content")] = False,
    chunk_strategy: Annotated[str, Field(description="Chunking strategy: 'topic', 'regex', 'sentence'")] = "topic",
    chunk_size: Annotated[int, Field(description="Maximum chunk size in tokens")] = 1000,
    overlap_rate: Annotated[float, Field(description="Overlap rate between chunks (0.0-1.0)")] = 0.1,
    user_agent: Annotated[Optional[str], Field(description="Custom user agent string")] = None,
    headers: Annotated[Optional[Dict[str, str]], Field(description="Custom HTTP headers")] = None,
    enable_caching: Annotated[bool, Field(description="Whether to enable caching")] = True,
    cache_mode: Annotated[str, Field(description="Cache mode: 'enabled', 'disabled', 'bypass'")] = "enabled",
    execute_js: Annotated[Optional[str], Field(description="JavaScript code to execute")] = None,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete")] = False,
    simulate_user: Annotated[bool, Field(description="Simulate human-like browsing behavior")] = False,
    use_undetected_browser: Annotated[bool, Field(description="Use undetected browser mode to bypass bot detection")] = False,
    auth_token: Annotated[Optional[str], Field(description="Authentication token")] = None,
    cookies: Annotated[Optional[Dict[str, str]], Field(description="Custom cookies")] = None,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long'")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider for summarization (auto-detected if not specified)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model for summarization (auto-detected if not specified)")] = None
) -> CrawlResponse:
    """
    Enhanced crawling with multiple fallback strategies for difficult sites.
    
    Uses multiple fallback strategies when normal crawling fails. Same parameters as crawl_url 
    but with enhanced reliability for sites with aggressive anti-bot protection.
    """
    import asyncio
    import random
    from urllib.parse import urlparse
    
    # Detect HackerNews or similar problematic sites
    domain = urlparse(url).netloc.lower()
    is_hn = 'ycombinator.com' in domain
    is_reddit = 'reddit.com' in domain
    is_social_media = any(site in domain for site in ['twitter.com', 'facebook.com', 'linkedin.com'])
    
    # Enhanced strategies for difficult sites
    strategies = []
    
    # Strategy 1: Human-like browsing with realistic headers
    realistic_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]
    
    realistic_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }
    
    if is_hn:
        # HackerNews specific selectors
        hn_css_selectors = [
            ".fatitem, .athing, .comment",  # Main content and comments
            ".storylink, .titleline a",     # Story titles
            ".comtr, .comment-tree",        # Comment threads
            "table.comment-tree",           # Comment structure
        ]
        
        # Try with HN-specific CSS selector if not provided
        if not css_selector:
            css_selector = ".fatitem, .athing, .comtr"
    
    # Strategy 1: Realistic browser simulation
    strategies.append({
        "name": "realistic_browser",
        "params": {
            "user_agent": user_agent or random.choice(realistic_user_agents),
            "headers": {**realistic_headers, **(headers or {})},
            "wait_for_js": True,
            "simulate_user": True,
            "timeout": max(timeout, 30),
            "css_selector": css_selector,
            "wait_for_selector": wait_for_selector or (".fatitem" if is_hn else None),
            "cache_mode": "bypass"  # Fresh request
        }
    })
    
    # Strategy 2: Delayed request with different user agent
    strategies.append({
        "name": "delayed_request", 
        "params": {
            "user_agent": random.choice([ua for ua in realistic_user_agents if ua != strategies[0]["params"]["user_agent"]]),
            "headers": realistic_headers,
            "wait_for_js": wait_for_js,
            "timeout": timeout + 30,
            "css_selector": css_selector,
            "execute_js": "setTimeout(() => { window.scrollTo(0, document.body.scrollHeight/2); }, 2000);" if is_hn else execute_js,
            "cache_mode": "disabled"
        }
    })
    
    # Strategy 3: Mobile user agent
    strategies.append({
        "name": "mobile_agent",
        "params": {
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "headers": {
                **realistic_headers,
                "Sec-CH-UA-Mobile": "?1",
                "Sec-CH-UA-Platform": '"iOS"'
            },
            "wait_for_js": True,
            "timeout": timeout,
            "css_selector": css_selector
        }
    })
    
    last_error = None
    
    for i, strategy in enumerate(strategies):
        try:
            # Add random delay between attempts (1-5 seconds)
            if i > 0:
                delay = random.uniform(1, 5)
                await asyncio.sleep(delay)
            
            print(f"Attempting strategy {i+1}/{len(strategies)}: {strategy['name']}")
            
            # Prepare strategy-specific parameters
            strategy_params = {
                "url": url,
                "css_selector": strategy["params"].get("css_selector", css_selector),
                "xpath": xpath,
                "extract_media": extract_media,
                "take_screenshot": take_screenshot,
                "generate_markdown": generate_markdown,
                "wait_for_selector": strategy["params"].get("wait_for_selector", wait_for_selector),
                "timeout": strategy["params"].get("timeout", timeout),
                "max_depth": None,  # Force single page to avoid deep crawling issues
                "max_pages": max_pages,
                "include_external": include_external,
                "crawl_strategy": crawl_strategy,
                "url_pattern": url_pattern,
                "score_threshold": score_threshold,
                "content_filter": content_filter,
                "filter_query": filter_query,
                "chunk_content": chunk_content,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "overlap_rate": overlap_rate,
                "user_agent": strategy["params"].get("user_agent", user_agent),
                "headers": strategy["params"].get("headers", headers),
                "enable_caching": enable_caching,
                "cache_mode": strategy["params"].get("cache_mode", cache_mode),
                "execute_js": strategy["params"].get("execute_js", execute_js),
                "wait_for_js": strategy["params"].get("wait_for_js", wait_for_js),
                "simulate_user": strategy["params"].get("simulate_user", simulate_user),
                "use_undetected_browser": use_undetected_browser,
                "auth_token": auth_token,
                "cookies": cookies,
                "auto_summarize": auto_summarize,
                "max_content_tokens": max_content_tokens,
                "summary_length": summary_length,
                "llm_provider": llm_provider,
                "llm_model": llm_model
            }
            
            # Attempt crawl with current strategy
            result = await crawl_url(**strategy_params)
            
            # Check if we got meaningful content
            if result.success and result.content and len(result.content.strip()) > 100:
                # Add strategy info to extracted_data
                if result.extracted_data is None:
                    result.extracted_data = {}
                result.extracted_data.update({
                    "fallback_strategy_used": strategy["name"],
                    "fallback_attempt": i + 1,
                    "total_strategies_tried": len(strategies),
                    "site_type_detected": "hackernews" if is_hn else "reddit" if is_reddit else "social_media" if is_social_media else "general"
                })
                return result
            
            last_error = f"Strategy {strategy['name']}: Content too short or empty"
            
        except Exception as e:
            last_error = f"Strategy {strategy['name']}: {str(e)}"
            print(f"Strategy {strategy['name']} failed: {e}")
            continue
    
    # All strategies failed, return error with details
    return CrawlResponse(
        success=False,
        url=url,
        error=f"All fallback strategies failed. Last error: {last_error}. "
               f"This site may have strong anti-bot protection. "
               f"Strategies attempted: {', '.join([s['name'] for s in strategies])}",
        extracted_data={
            "fallback_strategies_attempted": [s["name"] for s in strategies],
            "total_attempts": len(strategies),
            "site_type_detected": "hackernews" if is_hn else "reddit" if is_reddit else "social_media" if is_social_media else "general",
            "recommendations": [
                "Try accessing the site manually to check if it's available",
                "Consider using the site's API if available",
                "Try accessing during off-peak hours",
                "Use a VPN if the site is geo-blocked"
            ]
        }
    )


