"""
Utility tools for Crawl4AI MCP Server.

Contains complete configuration info, batch operations, and tool selection guidance.
"""

import asyncio
from typing import Any, Dict, List, Optional, Annotated
from pydantic import Field

# Import models from the models module
from ..models import CrawlResponse, CrawlRequest

# Import the internal crawl function and web batch crawl
from .web_crawling import _internal_crawl_url


# Complete tool selection guide with all implemented tools
TOOL_SELECTION_GUIDE = {
    "single_page_content": {
        "tools": ["crawl_url", "crawl_url_with_fallback"],
        "description": "Extract content from a single webpage",
        "use_cases": [
            "Basic web scraping and content extraction",
            "SPA (Single Page Application) crawling with JavaScript",
            "Difficult sites requiring fallback strategies",
            "Converting web content to markdown format"
        ],
        "complexity": "simple to moderate",
        "parameters": ["url", "wait_for_js", "extract_media", "generate_markdown", "timeout"]
    },
    "multi_page_crawling": {
        "tools": ["deep_crawl_site"],
        "description": "Crawl multiple related pages from a website",
        "use_cases": [
            "Documentation site scraping",
            "Website mapping and discovery",
            "Content inventory and analysis",
            "Link following with depth control"
        ],
        "complexity": "advanced",
        "parameters": ["url", "max_depth", "max_pages", "url_pattern", "crawl_strategy"]
    },
    "targeted_extraction": {
        "tools": ["intelligent_extract", "extract_entities", "extract_structured_data"],
        "description": "Extract specific data or entities from web pages",
        "use_cases": [
            "AI-powered content extraction with specific goals",
            "Contact information and entity extraction",
            "Product details and pricing information",
            "Named entity recognition (emails, phones, addresses)",
            "Schema-based structured data extraction"
        ],
        "complexity": "moderate to advanced",
        "parameters": ["url", "extraction_goal", "entity_types", "schema", "custom_patterns"]
    },
    "file_processing": {
        "tools": ["process_file", "enhanced_process_large_content", "get_supported_file_formats"],
        "description": "Process documents and large content with AI summarization",
        "use_cases": [
            "PDF document analysis and conversion",
            "Microsoft Office file processing",
            "ZIP archive extraction and processing",
            "Large document summarization and chunking",
            "ePub and eBook content extraction"
        ],
        "complexity": "moderate to advanced",
        "parameters": ["url", "max_size_mb", "auto_summarize", "chunking_strategy", "filtering_strategy"]
    },
    "video_content": {
        "tools": ["extract_youtube_transcript", "get_youtube_video_info", "batch_extract_youtube_transcripts", "get_youtube_api_setup_guide"],
        "description": "Extract and analyze YouTube video content",
        "use_cases": [
            "Video transcript extraction with timestamps",
            "Multi-language transcript support and translation",
            "Video metadata and information retrieval",
            "Batch processing of multiple YouTube videos",
            "AI-powered transcript summarization"
        ],
        "complexity": "moderate",
        "parameters": ["url", "languages", "translate_to", "auto_summarize", "include_timestamps"]
    },
    "search_operations": {
        "tools": ["search_google", "search_and_crawl", "batch_search_google", "get_search_genres"],
        "description": "Search and discover content on the web with advanced filtering",
        "use_cases": [
            "Google search with genre-specific filtering",
            "Search result crawling and content extraction",
            "Batch search operations with analysis",
            "Academic, news, technical content discovery",
            "Competitive analysis and research",
            "Recent content filtering with date ranges",
            "Time-sensitive searches (last week, month, etc.)"
        ],
        "complexity": "moderate to advanced",
        "parameters": ["query", "search_genre", "num_results", "crawl_top_results", "recent_days", "auto_summarize", "summary_length"]
    },
    "batch_operations": {
        "tools": ["batch_crawl", "batch_extract_youtube_transcripts", "batch_search_google"],
        "description": "Process multiple URLs, videos, or queries simultaneously",
        "use_cases": [
            "Bulk website crawling and analysis",
            "Mass YouTube video processing",
            "Multiple search query execution with date filtering",
            "Efficient large-scale data collection",
            "Performance optimization for multiple operations",
            "Time-based batch analysis (recent trends, etc.)"
        ],
        "complexity": "advanced",
        "parameters": ["urls/queries", "max_concurrent", "base_timeout", "recent_days", "auto_summarize", "summary_length", "llm_provider"]
    },
    "configuration_and_info": {
        "tools": ["get_llm_config_info", "get_supported_file_formats", "get_search_genres", "get_youtube_api_setup_guide", "get_tool_selection_guide"],
        "description": "Get system configuration, capabilities, and usage guidance",
        "use_cases": [
            "System setup verification and troubleshooting",
            "Capability discovery and feature exploration",
            "Tool selection and workflow planning",
            "API configuration and status checking",
            "Usage guidance and best practices"
        ],
        "complexity": "simple",
        "parameters": ["None (information retrieval only)"]
    }
}

# Comprehensive workflow patterns
WORKFLOW_GUIDE = {
    "research_workflow": {
        "steps": ["search_google", "search_and_crawl", "intelligent_extract"],
        "description": "Comprehensive research workflow with targeted extraction",
        "use_case": "Academic research, competitive analysis, market research",
        "estimated_time": "Medium (5-15 minutes)",
        "output": "Search results + full content + extracted insights"
    },
    "documentation_analysis": {
        "steps": ["deep_crawl_site", "enhanced_process_large_content"],
        "description": "Analyze large documentation sites with content processing",
        "use_case": "Technical documentation analysis, API documentation review",
        "estimated_time": "Long (10-30 minutes)",
        "output": "Site structure + processed content + summaries"
    },
    "competitive_analysis": {
        "steps": ["search_google", "batch_crawl", "extract_structured_data"],
        "description": "Analyze competitor websites and extract structured information",
        "use_case": "Business intelligence, market analysis, product comparison",
        "estimated_time": "Long (15-45 minutes)",
        "output": "Competitor data + structured insights + analysis"
    },
    "content_research": {
        "steps": ["search_and_crawl", "process_file", "extract_youtube_transcript"],
        "description": "Multi-format content research and analysis",
        "use_case": "Content creation, educational research, media analysis",
        "estimated_time": "Medium (10-20 minutes)",
        "output": "Web content + documents + video transcripts"
    },
    "bulk_data_collection": {
        "steps": ["batch_search_google", "batch_crawl", "batch_extract_youtube_transcripts"],
        "description": "Large-scale data collection across multiple sources",
        "use_case": "Dataset creation, mass content analysis, research corpus building",
        "estimated_time": "Very Long (30+ minutes)",
        "output": "Large datasets + batch processing results + aggregated analysis"
    }
}

# Updated complexity classification
COMPLEXITY_GUIDE = {
    "simple": {
        "tools": ["crawl_url", "get_llm_config_info", "get_supported_file_formats", "get_search_genres", "get_youtube_api_setup_guide", "get_tool_selection_guide"],
        "description": "Single-step operations with minimal configuration",
        "typical_time": "1-3 minutes",
        "skill_level": "Beginner"
    },
    "moderate": {
        "tools": ["intelligent_extract", "extract_entities", "extract_structured_data", "process_file", "search_google", "extract_youtube_transcript", "get_youtube_video_info"],
        "description": "Operations requiring specific parameters or domain knowledge",
        "typical_time": "3-10 minutes",
        "skill_level": "Intermediate"
    },
    "advanced": {
        "tools": ["deep_crawl_site", "enhanced_process_large_content", "search_and_crawl", "crawl_url_with_fallback"],
        "description": "Complex operations with multiple steps and advanced configuration",
        "typical_time": "10-30 minutes",
        "skill_level": "Advanced"
    },
    "batch": {
        "tools": ["batch_crawl", "batch_search_google", "batch_extract_youtube_transcripts"],
        "description": "High-volume operations requiring careful resource management",
        "typical_time": "15-60+ minutes",
        "skill_level": "Expert"
    }
}

# Performance characteristics
PERFORMANCE_GUIDE = {
    "fast": {
        "tools": ["crawl_url", "get_llm_config_info", "get_supported_file_formats"],
        "typical_response_time": "< 30 seconds",
        "resource_usage": "Low"
    },
    "medium": {
        "tools": ["intelligent_extract", "extract_entities", "process_file", "search_google"],
        "typical_response_time": "30 seconds - 3 minutes",
        "resource_usage": "Medium"
    },
    "slow": {
        "tools": ["deep_crawl_site", "enhanced_process_large_content", "search_and_crawl"],
        "typical_response_time": "3-15 minutes",
        "resource_usage": "High"
    },
    "very_slow": {
        "tools": ["batch_crawl", "batch_search_google", "batch_extract_youtube_transcripts"],
        "typical_response_time": "15+ minutes",
        "resource_usage": "Very High"
    }
}


# MCP Tool implementations
async def batch_crawl(
    urls: Annotated[List[str], Field(description="List of URLs to crawl")],
    config: Annotated[Optional[Dict[str, Any]], Field(description="Optional configuration parameters (default: None)")] = None,
    base_timeout: Annotated[int, Field(description="Base timeout in seconds, adjusted based on URL count (default: 30)")] = 30
) -> List[CrawlResponse]:
    """
    Crawl multiple URLs in batch.
    
    Process multiple URLs concurrently for efficiency. Timeout auto-scales based on URL count.
    
    Parameters:
    - urls: List of URLs to crawl (required)
    - config: Optional configuration parameters (default: None)
    - base_timeout: Base timeout in seconds, adjusted based on URL count (default: 30)
    
    Example:
    {"urls": ["https://example.com/page1", "https://example.com/page2"], "config": {"generate_markdown": true}}
    
    Returns List of CrawlResponse objects for each URL.
    """
    if not urls:
        return []
    
    # Prepare default configuration
    default_config = {
        "generate_markdown": True,
        "extract_media": False,
        "wait_for_js": False,
        "auto_summarize": False
    }
    
    if config:
        default_config.update(config)
    
    # Calculate dynamic timeout based on URL count  
    # Base timeout + additional time per URL (10s per additional URL after the first)
    dynamic_timeout = base_timeout + max(0, (len(urls) - 1) * 10)
    default_config["timeout"] = dynamic_timeout
    
    # Limit concurrent processing to be respectful to servers and prevent hangs
    max_concurrent = min(len(urls), 2)  # Reduced to 2 concurrent requests for stability
    semaphore = asyncio.Semaphore(max_concurrent)

    async def crawl_single_url(url: str) -> CrawlResponse:
        async with semaphore:
            try:
                # Create crawl request with merged configuration
                request = CrawlRequest(url=url, **default_config)
                # Add individual URL timeout to prevent hangs
                result = await asyncio.wait_for(
                    _internal_crawl_url(request),
                    timeout=dynamic_timeout
                )
                return result
            except asyncio.TimeoutError:
                # Handle individual URL timeouts
                return CrawlResponse(
                    success=False,
                    url=url,
                    error=f"Individual URL timeout after {dynamic_timeout}s: {url}"
                )
            except Exception as e:
                # Return error response for failed URLs
                return CrawlResponse(
                    success=False,
                    url=url,
                    error=f"Batch crawl error for {url}: {str(e)}"
                )
    
    # Process all URLs concurrently with semaphore control and global timeout
    tasks = [crawl_single_url(url) for url in urls]
    try:
        # Add overall batch timeout to prevent infinite hangs
        batch_timeout = dynamic_timeout * 2  # Allow extra time for batch processing
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=batch_timeout
        )
    except asyncio.TimeoutError:
        # If batch times out, return timeout errors for all URLs
        results = [
            CrawlResponse(
                success=False,
                url=url,
                error=f"Batch timeout after {batch_timeout}s"
            ) for url in urls
        ]
    
    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(CrawlResponse(
                success=False,
                url=urls[i],
                error=f"Batch processing exception: {str(result)}"
            ))
        else:
            processed_results.append(result)
    
    return processed_results


async def get_tool_selection_guide() -> Dict[str, Any]:
    """
    Get comprehensive tool selection guide for AI agents.
    
    Provides complete mapping of use cases to appropriate tools, workflows, and complexity guides.
    Essential for tool selection, workflow planning, and understanding capabilities.
    
    Parameters: None
    
    Returns dictionary with tool selection guide, workflows, and complexity mapping.
    """
    try:
        return {
            "success": True,
            "version": "2024.12",
            "last_updated": "2024-12-20",
            "total_tools": 20,
            "tool_selection_guide": TOOL_SELECTION_GUIDE,
            "workflow_guide": WORKFLOW_GUIDE,
            "complexity_guide": COMPLEXITY_GUIDE,
            "performance_guide": PERFORMANCE_GUIDE,
            "guide_categories": [
                "single_page_content",
                "multi_page_crawling", 
                "targeted_extraction",
                "file_processing",
                "video_content",
                "search_operations",
                "batch_operations",
                "configuration_and_info"
            ],
            "quick_reference": {
                "basic_crawling": {
                    "tool": "crawl_url",
                    "description": "Simple single-page content extraction",
                    "example": '{"url": "https://example.com"}'
                },
                "difficult_sites": {
                    "tool": "crawl_url_with_fallback", 
                    "description": "Crawling with multiple fallback strategies",
                    "example": '{"url": "https://difficult-site.com", "wait_for_js": true, "simulate_user": true}'
                },
                "multiple_pages": {
                    "tool": "deep_crawl_site",
                    "description": "Multi-page crawling with link following",
                    "example": '{"url": "https://docs.site.com", "max_depth": 2, "max_pages": 10}'
                },
                "specific_data": {
                    "tool": "intelligent_extract",
                    "description": "AI-powered extraction of specific information",
                    "example": '{"url": "https://company.com", "extraction_goal": "contact information and pricing"}'
                },
                "contact_info": {
                    "tool": "extract_entities",
                    "description": "Extract emails, phones, addresses using patterns",
                    "example": '{"url": "https://company.com", "entity_types": ["emails", "phones"]}'
                },
                "structured_data": {
                    "tool": "extract_structured_data",
                    "description": "Extract data matching a predefined schema",
                    "example": '{"request": {"url": "https://shop.com", "schema": {"title": "string", "price": "number"}}}'
                },
                "documents": {
                    "tool": "process_file",
                    "description": "Process PDF, Office files, and archives",
                    "example": '{"url": "https://example.com/document.pdf", "auto_summarize": true}'
                },
                "large_content": {
                    "tool": "enhanced_process_large_content",
                    "description": "Advanced processing for large documents/content",
                    "example": '{"url": "https://long-article.com", "chunking_strategy": "topic", "summarize_chunks": true}'
                },
                "youtube": {
                    "tool": "extract_youtube_transcript",
                    "description": "Extract video transcripts with timestamps",
                    "example": '{"url": "https://youtube.com/watch?v=abc123", "auto_summarize": true}'
                },
                "search": {
                    "tool": "search_google",
                    "description": "Search Google with genre filtering and date ranges",
                    "example": '{"request": {"query": "AI research", "search_genre": "academic"}, "recent_days": 30}'
                },
                "search_crawl": {
                    "tool": "search_and_crawl",
                    "description": "Search and automatically crawl top results with date filtering",
                    "example": '{"search_query": "competitor analysis", "crawl_top_results": 5, "recent_days": 7}'
                },
                "multiple_urls": {
                    "tool": "batch_crawl",
                    "description": "Process multiple URLs efficiently",
                    "example": '{"urls": ["https://site1.com", "https://site2.com"], "config": {"generate_markdown": true}}'
                },
                "system_info": {
                    "tool": "get_llm_config_info",
                    "description": "Get current system configuration and status",
                    "example": "No parameters required"
                }
            },
            "decision_tree": {
                "web_content": {
                    "single_page": {
                        "simple_site": {
                            "tool": "crawl_url",
                            "when": "Standard HTML sites, blogs, news articles"
                        },
                        "spa_or_js": {
                            "tool": "crawl_url",
                            "parameters": {"wait_for_js": True},
                            "when": "React, Vue, Angular apps requiring JavaScript execution"
                        },
                        "difficult_site": {
                            "tool": "crawl_url_with_fallback",
                            "when": "Sites with bot protection, complex JavaScript, or frequent failures"
                        }
                    },
                    "multiple_pages": {
                        "tool": "deep_crawl_site",
                        "when": "Documentation sites, wikis, multi-page content exploration"
                    },
                    "specific_data": {
                        "ai_extraction": {
                            "tool": "intelligent_extract",
                            "when": "Need to extract specific information described in natural language"
                        },
                        "regex_patterns": {
                            "tool": "extract_entities",
                            "when": "Well-defined patterns like emails, phones, URLs, addresses"
                        },
                        "structured_schema": {
                            "tool": "extract_structured_data",
                            "when": "Need data in a specific JSON schema format"
                        }
                    }
                },
                "documents": {
                    "standard_size": {
                        "tool": "process_file",
                        "when": "PDF, Word, Excel files under 100MB"
                    },
                    "large_documents": {
                        "tool": "enhanced_process_large_content",
                        "when": "Large PDFs, technical documents requiring chunking/summarization"
                    }
                },
                "youtube": {
                    "single_video": {
                        "tool": "extract_youtube_transcript",
                        "when": "Individual video transcript extraction"
                    },
                    "video_info": {
                        "tool": "get_youtube_video_info",
                        "when": "Need video metadata and transcript availability info"
                    },
                    "multiple_videos": {
                        "tool": "batch_extract_youtube_transcripts",
                        "when": "Process many videos efficiently"
                    }
                },
                "search": {
                    "search_only": {
                        "tool": "search_google",
                        "when": "Need search results with URLs and snippets only"
                    },
                    "search_and_content": {
                        "tool": "search_and_crawl",
                        "when": "Need both search results and full page content"
                    },
                    "multiple_queries": {
                        "tool": "batch_search_google",
                        "when": "Research requiring multiple search queries"
                    }
                },
                "batch_processing": {
                    "multiple_urls": {
                        "tool": "batch_crawl",
                        "when": "Process many websites efficiently"
                    },
                    "multiple_videos": {
                        "tool": "batch_extract_youtube_transcripts",
                        "when": "Extract transcripts from many YouTube videos"
                    },
                    "multiple_searches": {
                        "tool": "batch_search_google",
                        "when": "Execute multiple search queries with analysis"
                    }
                }
            },
            "best_practices": {
                "parameter_optimization": [
                    "Use appropriate timeouts based on site complexity",
                    "Enable wait_for_js for Single Page Applications",
                    "Use search genres for targeted results",
                    "Use recent_days parameter for time-sensitive searches",
                    "Enable auto_summarize for large content",
                    "Set appropriate batch sizes for efficiency"
                ],
                "error_handling": [
                    "Always check success field in responses",
                    "Use fallback tools for difficult sites",
                    "Implement retry logic for batch operations",
                    "Monitor rate limits for search operations"
                ],
                "performance_tips": [
                    "Use batch operations for multiple items",
                    "Cache results when possible",
                    "Use appropriate chunk sizes for large content",
                    "Optimize concurrent request limits"
                ]
            },
            "troubleshooting": {
                "common_issues": {
                    "timeout_errors": "Increase timeout parameter or use crawl_url_with_fallback",
                    "javascript_sites": "Set wait_for_js=true or wait_for_selector parameter",
                    "rate_limiting": "Reduce concurrent requests or add delays",
                    "large_content": "Use enhanced_process_large_content with chunking",
                    "search_failures": "Check search genre compatibility and query formatting"
                },
                "debugging_steps": [
                    "Check tool selection guide for appropriate tool",
                    "Verify required parameters are provided",
                    "Test with simple examples first",
                    "Check system configuration with get_llm_config_info",
                    "Use get_supported_file_formats for file processing issues"
                ]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate tool selection guide: {str(e)}"
        }


async def get_llm_config_info() -> Dict[str, Any]:
    """
    Get information about the current LLM configuration.
    
    Provides details about available LLM providers, models, and API key status.
    
    Parameters: None
    
    Returns dictionary with LLM configuration details including available providers and models.
    """
    try:
        # Import config functions
        try:
            from ..config import get_llm_config
        except ImportError:
            return {
                "success": False,
                "error": "LLM configuration module not available"
            }
        
        # Get current LLM configuration
        current_config = get_llm_config()
        
        return {
            "success": True,
            "current_configuration": {
                "provider": getattr(current_config, 'provider', 'Not configured'),
                "model": getattr(current_config, 'model', 'Not configured'), 
                "api_token_configured": bool(getattr(current_config, 'api_token', None)),
                "base_url": getattr(current_config, 'base_url', None)
            },
            "supported_providers": [
                "openai",
                "anthropic", 
                "google",
                "ollama",
                "azure",
                "together",
                "groq"
            ],
            "supported_features": [
                "Content summarization for large documents",
                "Search result analysis and insights", 
                "Batch processing with AI enhancement",
                "Multiple model support for different tasks",
                "Custom API endpoints and configurations",
                "Intelligent content extraction",
                "YouTube transcript summarization",
                "File content analysis and processing"
            ],
            "configuration_status": {
                "properly_configured": bool(getattr(current_config, 'provider', None) and getattr(current_config, 'api_token', None)),
                "configuration_source": "Environment variables and MCP settings",
                "auto_detection": "Provider and model auto-detected based on configuration"
            },
            "usage_statistics": {
                "ai_enabled_tools": [
                    "intelligent_extract",
                    "enhanced_process_large_content", 
                    "process_file (with auto_summarize)",
                    "extract_youtube_transcript (with auto_summarize)",
                    "search result analysis in batch operations"
                ],
                "tools_requiring_llm": 5,
                "total_tools": 20,
                "ai_enhancement_percentage": "25%"
            },
            "setup_guidance": {
                "environment_variables": [
                    "OPENAI_API_KEY for OpenAI models",
                    "ANTHROPIC_API_KEY for Claude models", 
                    "GOOGLE_API_KEY for Gemini models",
                    "Or configure via MCP settings"
                ],
                "recommended_models": {
                    "general_purpose": "gpt-4o, claude-3-sonnet",
                    "fast_processing": "gpt-3.5-turbo, claude-3-haiku",
                    "large_content": "gpt-4-turbo, claude-3-opus",
                    "cost_effective": "gpt-3.5-turbo, local ollama models"
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get LLM configuration: {str(e)}"
        }