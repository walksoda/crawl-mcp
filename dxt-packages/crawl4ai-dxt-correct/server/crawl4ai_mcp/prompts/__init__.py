"""
Prompts module for Crawl4AI MCP Server.

Contains all MCP prompt definitions for various crawling and processing tasks.
"""

from .mcp_prompts import (
    crawl_website_prompt,
    analyze_crawl_results_prompt,
    batch_crawl_setup_prompt,
    process_file_prompt
)

__all__ = [
    'crawl_website_prompt',
    'analyze_crawl_results_prompt', 
    'batch_crawl_setup_prompt',
    'process_file_prompt'
]