"""
Crawl-related models for Crawl4AI MCP Server.

Contains request and response models for web crawling operations.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .base import BaseRequest, BaseResponse


class CrawlRequest(BaseRequest):
    """Request model for crawling operations."""
    url: str = Field(..., description="URL to crawl")
    css_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    xpath: Optional[str] = Field(None, description="XPath selector for content extraction")
    extract_media: bool = Field(False, description="Whether to extract media files")
    take_screenshot: bool = Field(False, description="Whether to take a screenshot")
    generate_markdown: bool = Field(True, description="Whether to generate markdown")
    include_cleaned_html: bool = Field(False, description="Include cleaned HTML in content field (default: False, only markdown returned)")
    wait_for_selector: Optional[str] = Field(None, description="Wait for specific element")
    timeout: int = Field(60, description="Request timeout in seconds")
    
    # Deep crawling parameters
    max_depth: Optional[int] = Field(None, description="Maximum crawling depth (None for single page)")
    max_pages: Optional[int] = Field(10, description="Maximum number of pages to crawl")
    include_external: bool = Field(False, description="Whether to follow external domain links")
    crawl_strategy: str = Field("bfs", description="Crawling strategy: 'bfs', 'dfs', or 'best_first'")
    url_pattern: Optional[str] = Field(None, description="URL pattern filter (e.g., '*docs*')")
    score_threshold: float = Field(0.3, description="Minimum score for URLs to be crawled")
    
    # Advanced content processing
    content_filter: Optional[str] = Field(None, description="Content filter type: 'bm25', 'pruning', 'llm'")
    filter_query: Optional[str] = Field(None, description="Query for BM25 content filtering")
    chunk_content: bool = Field(False, description="Whether to chunk large content")
    chunk_strategy: str = Field("topic", description="Chunking strategy: 'topic', 'regex', 'sentence'")
    chunk_size: int = Field(1000, description="Maximum chunk size in tokens")
    overlap_rate: float = Field(0.1, description="Overlap rate between chunks (0.0-1.0)")
    
    # Browser configuration
    user_agent: Optional[str] = Field(None, description="Custom user agent string")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")
    enable_caching: bool = Field(True, description="Whether to enable caching")
    cache_mode: str = Field("enabled", description="Cache mode: 'enabled', 'disabled', 'bypass'")
    
    # JavaScript and interaction
    execute_js: Optional[str] = Field(None, description="JavaScript code to execute")
    wait_for_js: bool = Field(False, description="Wait for JavaScript to complete")
    simulate_user: bool = Field(False, description="Simulate human-like browsing behavior")
    use_undetected_browser: bool = Field(False, description="Use undetected browser mode to bypass bot detection")
    
    # Authentication
    auth_token: Optional[str] = Field(None, description="Authentication token")
    cookies: Optional[Dict[str, str]] = Field(None, description="Custom cookies")
    
    # Auto-summarization for large content
    auto_summarize: bool = Field(False, description="Automatically summarize large content using LLM")
    max_content_tokens: int = Field(15000, description="Maximum tokens before triggering auto-summarization")
    summary_length: str = Field("medium", description="Summary length: 'short', 'medium', 'long'")
    llm_provider: Optional[str] = Field(None, description="LLM provider for summarization (auto-detected if not specified)")
    llm_model: Optional[str] = Field(None, description="Specific LLM model for summarization (auto-detected if not specified)")


class CrawlResponse(BaseResponse):
    """Response model for crawling operations."""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    markdown: Optional[str] = None
    media: Optional[List[Dict[str, str]]] = None
    screenshot: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None


class LargeContentRequest(BaseRequest):
    """Request model for large content processing operations."""
    url: str = Field(..., description="URL to process")
    chunking_strategy: str = Field("topic", description="Chunking strategy: 'topic', 'sentence', 'overlap', 'regex'")
    filtering_strategy: str = Field("bm25", description="Filtering strategy: 'bm25', 'pruning', 'llm'")
    filter_query: Optional[str] = Field(None, description="Query for BM25 filtering")
    max_chunk_tokens: int = Field(8000, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(500, description="Token overlap between chunks")
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold for relevant chunks")
    extract_top_chunks: int = Field(10, description="Number of top relevant chunks to extract")
    summarize_chunks: bool = Field(True, description="Whether to summarize individual chunks")
    merge_strategy: str = Field("hierarchical", description="Strategy for merging chunk summaries")
    final_summary_length: str = Field("medium", description="Final summary length: 'short', 'medium', 'long'")


class LargeContentResponse(BaseResponse):
    """Response model for large content processing operations."""
    url: str
    original_content_length: int
    filtered_content_length: int
    total_chunks: int
    relevant_chunks: int
    processing_method: str
    chunking_strategy_used: str
    filtering_strategy_used: str
    chunks: List[Dict[str, Any]]
    chunk_summaries: Optional[List[str]] = None
    merged_summary: Optional[str] = None
    final_summary: Optional[str] = None