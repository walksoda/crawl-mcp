# API Reference

Complete reference for all MCP tools available in the Crawl4AI MCP Server.

## Tool Selection Guide

### Choose the Right Tool for Your Task

| **Use Case** | **Recommended Tool** | **Key Features** |
|-------------|---------------------|------------------|
| Single webpage | `crawl_url` | Basic crawling, JS support |
| Multiple pages (up to 10) | `deep_crawl_site` | Site mapping, link following |
| Search + Crawling | `search_and_crawl` | Google search + auto-crawl |
| Difficult sites | `crawl_url_with_fallback` | Multiple retry strategies |
| Structured data | `extract_structured_data` | CSS/LLM schemas |
| Intelligent extraction | `intelligent_extract` | Goal-based, filter-aware extraction |
| Entity extraction | `extract_entities` | Emails, phones, URLs, dates, IPs, prices |
| File processing | `process_file` | PDF, Office, ZIP conversion |
| Large content | `enhanced_process_large_content` | Chunking and filtering strategies |
| YouTube transcript | `extract_youtube_transcript` | Subtitle extraction |
| YouTube comments | `extract_youtube_comments` | Comment extraction with sorting |
| YouTube transcripts (batch) | `batch_extract_youtube_transcripts` | Up to 3 videos at once |
| Video metadata | `get_youtube_video_info` | Title, description, summary |
| Batch crawling | `batch_crawl` | Up to 3 URLs concurrently |
| Multi-URL with config | `multi_url_crawl` | Up to 5 URLs, per-URL config |
| Batch search | `batch_search_google` | Up to 3 queries concurrently |
| Search genres | `get_search_genres` | List available genre filters |
| Supported file formats | `get_supported_file_formats` | List supported formats |

### Performance Guidelines

- **Deep Crawling**: Limited to 10 pages max (stability focused)
- **Batch Processing**: Concurrent limits enforced
- **Timeout Calculation**: `pages x base_timeout` recommended
- **Large Files**: 100MB maximum size limit
- **Retry Strategy**: Manual retry recommended on first failure

### Best Practices

**For JavaScript-Heavy Sites:**
- Always use `wait_for_js: true`
- Increase timeout to 30-60 seconds
- Use `wait_for_selector` for specific elements

**For AI Features:**
- Use `extract_structured_data` with LLM mode for semantic extraction
- Fallback to CSS extraction if LLM unavailable
- Customize extraction schemas based on needs

## Web Crawling Tools

### `crawl_url`

Web crawling with JavaScript support, screenshot capture, and optional content summarization.

**Parameters:**
- `url` (str): URL to crawl
- `css_selector` (Optional[str]): CSS selector for targeted extraction (default: None)
- `extract_media` (bool): Extract images and videos (default: False)
- `take_screenshot` (bool): Take a screenshot of the page (default: False)
- `generate_markdown` (bool): Generate markdown from page content (default: True)
- `include_cleaned_html` (bool): Include cleaned HTML in response (default: False)
- `wait_for_selector` (Optional[str]): Wait for a specific element to load (default: None)
- `timeout` (int): Timeout in seconds (default: 60)
- `wait_for_js` (bool): Wait for JavaScript to finish executing (default: False)
- `auto_summarize` (bool): Auto-summarize large content using LLM (default: False)
- `use_undetected_browser` (bool): Use undetected browser to bypass bot detection (default: False)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)

**Response Behavior:**
- By default, returns markdown content only to reduce token usage
- Set `include_cleaned_html=True` to also receive cleaned HTML content
- Token limit: 25000 tokens (automatically truncated with recommendations if exceeded)

### `deep_crawl_site`

Dedicated tool for comprehensive site mapping and recursive crawling.

**Parameters:**
- `url`: Starting URL
- `max_depth`: Maximum crawling depth (default: 2, recommended: 1-3)
- `max_pages`: Maximum number of pages to crawl (default: 5, max: 10)
- `crawl_strategy`: Crawling strategy ('bfs', 'dfs', 'best_first')
- `include_external`: Include external links in crawl
- `url_pattern`: URL filter pattern (e.g., '*docs*', '*blog*')
- `score_threshold`: Minimum relevance score (0.0-1.0)
- `extract_media`: Extract images and videos from pages
- `base_timeout`: Base timeout per page in seconds

### `crawl_url_with_fallback`

Crawl with fallback strategies for anti-bot sites. Use content_offset/content_limit for pagination.

**Parameters:**
- `url` (str): URL to crawl
- `css_selector` (Optional[str]): CSS selector (default: None)
- `extract_media` (bool): Extract media (default: False)
- `take_screenshot` (bool): Take screenshot (default: False)
- `generate_markdown` (bool): Generate markdown (default: True)
- `wait_for_selector` (Optional[str]): Element to wait for (default: None)
- `timeout` (int): Timeout in seconds (default: 60)
- `wait_for_js` (bool): Wait for JavaScript (default: False)
- `auto_summarize` (bool): Auto-summarize content (default: False)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)

## Data Extraction Tools

### `extract_structured_data`

Traditional structured data extraction using CSS selectors or LLM schemas.

**Parameters:**
- `url`: Target URL
- `extraction_type`: Extraction method ('css', 'llm', 'table')
- `css_selectors`: CSS selectors for targeted extraction
- `extraction_schema`: Schema definition for LLM extraction
- `generate_markdown`: Generate markdown from page content
- `wait_for_js`: Wait for JavaScript before extraction
- `timeout`: Timeout in seconds (default: 30)
- `use_llm_table_extraction`: Use LLM for table extraction
- `table_chunking_strategy`: Table chunking method ('intelligent', 'fixed', 'semantic')

### `intelligent_extract`

Goal-based content extraction with configurable filtering and optional LLM enrichment.

**Parameters:**
- `url`: Target URL
- `extraction_goal`: Natural language description of what to extract
- `content_filter`: Filter strategy ('bm25', 'pruning', 'llm')
- `filter_query`: Query string for filter-based strategies
- `chunk_content`: Enable content chunking for large documents
- `use_llm`: Use LLM for semantic extraction
- `llm_provider`: LLM provider (auto-detected if not specified)
- `llm_model`: Specific LLM model (auto-detected if not specified)
- `custom_instructions`: Additional instructions for the extraction process

### `extract_entities`

Extract specific entity types from a URL using pattern matching or LLM.

**Parameters:**
- `url`: Target URL
- `entity_types` (List): Entity types to extract — supported values: 'email', 'phone', 'url', 'date', 'ip', 'price'
- `custom_patterns`: Custom regex patterns for entity extraction
- `include_context`: Include surrounding context for each entity
- `deduplicate`: Remove duplicate entities from results
- `use_llm`: Use LLM for enhanced entity detection
- `llm_provider`: LLM provider (auto-detected if not specified)
- `llm_model`: Specific LLM model (auto-detected if not specified)

## File Processing Tools

### `process_file`

Convert various file formats to Markdown using Microsoft MarkItDown.

**Parameters:**
- `url`: File URL (PDF, Office, ZIP, etc.)
- `max_size_mb`: Maximum file size limit (default: 100MB)
- `extract_all_from_zip`: Extract all files from ZIP archives
- `include_metadata`: Include file metadata in response
- `auto_summarize`: Automatically summarize large content using LLM
- `max_content_tokens`: Maximum tokens before triggering auto-summarization
- `summary_length`: Summary length setting ('short', 'medium', 'long')
- `llm_provider`: LLM provider for summarization (auto-detected if not specified)
- `llm_model`: Specific LLM model for summarization (auto-detected if not specified)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)

**Supported Formats:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **Archives**: .zip
- **Web/Text**: .html, .htm, .txt, .md, .csv, .rtf
- **eBooks**: .epub

### `get_supported_file_formats`

Returns the list of supported file formats and their capabilities. No parameters required.

### `enhanced_process_large_content`

Process large content with advanced chunking and filtering strategies.

**Parameters:**
- `url`: Target URL or file URL
- `chunking_strategy`: How to split content ('topic', 'sentence', 'overlap', 'regex')
- `filtering_strategy`: How to filter chunks ('bm25', 'pruning', 'llm')
- `filter_query`: Query string for filter-based strategies
- `max_chunk_tokens`: Maximum tokens per chunk
- `chunk_overlap`: Overlap size between consecutive chunks
- `extract_top_chunks`: Number of top-ranked chunks to return
- `similarity_threshold`: Minimum similarity score for chunk inclusion
- `summarize_chunks`: Summarize individual chunks before merging
- `merge_strategy`: How to merge processed chunks ('hierarchical', 'linear')
- `final_summary_length`: Length of the final merged summary

## YouTube Processing Tools

### `extract_youtube_transcript`

Extract transcripts from YouTube videos with language preferences and optional translation.

**Parameters:**
- `url`: YouTube video URL
- `languages`: Preferred languages in order of preference (default: ["ja", "en"])
- `translate_to`: Target language for translation (optional)
- `include_timestamps`: Include timestamps in transcript
- `preserve_formatting`: Preserve original formatting
- `include_metadata`: Include video metadata
- `auto_summarize`: Automatically summarize large content using LLM
- `max_content_tokens`: Maximum tokens before triggering auto-summarization
- `summary_length`: Summary length setting ('short', 'medium', 'long')
- `llm_provider`: LLM provider for summarization (auto-detected if not specified)
- `llm_model`: Specific LLM model for summarization (auto-detected if not specified)
- `enable_crawl_fallback`: Fall back to page crawl if transcript API fails
- `fallback_timeout`: Timeout in seconds for crawl fallback
- `enrich_metadata`: Enrich video metadata with additional details
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)

### `extract_youtube_comments`

Extract and sort comments from a YouTube video.

**Parameters:**
- `url`: YouTube video URL
- `sort_by`: Comment sort order ('popular' or 'recent')
- `max_comments`: Maximum number of comments to retrieve (1-1000, default: 300)
- `comment_offset`: Starting offset for comment retrieval
- `include_replies`: Include replies to top-level comments
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)

### `batch_extract_youtube_transcripts`

Extract transcripts from multiple YouTube videos in a single request (up to 3 videos).

**Parameters:**
- `request` (Dict): Request object with the following fields:
  - `urls`: List of YouTube video URLs (max 3)
  - `languages`: Preferred languages in order of preference
  - `include_timestamps`: Include timestamps in transcripts

### `get_youtube_video_info`

Get metadata and transcript information for a YouTube video.

**Parameters:**
- `video_url`: YouTube video URL
- `summarize_transcript`: Generate a summary of the transcript
- `max_tokens`: Maximum tokens for transcript processing
- `llm_provider`: LLM provider for summarization (auto-detected if not specified)
- `llm_model`: Specific LLM model for summarization (auto-detected if not specified)
- `summary_length`: Summary length setting ('short', 'medium', 'long')
- `include_timestamps`: Include timestamps in transcript output

**Returns:**
- Available transcript languages
- Manual/auto-generated distinction
- Translatable language information

## Google Search Tools

### `search_google`

Perform Google search with genre filtering and metadata extraction.

**Parameters:**
- `request` (Dict): Request object with the following fields:
  - `query`: Search query string
  - `num_results`: Number of results to return (1-100, default: 10)
  - `search_genre`: Content genre filter (optional)
  - `language`: Search language (default: "en")
  - `region`: Search region (default: "us")
  - `recent_days`: Limit results to the past N days (optional)
  - `content_limit`: Maximum characters to return per result
  - `content_offset`: Start position for content retrieval

**Features:**
- Automatic title and snippet extraction from search results
- 7 optimized search genres using Google official operators
- URL classification and domain analysis
- Safe search enforced by default

### `search_and_crawl`

Perform Google search and automatically crawl top results.

**Parameters:**
- `request` (Dict): Request object with the following fields:
  - `search_query` (required): Google search query
  - `crawl_top_results`: Number of top results to crawl (1-10, default: 3)
  - `search_genre`: Content genre filter (optional)
  - `recent_days`: Limit search results to the past N days (optional)
  - `generate_markdown`: Generate markdown content from crawled pages
  - `max_content_per_page`: Maximum content characters per crawled page

**Returns:**
- Complete search metadata and crawled content
- Success rates and processing statistics
- Integrated analysis of search and crawl results

### `batch_search_google`

Perform multiple Google searches concurrently (up to 3 queries).

**Parameters:**
- `request` (Dict): Request object with the following fields:
  - `queries`: List of search query strings (max 3)
  - `num_results_per_query`: Number of results per query
  - `search_genre`: Content genre filter applied to all queries
  - `recent_days`: Limit results to the past N days (optional)

### `get_search_genres`

Returns the list of available search genre filters. No parameters required.

## Batch Operations

### `batch_crawl`

Crawl multiple URLs concurrently in a single request (up to 3 URLs).

**Parameters:**
- `urls` (List): List of URLs to crawl (max 3)
- `base_timeout`: Timeout in seconds per URL (default: 30)
- `generate_markdown`: Generate markdown from page content
- `extract_media`: Extract images and videos from pages
- `wait_for_js`: Wait for JavaScript to finish executing

### `multi_url_crawl`

Crawl multiple URLs with per-URL configuration options (up to 5 URLs).

**Parameters:**
- `url_configurations` (Dict): Per-URL configuration map (max 5 URLs)
- `pattern_matching`: URL pattern matching mode ('wildcard' or 'regex')
- `default_config`: Default configuration applied to all URLs
- `base_timeout`: Base timeout in seconds per URL
- `max_concurrent`: Maximum number of concurrent crawl operations

## Tool Categories

### By Complexity
- **Simple**: `crawl_url`, `process_file`, `get_youtube_video_info`, `get_search_genres`, `get_supported_file_formats`
- **Moderate**: `deep_crawl_site`, `search_google`, `extract_youtube_transcript`, `extract_youtube_comments`, `batch_crawl`
- **Advanced**: `extract_structured_data`, `intelligent_extract`, `extract_entities`, `search_and_crawl`, `batch_search_google`, `multi_url_crawl`, `batch_extract_youtube_transcripts`, `enhanced_process_large_content`

### By Use Case
- **Content Discovery**: `search_google`, `search_and_crawl`, `batch_search_google`, `get_search_genres`
- **Data Extraction**: `crawl_url`, `extract_structured_data`, `intelligent_extract`, `extract_entities`
- **Media Processing**: `extract_youtube_transcript`, `extract_youtube_comments`, `batch_extract_youtube_transcripts`, `process_file`, `get_supported_file_formats`
- **Site Analysis**: `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`, `multi_url_crawl`
- **Large Content**: `enhanced_process_large_content`, `get_youtube_video_info`

## Integration Examples

For detailed configuration examples, see [Configuration Examples](CONFIGURATION_EXAMPLES.md).

For HTTP API integration, see [HTTP Integration Guide](HTTP_INTEGRATION.md).

For advanced usage patterns, see [Advanced Usage Guide](ADVANCED_USAGE.md).
