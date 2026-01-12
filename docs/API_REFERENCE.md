# API Reference

Complete reference for all MCP tools available in the Crawl4AI MCP Server.

## 🛠️ Tool Selection Guide

### 📋 **Choose the Right Tool for Your Task**

| **Use Case** | **Recommended Tool** | **Key Features** |
|-------------|---------------------|------------------|
| Single webpage | `crawl_url` | Basic crawling, JS support |
| Multiple pages (up to 5) | `deep_crawl_site` | Site mapping, link following |
| Search + Crawling | `search_and_crawl` | Google search + auto-crawl |
| Difficult sites | `crawl_url_with_fallback` | Multiple retry strategies |
| Structured data | `extract_structured_data` | CSS/XPath/LLM schemas |
| File processing | `process_file` | PDF, Office, ZIP conversion |
| YouTube content | `extract_youtube_transcript` | Subtitle extraction |
| Video metadata | `get_youtube_video_info` | Title, description, etc. |

### ⚡ **Performance Guidelines**

- **Deep Crawling**: Limited to 5 pages max (stability focused)
- **Batch Processing**: Concurrent limits enforced
- **Timeout Calculation**: `pages × base_timeout` recommended
- **Large Files**: 100MB maximum size limit
- **Retry Strategy**: Manual retry recommended on first failure

### 🎯 **Best Practices**

**For JavaScript-Heavy Sites:**
- Always use `wait_for_js: true`
- Set `simulate_user: true` for better compatibility
- Increase timeout to 30-60 seconds
- Use `wait_for_selector` for specific elements

**For AI Features:**
- Use `extract_structured_data` with LLM mode for semantic extraction
- Fallback to CSS/XPath extraction if LLM unavailable
- Customize extraction schemas based on needs

## 🔧 Web Crawling Tools

### `crawl_url`

Advanced web crawling with deep crawling support, intelligent filtering, and automatic summarization for large content.

**Key Parameters:**
- `url`: Target URL to crawl
- `max_depth`: Maximum crawling depth (None for single page)
- `crawl_strategy`: Strategy type ('bfs', 'dfs', 'best_first')
- `content_filter`: Filter type ('bm25', 'pruning', 'llm')
- `chunk_content`: Enable content chunking for large documents
- `execute_js`: Custom JavaScript code execution
- `user_agent`: Custom user agent string
- `headers`: Custom HTTP headers
- `cookies`: Authentication cookies
- `include_cleaned_html`: Include cleaned HTML in response (default: False, markdown only)
- `auto_summarize`: Automatically summarize large content using LLM
- `max_content_tokens`: Maximum tokens before triggering auto-summarization (default: 15000)
- `summary_length`: Summary length setting ('short', 'medium', 'long')
- `llm_provider`: LLM provider for summarization (auto-detected if not specified)
- `llm_model`: Specific LLM model for summarization (auto-detected if not specified)

**Response Behavior:**
- By default, returns markdown content only to reduce token usage
- Set `include_cleaned_html=True` to also receive cleaned HTML content
- Token limit: 25000 tokens (automatically truncated with recommendations if exceeded)

### `deep_crawl_site`

Dedicated tool for comprehensive site mapping and recursive crawling.

**Parameters:**
- `url`: Starting URL
- `max_depth`: Maximum crawling depth (recommended: 1-3)
- `max_pages`: Maximum number of pages to crawl
- `crawl_strategy`: Crawling strategy ('bfs', 'dfs', 'best_first')
- `url_pattern`: URL filter pattern (e.g., '*docs*', '*blog*')
- `score_threshold`: Minimum relevance score (0.0-1.0)

### `crawl_url_with_fallback`

Robust crawling with multiple fallback strategies for maximum reliability.

## 🧠 Data Extraction Tools

### `extract_structured_data`

Traditional structured data extraction using CSS/XPath selectors or LLM schemas.

## 📄 File Processing Tools

### `process_file`

**📄 File Processing**: Convert various file formats to Markdown using Microsoft MarkItDown.

**Parameters:**
- `url`: File URL (PDF, Office, ZIP, etc.)
- `max_size_mb`: Maximum file size limit (default: 100MB)
- `extract_all_from_zip`: Extract all files from ZIP archives
- `include_metadata`: Include file metadata in response

**Supported Formats:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **Archives**: .zip
- **Web/Text**: .html, .htm, .txt, .md, .csv, .rtf
- **eBooks**: .epub

## 📺 YouTube Processing Tools

### `extract_youtube_transcript`

**📺 YouTube Processing**: Extract transcripts from YouTube videos with language preferences and translation using youtube-transcript-api v1.1.0+.

**✅ Stable and reliable - No authentication required!**

**Parameters:**
- `url`: YouTube video URL
- `languages`: Preferred languages in order of preference (default: ["ja", "en"])
- `translate_to`: Target language for translation (optional)
- `include_timestamps`: Include timestamps in transcript
- `preserve_formatting`: Preserve original formatting
- `include_metadata`: Include video metadata

### `get_youtube_video_info`

**📋 YouTube Info**: Get available transcript information for a YouTube video without extracting the full transcript.

**Parameters:**
- `video_url`: YouTube video URL

**Returns:**
- Available transcript languages
- Manual/auto-generated distinction
- Translatable language information

## 🔍 Google Search Tools

### `search_google`

**🔍 Google Search**: Perform Google search with genre filtering and metadata extraction.

**Parameters:**
- `query`: Search query string
- `num_results`: Number of results to return (1-100, default: 10)
- `language`: Search language (default: "en")
- `region`: Search region (default: "us")
- `search_genre`: Content genre filter (optional)
- `safe_search`: Safe search enabled (always True for security)

**Features:**
- Automatic title and snippet extraction from search results
- 7 optimized search genres using Google official operators
- URL classification and domain analysis
- Safe search enforced by default

### `search_and_crawl`

**🔍 Integrated Search+Crawl**: Perform Google search and automatically crawl top results.

**Parameters:**
- `search_query`: Google search query
- `num_search_results`: Number of search results (1-20, default: 5)
- `crawl_top_results`: Number of top results to crawl (1-10, default: 3)
- `extract_media`: Extract media from crawled pages
- `generate_markdown`: Generate markdown content
- `search_genre`: Content genre filter (optional)

**Returns:**
- Complete search metadata and crawled content
- Success rates and processing statistics
- Integrated analysis of search and crawl results

## 📚 MCP Resources

### Available Resources

- `uri://crawl4ai/config`: Default crawler configuration options
- `uri://crawl4ai/examples`: Usage examples and sample requests

## 🎯 MCP Prompts

### Available Prompts

- `crawl_website_prompt`: Guided website crawling workflows
- `analyze_crawl_results_prompt`: Crawl result analysis

## 📊 Tool Categories

### By Complexity
- **Simple**: `crawl_url`, `process_file`, `get_youtube_video_info`
- **Moderate**: `deep_crawl_site`, `search_google`, `extract_youtube_transcript`
- **Advanced**: `extract_structured_data`, `search_and_crawl`, `batch_crawl`

### By Use Case
- **Content Discovery**: `search_google`, `search_and_crawl`
- **Data Extraction**: `crawl_url`, `extract_structured_data`
- **Media Processing**: `extract_youtube_transcript`, `process_file`
- **Site Analysis**: `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`

## 🔧 Integration Examples

For detailed configuration examples, see [Configuration Examples](CONFIGURATION_EXAMPLES.md).

For HTTP API integration, see [HTTP Integration Guide](HTTP_INTEGRATION.md).

For advanced usage patterns, see [Advanced Usage Guide](ADVANCED_USAGE.md).