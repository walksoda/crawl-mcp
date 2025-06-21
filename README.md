# Crawl4AI MCP Server

A comprehensive Model Context Protocol (MCP) server that wraps the powerful crawl4ai library, providing advanced web crawling, content extraction, and AI-powered analysis capabilities through the standardized MCP interface.

## 🚀 Key Features

### Core Capabilities
- **Advanced Web Crawling** with JavaScript execution support
- **Deep Crawling** with configurable depth and multiple strategies (BFS, DFS, Best-First)
- **AI-Powered Content Extraction** using LLM-based analysis
- **📄 File Processing** with Microsoft MarkItDown integration
  - PDF, Office documents, ZIP archives, and more
  - Automatic file format detection and conversion
  - Batch processing of archive contents
- **📺 YouTube Transcript Extraction** (youtube-transcript-api v1.1.0+)
  - No authentication required - works out of the box
  - Stable and reliable transcript extraction
  - Support for both auto-generated and manual captions
  - Multi-language support with priority settings
  - Timestamped segment information and clean text output
  - Batch processing for multiple videos
- **Entity Extraction** with 9 built-in patterns (emails, phones, URLs, dates, etc.)
- **Intelligent Content Filtering** (BM25, pruning, LLM-based)
- **Content Chunking** for large document processing
- **Screenshot Capture** and media extraction

### Advanced Features
- **🔍 Google Search Integration** with genre-based filtering and metadata extraction
  - 31 search genres (academic, programming, news, etc.)
  - Automatic title and snippet extraction from search results  
  - Safe search enabled by default for security
  - Batch search capabilities with result analysis
- **Multiple Extraction Strategies**: CSS selectors, XPath, regex patterns, LLM-based
- **Browser Automation**: Custom user agents, headers, cookies, authentication
- **Caching System** with multiple modes for performance optimization
- **Custom JavaScript Execution** for dynamic content interaction
- **Structured Data Export** in multiple formats (JSON, Markdown, HTML)

## 📦 Installation

### Quick Setup

**Linux/macOS:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup_windows.bat
```

### Manual Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Playwright browser dependencies (Linux/WSL):**
```bash
sudo apt-get update
sudo apt-get install libnss3 libnspr4 libasound2 libatk-bridge2.0-0 libdrm2 libgtk-3-0 libgbm1
```

## 🖥️ Usage

### Start the MCP Server

**STDIO transport (default):**
```bash
python -m crawl4ai_mcp.server
```

**HTTP transport:**
```bash
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000
```

### Claude Desktop Integration

#### 🎯 Pure StreamableHTTP Usage (Recommended)

1. **Start Server**:
   ```bash
   ./scripts/start_pure_http_server.sh
   ```

2. **Apply Configuration**:
   - Copy `configs/claude_desktop_config_pure_http.json` to Claude Desktop's config directory
   - Or add the following to your existing config:
   ```json
   {
     "mcpServers": {
       "crawl4ai-pure-http": {
         "url": "http://127.0.0.1:8000/mcp"
       }
     }
   }
   ```

3. **Restart Claude Desktop**: Apply settings

4. **Start Using**: crawl4ai tools are now available in chat

#### 🔄 Traditional STDIO Usage

1. Copy the configuration:
   ```bash
   cp configs/claude_desktop_config.json ~/.config/claude-desktop/claude_desktop_config.json
   ```

2. Restart Claude Desktop to enable the crawl4ai tools

#### 📂 Configuration File Locations

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/claude-desktop/claude_desktop_config.json
```

## 🌐 HTTP API Access

This MCP server supports multiple HTTP protocols, allowing you to choose the optimal implementation for your use case.

### 🎯 Pure StreamableHTTP (Recommended)

**Pure JSON HTTP protocol without Server-Sent Events (SSE)**

#### Server Startup
```bash
# Method 1: Using startup script
./scripts/start_pure_http_server.sh

# Method 2: Direct startup
python examples/simple_pure_http_server.py --host 127.0.0.1 --port 8000

# Method 3: Background startup
nohup python examples/simple_pure_http_server.py --port 8000 > server.log 2>&1 &
```

#### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

#### Usage Steps
1. **Start Server**: `./scripts/start_pure_http_server.sh`
2. **Apply Configuration**: Use `configs/claude_desktop_config_pure_http.json`
3. **Restart Claude Desktop**: Apply settings

#### Verification
```bash
# Health check
curl http://127.0.0.1:8000/health

# Complete test
python examples/pure_http_test.py
```

### 🔄 Legacy HTTP (SSE Implementation)

**Traditional FastMCP StreamableHTTP protocol (with SSE)**

#### Server Startup
```bash
# Method 1: Command line
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8001

# Method 2: Environment variables
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8001
python -m crawl4ai_mcp.server
```

#### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### 📊 Protocol Comparison

| Feature | Pure StreamableHTTP | Legacy HTTP (SSE) | STDIO |
|---------|---------------------|-------------------|-------|
| Response Format | Plain JSON | Server-Sent Events | Binary |
| Configuration Complexity | Low (URL only) | Low (URL only) | High (Process management) |
| Debug Ease | High (curl compatible) | Medium (SSE parser needed) | Low |
| Independence | High | High | Low |
| Performance | High | Medium | High |

### 🚀 HTTP Usage Examples

#### Pure StreamableHTTP
```bash
# Initialize
SESSION_ID=$(curl -s -X POST http://127.0.0.1:8000/mcp/initialize \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' \
  -D- | grep -i mcp-session-id | cut -d' ' -f2 | tr -d '\r')

# Execute tool
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":"crawl","method":"tools/call","params":{"name":"crawl_url","arguments":{"url":"https://example.com"}}}'
```

#### Legacy HTTP
```bash
curl -X POST "http://127.0.0.1:8001/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

### 📚 Detailed Documentation

- **Pure StreamableHTTP**: [PURE_STREAMABLE_HTTP.md](PURE_STREAMABLE_HTTP.md)
- **HTTP Server Usage**: [HTTP_SERVER_USAGE.md](HTTP_SERVER_USAGE.md)
- **Legacy HTTP API**: [HTTP_API_GUIDE.md](HTTP_API_GUIDE.md)

### Starting the HTTP Server

**Method 1: Command Line**
```bash
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000
```

**Method 2: Environment Variables**
```bash
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8000
python -m crawl4ai_mcp.server
```

**Method 3: Docker (if available)**
```bash
docker run -p 8000:8000 crawl4ai-mcp --transport http --port 8000
```

### Basic Endpoint Information

Once running, the HTTP API provides:
- **Base URL**: `http://127.0.0.1:8000`
- **OpenAPI Documentation**: `http://127.0.0.1:8000/docs`
- **Tool Endpoints**: `http://127.0.0.1:8000/tools/{tool_name}`
- **Resource Endpoints**: `http://127.0.0.1:8000/resources/{resource_uri}`

All MCP tools (crawl_url, intelligent_extract, process_file, etc.) are accessible via HTTP POST requests with JSON payloads matching the tool parameters.

### Quick HTTP Example

```bash
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

For detailed HTTP API documentation, examples, and integration guides, see the [HTTP API Guide](docs/http_api_guide.md).

## 🛠️ MCP Tools

### `crawl_url`
Advanced web crawling with deep crawling support and intelligent filtering.

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

### `deep_crawl_site`
Dedicated tool for comprehensive site mapping and recursive crawling.

**Parameters:**
- `url`: Starting URL
- `max_depth`: Maximum crawling depth (recommended: 1-3)
- `max_pages`: Maximum number of pages to crawl
- `crawl_strategy`: Crawling strategy ('bfs', 'dfs', 'best_first')
- `url_pattern`: URL filter pattern (e.g., '*docs*', '*blog*')
- `score_threshold`: Minimum relevance score (0.0-1.0)

### `intelligent_extract`
AI-powered content extraction with advanced filtering and analysis.

**Parameters:**
- `url`: Target URL
- `extraction_goal`: Description of extraction target
- `content_filter`: Filter type for content quality
- `use_llm`: Enable LLM-based intelligent extraction
- `llm_provider`: LLM provider (openai, claude, etc.)
- `custom_instructions`: Detailed extraction instructions

### `extract_entities`
High-speed entity extraction using regex patterns.

**Built-in Entity Types:**
- `emails`: Email addresses
- `phones`: Phone numbers
- `urls`: URLs and links
- `dates`: Date formats
- `ips`: IP addresses
- `social_media`: Social media handles (@username, #hashtag)
- `prices`: Price information
- `credit_cards`: Credit card numbers
- `coordinates`: Geographic coordinates

### `extract_structured_data`
Traditional structured data extraction using CSS/XPath selectors or LLM schemas.

### `batch_crawl`
Parallel processing of multiple URLs with unified reporting.

### `crawl_url_with_fallback`
Robust crawling with multiple fallback strategies for maximum reliability.

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

### `get_supported_file_formats`
**📋 Format Information**: Get comprehensive list of supported file formats and their capabilities.

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

### `batch_extract_youtube_transcripts`
**📺 Batch YouTube Processing**: Extract transcripts from multiple YouTube videos in parallel.

**✅ Enhanced performance with controlled concurrency for stable batch processing.**

**Parameters:**
- `urls`: List of YouTube video URLs
- `languages`: Preferred languages list
- `translate_to`: Target language for translation (optional)
- `include_timestamps`: Include timestamps in transcript
- `max_concurrent`: Maximum concurrent requests (1-5, default: 3)

### `get_youtube_video_info`
**📋 YouTube Info**: Get available transcript information for a YouTube video without extracting the full transcript.

**Parameters:**
- `video_url`: YouTube video URL

**Returns:**
- Available transcript languages
- Manual/auto-generated distinction
- Translatable language information

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
- 31 available search genres for content filtering
- URL classification and domain analysis
- Safe search enforced by default

### `batch_search_google`
**🔍 Batch Google Search**: Perform multiple Google searches with comprehensive analysis.

**Parameters:**
- `queries`: List of search queries
- `num_results_per_query`: Results per query (1-100, default: 10)
- `max_concurrent`: Maximum concurrent searches (1-5, default: 3)
- `language`: Search language (default: "en")
- `region`: Search region (default: "us")
- `search_genre`: Content genre filter (optional)

**Returns:**
- Individual search results for each query
- Cross-query analysis and statistics
- Domain distribution and result type analysis

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

### `get_search_genres`
**📋 Search Genres**: Get comprehensive list of available search genres and their descriptions.

**Returns:**
- 31 available search genres with descriptions
- Categorized genre lists (Academic, Technical, News, etc.)
- Usage examples for each genre type

## 📚 Resources

- `uri://crawl4ai/config`: Default crawler configuration options
- `uri://crawl4ai/examples`: Usage examples and sample requests

## 🎯 Prompts

- `crawl_website_prompt`: Guided website crawling workflows
- `analyze_crawl_results_prompt`: Crawl result analysis
- `batch_crawl_setup_prompt`: Batch crawling setup

## 🔧 Configuration Examples

### 🔍 Google Search Examples

#### Basic Google Search
```json
{
    "query": "python machine learning tutorial",
    "num_results": 10,
    "language": "en",
    "region": "us"
}
```

#### Genre-Filtered Search
```json
{
    "query": "machine learning research",
    "num_results": 15,
    "search_genre": "academic",
    "language": "en"
}
```

#### Batch Search with Analysis
```json
{
    "queries": [
        "python programming tutorial",
        "web development guide", 
        "data science introduction"
    ],
    "num_results_per_query": 5,
    "max_concurrent": 3,
    "search_genre": "education"
}
```

#### Integrated Search and Crawl
```json
{
    "search_query": "python official documentation",
    "num_search_results": 10,
    "crawl_top_results": 5,
    "extract_media": false,
    "generate_markdown": true,
    "search_genre": "documentation"
}
```

### Basic Deep Crawling
```json
{
    "url": "https://docs.example.com",
    "max_depth": 2,
    "max_pages": 20,
    "crawl_strategy": "bfs"
}
```

### AI-Driven Content Extraction
```json
{
    "url": "https://news.example.com",
    "extraction_goal": "article summary and key points",
    "content_filter": "llm",
    "use_llm": true,
    "custom_instructions": "Extract main article content, summarize key points, and identify important quotes"
}
```

### 📄 File Processing Examples

#### PDF Document Processing
```json
{
    "url": "https://example.com/document.pdf",
    "max_size_mb": 50,
    "include_metadata": true
}
```

#### Office Document Processing
```json
{
    "url": "https://example.com/report.docx",
    "max_size_mb": 25,
    "include_metadata": true
}
```

#### ZIP Archive Processing
```json
{
    "url": "https://example.com/documents.zip",
    "max_size_mb": 100,
    "extract_all_from_zip": true,
    "include_metadata": true
}
```

#### Automatic File Detection
The `crawl_url` tool automatically detects file formats and routes to appropriate processing:
```json
{
    "url": "https://example.com/mixed-content.pdf",
    "generate_markdown": true
}
```

### 📺 YouTube Video Processing Examples

**✅ Stable youtube-transcript-api v1.1.0+ integration - No setup required!**

#### Basic Transcript Extraction
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["ja", "en"],
    "include_timestamps": true,
    "include_metadata": true
}
```

#### Auto-Translation Feature
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["en"],
    "translate_to": "ja",
    "include_timestamps": false
}
```

#### Batch Video Processing
```json
{
    "urls": [
        "https://www.youtube.com/watch?v=VIDEO_ID1",
        "https://www.youtube.com/watch?v=VIDEO_ID2",
        "https://youtu.be/VIDEO_ID3"
    ],
    "languages": ["ja", "en"],
    "max_concurrent": 3
}
```

#### Automatic YouTube Detection
The `crawl_url` tool automatically detects YouTube URLs and extracts transcripts:
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "generate_markdown": true
}
```

#### Video Information Lookup
```json
{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Entity Extraction
```json
{
    "url": "https://company.com/contact",
    "entity_types": ["emails", "phones", "social_media"],
    "include_context": true,
    "deduplicate": true
}
```

### Authenticated Crawling
```json
{
    "url": "https://private.example.com",
    "auth_token": "Bearer your-token",
    "cookies": {"session_id": "abc123"},
    "headers": {"X-API-Key": "your-key"}
}
```

## 🏗️ Project Structure

```
crawl4ai_mcp/
├── __init__.py              # Package initialization
├── server.py                # Main MCP server (1,184+ lines)
├── strategies.py            # Additional extraction strategies
└── suppress_output.py       # Output suppression utilities

config/
├── claude_desktop_config_windows.json  # Claude Desktop config (Windows)
├── claude_desktop_config_script.json   # Script-based config
└── claude_desktop_config.json          # Basic config

docs/
├── README_ja.md             # Japanese documentation
├── setup_instructions_ja.md # Detailed setup guide
└── troubleshooting_ja.md    # Troubleshooting guide

scripts/
├── setup.sh                 # Linux/macOS setup
├── setup_windows.bat        # Windows setup
└── run_server.sh            # Server startup script
```

## 🔍 Troubleshooting

### Common Issues

**ModuleNotFoundError:**
- Ensure virtual environment is activated
- Verify PYTHONPATH is set correctly
- Install dependencies: `pip install -r requirements.txt`

**Playwright Browser Errors:**
- Install system dependencies: `sudo apt-get install libnss3 libnspr4 libasound2`
- For WSL: Ensure X11 forwarding or headless mode

**JSON Parsing Errors:**
- **Resolved**: Output suppression implemented in latest version
- All crawl4ai verbose output is now properly suppressed

For detailed troubleshooting, see [`troubleshooting_ja.md`](troubleshooting_ja.md).

## 🚀 Performance Features

- **Intelligent Caching**: 15-minute self-cleaning cache with multiple modes
- **Async Architecture**: Built on asyncio for high performance
- **Memory Management**: Adaptive concurrency based on system resources
- **Rate Limiting**: Configurable delays and request throttling
- **Parallel Processing**: Concurrent crawling of multiple URLs

## 🛡️ Security Features

- **Output Suppression**: Complete isolation of crawl4ai output from MCP JSON
- **Authentication Support**: Token-based and cookie authentication
- **Secure Headers**: Custom header support for API access
- **Error Isolation**: Comprehensive error handling with helpful suggestions

## 📋 Dependencies

- `crawl4ai>=0.3.0` - Advanced web crawling library
- `fastmcp>=0.1.0` - MCP server framework
- `pydantic>=2.0.0` - Data validation and serialization
- `markitdown>=0.0.1a2` - File processing and conversion (Microsoft)
- `googlesearch-python>=1.3.0` - Google search functionality
- `aiohttp>=3.8.0` - Asynchronous HTTP client for metadata extraction
- `beautifulsoup4>=4.12.0` - HTML parsing for title/snippet extraction
- `youtube-transcript-api>=1.1.0` - Stable YouTube transcript extraction
- `asyncio` - Asynchronous programming support
- `typing-extensions` - Extended type hints

**✅ YouTube Features Status:**
- YouTube transcript extraction is stable and reliable with v1.1.0+
- No authentication or API keys required
- Works out of the box after installation

## 📄 License

MIT License

## 🤝 Contributing

This project implements the Model Context Protocol specification and is compatible with any MCP-compliant client. Built with the FastMCP framework for easy extension and modification.

For detailed feature documentation in Japanese, see [`README_ja.md`](README_ja.md).