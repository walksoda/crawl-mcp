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
- **Entity Extraction** with 9 built-in patterns (emails, phones, URLs, dates, etc.)
- **Intelligent Content Filtering** (BM25, pruning, LLM-based)
- **Content Chunking** for large document processing
- **Screenshot Capture** and media extraction

### Advanced Features
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

For Windows with WSL, copy the configuration:
```bash
cp claude_desktop_config_windows.json %APPDATA%\Claude\claude_desktop_config.json
```

Then restart Claude Desktop to enable the crawl4ai tools.

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

## 📚 Resources

- `uri://crawl4ai/config`: Default crawler configuration options
- `uri://crawl4ai/examples`: Usage examples and sample requests

## 🎯 Prompts

- `crawl_website_prompt`: Guided website crawling workflows
- `analyze_crawl_results_prompt`: Crawl result analysis
- `batch_crawl_setup_prompt`: Batch crawling setup

## 🔧 Configuration Examples

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
- `asyncio` - Asynchronous programming support
- `typing-extensions` - Extended type hints

## 📄 License

MIT License

## 🤝 Contributing

This project implements the Model Context Protocol specification and is compatible with any MCP-compliant client. Built with the FastMCP framework for easy extension and modification.

For detailed feature documentation in Japanese, see [`README_ja.md`](README_ja.md).