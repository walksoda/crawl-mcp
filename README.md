# Crawl-MCP: Unofficial MCP Server for crawl4ai

> **⚠️ Important**: This is an **unofficial** MCP server implementation for the excellent [crawl4ai](https://github.com/unclecode/crawl4ai) library.  
> **Not affiliated** with the original crawl4ai project.

A comprehensive Model Context Protocol (MCP) server that wraps the powerful crawl4ai library with advanced AI capabilities. Extract and analyze content from **any source**: web pages, PDFs, Office documents, YouTube videos, and more. Features intelligent summarization to dramatically reduce token usage while preserving key information.

## 🌟 Key Features

- **🔍 Google Search Integration** - 7 optimized search genres with Google official operators
- **🔍 Advanced Web Crawling**: JavaScript support, deep site mapping, entity extraction
- **🌐 Universal Content Extraction**: Web pages, PDFs, Word docs, Excel, PowerPoint, ZIP archives
- **🤖 AI-Powered Summarization**: Smart token reduction (up to 88.5%) while preserving essential information
- **🎬 YouTube Integration**: Extract video transcripts and summaries without API keys  
- **⚡ Production Ready**: 19 specialized tools with comprehensive error handling

## 🚀 Quick Start

### Prerequisites (Required First)

- Python 3.11 以上（FastMCP が Python 3.11+ を要求）

**Install system dependencies for Playwright:**

**Ubuntu 24.04 LTS (Manual Required):**
```bash
# Manual setup required due to t64 library transition
sudo apt update && sudo apt install -y \
  libnss3 libatk-bridge2.0-0 libxss1 libasound2t64 \
  libgbm1 libgtk-3-0t64 libxshmfence-dev libxrandr2 \
  libxcomposite1 libxcursor1 libxdamage1 libxi6 \
  fonts-noto-color-emoji fonts-unifont python3-venv python3-pip

python3 -m venv venv && source venv/bin/activate
pip install playwright==1.55.0 && playwright install chromium
sudo playwright install-deps
```

**Other Linux/macOS:**
```bash
sudo bash scripts/prepare_for_uvx_playwright.sh
```

**Windows (as Administrator):**
```powershell
scripts/prepare_for_uvx_playwright.ps1
```

### Installation

**UVX (Recommended - Easiest):**
```bash
# After system preparation above - that's it!
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

**Docker (Production-Ready):**
```bash
# Clone the repository
git clone https://github.com/walksoda/crawl-mcp
cd crawl-mcp

# Build and run with Docker Compose (STDIO mode)
docker-compose up --build

# Or build and run HTTP mode on port 8000
docker-compose --profile http up --build crawl4ai-mcp-http

# Or build manually
docker build -t crawl4ai-mcp .
docker run -it crawl4ai-mcp
```

**Docker Features:**
- 🔧 **Multi-Browser Support**: Chromium, Firefox, Webkit headless browsers
- 🐧 **Google Chrome**: Additional Chrome Stable for compatibility
- ⚡ **Optimized Performance**: Pre-configured browser flags for Docker
- 🔒 **Security**: Non-root user execution
- 📦 **Complete Dependencies**: All required libraries included

### Claude Desktop Setup

**UVX Installation:**
Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "crawl-mcp": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "en"
      }
    }
  }
}
```

**Docker HTTP Mode:**
```json
{
  "mcpServers": {
    "crawl-mcp": {
      "transport": "http",
      "baseUrl": "http://localhost:8000"
    }
  }
}
```

**For Japanese interface:**
```json
"env": {
  "CRAWL4AI_LANG": "ja"
}
```

## 📖 Documentation

| Topic | Description |
|-------|-------------|
| **[Installation Guide](docs/INSTALLATION.md)** | Complete installation instructions for all platforms |
| **[API Reference](docs/API_REFERENCE.md)** | Full tool documentation and usage examples |
| **[Configuration Examples](docs/CONFIGURATION_EXAMPLES.md)** | Platform-specific setup configurations |
| **[HTTP Integration](docs/HTTP_INTEGRATION.md)** | HTTP API access and integration methods |
| **[Advanced Usage](docs/ADVANCED_USAGE.md)** | Power user techniques and workflows |
| **[Development Guide](docs/DEVELOPMENT.md)** | Contributing and development setup |

### Language-Specific Documentation

- **English**: [docs/](docs/) directory
- **日本語**: [docs/ja/](docs/ja/) directory

## 🛠️ Tool Overview

### Web Crawling (3)
- `crawl_url` - Extract web page content with JavaScript support
- `deep_crawl_site` - Crawl multiple pages from a site with configurable depth
- `crawl_url_with_fallback` - Crawl with fallback strategies for anti-bot sites

### Data Extraction (3)
- `intelligent_extract` - Extract specific data from web pages using LLM
- `extract_entities` - Extract entities (emails, phones, etc.) from web pages
- `extract_structured_data` - Extract structured data using CSS selectors or LLM

### YouTube (4)
- `extract_youtube_transcript` - Extract YouTube transcripts with timestamps
- `batch_extract_youtube_transcripts` - Extract transcripts from multiple YouTube videos (max 3)
- `get_youtube_video_info` - Get YouTube video metadata and transcript availability
- `extract_youtube_comments` - Extract YouTube video comments with pagination

### Search (4)
- `search_google` - Search Google with genre filtering
- `batch_search_google` - Perform multiple Google searches (max 3)
- `search_and_crawl` - Search Google and crawl top results
- `get_search_genres` - Get available search genres

### File Processing (3)
- `process_file` - Convert PDF, Word, Excel, PowerPoint, ZIP to markdown
- `get_supported_file_formats` - Get supported file formats and capabilities
- `enhanced_process_large_content` - Process large content with chunking and BM25 filtering

### Batch Operations (2)
- `batch_crawl` - Crawl multiple URLs with fallback (max 3 URLs)
- `multi_url_crawl` - Multi-URL crawl with pattern-based config (max 5 URL patterns)

## 🎯 Common Use Cases

**Content Research:**
```bash
search_and_crawl → extract_structured_data → analysis
```

**Documentation Mining:**
```bash
deep_crawl_site → batch processing → extraction
```

**Media Analysis:**
```bash
extract_youtube_transcript → summarization workflow
```

**Site Mapping:**
```bash
batch_crawl → multi_url_crawl → comprehensive data
```

## 🚨 Quick Troubleshooting

**Installation Issues:**
1. Re-run setup scripts with proper privileges
2. Try development installation method
3. Check browser dependencies are installed

**Performance Issues:**
- Use `wait_for_js: true` for JavaScript-heavy sites
- Increase timeout for slow-loading pages
- Use `extract_structured_data` for targeted extraction

**Configuration Issues:**
- Check JSON syntax in `claude_desktop_config.json`
- Verify file paths are absolute
- Restart Claude Desktop after configuration changes

## 🏗️ Project Structure

- **Original Library**: [crawl4ai](https://github.com/unclecode/crawl4ai) by unclecode
- **MCP Wrapper**: This repository (walksoda)
- **Implementation**: Unofficial third-party integration

## 📄 License

This project is an unofficial wrapper around the crawl4ai library. Please refer to the original [crawl4ai license](https://github.com/unclecode/crawl4ai) for the underlying functionality.

## 🤝 Contributing

See our [Development Guide](docs/DEVELOPMENT.md) for contribution guidelines and development setup instructions.

## 🔗 Related Projects

- [crawl4ai](https://github.com/unclecode/crawl4ai) - The underlying web crawling library
- [Model Context Protocol](https://modelcontextprotocol.io/) - The standard this server implements
- [Claude Desktop](https://docs.anthropic.com/claude/docs/claude-desktop) - Primary client for MCP servers
