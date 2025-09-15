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
- **⚡ Production Ready**: 21 specialized tools with comprehensive error handling

## 🚀 Quick Start

### Prerequisites (Required First)

**Install system dependencies for Playwright:**

**Linux/macOS:**
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

### Web Crawling
- `crawl_url` - Single page crawling with JavaScript support
- `deep_crawl_site` - Multi-page site mapping and exploration
- `crawl_url_with_fallback` - Robust crawling with retry strategies
- `batch_crawl` - Process multiple URLs simultaneously

### AI-Powered Analysis
- `intelligent_extract` - Semantic content extraction with custom instructions
- `auto_summarize` - LLM-based summarization for large content
- `extract_entities` - Pattern-based entity extraction (emails, phones, URLs, etc.)

### Media Processing
- `process_file` - Convert PDFs, Office docs, ZIP archives to markdown
- `extract_youtube_transcript` - Multi-language transcript extraction
- `batch_extract_youtube_transcripts` - Process multiple videos

### Search Integration
- `search_google` - Genre-filtered Google search with metadata
- `search_and_crawl` - Combined search and content extraction
- `batch_search_google` - Multiple search queries with analysis

## 🎯 Common Use Cases

**Content Research:**
```bash
search_and_crawl → intelligent_extract → structured analysis
```

**Documentation Mining:**
```bash
deep_crawl_site → batch processing → comprehensive extraction
```

**Media Analysis:**
```bash
extract_youtube_transcript → auto_summarize → insight generation
```

**Competitive Intelligence:**
```bash
batch_crawl → extract_entities → comparative analysis
```

## 🚨 Quick Troubleshooting

**Installation Issues:**
1. Run system diagnostics: Use `get_system_diagnostics` tool
2. Re-run setup scripts with proper privileges
3. Try development installation method

**Performance Issues:**
- Use `wait_for_js: true` for JavaScript-heavy sites
- Increase timeout for slow-loading pages
- Enable `auto_summarize` for large content

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