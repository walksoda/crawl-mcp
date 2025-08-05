# Crawl-MCP: Unofficial MCP Server for crawl4ai

> **⚠️ Important**: This is an **unofficial** MCP server implementation for the excellent [crawl4ai](https://github.com/unclecode/crawl4ai) library.  
> **Not affiliated** with the original crawl4ai project.

A comprehensive Model Context Protocol (MCP) server that wraps the powerful crawl4ai library. This server provides advanced web crawling, content extraction, and AI-powered analysis capabilities through the standardized MCP interface.

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

### Claude Desktop Setup

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

**For Japanese interface:**
```json
"env": {
  "CRAWL4AI_LANG": "ja"
}
```

## ✨ Key Features

- **🌐 Advanced Web Crawling** - JavaScript-heavy sites, SPAs, dynamic content
- **🧠 AI-Powered Extraction** - LLM-based content analysis and summarization
- **📄 File Processing** - PDF, Office documents, ZIP archives (Microsoft MarkItDown)
- **📺 YouTube Transcripts** - Multi-language, timestamped extraction
- **🔍 Google Search Integration** - 31 search genres with metadata extraction
- **🔄 Batch Processing** - Multiple URLs, search queries, video transcripts
- **🌍 Multi-language** - English and Japanese interface support

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