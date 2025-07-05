# Crawl4AI Extension for Claude Desktop

MCP wrapper for crawl4ai library. This extension provides advanced web crawling and content extraction capabilities for Claude Desktop via MCP (Model Context Protocol).

**Built by**: [walksoda](https://github.com/walksoda/crawl-mcp)  
**Powered by**: [unclecode's crawl4ai](https://github.com/unclecode/crawl4ai)

## Features

- **Web Crawling** provides advanced crawling with JavaScript support and depth control
- **Content Extraction** offers AI-powered intelligent content extraction and filtering
- **YouTube Transcripts** can extract transcripts from YouTube videos with no API key required
- **Google Search** enables search with genre filtering and metadata extraction
- **File Processing** converts PDFs, Office documents, and archives to markdown
- **Batch Operations** process multiple URLs, videos, or search queries concurrently
- **Entity Extraction** finds emails, phones, URLs, dates, and other patterns
- **Structured Data** extraction uses CSS selectors, XPath, or LLM schemas

## Installation

1. Download the `.dxt` package
2. Install via Claude Desktop's extension manager
3. The extension will automatically install Python dependencies
4. Optionally configure API keys for enhanced LLM features

## Configuration

### API Keys (Optional)
Configure these in Claude Desktop's extension settings. The following options are available:

- **OpenAI API Key** enables advanced LLM-based content analysis
- **Anthropic API Key** provides Claude-powered content processing
- **Google API Key** enhances search functionality (basic search works without key)

### Server Settings
- **Log Level** controls verbosity (DEBUG, INFO, WARNING, ERROR)

## Usage

Once installed, you can use natural language to request various operations. Examples include:

- "Extract content from this website: https://example.com"
- "Get the transcript from this YouTube video"
- "Search for recent AI research papers and analyze the top results"
- "Convert this PDF to markdown and summarize it"
- "Find all email addresses on this company's contact page"

## Core Tools

- `crawl_url` - Extract content from web pages
- `intelligent_extract` - AI-powered content analysis
- `extract_youtube_transcript` - YouTube video transcription
- `search_google` - Google search with filtering
- `process_file` - Document conversion and processing
- `extract_entities` - Pattern-based data extraction
- `batch_crawl` - Process multiple URLs simultaneously

## Platform Support

- Supported on Windows 10/11 (x64)
- Supported on macOS (Intel & Apple Silicon)
- Supported on Linux (Ubuntu, Debian, CentOS, x64)

## System Requirements

- Python 3.8.0 or higher
- 512MB RAM minimum
- 1GB disk space
- Internet connection for crawling

## Version History

### v1.0.7
- Fixed FastMCP compatibility issue (@prompt() decorator syntax)
- Complete FastMCP framework compatibility (@tool() and @prompt() decorators)
- Final stable production release
- All server startup issues resolved

### v1.0.6
- Fixed FastMCP compatibility issue (@tool() decorator syntax)
- Resolved server startup errors after dependency installation
- Full compatibility with latest FastMCP framework
- Stable production-ready release

### v1.0.5
- Enhanced automatic dependency installation with retry logic
- Improved error handling and user feedback
- Robust dependency resolution from requirements.txt
- Better logging for troubleshooting installation issues

### v1.0.4
- Built according to official DXT specification
- Fixed all file path resolution issues
- Proper ${__dirname} variable usage
- Enhanced user configuration with sensitive field handling
- Improved cross-platform compatibility
- Added automatic dependency installation

### v1.0.3
- Simplified package structure for better compatibility
- Fixed file path resolution issues on all platforms
- Improved DXT deployment reliability

### v1.0.2
- Fixed Python module import issues (ModuleNotFoundError)
- Added support for both relative and absolute imports
- Improved direct execution compatibility

### v1.0.1
- Fixed file path issues on Windows
- Improved cross-platform compatibility
- Added automatic dependency installation

### v1.0.0
- Initial release

## Troubleshooting

### Common Issues

**Dependency installation issues** may require the following steps:
- On Windows: Install Visual C++ Build Tools if needed
- On Linux: Install system packages: `sudo apt-get install libnss3 libnspr4 libasound2`
- On macOS: Should work out of the box

**Playwright browser issues** can be resolved as follows:
- The extension will attempt to install Chromium automatically
- If this fails, basic functionality will still work
- You can manually install with: `playwright install chromium`

## Support

For issues or questions, please visit: https://github.com/walksoda/crawl-mcp

## License

MIT License