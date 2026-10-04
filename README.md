# Crawl-MCP: Unofficial MCP Server for crawl4ai

> This is an unofficial MCP server for the [crawl4ai](https://github.com/unclecode/crawl4ai) library. It is not affiliated with the crawl4ai project.

A Model Context Protocol (MCP) server that wraps crawl4ai. It extracts content from web pages, PDFs, Office documents, YouTube videos, and web search results (Google Custom Search, Brave, or Exa). To keep token usage low, it can summarize content with an LLM (`auto_summarize`, opt-in) and write full results to disk (`output_path`).

## Key Features

- Web search through Google Custom Search, Brave, or Exa, with automatic fallback between engines and 6 genres (file type and language filters)
- Web crawling with JavaScript rendering, recursive site crawls, and entity extraction
- Content extraction from web pages, PDFs, Word, Excel, PowerPoint, and ZIP archives
- LLM summarization when `auto_summarize=true`. Without it, large responses are truncated at about 25000 tokens, and `output_path` saves the full content
- Disk persistence: full results go to a file and the response carries only metadata, so an agent reads the file when it needs the content
- YouTube transcripts and summaries without an API key
- 21 tools that return structured error dicts on failure

## Quick Start

### Prerequisites (Required First)

- Python 3.11 or later (FastMCP requires Python 3.11+)

Install the system dependencies for Playwright.

Ubuntu 24.04 LTS needs a manual setup:

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

Other Linux distributions (installs Chromium and its system libraries):

```bash
uvx --from playwright==1.55.0 playwright install --with-deps chromium
```

macOS / Windows:

```bash
uvx --from playwright==1.55.0 playwright install chromium
```

If Playwright is already installed in a Python environment, `python -m playwright install --with-deps chromium` (Linux) or `playwright install chromium` (macOS/Windows) works as well.

### Installation

UVX (recommended):

```bash
# After system preparation above - that's it!
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

Docker:

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

The Docker image includes headless Chromium, Firefox, and WebKit, plus Google Chrome Stable. Browser flags are preset for running in a container, the server runs as a non-root user, and all required system libraries are installed.

### Claude Desktop Setup

For a UVX installation, add this to your `claude_desktop_config.json`:

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
      ]
    }
  }
}
```

For Docker in HTTP mode:

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

## Documentation

| Topic | Description |
|-------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Installation on each platform |
| [API Reference](docs/API_REFERENCE.md) | Parameters and behavior of every tool |
| [Configuration Examples](docs/CONFIGURATION_EXAMPLES.md) | Client and platform configurations |
| [HTTP Integration](docs/HTTP_INTEGRATION.md) | Running the server over HTTP and calling it |
| [Usage Patterns](docs/ADVANCED_USAGE.md) | Workflows that combine tools |
| [Development Guide](docs/DEVELOPMENT.md) | Development setup and contributing |

### Language-Specific Documentation

- English: [docs/](docs/)
- 日本語: [docs/ja/](docs/ja/)

## Tool Overview

### Web Crawling (3)
- `crawl_url` - Extract web page content with JavaScript support (remote `.html`/`.htm` URLs are rendered in the browser; document URLs are converted as files)
- `deep_crawl_site` - Crawl multiple pages from a site with configurable depth (`max_depth` 1-2, `max_pages` 1-10)
- `crawl_url_with_fallback` - Crawl with fallback strategies for anti-bot sites

### Data Extraction (3)
- `intelligent_extract` - Extract specific data from web pages using LLM
- `extract_entities` - Extract entities (emails, phones, etc.) from web pages
- `extract_structured_data` - Extract structured data using CSS selectors or LLM

### YouTube (4)
- `extract_youtube_transcript` - Extract YouTube transcripts with timestamps and timezone-converted publish time (`timezone`)
- `batch_extract_youtube_transcripts` - Extract transcripts from multiple YouTube videos (max 3)
- `get_youtube_video_info` - Get YouTube video metadata (including `published_at` in the requested `timezone`) and transcript availability
- `extract_youtube_comments` - Extract YouTube video comments with pagination

### Search (6)
- `search_web` - Web search with optional genre filtering
- `batch_search_web` - Run multiple web searches (max 3)
- `search_and_crawl` - Web search and crawl of the top results
- `get_search_genres` - Get available search genres and the engine configuration
- `search_google`, `batch_search_google` - Deprecated aliases of `search_web` and `batch_search_web`

The search tools need an API key for at least one engine: `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` (Google Custom Search, closed to new customers and ending on 2027-01-01), `BRAVE_SEARCH_API_KEY`, or `EXA_API_KEY`. `SEARCH_ENGINE` sets the default engine (`auto` tries the configured engines in `SEARCH_ENGINE_ORDER`, default `google_custom,brave,exa`), and the `engine` request key overrides it for one call. See the [API Reference](docs/API_REFERENCE.md#search-tools).

`search_web`, `batch_search_web` and `search_and_crawl` accept the same `search_genre` values: `pdf`, `documents`, `presentations`, `spreadsheets`, `japanese`, `english`. Matching is case-insensitive. Any other value returns an `invalid_search_genre` error before the search runs. Exa cannot filter by file type, so `auto` skips Exa for the file type genres.

`result_type` adds Brave result sections to a search: `discussions` (forum posts with forum name, answer count and score), `faq` (question and answer excerpts), `videos` and `news`. Several can be requested at once, for example `["web", "discussions", "faq"]`; the extra sections come back in `extra_results`. They need the `brave` engine. `discussions`, `faq` and `videos` share one Brave request and appear for some queries only; `news` is a separate request to Brave News Search.

### File Processing (3)
- `process_file` - Convert PDF, Word, Excel, PowerPoint, ZIP to markdown
- `get_supported_file_formats` - Get supported file formats and capabilities
- `enhanced_process_large_content` - Process large content with chunking and BM25 filtering

### Batch Operations (2)
- `batch_crawl` - Crawl multiple URLs sequentially with fallback (max 3 URLs)
- `multi_url_crawl` - Multi-URL crawl with pattern-based config (max 5 URL patterns, processed sequentially)

## Persist Large Results to Disk

All information-gathering tools accept an optional `output_path` parameter. The tool writes the full content to disk and returns only metadata. An LLM can then fetch large pages, long YouTube transcripts, or whole batches without filling its context, and read the saved file when it needs the content.

How it works:

- Single-file tools (e.g. `crawl_url`, `extract_youtube_transcript`) write one `.md` (or `.json` for JSON-kind tools). Pass an absolute file path; the extension is added if omitted. An existing regular file at that path is rejected unless `overwrite=true`.
- Batch tools (`batch_crawl`, `multi_url_crawl`, `deep_crawl_site`, `search_and_crawl`, `batch_extract_youtube_transcripts`) expect an absolute directory path and write one `.md` per URL plus `index.json`. Any non-existent path is treated as a directory and created, including names containing dots such as `/tmp/run.v1`. If the path already exists as a regular file, the call is rejected. `batch_crawl` / `multi_url_crawl` keep their `list` return shape and embed an `output_file` key on each success item.
- Request-dict tools (`search_google`, `batch_search_google`, `search_and_crawl`, `batch_extract_youtube_transcripts`) read the persistence keys directly from their request dict.
- Common parameters: `output_path` (absolute; default `None`, `""` also skips persistence), `include_content_in_response` (default `false`; when `true`, the content is also included in the response, still subject to any `content_limit`/`content_offset`/`max_content_per_page` slicing), `overwrite` (default `false`).
- Writes are atomic per file (temp file + `os.replace`); parent directories are auto-created; the full unsliced payload is written before any slicing or tool-internal truncation, so the file is complete even when the response is sliced.
- Batch dict tools (`deep_crawl_site`, `search_and_crawl`, `batch_extract_youtube_transcripts`) skip per-item persistence for items that report `success=false`; these still appear in `index.json` with `file: null`, so callers can see every attempted item. (`batch_crawl` / `multi_url_crawl` also skip failed items and record them in `index.json` with `output_file: null`.)

Single Markdown file:

```json
{
  "tool": "crawl_url",
  "arguments": {
    "url": "https://example.com/long-article",
    "output_path": "/tmp/crawl_out/article.md"
  }
}
```

JSON structured extraction (the extension is added automatically):

```json
{
  "tool": "extract_structured_data",
  "arguments": {
    "url": "https://example.com/products",
    "extraction_type": "css",
    "css_selectors": {"price": ".price", "name": "h1"},
    "output_path": "/tmp/crawl_out/products"
  }
}
```

Batch directory mode:

```json
{
  "tool": "batch_crawl",
  "arguments": {
    "urls": ["https://a.example", "https://b.example"],
    "output_path": "/tmp/crawl_out/batch_run1"
  }
}
```

Each persisted markdown file begins with a YAML frontmatter block containing `url`, `title`, `fetched_at`, and `source_tool`.

## Common Use Cases

Content research:

```bash
search_and_crawl → extract_structured_data → analysis
```

Collecting documentation:

```bash
deep_crawl_site → batch processing → extraction
```

Video analysis:

```bash
extract_youtube_transcript → summarization workflow
```

Site mapping:

```bash
batch_crawl → multi_url_crawl → aggregated data
```

## Quick Troubleshooting

Installation problems:

1. Re-run the Playwright install commands above with proper privileges
2. Try the development installation method
3. Check that the browser dependencies are installed

Slow or incomplete pages:

- Use `wait_for_js: true` for JavaScript-heavy sites
- Increase timeout for slow-loading pages
- Use `extract_structured_data` for targeted extraction

Configuration problems:

- Check JSON syntax in `claude_desktop_config.json`
- Verify file paths are absolute
- Restart Claude Desktop after configuration changes

## Project Structure

The underlying library is [crawl4ai](https://github.com/unclecode/crawl4ai) by unclecode. This repository (walksoda) is an unofficial third-party MCP wrapper around it.

## License

This project is an unofficial wrapper around the crawl4ai library. See the [crawl4ai license](https://github.com/unclecode/crawl4ai) for the underlying functionality.

## Contributing

See the [Development Guide](docs/DEVELOPMENT.md) for development setup and contribution guidelines.

## Related Projects

- [crawl4ai](https://github.com/unclecode/crawl4ai) - The underlying web crawling library
- [Model Context Protocol](https://modelcontextprotocol.io/) - The standard this server implements
- [Claude Desktop](https://docs.anthropic.com/claude/docs/claude-desktop) - Primary client for MCP servers
