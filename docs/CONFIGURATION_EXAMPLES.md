# Configuration Examples

MCP client configurations for different installation methods and platforms, plus the environment variables the server reads.

## UVX (recommended)

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "crawl-mcp": {
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

Claude Code, from the command line:

```bash
claude mcp add crawl-mcp -- uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

Or in a project's `.mcp.json`, where Claude Code expects a `type` field:

```json
{
  "mcpServers": {
    "crawl-mcp": {
      "type": "stdio",
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

## Local checkout

Point the client at the Python interpreter inside the checkout's virtual environment. With the package installed in that environment (`pip install -e .`), no `PYTHONPATH` or working directory is needed.

Linux and macOS:

```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"]
    }
  }
}
```

Windows:

```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "C:\\path\\to\\crawl-mcp\\venv\\Scripts\\python.exe",
      "args": ["-m", "crawl4ai_mcp.server"]
    }
  }
}
```

If you run the code without installing it, set `PYTHONPATH` to the repository root (not to `site-packages`):

```json
{
  "mcpServers": {
    "crawl4ai-source": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/crawl-mcp"
      }
    }
  }
}
```

### WSL (Windows Subsystem for Linux)

Claude Desktop on Windows can start the server inside WSL:

```json
{
  "mcpServers": {
    "crawl4ai-wsl": {
      "command": "wsl",
      "args": [
        "/home/user/crawl-mcp/venv/bin/python",
        "-m",
        "crawl4ai_mcp.server"
      ]
    }
  }
}
```

Variables in an `env` block are set for `wsl.exe` on the Windows side. They reach the Linux process only if they are listed in `WSLENV`. Put API keys in the `.env` file at the repository root instead; the server loads it at startup.

## HTTP transport

Transport, host, and port are command-line flags. Environment variables do not set them.

```bash
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000
```

`--transport http` is an alias for `streamable-http`. The endpoint is `http://127.0.0.1:8000/mcp`. Start the server first, then point the client at the URL:

```json
{
  "mcpServers": {
    "crawl4ai-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

To use another port, change both the `--port` flag and the URL. `--transport sse` starts the legacy SSE transport at `/sse` for older clients. One HTTP server can be shared by several clients, and a client can list a STDIO entry and an HTTP entry side by side. The [HTTP Integration Guide](HTTP_INTEGRATION.md) covers curl and Python clients.

## Environment variables

The server loads a `.env` file from the repository root, or from the current directory, without overriding variables that are already set. `.env.example` is a template. You can also set the variables in the `env` block of the MCP client configuration.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | none | OpenAI API key |
| `ANTHROPIC_API_KEY` | none | Anthropic API key |
| `AZURE_OPENAI_API_KEY` | none | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | none | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | API version used by the summarization client for Azure OpenAI |
| `MCP_LLM_CONFIG` | none | Full LLM configuration as JSON (see below). Replaces the built-in provider list |
| `SEARCH_ENGINE` | `auto` | Default search engine: `auto`, `google_custom`, `brave`, or `exa`. The `engine` request key overrides it for one call |
| `SEARCH_ENGINE_ORDER` | `google_custom,brave,exa` | Order in which `auto` tries the engines that have API keys. Unknown names are ignored |
| `GOOGLE_SEARCH_API_KEY` | none | Google Custom Search API key. The API is closed to new customers and ends on 2027-01-01 |
| `GOOGLE_SEARCH_ENGINE_ID` | none | Custom Search engine ID |
| `BRAVE_SEARCH_API_KEY` | none | Brave Search API key |
| `EXA_API_KEY` | none | Exa API key |
| `CUSTOM_SEARCH_API_RPM` | `100` | Requests per minute for the Custom Search API, enforced by the server. Brave and Exa enforce their own limits and answer HTTP 429, which makes `auto` move to the next engine |
| `GOOGLE_SEARCH_MODE` | none | Read only when `SEARCH_ENGINE` is unset: `custom_search_only` means `google_custom`. `hybrid` and `googlesearch_only` mean `auto` (googlesearch-python was removed) |
| `CRAWL4AI_BROWSER_TYPE` | unset | `chromium`, `firefox`, or `webkit` forces one browser. When unset, the crawler tries WebKit and then Chromium. The undetected browser always uses Chromium |
| `CRAWL4AI_CACHE_TTL_SECONDS` | `3600` | Seconds a fetched page counts as fresh. After that, the next crawl of the URL bypasses the cache and fetches it again. `0` disables expiry, so cached pages are always reused |
| `CRAWL4AI_MCP_LOG_LEVEL` | unset (no logging) | `DEBUG`, `INFO`, `WARNING`, or `ERROR` writes log records to stderr. The server reads it before it loads `.env`, so set it in the process environment or the client's `env` block |

`FASTMCP_LOG_LEVEL` has no effect on this server; use `CRAWL4AI_MCP_LOG_LEVEL` instead. Diagnostics go to stderr.

### LLM settings

Without any configuration, the default provider is `openai` with model `gpt-4.1`. Providers whose API key variable is set become available. If the requested provider has no key, the server tries `openai`, `aoai`, `anthropic`, and `ollama` in that order. Ollama needs no key and uses `http://localhost:11434`.

To change the default provider or model, set `MCP_LLM_CONFIG` to a JSON string:

```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/walksoda/crawl-mcp", "crawl-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MCP_LLM_CONFIG": "{\"default_provider\": \"anthropic\", \"default_model\": \"claude-3-5-sonnet-20241022\", \"providers\": {\"anthropic\": {\"api_key_env\": \"ANTHROPIC_API_KEY\", \"models\": [\"claude-3-5-sonnet-20241022\"]}}}"
      }
    }
  }
}
```

Each provider entry accepts `api_key` or `api_key_env`, `base_url` or `base_url_env`, `api_version`, `models`, and `extra_headers`. Provider names are `openai`, `anthropic`, `ollama`, and `aoai` (Azure OpenAI).

If `MCP_LLM_CONFIG` is not set, the server also looks for a `claude_desktop_config.json` in the repository root or the current directory and reads `mcpServers.crawl4ai.llm_config` from it. `configs/claude_desktop_config.json` shows that format; its paths are examples and must be changed before use.

A configuration that only sets API keys:

```json
{
  "mcpServers": {
    "crawl4ai-llm": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here",
        "ANTHROPIC_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Configuration file locations

Claude Desktop reads its configuration from:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

Create the directory if it does not exist, then add the server entry to the file. On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "$env:APPDATA\Claude"
notepad "$env:APPDATA\Claude\claude_desktop_config.json"
```

On macOS:

```bash
mkdir -p ~/Library/Application\ Support/Claude
open -e ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Restart Claude Desktop after editing the file.

## Tool parameter examples

These are `arguments` objects for single tool calls. The [API Reference](API_REFERENCE.md) lists all parameters.

`crawl_url` for a basic page:

```json
{
  "url": "https://example.com",
  "wait_for_js": true,
  "timeout": 30,
  "generate_markdown": true
}
```

`crawl_url` for a JavaScript-heavy page:

```json
{
  "url": "https://spa.example.com",
  "wait_for_js": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded",
  "generate_markdown": true
}
```

`deep_crawl_site` (`max_depth` 1-2, `max_pages` 1-10; other values return an error):

```json
{
  "url": "https://example.com",
  "max_depth": 2,
  "max_pages": 10,
  "crawl_strategy": "bfs"
}
```

`intelligent_extract` with BM25 filtering:

```json
{
  "url": "https://example.com/article",
  "extraction_goal": "Summarize the main arguments",
  "content_filter": "bm25",
  "filter_query": "important content keywords",
  "chunk_content": true
}
```

`process_file` with summarization turned on:

```json
{
  "url": "https://example.com/report.pdf",
  "auto_summarize": true,
  "summary_length": "medium"
}
```

## Checking a configuration

```bash
# Check the JSON syntax
python -m json.tool claude_desktop_config.json

# Check that the server starts
python -m crawl4ai_mcp.server --help

# Check the UVX installation
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --help
```

`--help` lists `--transport` (`stdio`, `streamable-http`, `http`, `sse`), `--host`, and `--port`. An invalid argument prints an error to stderr and exits with status 2.

The server has no health-check endpoint. To check a running HTTP server, call a cheap tool such as `get_supported_file_formats`; the [HTTP Integration Guide](HTTP_INTEGRATION.md) shows how.

## Tips

1. UVX is the simplest setup for most users.
2. Use absolute paths in client configurations.
3. Keep API keys in the `env` block or in `.env`, not in shared files.
4. Validate the JSON before restarting the client.
5. Over STDIO, anything a wrapper script prints to stdout breaks the MCP stream. Send such output to stderr.

## Related documentation

- [Installation Guide](INSTALLATION.md)
- [HTTP Integration Guide](HTTP_INTEGRATION.md)
- [API Reference](API_REFERENCE.md)
- [Usage Patterns](ADVANCED_USAGE.md)
- [Development Guide](DEVELOPMENT.md)
