# HTTP Integration Guide

This guide covers running the Crawl4AI MCP Server over HTTP and calling its tools from HTTP clients.

## Overview

The server is a [FastMCP](https://gofastmcp.com) application. The transport is chosen with the `--transport` flag of `python -m crawl4ai_mcp.server` (or the `crawl-mcp` entry point):

| `--transport` | Endpoint | Notes |
|---------------|----------|-------|
| `stdio` (default) | stdin/stdout | For MCP clients that launch the server as a subprocess |
| `streamable-http` (alias: `http`) | `http://<host>:<port>/mcp` | Use this for HTTP. Current MCP Streamable HTTP transport |
| `sse` | `http://<host>:<port>/sse` | Legacy MCP SSE transport, for older clients only |

Other flags: `--host` (default `127.0.0.1`) and `--port` (default `8000`). Transport, host, and port are set only by these flags; no environment variables control them. `--help` lists the flags, and an invalid value exits with status 2 and an error on stderr.

Every HTTP request is an MCP JSON-RPC message. The server does not expose REST-style endpoints such as `/tools/{name}`, and it has no `/health`, `/docs`, `/stats`, or `/status` routes.

## Streamable HTTP (Recommended)

### Server Startup

```bash
# Development checkout (inside the project's virtual environment)
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000

# Installed package / UVX
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --transport streamable-http --host 127.0.0.1 --port 8000

# Background startup
nohup python -m crawl4ai_mcp.server --transport streamable-http --port 8000 > server.log 2>&1 &

# Helper script for a development checkout
./scripts/run_http_server.sh --port 8000
```

`--transport http` is accepted as an alias for `streamable-http`.

`scripts/run_http_server.sh` runs the same streamable-http server with the project's `venv` (or `.venv`) and prints the `/mcp` endpoint. Its options are `--host`, `--port`, `--external` (binds `0.0.0.0`), and `--log-level` (sets `CRAWL4AI_MCP_LOG_LEVEL`, so log records go to stderr).

### MCP Client Configuration

For clients that support remote (URL-based) MCP servers:

```json
{
  "mcpServers": {
    "crawl4ai-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Claude Code:

```bash
claude mcp add --transport http crawl4ai http://127.0.0.1:8000/mcp
```

The server must already be running; the client only connects to the URL.

### Calling Tools with curl

A session takes three steps: `initialize`, then the `notifications/initialized` notification, then `tools/call`. Every request must send `Accept: application/json, text/event-stream`. After `initialize`, every request must also send the `mcp-session-id` header returned by the server. Responses come back as Server-Sent Events (`event: message` / `data: {...}`).

```bash
BASE=http://127.0.0.1:8000/mcp

# 1. Initialize (the session ID is returned in the mcp-session-id response header)
SESSION_ID=$(curl -s -D - -o /dev/null -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}' \
  | grep -i '^mcp-session-id:' | awk '{print $2}' | tr -d '\r')

# 2. Confirm initialization
curl -s -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. Call a tool
curl -s -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"crawl_url","arguments":{"url":"https://example.com","content_limit":2000}}}'
```

List the available tools with `{"jsonrpc":"2.0","id":3,"method":"tools/list"}`, using the same headers.

### Tool Argument Examples

Only the JSON-RPC `params` change between calls. Each `arguments` object must use the tool's real parameter names.

Web crawling on a JavaScript-heavy page:

```json
{"name": "crawl_url", "arguments": {"url": "https://spa-example.com", "wait_for_js": true, "timeout": 60}}
```

Saving the full page to disk and returning only metadata:

```json
{"name": "crawl_url", "arguments": {"url": "https://example.com/long-article", "output_path": "/tmp/article.md"}}
```

Large pages are not summarized unless `auto_summarize` is `true`. Without it, a response larger than about 25,000 tokens is truncated and includes recommendations: use `content_limit`/`content_offset` to page through the content, or `output_path` to write the full text to a file.

Web search. The search tools take a single `request` object (`engine` is optional):

```json
{"name": "search_web", "arguments": {"request": {"query": "python web scraping", "num_results": 10}}}
```

File processing:

```json
{"name": "process_file", "arguments": {"url": "https://example.com/document.pdf", "max_size_mb": 50, "include_metadata": true}}
```

Remote `.html`/`.htm` URLs passed to `crawl_url` are rendered in the browser. Document URLs (PDF, DOCX, XLSX, PPTX, ZIP, ...) are converted as files and do not use the browser fallback.

YouTube transcript:

```json
{"name": "extract_youtube_transcript", "arguments": {"url": "https://www.youtube.com/watch?v=VIDEO_ID", "languages": ["en", "ja"], "include_timestamps": true, "timezone": "Asia/Tokyo"}}
```

`timezone` is an IANA name (default `UTC`). It is used to convert the publish time, which is returned under `metadata.published_at`.

See the [API Reference](API_REFERENCE.md) for every tool's parameters.

## Python Integration

Use an MCP client library instead of hand-written HTTP calls. `fastmcp` is already a dependency of this project:

```python
import asyncio
import json

from fastmcp import Client


async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        tools = await client.list_tools()
        print([tool.name for tool in tools])

        result = await client.call_tool(
            "crawl_url", {"url": "https://example.com", "content_limit": 2000}
        )
        data = json.loads(result.content[0].text)
        print(data["success"], data["title"])


asyncio.run(main())
```

The official `mcp` Python SDK (`mcp.client.streamable_http.streamablehttp_client`) and the TypeScript SDK (`@modelcontextprotocol/sdk`, `StreamableHTTPClientTransport`) work the same way.

## Docker

The image's default command runs the server over STDIO. For HTTP, override the command and bind to `0.0.0.0` so the port is reachable from outside the container:

```bash
docker run -p 8000:8000 crawl4ai-mcp:latest \
  python -m crawl4ai_mcp.server --transport http --host 0.0.0.0 --port 8000

# or, with docker-compose
docker-compose --profile http up crawl4ai-mcp-http
```

The `crawl4ai-mcp-http` service has a healthcheck that sends a plain GET to `/mcp`. A running server answers it with a 4xx status, so any status below 500 counts as healthy.

## STDIO Protocol

```bash
python -m crawl4ai_mcp.server
```

STDIO carries newline-delimited JSON-RPC messages. While the server runs over STDIO, stdout carries only MCP messages; all diagnostics go to stderr. A wrapper script that prints to stdout will corrupt the protocol stream.

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

## Transport Comparison

| Feature | Streamable HTTP | SSE (legacy) | STDIO |
|---------|-----------------|--------------|-------|
| Endpoint | `/mcp` | `/sse` | stdin/stdout |
| Message format | JSON-RPC (SSE-framed responses) | JSON-RPC over SSE | Newline-delimited JSON-RPC |
| Process management | Run the server yourself | Run the server yourself | Client launches the server |
| Shared by multiple clients | Yes | Yes | No |

## Security Considerations

The server implements neither CORS nor authentication. It binds to `127.0.0.1` by default. If you expose it beyond localhost, put it behind a reverse proxy (for example nginx or Caddy) that handles TLS, authentication, and CORS.

## Troubleshooting

If the port is already in use:

```bash
lsof -i :8000
kill <PID>
```

HTTP 406 Not Acceptable means the request is missing `Accept: application/json, text/event-stream`.

For "Missing session ID" or HTTP 400, send the `mcp-session-id` header returned by `initialize` on every later request.

If the connection is refused:

- Check that the server is running and that host and port match
- Inside Docker, start the server with `--host 0.0.0.0`
- Check firewall settings

For more troubleshooting information, see the [Installation Guide](INSTALLATION.md#troubleshooting).

## Related Documentation

- [Configuration Examples](CONFIGURATION_EXAMPLES.md)
- [API Reference](API_REFERENCE.md)
- [Installation Guide](INSTALLATION.md)
