# HTTP Integration Guide

This guide covers HTTP API access for the Crawl4AI MCP Server, supporting multiple HTTP protocols for different use cases.

## 🌐 Overview

The MCP server supports multiple HTTP protocols, allowing you to choose the optimal implementation:

- **Pure StreamableHTTP** (Recommended): Plain JSON HTTP protocol without Server-Sent Events
- **Legacy HTTP**: Traditional FastMCP StreamableHTTP protocol with SSE
- **STDIO**: Binary protocol for direct integration

## 🎯 Pure StreamableHTTP (Recommended)

**Pure JSON HTTP protocol without Server-Sent Events (SSE)**

### Server Startup

```bash
# Method 1: Using startup script
./scripts/start_pure_http_server.sh

# Method 2: Direct startup
python examples/simple_pure_http_server.py --host 127.0.0.1 --port 8000

# Method 3: Background startup
nohup python examples/simple_pure_http_server.py --port 8000 > server.log 2>&1 &
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### Usage Steps

1. **Start Server**: `./scripts/start_pure_http_server.sh`
2. **Apply Configuration**: Use `configs/claude_desktop_config_pure_http.json`
3. **Restart Claude Desktop**: Apply settings

### Verification

```bash
# Health check
curl http://127.0.0.1:8000/health

# Complete test
python examples/pure_http_test.py
```

### Pure StreamableHTTP Usage Example

```bash
# Initialize session
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

## 🔄 Legacy HTTP (SSE Implementation)

**Traditional FastMCP StreamableHTTP protocol (with SSE)**

### Server Startup

```bash
# Method 1: Command line
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8001

# Method 2: Environment variables
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8001
python -m crawl4ai_mcp.server
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### Legacy HTTP Usage Example

```bash
curl -X POST "http://127.0.0.1:8001/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

## 🖥️ STDIO Protocol

### Usage Methods

**STDIO transport (default):**
```bash
python -m crawl4ai_mcp.server
```

**Claude Desktop Configuration:**
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

## 📊 Protocol Comparison

| Feature | Pure StreamableHTTP | Legacy HTTP (SSE) | STDIO |
|---------|---------------------|-------------------|-------|
| Response Format | Plain JSON | Server-Sent Events | Binary |
| Configuration Complexity | Low (URL only) | Low (URL only) | High (Process management) |
| Debug Ease | High (curl compatible) | Medium (SSE parser needed) | Low |
| Independence | High | High | Low |
| Performance | High | Medium | High |

## 🚀 Server Startup Options

### Method 1: Command Line

```bash
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000
```

### Method 2: Environment Variables

```bash
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8000
python -m crawl4ai_mcp.server
```

### Method 3: Docker (if available)

```bash
docker run -p 8000:8000 crawl4ai-mcp --transport http --port 8000
```

## 🔗 API Endpoints

Once running, the HTTP API provides:

- **Base URL**: `http://127.0.0.1:8000`
- **OpenAPI Documentation**: `http://127.0.0.1:8000/docs`
- **Tool Endpoints**: `http://127.0.0.1:8000/tools/{tool_name}`
- **Resource Endpoints**: `http://127.0.0.1:8000/resources/{resource_uri}`

All MCP tools (crawl_url, intelligent_extract, process_file, etc.) are accessible via HTTP POST requests with JSON payloads matching the tool parameters.

## 🛠️ Tool Usage Examples

### Basic Web Crawling

```bash
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

### Advanced Crawling with JavaScript

```bash
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://spa-example.com",
    "wait_for_js": true,
    "simulate_user": true,
    "timeout": 30
  }'
```

### Google Search

```bash
curl -X POST "http://127.0.0.1:8000/tools/search_google" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "python web scraping",
    "num_results": 10,
    "search_genre": "programming"
  }'
```

### File Processing

```bash
curl -X POST "http://127.0.0.1:8000/tools/process_file" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "max_size_mb": 50,
    "include_metadata": true
  }'
```

### YouTube Transcript Extraction

```bash
curl -X POST "http://127.0.0.1:8000/tools/extract_youtube_transcript" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["en", "ja"],
    "include_timestamps": true
  }'
```

## 🔧 Integration with Applications

### Python Integration

```python
import requests
import json

# Basic usage
def crawl_url(url, **kwargs):
    response = requests.post(
        "http://127.0.0.1:8000/tools/crawl_url",
        headers={"Content-Type": "application/json"},
        json={"url": url, **kwargs}
    )
    return response.json()

# Example usage
result = crawl_url("https://example.com", generate_markdown=True)
```

### JavaScript/Node.js Integration

```javascript
async function crawlUrl(url, options = {}) {
  const response = await fetch('http://127.0.0.1:8000/tools/crawl_url', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url, ...options })
  });
  
  return await response.json();
}

// Example usage
const result = await crawlUrl('https://example.com', { 
  generate_markdown: true 
});
```

## 🔍 Debugging and Monitoring

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Server Logs

```bash
# View logs in real-time
tail -f server.log

# Search for errors
grep -i error server.log
```

### Performance Monitoring

```bash
# Monitor requests
curl -s http://127.0.0.1:8000/stats

# Check server status
curl -s http://127.0.0.1:8000/status
```

## 🔒 Security Considerations

### CORS Configuration

For web applications, ensure proper CORS headers are configured:

```python
# Example CORS configuration
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
}
```

### Authentication

For production use, implement authentication:

```bash
# Example with API key
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"url": "https://example.com"}'
```

## 📚 Additional Resources

- **Pure StreamableHTTP Details**: [PURE_STREAMABLE_HTTP.md](PURE_STREAMABLE_HTTP.md)
- **HTTP Server Usage**: [HTTP_SERVER_USAGE.md](HTTP_SERVER_USAGE.md)
- **Legacy HTTP API**: [HTTP_API_GUIDE.md](HTTP_API_GUIDE.md)
- **Configuration Examples**: [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Connection Refused:**
- Check if server is running
- Verify port and host configuration
- Check firewall settings

**JSON Parse Errors:**
- Ensure proper Content-Type headers
- Validate JSON payload format
- Check for special characters in data

For more troubleshooting information, see the [Installation Guide](INSTALLATION.md#troubleshooting).