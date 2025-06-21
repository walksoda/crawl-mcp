# CHANGELOG

## [1.2.0] - 2025-06-21

### 🎯 Major Features

#### Pure StreamableHTTP Implementation
- **Added**: Pure StreamableHTTP server implementation without Server-Sent Events (SSE)
- **Added**: `simple_pure_http_server.py` - Lightweight JSON-RPC 2.0 compliant HTTP server
- **Added**: `pure_http_test.py` - Comprehensive testing suite for Pure StreamableHTTP
- **Added**: `start_pure_http_server.sh` - Convenient startup script

#### Configuration Simplification
- **Added**: `claude_desktop_config_pure_http.json` - Simplified URL-only configuration
- **Improved**: Claude Desktop setup now requires only URL configuration instead of complex process management
- **Added**: `HTTP_SERVER_USAGE.md` - Complete usage guide for HTTP server deployment

#### Documentation Updates
- **Updated**: `README.md` and `README_ja.md` with Pure StreamableHTTP sections
- **Added**: `PURE_STREAMABLE_HTTP.md` - Comprehensive implementation guide
- **Added**: Protocol comparison tables and migration guides
- **Updated**: HTTP API documentation with new endpoints and examples

### 🔄 YouTube Transcript Migration
- **Migrated**: From YouTube Data API v3 to `youtube-transcript-api v1.1.0+`
- **Removed**: Complex OAuth authentication requirements
- **Simplified**: Configuration and setup process
- **Improved**: Stability and reliability of transcript extraction

### 🌐 HTTP Protocol Enhancement
- **Added**: Multiple HTTP protocol support (Pure StreamableHTTP + Legacy SSE)
- **Improved**: Server independence - run separately from Claude Desktop
- **Added**: Session management with UUID-based authentication
- **Enhanced**: Error handling and debugging capabilities

### 📊 Technical Improvements
- **Added**: Health check endpoints (`/health`)
- **Improved**: JSON-RPC 2.0 compliance
- **Added**: CORS support for web client integration
- **Enhanced**: Concurrent request handling

### 🛠️ Developer Experience
- **Added**: curl-compatible API testing
- **Improved**: Debug ease with plain JSON responses
- **Added**: Multiple startup methods (script, direct, background)
- **Enhanced**: Error messages and troubleshooting guides

### 📁 New Files
- `simple_pure_http_server.py` - Pure StreamableHTTP server
- `pure_http_test.py` - Testing client
- `start_pure_http_server.sh` - Startup script
- `claude_desktop_config_pure_http.json` - Simplified configuration
- `HTTP_SERVER_USAGE.md` - Usage documentation
- `PURE_STREAMABLE_HTTP.md` - Implementation guide

### 🔧 Configuration Changes

#### Old Configuration (Complex)
```json
{
  "mcpServers": {
    "crawl4ai-http": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server", "--transport", "http"],
      "cwd": "/path/to/project",
      "env": {"PYTHONPATH": "/path/to/venv/lib/python3.10/site-packages"}
    }
  }
}
```

#### New Configuration (Simple)
```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### 🆚 Protocol Comparison

| Feature | Pure StreamableHTTP | Legacy HTTP (SSE) | STDIO |
|---------|---------------------|-------------------|-------|
| Response Format | Plain JSON | Server-Sent Events | Binary |
| Configuration | Low (URL only) | Low (URL only) | High (Process) |
| Debug Ease | High (curl) | Medium (SSE parser) | Low |
| Independence | High | High | Low |
| Performance | High | Medium | High |

### 📈 Migration Path

1. **For New Users**: Use Pure StreamableHTTP with URL configuration
2. **For Existing Users**: 
   - Option 1: Migrate to Pure StreamableHTTP for simplified setup
   - Option 2: Continue with existing STDIO setup
   - Option 3: Use Legacy HTTP for compatibility

### 🎉 Benefits

- **Simplified Setup**: URL-only configuration vs complex process management
- **Better Debugging**: curl and standard HTTP tools compatible
- **Server Independence**: Run server separately, multiple clients possible
- **Improved Reliability**: Dedicated server process with better error handling
- **Enhanced Security**: Session-based authentication with UUID tokens

---

## [1.1.0] - Previous Release

### Features
- YouTube Data API v3 integration
- FastMCP StreamableHTTP with SSE
- Complex Claude Desktop configuration
- Process-based server management

### Limitations
- Required OAuth authentication for YouTube
- Complex setup process
- SSE parsing requirements
- Tightly coupled server-client architecture