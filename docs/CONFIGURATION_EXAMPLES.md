# Configuration Examples

Comprehensive configuration examples for different installation methods, platforms, and use cases.

## 🚀 UVX Installation Configurations

### Basic UVX Configuration (Recommended)

**For English environment:**
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

**For Japanese environment:**
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
        "CRAWL4AI_LANG": "ja"
      }
    }
  }
}
```

### UVX with Debug Logging

```json
{
  "mcpServers": {
    "crawl-mcp-debug": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "en",
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

## 🖥️ Development Environment Configurations

### Local Development Setup

**Linux/macOS:**
```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "C:\\path\\to\\your\\crawl\\venv\\Scripts\\python.exe",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "C:\\path\\to\\your\\crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### Development with Custom Python Path

```json
{
  "mcpServers": {
    "crawl4ai-custom": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "PYTHONPATH": "/path/to/crawl/venv/lib/python3.11/site-packages"
      }
    }
  }
}
```

## 🌐 HTTP Transport Configurations

### HTTP STDIO Configuration

```json
{
  "mcpServers": {
    "crawl4ai-stdio": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "PYTHONPATH": "/home/user/prj/crawl/venv/lib/python3.11/site-packages"
      }
    }
  }
}
```

### Legacy HTTP Configuration

```json
{
  "mcpServers": {
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### Pure StreamableHTTP Configuration

```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### HTTP with Custom Port

```json
{
  "mcpServers": {
    "crawl4ai-http-custom": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server", "--transport", "http", "--port", "8080"],
      "cwd": "/path/to/project",
      "env": {
        "PYTHONPATH": "/path/to/venv/lib/python3.11/site-packages"
      }
    }
  }
}
```

## 🖥️ Platform-Specific Configurations

### WSL (Windows Subsystem for Linux)

```json
{
  "mcpServers": {
    "crawl4ai-wsl": {
      "command": "wsl",
      "args": [
        "-e",
        "bash",
        "-c",
        "cd /home/user/prj/crawl && source venv/bin/activate && PYTHONPATH=/home/user/prj/crawl:$PYTHONPATH python -m crawl4ai_mcp.server"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### WSL with Simple Command

```json
{
  "mcpServers": {
    "crawl4ai-wsl-simple": {
      "command": "wsl",
      "args": [
        "/home/user/prj/crawl/venv/bin/python",
        "-m",
        "crawl4ai_mcp.server"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

### macOS Configuration

```json
{
  "mcpServers": {
    "crawl4ai-macos": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {}
    }
  }
}
```

## 🔧 Environment Variables

### Available Environment Variables

```json
{
  "mcpServers": {
    "crawl4ai-full-env": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "en",
        "FASTMCP_LOG_LEVEL": "INFO",
        "MCP_TRANSPORT": "stdio",
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": "8000"
      }
    }
  }
}
```

### Production Environment

```json
{
  "mcpServers": {
    "crawl4ai-production": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "en",
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

## 🛠️ Advanced Configuration Examples

### Configuration with LLM Settings

```json
{
  "mcpServers": {
    "crawl4ai-llm": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "PYTHONPATH": "/path/to/crawl/venv/lib/python3.11/site-packages",
        "OPENAI_API_KEY": "your-api-key-here",
        "ANTHROPIC_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Multi-Instance Configuration

```json
{
  "mcpServers": {
    "crawl4ai-primary": {
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
    },
    "crawl4ai-secondary": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### Testing Configuration

```json
{
  "mcpServers": {
    "crawl4ai-test": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG",
        "PYTEST_CURRENT_TEST": "true"
      }
    }
  }
}
```

## 📂 Configuration File Locations

### Platform-Specific Paths

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/claude-desktop/claude_desktop_config.json
```

### Configuration File Setup Commands

**Linux/macOS:**
```bash
# Copy configuration file
cp configs/claude_desktop_config.json ~/.config/claude-desktop/claude_desktop_config.json

# Create directory if it doesn't exist
mkdir -p ~/.config/claude-desktop/
```

**Windows PowerShell:**
```powershell
# Copy configuration file
Copy-Item "configs\claude_desktop_config.json" "$env:APPDATA\Claude\claude_desktop_config.json"

# Create directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "$env:APPDATA\Claude"
```

## 🔍 Tool Parameter Configuration Examples

### Basic Web Crawling Configuration

```json
{
  "wait_for_js": true,
  "simulate_user": true, 
  "timeout": 30,
  "generate_markdown": true
}
```

### JavaScript-Heavy Sites Configuration

```json
{
  "wait_for_js": true,
  "simulate_user": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded",
  "execute_js": "window.scrollTo(0, document.body.scrollHeight);",
  "generate_markdown": true
}
```

### Advanced Crawling Configuration

```json
{
  "url": "https://example.com",
  "max_depth": 2,
  "crawl_strategy": "bfs",
  "content_filter": "bm25",
  "filter_query": "important content keywords",
  "chunk_content": true,
  "auto_summarize": true,
  "summary_length": "medium"
}
```

## 🚨 Troubleshooting Configurations

### Debug Configuration

```json
{
  "mcpServers": {
    "crawl4ai-debug": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG",
        "PYTHONPATH": "/home/user/prj/crawl",
        "DEBUG": "1"
      }
    }
  }
}
```

### Error Isolation Configuration

```json
{
  "mcpServers": {
    "crawl4ai-isolated": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "en",
        "FASTMCP_LOG_LEVEL": "ERROR",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## 📊 Configuration Validation

### Validation Commands

```bash
# Test configuration syntax
python -m json.tool claude_desktop_config.json

# Test server startup
python -m crawl4ai_mcp.server --help

# Test UVX installation
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --help
```

### Health Check Configuration

```json
{
  "mcpServers": {
    "crawl4ai-health": {
      "url": "http://127.0.0.1:8000/mcp",
      "timeout": 30000
    }
  }
}
```

## 🔗 Related Documentation

- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **HTTP Integration**: [HTTP_INTEGRATION.md](HTTP_INTEGRATION.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Advanced Usage**: [ADVANCED_USAGE.md](ADVANCED_USAGE.md)
- **Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)

## 💡 Configuration Tips

1. **Use UVX for simplicity**: Recommended for most users
2. **Debug locally first**: Start with development environment
3. **Check file paths**: Ensure all paths are absolute and correct
4. **Environment variables**: Use them for sensitive data
5. **Test configurations**: Validate JSON syntax before use
6. **Platform considerations**: Different paths for different OSes
7. **Log levels**: Use DEBUG for troubleshooting, ERROR for production
