# Development Guide

Complete guide for developers contributing to and working with the Crawl4AI MCP Server project.

## 🏗️ Project Architecture

### Repository Structure

```
crawl-mcp/
├── crawl4ai_mcp/              # Main server implementation
│   ├── server.py              # Primary MCP server
│   ├── config.py              # Configuration management
│   ├── tools/                 # MCP tool implementations
│   ├── prompts/               # MCP prompt definitions
│   ├── models/                # Data models and schemas
│   ├── file_processor.py      # File processing logic
│   ├── youtube_processor.py   # YouTube integration
│   ├── google_search_processor.py # Google search integration
│   └── strategies.py          # Crawling strategies
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore              # Docker build exclusions
├── scripts/                   # Setup and utility scripts
├── configs/                   # Configuration examples
├── examples/                  # Usage examples and tests
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── pyproject.toml            # UVX packaging configuration
└── CLAUDE.md                 # Developer instructions
```

### Key Components

**Server Implementation:**
- `server.py` - Main FastMCP server with all MCP tools
- `config.py` - Environment and configuration management
- `suppress_output.py` - Output suppression for clean execution

**Core Processors:**
- `file_processor.py` - Microsoft MarkItDown integration
- `youtube_processor.py` - YouTube transcript extraction
- `google_search_processor.py` - Google search integration
- `strategies.py` - Crawling strategy implementations

**MCP Framework:**
- `tools/` - Individual tool implementations
- `prompts/` - Workflow prompt definitions
- `models/` - Request/response schemas

## 🔄 Development & Distribution

This project supports **multiple distribution methods**:

1. **Development**: `crawl4ai_mcp/server.py`
   - Direct Python execution for development
   - Virtual environment setup
   - Easy debugging and modification

2. **UVX Distribution**: PyPI package
   - Distributed via GitHub releases
   - Automatic UVX compatibility
   - Easy installation for end users

3. **Docker Distribution**: Container deployment
   - Production-ready container images
   - Multi-browser headless support
   - Easy scaling and deployment

### Development Workflow

**Standard development process**:

```bash
# 1. Set up development environment
source ./venv/bin/activate

# 2. Develop and test
vim crawl4ai_mcp/server.py
python -m crawl4ai_mcp.server

# 3. Test with Docker (optional)
docker-compose up --build

# 4. Update version
vim pyproject.toml  # Update version number

# 5. Commit and tag
git add .
git commit -m "feat: Add new functionality"
git tag -a v0.1.4 -m "Release v0.1.4"

# 6. Push (triggers automatic distribution)
git push origin main --tags
```

### Changes Requiring Synchronization

- **Tool descriptions** - Ensure LLM can properly select tools
- **New features** - New MCP tools or parameters
- **Bug fixes** - Security fixes and behavior improvements
- **Dependencies** - requirements.txt updates
- **Configuration** - Default values and timeout adjustments
- **Packaging** - pyproject.toml for UVX compatibility

## 🛠️ Development Environment Setup

### Prerequisites

**System Dependencies:**
```bash
# Linux/macOS
sudo bash scripts/prepare_for_uvx_playwright.sh

# Windows (as Administrator)
powershell -ExecutionPolicy Bypass -File scripts/prepare_for_uvx_playwright.ps1
```

### Local Development Setup

**Method 1: UV Package Manager (Recommended)**
```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
uv sync
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

**Method 2: Traditional Virtual Environment**
```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
python -m playwright install chromium
```

### Required Virtual Environment

**⚠️ CRITICAL**: Always use the `./venv` virtual environment for development:

```bash
# Activate virtual environment
source venv/bin/activate

# Verify correct environment
which python  # Should point to ./venv/bin/python
python -c "import sys; print(sys.prefix)"  # Should show venv path
```

## 🧪 Testing and Quality Assurance

### Test Commands

**Basic Server Testing:**
```bash
# Test main server startup
python -m crawl4ai_mcp.server

# Test HTTP server
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000

# Test with MCP client
claude mcp test crawl4ai
```

**HTTP API Testing:**
```bash
# Pure StreamableHTTP test
python examples/pure_http_test.py

# Legacy HTTP test
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Health check
curl http://127.0.0.1:8000/health
```

**YouTube Integration Testing:**
```bash
# Direct API test
python test_youtube_transcript_direct.py "https://www.youtube.com/watch?v=VIDEO_ID"

# MCP integration test
python test_mcp_youtube.py
```

### Pre-commit Quality Checks

**Required before commits:**
```bash
# 1. Lint commit messages with textlint
npx textlint commit-message.txt

# 2. Test Docker build
docker build -t crawl4ai-mcp:test . --quiet

# 3. Verify Docker functionality
echo "Docker build: $(docker build -t crawl4ai-mcp:test . --quiet > /dev/null 2>&1 && echo "✅" || echo "❌")"

# 4. Test UVX installation
python -m crawl4ai_mcp.server --help
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --help
```

## 🔧 Development Tools and Configuration

### Environment Variables

```bash
# Development logging
export FASTMCP_LOG_LEVEL=DEBUG

# Language settings
export CRAWL4AI_LANG=en  # or 'ja' for Japanese

# API keys (development)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://resource.openai.azure.com"

# Python environment
export PYTHONPATH="/path/to/crawl:$PYTHONPATH"
export PYTHONUNBUFFERED=1
```

### Debug Configuration

**High-verbosity debugging:**
```json
{
  "mcpServers": {
    "crawl4ai-debug": {
      "command": "/path/to/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG",
        "PYTHONPATH": "/path/to/crawl",
        "DEBUG": "1",
        "PYTHONUNBUFFERED": "1",
        "PLAYWRIGHT_DEBUG": "1"
      }
    }
  }
}
```

### Code Style and Standards

**Python Code Standards:**
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Document functions with docstrings
- Handle errors gracefully with try/catch
- Use async/await for I/O operations

**Commit Message Standards:**
- English language only
- No emojis in commit messages
- Format: `type: brief description`
- Examples:
  - `feat: Add YouTube batch processing`
  - `fix: Resolve memory leak in large file processing`
  - `docs: Update API reference documentation`

## 📦 Build and Packaging

### Docker Container Management

**Building Docker Images:**
```bash
# Build development image
docker build -t crawl4ai-mcp:dev .

# Build production image with multi-browser support
docker build -t crawl4ai-mcp:latest .

# Verify image contents
docker images | grep crawl4ai-mcp
```

**Testing Docker Containers:**
```bash
# Test STDIO mode (default)
docker-compose up --build

# Test HTTP mode
docker-compose --profile http up --build crawl4ai-mcp-http

# Test individual container
docker run -it crawl4ai-mcp:latest
```

**Version Management:**
```bash
# Update version in pyproject.toml
vim pyproject.toml

# Update Docker tags
docker tag crawl4ai-mcp:latest crawl4ai-mcp:v0.1.4

# Push to registry (if applicable)
# docker push crawl4ai-mcp:v0.1.4

# Tag git release
git tag -a v0.1.4 -m "Release v0.1.4 - Docker support"
git push origin v0.1.4
```

### UVX Compatibility

**pyproject.toml Configuration:**
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "crawl-mcp"
version = "1.0.0"
dependencies = [
    "crawl4ai>=0.7.2",
    "playwright>=1.54.0",
    "fastmcp>=0.2.0"
]

[project.scripts]
crawl-mcp = "crawl4ai_mcp.server:main"
```

## 🔍 Debugging and Troubleshooting

### Common Development Issues

**Import Errors:**
```bash
# Ensure virtual environment is active
source venv/bin/activate

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Install missing dependencies
pip install -r requirements.txt
```

**Browser Issues:**
```bash 
# Reinstall Playwright browsers
python -m playwright install chromium

# System dependencies (Linux)
sudo apt-get install libnss3 libnspr4 libasound2 libatk-bridge2.0-0

# Check browser installation
python -c "from playwright.sync_api import sync_playwright; print('Browser check: OK')"
```

**Server Startup Issues:**
```bash
# Check port availability
lsof -i :8000

# Kill existing process
kill -9 $(lsof -t -i:8000)

# Check server logs
export FASTMCP_LOG_LEVEL=DEBUG
python -m crawl4ai_mcp.server
```

### Performance Profiling

**Memory and CPU Monitoring:**
```bash
# Monitor memory usage
python -m memory_profiler crawl4ai_mcp/server.py

# CPU profiling
python -m cProfile -o profile.prof -m crawl4ai_mcp.server

# Analyze profile
python -m pstats profile.prof
```

**Request Tracing:**
```bash
# Enable request tracing
export CRAWL4AI_TRACE=1
export FASTMCP_LOG_LEVEL=DEBUG

# Monitor request patterns
tail -f server.log | grep -E "(REQUEST|RESPONSE|ERROR)"
```

## 🤝 Contributing Guidelines

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/crawl-mcp.git
   cd crawl-mcp
   git remote add upstream https://github.com/walksoda/crawl-mcp.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Develop and Test**
   ```bash
   # Make changes to crawl4ai_mcp/server.py
   # Test thoroughly
   python -m crawl4ai_mcp.server
   ```

4. **Test and Build**
   ```bash
   # Test with Docker
   docker-compose up --build
   
   # Test UVX installation
   uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --help
   ```

5. **Quality Checks**
   ```bash
   # Run tests
   python examples/pure_http_test.py
   
   # Test Docker build
   docker build -t crawl4ai-mcp:test .
   
   # Lint commit message
   npx textlint commit-message.txt
   ```

6. **Submit Pull Request**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   git push origin feature/your-feature-name
   ```

### Code Review Process

**Before Submitting:**
- [ ] Changes tested in development environment
- [ ] Docker containers build successfully
- [ ] Documentation updated if needed
- [ ] Commit messages follow standards
- [ ] No breaking changes without discussion

**Review Criteria:**
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Security considerations
- Backward compatibility

## 📊 Monitoring and Analytics

### Development Metrics

**Server Performance:**
```bash
# Request processing time
grep "Processing time" server.log | awk '{print $4}' | sort -n

# Memory usage patterns
ps aux | grep crawl4ai_mcp | awk '{print $6}' # RSS memory

# Success/failure rates
grep -c "SUCCESS" server.log
grep -c "ERROR" server.log
```

**Tool Usage Statistics:**
```bash
# Most used tools
grep "Tool called:" server.log | awk '{print $3}' | sort | uniq -c | sort -nr

# Average response sizes
grep "Response size:" server.log | awk '{sum+=$3; count++} END {print sum/count}'
```

## 🔗 Related Resources

- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Configuration Examples**: [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
- **Advanced Usage**: [ADVANCED_USAGE.md](ADVANCED_USAGE.md)
- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **HTTP Integration**: [HTTP_INTEGRATION.md](HTTP_INTEGRATION.md)

## 💡 Development Best Practices

1. **Always use virtual environment** - Essential for dependency isolation
2. **Test multiple deployment methods** - UVX and Docker must work correctly
3. **Test before committing** - Ensure all functionality works
4. **Follow commit message standards** - English only, no emojis, clear descriptions
5. **Profile performance regularly** - Monitor memory and CPU usage
6. **Document thoroughly** - Update docs with new features
7. **Use debug logging during development** - `FASTMCP_LOG_LEVEL=DEBUG`
8. **Test edge cases** - Large files, network issues, rate limits
9. **Monitor resource usage** - Memory leaks and performance degradation
10. **Version dependencies carefully** - Pin versions for stability