# Installation Guide

This guide provides detailed installation instructions for the Crawl4AI MCP Server.

## 🔧 Prerequisites Setup (Required First)

**Before using any installation method, you MUST prepare system dependencies for Playwright:**

### 🐧 Linux/macOS

```bash
# Install system dependencies for Playwright (required for all methods)
sudo bash scripts/prepare_for_uvx_playwright.sh

# For Japanese language (optional)
export CRAWL4AI_LANG=ja
sudo bash scripts/prepare_for_uvx_playwright.sh
```

### 🪟 Windows

```powershell
# Run as Administrator in PowerShell (required for all methods)
scripts/prepare_for_uvx_playwright.ps1

# If execution policy blocks the script, use:
# powershell -ExecutionPolicy Bypass -File "scripts/prepare_for_uvx_playwright.ps1"
```

### System Preparation Features

- **Cross-platform**: Linux (apt/yum/pacman/apk) + Windows  
- **Minimal Dependencies**: Only installs essential Playwright system libraries
- **UVX Optimized**: Designed specifically for UVX execution environment
- **Multi-language**: English (default) + Japanese (`CRAWL4AI_LANG=ja`)
- **Version Synchronization**: Automatically reads Playwright version from requirements.txt for consistency
- **Smart Installation**: Manual installation instructions use correct pinned versions
- **Enhanced Error Handling**: Improved Chromium version compatibility messages for MCP clients

## 🚀 Installation Methods

### Method 1: UVX (Recommended - Easiest & Production Ready) ⭐

**Most convenient single-command installation:**

```bash
# After system preparation above - that's it!
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

**✅ Benefits:** Zero configuration, automatic dependency management, isolated environment

### Method 2: Development Environment

```bash
# After system preparation above, create development environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # Uses pinned versions for stability
python -m playwright install chromium
python -m crawl4ai_mcp.server
```

**Use case:** Local development and customization

### Method 3: Direct Installation

```bash
# After system preparation above
pip install -r requirements.txt  # Recommended: uses pinned versions
# Alternative: pip install crawl4ai==0.7.2 playwright==1.54.0
python -m playwright install chromium
python -m crawl4ai_mcp.server
```

**Use case:** Global installation or system-wide deployment

## 🔧 Development Setup

### Local Development Setup

```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
uv sync
```

### Quick Setup (Traditional)

**Linux/macOS:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup_windows.bat
```

### Manual Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt  # Installs pinned versions for stability
```

3. **Install Playwright browser dependencies (Linux/WSL):**
```bash
sudo apt-get update
sudo apt-get install libnss3 libnspr4 libasound2 libatk-bridge2.0-0 libdrm2 libgtk-3-0 libgbm1
```

## ⚙️ Claude Desktop Integration

### Basic Configuration

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

### Configuration File Locations

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

## 🔍 Troubleshooting

### Common Installation Issues

If installation fails:

1. **Check Chromium**: Run diagnostics with `get_system_diagnostics` tool
2. **Browser Issues**: Re-run the Chromium setup scripts above
3. **Permissions**: Ensure scripts run with proper privileges (sudo/Administrator)
4. **Alternative Methods**: Try Method 2 (Development) or Method 3 (Direct) if UVX fails
5. **UVX Success**: After system preparation, UVX (Method 1) typically works reliably

### Specific Error Solutions

**ModuleNotFoundError:**
- Ensure virtual environment is activated
- Verify PYTHONPATH is set correctly
- Install dependencies: `pip install -r requirements.txt`

**Playwright Browser Errors:**
- Install system dependencies: `sudo apt-get install libnss3 libnspr4 libasound2`
- For WSL: Ensure X11 forwarding or headless mode

**JSON Parsing Errors:**
- **Resolved**: Output suppression implemented in latest version
- All crawl4ai verbose output is now properly suppressed

### PowerShell Execution Policy (Windows)

If you encounter execution policy errors on Windows:

```powershell
# Option 1: Bypass execution policy for this script only
powershell -ExecutionPolicy Bypass -File "scripts/prepare_for_uvx_playwright.ps1"

# Option 2: Temporarily change execution policy (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then restore after setup
Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser
```

## 📚 Next Steps

After successful installation:

1. **Verify Installation**: Run `python -m crawl4ai_mcp.server --help`
2. **Test Basic Functionality**: Try a simple crawl operation
3. **Configure Claude Desktop**: Add the MCP server configuration
4. **Explore Documentation**: Check the [API Reference](API_REFERENCE.md) for available tools

For detailed usage instructions, see the main [README](../README.md) or [Advanced Usage Guide](ADVANCED_USAGE.md).