# Installation Guide

This guide provides detailed installation instructions for the Crawl4AI MCP Server.

## 🔧 Prerequisites Setup (Required First)

**Python 3.11 or later is required (FastMCP dependency baseline)**

**Before using any installation method, you MUST prepare system dependencies for Playwright:**

### 🐧 Linux/macOS

#### Ubuntu 24.04 LTS Users (Manual Installation Required)

**⚠️ IMPORTANT:** Due to Ubuntu 24.04's t64 library transition, the automated setup scripts are deprecated. Please use manual installation:

```bash
# Ubuntu 24.04 LTS Manual Setup (required due to t64 transition)
sudo apt update && sudo apt install -y \
  libnss3 \
  libatk-bridge2.0-0 \
  libxss1 \
  libasound2t64 \
  libgbm1 \
  libgtk-3-0t64 \
  libxshmfence-dev \
  libxrandr2 \
  libxcomposite1 \
  libxcursor1 \
  libxdamage1 \
  libxi6 \
  fonts-noto-color-emoji \
  fonts-unifont \
  python3-venv \
  python3-pip

# Install Playwright and browsers
python3 -m venv venv
source venv/bin/activate
pip install playwright==1.55.0
playwright install chromium
sudo playwright install-deps
```

#### Other Linux Distributions/macOS (Automated Script)

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
# Alternative: pip install crawl4ai==0.7.4 playwright==1.55.0
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

**Ubuntu 24.04 LTS:**
```bash
sudo apt update && sudo apt install -y \
  libnss3 libatk-bridge2.0-0 libxss1 libasound2t64 \
  libgbm1 libgtk-3-0t64 libxshmfence-dev libxrandr2 \
  libxcomposite1 libxcursor1 libxdamage1 libxi6 \
  fonts-noto-color-emoji fonts-unifont
```

**Other Linux distributions:**
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

**Playwright Browser Errors (Ubuntu 24.04 LTS):**
- Use t64 library names: `sudo apt-get install libnss3 libnspr4 libasound2t64 libgtk-3-0t64`
- Manual installation required due to t64 transition (see Prerequisites section)
- For WSL: Ensure X11 forwarding or headless mode

**Playwright Browser Errors (Other Linux):**
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
