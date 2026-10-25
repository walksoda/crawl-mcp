# Installation Guide

How to install the Crawl4AI MCP Server and connect it to an MCP client.

## Prerequisites

- Python 3.11 or later (`requires-python = ">=3.11"`; tested on 3.11 and 3.12)
- Playwright Chromium and its system libraries. The project pins `playwright==1.64.1`. Install the browser build that matches that version before first use.

### Install Chromium for Playwright

On Linux, this installs the browser and the system libraries and requires sudo:

```bash
uvx --from playwright==1.64.1 playwright install --with-deps chromium
```

On macOS and Windows:

```bash
uvx --from playwright==1.64.1 playwright install chromium
```

If you already have an environment with the package installed (see Methods 2 and 3), you can run the same command inside it instead: `python -m playwright install --with-deps chromium` (Linux) or `python -m playwright install chromium`.

#### Manual system libraries (Ubuntu 24.04 LTS)

If `--with-deps` is not available to you (for example, no sudo during the Playwright step), install the libraries manually. Ubuntu 24.04 uses the `t64` package names:

```bash
sudo apt update && sudo apt install -y \
  libnss3 libnspr4 libatk-bridge2.0-0 libxss1 libasound2t64 \
  libgbm1 libgtk-3-0t64 libxshmfence-dev libxrandr2 \
  libxcomposite1 libxcursor1 libxdamage1 libxi6 \
  fonts-noto-color-emoji fonts-unifont
```

On older Debian/Ubuntu releases, use the non-`t64` names (`libasound2`, `libgtk-3-0`).

## Installation Methods

### Method 1: UVX (Recommended)

```bash
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

UVX resolves the dependencies and runs the server in its own isolated environment, so there is no virtual environment to set up.

### Method 2: Development Environment

```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
python3 -m venv venv && source venv/bin/activate
pip install -e .
python -m playwright install chromium
python -m crawl4ai_mcp.server
```

Use this method for local development. See the [Development Guide](DEVELOPMENT.md) for test dependencies.

### Method 3: pip Installation

```bash
pip install git+https://github.com/walksoda/crawl-mcp
# or, from a cloned repository: pip install .
python -m playwright install chromium
crawl-mcp
```

Use this method to install into an existing Python environment.

`pyproject.toml` defines the dependencies with constrained versions (for example `crawl4ai>=0.9.5,<0.10`, `fastmcp>=2.14.7,<3`, `playwright==1.64.1`). `requirements.txt` mirrors them for Docker builds. The Anthropic SDK (`anthropic>=0.40.0`) is a regular dependency, so Anthropic summarization also works in UVX installs.

### Setup Scripts (Traditional)

The repository also contains setup scripts. They create `venv`, install `requirements.txt` and Playwright's Chromium, and print the commands to start the server. `requirements.txt` does not install the package itself, so run the server from the repository root. On Linux, the scripts do not install system libraries; see Prerequisites if Chromium fails to start.

Linux/macOS:

```bash
bash scripts/setup.sh
```

Windows:

```cmd
scripts\setup_windows.bat
```

## Claude Desktop Integration

### Basic Configuration

Add to your `claude_desktop_config.json`:

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

### Configuration File Locations

Windows:

```
%APPDATA%\Claude\claude_desktop_config.json
```

macOS:

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

## Troubleshooting

### Common Installation Issues

1. Browser not found, or executable doesn't exist: re-run the Chromium install command above with `playwright==1.64.1`. A different Playwright version downloads a different Chromium build.
2. Missing system libraries on Linux: use `--with-deps`, or install the libraries manually (see Prerequisites).
3. Permission errors: `--with-deps` needs sudo on Linux.
4. If UVX fails, try Method 2 or Method 3.

### Specific Error Solutions

`ModuleNotFoundError`: activate the virtual environment and install the package with `pip install -e .` (or `pip install .`). If you run from a source checkout without installing, run from the repository root.

Playwright browser errors on Linux or WSL: on Ubuntu 24.04, use the `t64` library names (`libasound2t64`, `libgtk-3-0t64`). The server runs Chromium headless, so WSL does not need an X server.

JSON parsing errors in the MCP client (STDIO): over STDIO, the server keeps stdout for MCP messages only. All diagnostics, including the `.env` loading notices, go to stderr. If a client still reports invalid JSON, check for a wrapper script or shell profile that prints to stdout before the server starts.

Server logs: logging is off by default. Set `CRAWL4AI_MCP_LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, or `ERROR` in the `env` block of the client configuration to write log records to stderr. A value in `.env` is not used, because the server reads this variable before it loads `.env`.

Cached results that look like 16-character hash strings: crawl4ai's legacy cache migration could corrupt an existing cache database. The server disables that migration on startup, so upgrading crawl4ai no longer rewrites `~/.crawl4ai/crawl4ai.db`. If an earlier run already damaged the cache, you can delete the crawl4ai cache (`~/.crawl4ai/`); it is rebuilt on the next crawl.

## Next Steps

After installation:

1. Check the installation: run `python -m crawl4ai_mcp.server --help` (or `crawl-mcp --help`)
2. Configure your MCP client: add the server configuration shown above
3. Try a simple `crawl_url` call
4. See the [API Reference](API_REFERENCE.md) for the available tools

For usage, see the main [README](../README.md) or the [Usage Patterns](ADVANCED_USAGE.md).
