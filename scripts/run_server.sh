#!/bin/bash
# Crawl4AI MCP Server startup script (stdio transport)
#
# The server keeps stdout for MCP messages only; diagnostics go to stderr.
# Set CRAWL4AI_MCP_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR to enable logging.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "No virtual environment found. Run scripts/setup.sh first." >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="ignore"
export CRAWL4AI_VERBOSE="false"
export PLAYWRIGHT_SKIP_BROWSER_GC="1"

exec "$PYTHON" -m crawl4ai_mcp.server "$@"
