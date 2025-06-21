#!/bin/bash
# Crawl4AI MCP Server startup script with complete output suppression

cd /home/user/prj/crawl
source venv/bin/activate

# Set environment variables to suppress all non-JSON output
export PYTHONPATH="/home/user/prj/crawl:$PYTHONPATH"
export PYTHONWARNINGS="ignore"
export CRAWL4AI_VERBOSE="false"
export PLAYWRIGHT_SKIP_BROWSER_GC="1"

# Run the server with stderr completely suppressed to avoid JSON parsing errors
exec python -m crawl4ai_mcp.server "$@" 2>/dev/null