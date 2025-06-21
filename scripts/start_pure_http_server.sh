#!/bin/bash
# Pure StreamableHTTP MCP サーバー起動スクリプト

echo "🚀 Starting Pure StreamableHTTP MCP Server..."
echo "📋 Protocol: Pure JSON (no SSE)"
echo "🌐 Endpoint: http://127.0.0.1:8000/mcp"
echo "🛑 Stop: Ctrl+C"
echo ""

# 仮想環境があれば活性化
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# サーバー起動
python examples/simple_pure_http_server.py --host 127.0.0.1 --port 8000