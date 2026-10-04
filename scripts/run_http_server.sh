#!/bin/bash
# Crawl4AI MCP HTTPサーバー起動スクリプト (streamable-http, エンドポイント /mcp)

set -e

# プロジェクトルートに移動 (このスクリプトは scripts/ にある)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 仮想環境の確認 (venv を優先し、なければ .venv)
if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "仮想環境が見つかりません。scripts/setup.sh を実行してください。" >&2
    exit 1
fi

# デフォルト設定
HOST="127.0.0.1"
PORT="8000"

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --log-level)
            export CRAWL4AI_MCP_LOG_LEVEL="$2"
            shift 2
            ;;
        --external)
            HOST="0.0.0.0"
            echo "外部アクセス可能な設定で起動します"
            shift
            ;;
        --help|-h)
            echo "Crawl4AI MCP HTTPサーバー起動スクリプト"
            echo ""
            echo "使用方法:"
            echo "  $0 [オプション]"
            echo ""
            echo "オプション:"
            echo "  --host HOST        バインドするホスト (デフォルト: 127.0.0.1)"
            echo "  --port PORT        バインドするポート (デフォルト: 8000)"
            echo "  --log-level LEVEL  stderrへのログ出力 (DEBUG|INFO|WARNING|ERROR、デフォルト: 出力なし)"
            echo "  --external         外部アクセス許可 (--host 0.0.0.0と同等)"
            echo "  --help, -h         このヘルプを表示"
            echo ""
            echo "例:"
            echo "  $0                              # ローカルホストで起動"
            echo "  $0 --external                   # 外部からアクセス可能で起動"
            echo "  $0 --port 8080                  # ポート8080で起動"
            echo "  $0 --host 0.0.0.0 --port 9000  # カスタム設定で起動"
            exit 0
            ;;
        *)
            echo "不明なオプション: $1" >&2
            echo "ヘルプ: $0 --help" >&2
            exit 1
            ;;
    esac
done

echo "Crawl4AI MCP HTTPサーバーを起動中..."
echo "エンドポイント: http://$HOST:$PORT/mcp"
echo "停止: Ctrl+C"
echo ""

exec "$PYTHON" -m crawl4ai_mcp.server --transport streamable-http --host "$HOST" --port "$PORT"
