#!/bin/bash
# Crawl4AI MCP HTTPサーバー起動スクリプト

set -e

# プロジェクトディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 仮想環境の確認
if [ ! -d "venv" ]; then
    echo "❌ 仮想環境が見つかりません。setup.shを実行してください。"
    exit 1
fi

# 仮想環境をアクティベート
source venv/bin/activate

# 必要なパッケージの確認
echo "📦 依存関係を確認中..."
pip install -q -r requirements.txt

# デフォルト設定
HOST="127.0.0.1"
PORT="8000"
LOG_LEVEL="INFO"

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
            LOG_LEVEL="$2"
            shift 2
            ;;
        --external)
            HOST="0.0.0.0"
            echo "⚠️  外部アクセス可能な設定で起動します"
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
            echo "  --log-level LEVEL  ログレベル (DEBUG|INFO|WARNING|ERROR)"
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
            echo "❌ 不明なオプション: $1"
            echo "ヘルプ: $0 --help"
            exit 1
            ;;
    esac
done

echo "🚀 Crawl4AI MCP HTTPサーバーを起動中..."
echo "📍 アドレス: http://$HOST:$PORT"
echo "🛑 停止: Ctrl+C"
echo ""

# HTTPサーバーを起動
python run_http_server.py --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"