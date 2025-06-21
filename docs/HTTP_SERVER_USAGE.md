# Pure StreamableHTTP サーバー使用方法

## 🎯 概要

Pure StreamableHTTP実装により、Claude DesktopはHTTP URL経由でMCPサーバーに接続できます。サーバーを別途起動し、Claude Desktop側は単純にURLで接続する方式です。

## 🚀 サーバー起動方法

### 方法1: 起動スクリプト使用
```bash
./start_pure_http_server.sh
```

### 方法2: 直接起動
```bash
python simple_pure_http_server.py --host 127.0.0.1 --port 8000
```

### 方法3: バックグラウンド起動
```bash
nohup python simple_pure_http_server.py --port 8000 > server.log 2>&1 &
```

## 📱 Claude Desktop設定

### 設定ファイル: `claude_desktop_config_pure_http.json`

```json
{
  "mcpServers": {
    "crawl4ai-stdio": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "PYTHONPATH": "/home/user/prj/crawl/venv/lib/python3.10/site-packages"
      }
    },
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    },
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### 利用できるサーバー

1. **crawl4ai-stdio**: 従来のSTDIOプロトコル (プロセス起動)
2. **crawl4ai-pure-http**: 新しいPure StreamableHTTP (URL接続)
3. **crawl4ai-legacy-http**: 従来のSSE実装 (URL接続)

## 🔧 使用手順

### 1. サーバー起動
```bash
# ターミナル1でサーバー起動
./start_pure_http_server.sh
```

### 2. 設定ファイル適用
```bash
# Claude Desktopの設定ファイルを更新
cp claude_desktop_config_pure_http.json ~/.config/claude-desktop/claude_desktop_config.json
```

### 3. Claude Desktop再起動
Claude Desktopアプリケーションを再起動して設定を適用

## ✅ 動作確認方法

### サーバー状態確認
```bash
# ヘルスチェック
curl http://127.0.0.1:8000/health

# 期待される結果
{
  "status": "healthy",
  "server": "simple-pure-http",
  "version": "1.0.0",
  "protocol": "StreamableHTTP (pure JSON)"
}
```

### 完全なワークフローテスト
```bash
python pure_http_test.py
```

## 🆚 プロトコル比較

| プロトコル | 接続方式 | 設定 | 利点 |
|-----------|----------|------|------|
| STDIO | プロセス起動 | `command` + `args` | シンプル、デバッグ容易 |
| Pure HTTP | URL接続 | `url` のみ | サーバー独立、スケーラブル |
| Legacy HTTP | URL接続 | `url` のみ | FastMCP互換 |

## 🔍 トラブルシューティング

### サーバーが起動しない
```bash
# ポート使用状況確認
lsof -i :8000

# プロセス確認
ps aux | grep simple_pure_http_server
```

### Claude Desktopから接続できない
1. サーバーが起動しているか確認
2. ファイアウォール設定確認
3. Claude Desktop設定ファイルの構文確認

### デバッグモード
```bash
python simple_pure_http_server.py --host 127.0.0.1 --port 8000
# ログを確認してエラーを特定
```

## 📊 利点

### Pure StreamableHTTP の利点
- **独立性**: サーバーを独立して起動・管理
- **スケーラビリティ**: 複数のClaude Desktopから同じサーバーに接続可能
- **デバッグ**: curlやPostmanで直接テスト可能
- **開発**: サーバー側の変更時にClaude Desktop再起動不要

### URL設定の利点
- **シンプル**: 設定ファイルが大幅にシンプル化
- **柔軟性**: 異なるポートやホストに簡単に切り替え
- **運用**: 本番環境での運用に適している

## 🎯 推奨運用方法

### 開発時
```bash
# 開発用サーバー起動
python simple_pure_http_server.py --host 127.0.0.1 --port 8000
```

### 本番運用時
```bash
# バックグラウンド起動
nohup python simple_pure_http_server.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# systemd サービス化も推奨
```

これにより、真のStreamableHTTPプロトコルでの運用が可能になります！