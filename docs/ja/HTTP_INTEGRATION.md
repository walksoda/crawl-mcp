# HTTP統合ガイド

このガイドでは、Crawl4AI MCPサーバーのHTTP API アクセスについて、異なる使用ケースに対応する複数のHTTPプロトコルをカバーしています。

## 🌐 概要

MCPサーバーは複数のHTTPプロトコルをサポートしており、最適な実装を選択できます：

- **Pure StreamableHTTP**（推奨）: Server-Sent Eventsなしのプレーンな JSON HTTP プロトコル
- **Legacy HTTP**: SSE付きの従来の FastMCP StreamableHTTP プロトコル
- **STDIO**: 直接統合用のバイナリプロトコル

## 🎯 Pure StreamableHTTP（推奨）

**Server-Sent Events (SSE) なしのプレーンな JSON HTTP プロトコル**

### サーバー起動

```bash
# 方法1: 起動スクリプトを使用
./scripts/start_pure_http_server.sh

# 方法2: 直接起動
python examples/simple_pure_http_server.py --host 127.0.0.1 --port 8000

# 方法3: バックグラウンド起動
nohup python examples/simple_pure_http_server.py --port 8000 > server.log 2>&1 &
```

### Claude Desktop設定

```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### 使用手順

1. **サーバー起動**: `./scripts/start_pure_http_server.sh`
2. **設定適用**: `configs/claude_desktop_config_pure_http.json`を使用
3. **Claude Desktop再起動**: 設定を適用

### 検証

```bash
# ヘルスチェック
curl http://127.0.0.1:8000/health

# 完全テスト
python examples/pure_http_test.py
```

### Pure StreamableHTTP使用例

```bash
# セッション初期化
SESSION_ID=$(curl -s -X POST http://127.0.0.1:8000/mcp/initialize \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' \
  -D- | grep -i mcp-session-id | cut -d' ' -f2 | tr -d '\r')

# ツール実行
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":"crawl","method":"tools/call","params":{"name":"crawl_url","arguments":{"url":"https://example.com"}}}'
```

## 🔄 Legacy HTTP（SSE実装）

**SSE付きの従来の FastMCP StreamableHTTP プロトコル**

### サーバー起動

```bash
# 方法1: コマンドライン
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8001

# 方法2: 環境変数
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8001
python -m crawl4ai_mcp.server
```

### Claude Desktop設定

```json
{
  "mcpServers": {
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### Legacy HTTP使用例

```bash
curl -X POST "http://127.0.0.1:8001/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

## 🖥️ STDIOプロトコル

### 使用方法

**STDIOトランスポート（デフォルト）:**
```bash
python -m crawl4ai_mcp.server
```

**Claude Desktop設定:**
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
        "CRAWL4AI_LANG": "ja"
      }
    }
  }
}
```

## 📊 プロトコル比較

| 機能 | Pure StreamableHTTP | Legacy HTTP (SSE) | STDIO |
|---------|---------------------|-------------------|-------|
| レスポンス形式 | プレーンJSON | Server-Sent Events | バイナリ |
| 設定の複雑さ | 低（URLのみ） | 低（URLのみ） | 高（プロセス管理） |
| デバッグの容易さ | 高（curl互換） | 中（SSEパーサーが必要） | 低 |
| 独立性 | 高 | 高 | 低 |
| パフォーマンス | 高 | 中 | 高 |

## 🚀 サーバー起動オプション

### 方法1: コマンドライン

```bash
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000
```

### 方法2: 環境変数

```bash
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8000
python -m crawl4ai_mcp.server
```

### 方法3: Docker（利用可能な場合）

```bash
docker run -p 8000:8000 crawl4ai-mcp --transport http --port 8000
```

## 🔗 APIエンドポイント

実行後、HTTP APIは以下を提供します：

- **ベース URL**: `http://127.0.0.1:8000`
- **OpenAPI ドキュメント**: `http://127.0.0.1:8000/docs`
- **ツールエンドポイント**: `http://127.0.0.1:8000/tools/{tool_name}`
- **リソースエンドポイント**: `http://127.0.0.1:8000/resources/{resource_uri}`

すべてのMCPツール（crawl_url、extract_structured_data、process_file等）は、ツールパラメータと一致するJSONペイロードを持つHTTP POSTリクエストでアクセス可能です。

## 🛠️ ツール使用例

### 基本的なWebクローリング

```bash
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

### JavaScript対応の高度なクローリング

```bash
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://spa-example.com",
    "wait_for_js": true,
    "simulate_user": true,
    "timeout": 30
  }'
```

### Google検索

```bash
curl -X POST "http://127.0.0.1:8000/tools/search_google" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "python web scraping",
    "num_results": 10,
    "search_genre": "programming"
  }'
```

### ファイル処理

```bash
curl -X POST "http://127.0.0.1:8000/tools/process_file" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "max_size_mb": 50,
    "include_metadata": true
  }'
```

### YouTubeトランスクリプト抽出

```bash
curl -X POST "http://127.0.0.1:8000/tools/extract_youtube_transcript" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["ja", "en"],
    "include_timestamps": true
  }'
```

## 🔧 アプリケーションとの統合

### Python統合

```python
import requests
import json

# 基本的な使用方法
def crawl_url(url, **kwargs):
    response = requests.post(
        "http://127.0.0.1:8000/tools/crawl_url",
        headers={"Content-Type": "application/json"},
        json={"url": url, **kwargs}
    )
    return response.json()

# 使用例
result = crawl_url("https://example.com", generate_markdown=True)
```

### JavaScript/Node.js統合

```javascript
async function crawlUrl(url, options = {}) {
  const response = await fetch('http://127.0.0.1:8000/tools/crawl_url', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url, ...options })
  });
  
  return await response.json();
}

// 使用例
const result = await crawlUrl('https://example.com', { 
  generate_markdown: true 
});
```

## 🔍 デバッグとモニタリング

### ヘルスチェック

```bash
curl http://127.0.0.1:8000/health
```

### サーバーログ

```bash
# リアルタイムでログを表示
tail -f server.log

# エラーを検索
grep -i error server.log
```

### パフォーマンス監視

```bash
# リクエストを監視
curl -s http://127.0.0.1:8000/stats

# サーバーステータスを確認
curl -s http://127.0.0.1:8000/status
```

## 🔒 セキュリティ考慮事項

### CORS設定

Webアプリケーションの場合、適切なCORSヘッダーが設定されていることを確認：

```python
# CORS設定例
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
}
```

### 認証

本番環境では認証を実装：

```bash
# APIキーを使用した例
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"url": "https://example.com"}'
```

## 📚 追加リソース

- **Pure StreamableHTTP詳細**: [PURE_STREAMABLE_HTTP.md](PURE_STREAMABLE_HTTP.md)
- **HTTPサーバー使用法**: [HTTP_SERVER_USAGE.md](HTTP_SERVER_USAGE.md)
- **Legacy HTTP API**: [HTTP_API_GUIDE.md](HTTP_API_GUIDE.md)
- **設定例**: [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
- **APIリファレンス**: [API_REFERENCE.md](API_REFERENCE.md)

## 🚨 トラブルシューティング

### よくある問題

**ポートが既に使用中:**
```bash
# ポートを使用しているプロセスを検索
lsof -i :8000

# プロセスを終了
kill -9 <PID>
```

**接続拒否:**
- サーバーが実行されているか確認
- ポートとホスト設定を確認
- ファイアウォール設定を確認

**JSON解析エラー:**
- 適切なContent-Typeヘッダーを確認
- JSONペイロード形式を検証
- データ内の特殊文字を確認

詳細なトラブルシューティング情報については、[インストールガイド](INSTALLATION.md#トラブルシューティング)をご覧ください。