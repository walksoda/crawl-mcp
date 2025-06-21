# Pure StreamableHTTP Implementation

このドキュメントでは、Server-Sent Events (SSE) を使用しない純粋なStreamableHTTP実装について説明します。

## 🎯 概要

### 実装の背景

前回の実装では、FastMCPのHTTP機能を使用していましたが、実際にはServer-Sent Events (SSE) 形式のレスポンスが返されており、真のStreamableHTTPとは異なるアプローチでした。

**研究結果に基づく改善:**
- **StreamableHTTP**: 単一エンドポイント、プレーンJSON、必要時のみストリーミング
- **SSE方式**: 常にイベントストリーム形式、複雑なパーサー必要

## 🛠️ 新しい実装

### 主な特徴

1. **純粋なJSON HTTP API**
   - Server-Sent Events (SSE) 不使用
   - プレーンなJSONレスポンス
   - シンプルなクライアント実装

2. **JSON-RPC 2.0準拠**
   - MCPプロトコル完全対応
   - 標準的なリクエスト・レスポンス形式

3. **セッション管理**
   - 初期化プロセス
   - セッションIDベースの認証

4. **単一エンドポイント**
   - `/mcp` - すべてのMCP操作
   - `/mcp/initialize` - 初期化専用（オプション）

### Claude Desktop設定の簡素化

**従来の設定（複雑）:**
```json
{
  "mcpServers": {
    "crawl4ai-http": {
      "command": "python",
      "args": ["server.py", "--host", "127.0.0.1", "--port", "8000"],
      "cwd": "/path/to/project",
      "env": {
        "PYTHONPATH": "/path/to/venv/lib/python3.10/site-packages"
      }
    }
  }
}
```

**新しい設定（シンプル）:**
```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

## 🚀 使用方法

### サーバー起動

```bash
# Pure StreamableHTTP サーバーを起動
python pure_streamable_http_server.py

# ポート指定
python pure_streamable_http_server.py --port 8080

# 外部アクセス許可
python pure_streamable_http_server.py --host 0.0.0.0
```

### テスト実行

```bash
# Pure HTTP テストを実行
python pure_http_test.py
```

## 📡 API エンドポイント

### 1. ヘルスチェック

```http
GET /health
```

**レスポンス:**
```json
{
  "status": "healthy",
  "server": "crawl4ai-mcp-pure-http",
  "version": "1.0.0",
  "protocol": "StreamableHTTP (pure JSON)"
}
```

### 2. MCP初期化

```http
POST /mcp/initialize
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "init",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    }
  }
}
```

**レスポンス:**
```json
{
  "jsonrpc": "2.0",
  "id": "init",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {"listChanged": false},
      "prompts": {"listChanged": false},
      "resources": {"listChanged": false}
    },
    "serverInfo": {
      "name": "crawl4ai-mcp-server",
      "version": "1.0.0"
    }
  }
}
```

**ヘッダー:**
```
mcp-session-id: <セッションID>
```

### 3. ツール一覧取得

```http
POST /mcp
Content-Type: application/json
mcp-session-id: <セッションID>

{
  "jsonrpc": "2.0",
  "id": "tools-list",
  "method": "tools/list"
}
```

### 4. ツール実行

```http
POST /mcp
Content-Type: application/json
mcp-session-id: <セッションID>

{
  "jsonrpc": "2.0",
  "id": "crawl-test",
  "method": "tools/call",
  "params": {
    "name": "crawl_url",
    "arguments": {
      "url": "https://example.com",
      "generate_markdown": true
    }
  }
}
```

## 🔧 クライアント実装例

### Python クライアント

```python
import aiohttp
import json

class PureHTTPClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session_id = None
    
    async def initialize(self):
        """MCP初期化"""
        request_data = {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "my-client",
                    "version": "1.0.0"
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/mcp/initialize",
                json=request_data
            ) as resp:
                if resp.status == 200:
                    self.session_id = resp.headers.get('mcp-session-id')
                    result = await resp.json()
                    return result
    
    async def call_tool(self, tool_name, arguments):
        """ツール実行"""
        request_data = {
            "jsonrpc": "2.0",
            "id": f"call-{tool_name}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "mcp-session-id": self.session_id
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/mcp",
                json=request_data,
                headers=headers
            ) as resp:
                return await resp.json()
```

### JavaScript クライアント

```javascript
class PureHTTPClient {
    constructor(baseUrl = 'http://127.0.0.1:8000') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }
    
    async initialize() {
        const requestData = {
            jsonrpc: '2.0',
            id: 'init',
            method: 'initialize',
            params: {
                protocolVersion: '2024-11-05',
                capabilities: {},
                clientInfo: {
                    name: 'js-client',
                    version: '1.0.0'
                }
            }
        };
        
        const response = await fetch(`${this.baseUrl}/mcp/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (response.ok) {
            this.sessionId = response.headers.get('mcp-session-id');
            return await response.json();
        }
        
        throw new Error(`HTTP ${response.status}`);
    }
    
    async callTool(toolName, arguments) {
        const requestData = {
            jsonrpc: '2.0',
            id: `call-${toolName}`,
            method: 'tools/call',
            params: {
                name: toolName,
                arguments: arguments
            }
        };
        
        const response = await fetch(`${this.baseUrl}/mcp`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'mcp-session-id': this.sessionId
            },
            body: JSON.stringify(requestData)
        });
        
        return await response.json();
    }
}
```

## 🆚 SSE実装との比較

| 特徴 | SSE実装 | Pure StreamableHTTP |
|------|---------|---------------------|
| レスポンス形式 | `event: message\ndata: {...}` | `{...}` |
| パーサー | SSE専用パーサー必要 | 標準JSON |
| クライアント複雑度 | 高 | 低 |
| デバッグ | 困難 | 簡単 |
| 標準準拠 | FastMCP SSE | JSON-RPC 2.0 |
| ブラウザ対応 | EventSource API | fetch API |

## ✅ 利点

1. **シンプルさ**: 標準的なHTTP JSON APIとして動作
2. **デバッグ容易**: curlやPostmanで簡単にテスト可能
3. **クライアント実装簡単**: 特別なSSEパーサー不要
4. **標準準拠**: JSON-RPC 2.0プロトコル完全対応
5. **互換性**: 既存のHTTPツールチェーンと完全互換

## 🔄 従来実装からの移行

### 1. サーバー側
- `pure_streamable_http_server.py` を使用
- SSEパーサー削除
- プレーンJSONレスポンス

### 2. クライアント側
```python
# 従来 (SSE)
async def parse_sse_response(response_text):
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith('data: '):
            return json.loads(line[6:])

# 新実装 (Pure JSON)
result = await resp.json()  # 直接JSON解析
```

## 🧪 テスト

### 完全なワークフローテスト
```bash
python pure_http_test.py
```

### 個別機能テスト
```bash
# ヘルスチェック
curl http://127.0.0.1:8000/health

# 初期化
curl -X POST http://127.0.0.1:8000/mcp/initialize \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl-client","version":"1.0.0"}}}'

# ツール実行 (セッションID必須)
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: YOUR_SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":"crawl","method":"tools/call","params":{"name":"crawl_url","arguments":{"url":"https://httpbin.org/json"}}}'
```

## 📈 今後の拡張

1. **WebSocket対応**: 真のストリーミングが必要な場合
2. **バッチ処理**: 複数リクエストの一括処理
3. **認証拡張**: JWT等の本格的な認証システム
4. **OpenAPI仕様**: 自動ドキュメント生成

この実装により、FastMCPの真のStreamableHTTPプロトコルの精神に基づいた、シンプルで標準準拠のHTTP API を提供しています。