# 🌐 Crawl4AI MCP HTTPサーバー 完全ガイド

## 📋 概要

Crawl4AI MCP サーバーは、FastMCPのStreamableHTTPプロトコルを使用してHTTPアクセス可能なAPIサーバーとして動作できます。これにより、標準的なHTTPクライアントから直接MCPツールやプロンプトにアクセスできます。

## 🚀 サーバー起動方法

### 方法1: 専用スクリプト使用（推奨）

```bash
# シンプル起動（ローカルホストのみ）
./run_http_server.sh

# 外部からアクセス可能で起動
./run_http_server.sh --external

# カスタムポートで起動
./run_http_server.sh --port 8080

# デバッグモードで起動
./run_http_server.sh --log-level DEBUG
```

### 方法2: Pythonスクリプト直接実行

```bash
# 基本起動
python run_http_server.py

# カスタム設定
python run_http_server.py --host 0.0.0.0 --port 8000 --log-level INFO
```

### 方法3: MCPサーバー直接実行

```bash
# HTTPモード
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000

# 外部アクセス可能
python -m crawl4ai_mcp.server --transport http --host 0.0.0.0 --port 8000
```

## 🔌 HTTPエンドポイント

### 基本情報

- **ベースURL**: `http://127.0.0.1:8000` (デフォルト)
- **プロトコル**: HTTP/1.1
- **コンテンツタイプ**: `application/json`
- **認証**: 不要

### 主要エンドポイント

| エンドポイント | メソッド | 説明 |
|-------------|---------|------|
| `/` | GET | サーバー情報取得 |
| `/mcp/tools` | POST | 利用可能なツール一覧 |
| `/mcp/tools/{tool_name}` | POST | 特定ツールの実行 |
| `/mcp/prompts` | POST | 利用可能なプロンプト一覧 |
| `/mcp/prompts/{prompt_name}` | POST | 特定プロンプトの実行 |
| `/mcp/resources` | POST | 利用可能なリソース一覧 |
| `/mcp/resources/{resource_uri}` | POST | 特定リソースの取得 |

## 🛠️ 主要ツール一覧

### Webクローリング
- `crawl_url` - Webページのクローリングとコンテンツ抽出
- `deep_crawl_site` - サイト全体の深度クローリング
- `extract_structured_data` - 構造化データの抽出
- `intelligent_extract` - AI駆動のコンテンツ分析

### YouTube処理
- `extract_youtube_transcript` - YouTube動画の字幕抽出
- `batch_extract_youtube_transcripts` - 複数動画の一括処理
- `get_youtube_video_info` - 動画情報の取得

### ファイル処理
- `process_file` - PDF、Office、ZIP等のファイル処理
- `get_supported_file_formats` - サポート形式の確認

### 検索機能
- `search_google` - Google検索の実行
- `batch_search_google` - 複数クエリの一括検索
- `search_and_crawl` - 検索結果のクローリング

### 設定・情報
- `get_llm_config_info` - LLM設定情報の取得

## 📝 使用例

### 1. サーバー情報取得

```bash
curl -X GET http://127.0.0.1:8000/
```

```json
{
  "name": "crawl4ai-mcp",
  "version": "1.0.0",
  "capabilities": {
    "tools": true,
    "prompts": true,
    "resources": true
  }
}
```

### 2. ツール一覧取得

```bash
curl -X POST http://127.0.0.1:8000/mcp/tools \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 3. Webページクローリング

```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/crawl_url \
  -H "Content-Type: application/json" \
  -d '{
    "arguments": {
      "url": "https://example.com",
      "generate_markdown": true,
      "extract_media": false
    }
  }'
```

### 4. YouTube字幕抽出

```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/extract_youtube_transcript \
  -H "Content-Type: application/json" \
  -d '{
    "arguments": {
      "url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "languages": ["ja", "en"],
      "include_timestamps": true
    }
  }'
```

### 5. Google検索

```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/search_google \
  -H "Content-Type: application/json" \
  -d '{
    "arguments": {
      "query": "Python programming tutorial",
      "num_results": 5,
      "language": "en"
    }
  }'
```

### 6. ファイル処理

```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/process_file \
  -H "Content-Type: application/json" \
  -d '{
    "arguments": {
      "url": "https://example.com/document.pdf",
      "max_size_mb": 50
    }
  }'
```

## 🐍 Pythonクライアント例

### 基本的なクライアント

```python
import aiohttp
import asyncio
import json

class Crawl4AIMCPClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_tool(self, tool_name, arguments):
        async with self.session.post(
            f"{self.base_url}/mcp/tools/{tool_name}",
            json={"arguments": arguments}
        ) as resp:
            return await resp.json()

# 使用例
async def main():
    async with Crawl4AIMCPClient() as client:
        # Webページクローリング
        result = await client.call_tool("crawl_url", {
            "url": "https://example.com",
            "generate_markdown": True
        })
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # YouTube字幕抽出
        result = await client.call_tool("extract_youtube_transcript", {
            "url": "https://www.youtube.com/watch?v=VIDEO_ID",
            "languages": ["ja", "en"]
        })
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
```

### 高度なクライアント例

```python
import aiohttp
import asyncio
from typing import Dict, Any, List

class AdvancedCrawl4AIMCPClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """利用可能なツール一覧を取得"""
        async with self.session.post(f"{self.base_url}/mcp/tools") as resp:
            data = await resp.json()
            return data.get('tools', [])
    
    async def crawl_webpage(self, url: str, **kwargs) -> Dict[str, Any]:
        """Webページをクローリング"""
        arguments = {"url": url, **kwargs}
        return await self.call_tool("crawl_url", arguments)
    
    async def extract_youtube_transcript(self, url: str, languages=None, **kwargs) -> Dict[str, Any]:
        """YouTube字幕を抽出"""
        if languages is None:
            languages = ["ja", "en"]
        arguments = {"url": url, "languages": languages, **kwargs}
        return await self.call_tool("extract_youtube_transcript", arguments)
    
    async def search_google(self, query: str, num_results=5, **kwargs) -> Dict[str, Any]:
        """Google検索を実行"""
        arguments = {"query": query, "num_results": num_results, **kwargs}
        return await self.call_tool("search_google", arguments)
    
    async def process_file(self, url: str, **kwargs) -> Dict[str, Any]:
        """ファイルを処理"""
        arguments = {"url": url, **kwargs}
        return await self.call_tool("process_file", arguments)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """ツールを実行"""
        async with self.session.post(
            f"{self.base_url}/mcp/tools/{tool_name}",
            json={"arguments": arguments}
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

# 使用例
async def demo():
    async with AdvancedCrawl4AIMCPClient() as client:
        # 利用可能なツールを確認
        tools = await client.get_tools()
        print(f"利用可能なツール: {len(tools)}個")
        
        # Webページクローリング
        crawl_result = await client.crawl_webpage(
            "https://httpbin.org/json",
            generate_markdown=True
        )
        
        # YouTube字幕抽出
        youtube_result = await client.extract_youtube_transcript(
            "https://www.youtube.com/watch?v=UJnPNIoeqzI",
            languages=["ja", "en"],
            include_timestamps=True
        )
        
        # Google検索
        search_result = await client.search_google(
            "Python programming tutorial",
            num_results=3
        )
        
        print("🎉 すべての操作が完了しました！")

if __name__ == "__main__":
    asyncio.run(demo())
```

## 🔍 レスポンス形式

### 成功レスポンス

```json
{
  "isError": false,
  "content": {
    // ツール固有のレスポンスデータ
  }
}
```

### エラーレスポンス

```json
{
  "isError": true,
  "content": "エラーメッセージの詳細"
}
```

## 🧪 テスト

### 自動テストスイート実行

```bash
# サーバーが起動している状態で
python test_http_server.py

# カスタムURL指定
python test_http_server.py --url http://192.168.1.100:8000
```

### 手動テスト

```bash
# 基本接続テスト
curl -X GET http://127.0.0.1:8000/

# ツール一覧取得
curl -X POST http://127.0.0.1:8000/mcp/tools

# 簡単なクローリングテスト
curl -X POST http://127.0.0.1:8000/mcp/tools/crawl_url \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"url": "https://httpbin.org/json"}}'
```

## ⚙️ 設定とトラブルシューティング

### 環境変数

HTTP APIサーバーは、通常のMCPサーバーと同じ環境変数を使用します：

```bash
# LLM設定
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key

# ログレベル
FASTMCP_LOG_LEVEL=INFO
```

### 一般的な問題と解決方法

#### 1. ポートが使用中

```bash
# 別のポートを使用
./run_http_server.sh --port 8080
```

#### 2. 外部アクセスできない

```bash
# 外部アクセス許可
./run_http_server.sh --external

# ファイアウォール設定確認
sudo ufw status
```

#### 3. 依存関係エラー

```bash
# 依存関係再インストール
pip install -r requirements.txt
```

#### 4. レスポンスが遅い

- LLM機能を使用する場合は、APIキーが正しく設定されているか確認
- ネットワーク接続を確認
- ログレベルをDEBUGに設定して詳細を確認

## 🔐 セキュリティ考慮事項

### 本番環境での注意点

1. **外部公開時の注意**
   - 適切なファイアウォール設定
   - リバースプロキシの使用（nginx等）
   - HTTPS化の検討

2. **アクセス制御**
   - 必要に応じて認証機能の追加
   - レート制限の実装
   - IPアドレス制限

3. **ログ管理**
   - 適切なログローテーション
   - 機密情報のマスキング

## 🎯 パフォーマンス最適化

### 推奨設定

```bash
# 高負荷環境での起動例
python run_http_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level WARNING
```

### モニタリング

- CPU使用率の監視
- メモリ使用量の監視
- レスポンス時間の測定
- エラー率の追跡

## 📚 関連リソース

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Crawl4AI Documentation](https://crawl4ai.com/)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)

## 🤝 サポート

問題や質問がある場合は、以下を確認してください：

1. **ログの確認**: デバッグモードでサーバーを起動
2. **テストスイート実行**: `python test_http_server.py`
3. **環境の確認**: 依存関係とPythonバージョン
4. **ドキュメント**: この文書と関連リンク

---

このガイドを使用して、Crawl4AI MCP HTTPサーバーを効果的に活用してください！