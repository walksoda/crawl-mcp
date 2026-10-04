# HTTP統合ガイド

このガイドでは、Crawl4AI MCPサーバーをHTTPで起動し、HTTPクライアントからツールを呼び出す方法を説明します。

## 概要

このサーバーは [FastMCP](https://gofastmcp.com) アプリケーションです。トランスポートは `python -m crawl4ai_mcp.server`（または `crawl-mcp` エントリポイント）の `--transport` フラグで選びます。

| `--transport` | エンドポイント | 備考 |
|---------------|----------------|------|
| `stdio`（デフォルト） | stdin/stdout | サーバーをサブプロセスとして起動するMCPクライアント向け |
| `streamable-http`（別名: `http`） | `http://<host>:<port>/mcp` | HTTPで使うならこちら。現行のMCP Streamable HTTPトランスポート |
| `sse` | `http://<host>:<port>/sse` | 旧来のMCP SSEトランスポート。古いクライアント向け |

その他のフラグは `--host`（デフォルト `127.0.0.1`）と `--port`（デフォルト `8000`）です。トランスポート、ホスト、ポートはフラグでしか指定できず、環境変数では設定できません。

HTTPリクエストはすべてMCPのJSON-RPCメッセージです。`/tools/{name}` のようなREST形式のエンドポイントはなく、`/health`、`/docs`、`/stats`、`/status` といったルートも提供していません。

## Streamable HTTP（推奨）

### サーバーの起動

```bash
# 開発用チェックアウト（プロジェクトの仮想環境内で実行）
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000

# インストール済みパッケージ / UVX
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --transport streamable-http --host 127.0.0.1 --port 8000

# バックグラウンド起動
nohup python -m crawl4ai_mcp.server --transport streamable-http --port 8000 > server.log 2>&1 &
```

`--transport http` は `streamable-http` の別名として受け付けます。

### MCPクライアント設定

URLで指定するリモートMCPサーバーに対応したクライアントでは、次のように設定します。

```json
{
  "mcpServers": {
    "crawl4ai-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Claude Code では次のコマンドで追加します。

```bash
claude mcp add --transport http crawl4ai http://127.0.0.1:8000/mcp
```

クライアントはURLに接続するだけなので、サーバーは先に起動しておいてください。

### curlでツールを呼び出す

セッションは `initialize`、`notifications/initialized` 通知、`tools/call` の3段階で進めます。すべてのリクエストに `Accept: application/json, text/event-stream` ヘッダーが必要です。`initialize` 以降のリクエストには、サーバーが返した `mcp-session-id` ヘッダーも付けます。レスポンスはServer-Sent Events形式（`event: message` / `data: {...}`）で返ります。

```bash
BASE=http://127.0.0.1:8000/mcp

# 1. 初期化（セッションIDはレスポンスヘッダー mcp-session-id で返る）
SESSION_ID=$(curl -s -D - -o /dev/null -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}' \
  | grep -i '^mcp-session-id:' | awk '{print $2}' | tr -d '\r')

# 2. 初期化完了を通知
curl -s -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. ツールを呼び出す
curl -s -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"crawl_url","arguments":{"url":"https://example.com","content_limit":2000}}}'
```

ツールの一覧は、同じヘッダーで `{"jsonrpc":"2.0","id":3,"method":"tools/list"}` を送れば取れます。

### ツール引数の例

呼び出しごとに変わるのはJSON-RPCの `params` だけです。`arguments` には各ツールの実際のパラメータ名を使います。

JavaScriptの多いページをクロールする場合:
```json
{"name": "crawl_url", "arguments": {"url": "https://spa-example.com", "wait_for_js": true, "timeout": 60}}
```

ページ全文をディスクに保存し、レスポンスはメタデータだけにする場合:
```json
{"name": "crawl_url", "arguments": {"url": "https://example.com/long-article", "output_path": "/tmp/article.md"}}
```

大きなページは `auto_summarize` を `true` にしない限り要約されません。指定しなければ、約25,000トークンを超えるレスポンスは切り詰められ、対処方法の案内が付きます。`content_limit` / `content_offset` で分割取得するか、`output_path` で全文をファイルに保存してください。

Google検索の場合、検索系のツールは `request` オブジェクトを1つ受け取ります。
```json
{"name": "search_google", "arguments": {"request": {"query": "python web scraping", "num_results": 10}}}
```

ファイルを処理する場合:
```json
{"name": "process_file", "arguments": {"url": "https://example.com/document.pdf", "max_size_mb": 50, "include_metadata": true}}
```

`crawl_url` に渡したリモートの `.html` / `.htm` URLはブラウザでレンダリングされます。ドキュメントURL（PDF、DOCX、XLSX、PPTX、ZIPなど）はファイルとして変換し、ブラウザでのフォールバックは使いません。

YouTubeのトランスクリプトを取得する場合:
```json
{"name": "extract_youtube_transcript", "arguments": {"url": "https://www.youtube.com/watch?v=VIDEO_ID", "languages": ["ja", "en"], "include_timestamps": true, "timezone": "Asia/Tokyo"}}
```

`timezone` にはIANAタイムゾーン名を指定します（デフォルト `UTC`）。公開日時をこのタイムゾーンに変換した結果が `metadata.published_at` に入ります。

各ツールの全パラメータは [APIリファレンス](API_REFERENCE.md) を参照してください。

## Pythonから使う

Pythonからは、HTTPリクエストを手で組み立てるよりMCPクライアントライブラリを使うほうが簡単です。`fastmcp` はこのプロジェクトの依存関係に含まれています。

```python
import asyncio
import json

from fastmcp import Client


async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        tools = await client.list_tools()
        print([tool.name for tool in tools])

        result = await client.call_tool(
            "crawl_url", {"url": "https://example.com", "content_limit": 2000}
        )
        data = json.loads(result.content[0].text)
        print(data["success"], data["title"])


asyncio.run(main())
```

公式の `mcp` Python SDK（`mcp.client.streamable_http.streamablehttp_client`）やTypeScript SDK（`@modelcontextprotocol/sdk` の `StreamableHTTPClientTransport`）も同じように使えます。

## Docker

イメージのデフォルトコマンドは、STDIOでサーバーを起動します。HTTPで使うときはコマンドを上書きし、コンテナの外から届くよう `0.0.0.0` にバインドします。

```bash
docker run -p 8000:8000 crawl4ai-mcp:latest \
  python -m crawl4ai_mcp.server --transport http --host 0.0.0.0 --port 8000

# または docker-compose を使う場合
docker-compose --profile http up crawl4ai-mcp-http
```

## STDIOプロトコル

```bash
python -m crawl4ai_mcp.server
```

STDIOでは改行区切りのJSON-RPCメッセージをやり取りします。このときstdoutはMCPメッセージ専用で、診断出力はすべてstderrに出ます。stdoutに出力するラッパースクリプトを挟むとプロトコルが壊れます。

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

## トランスポートの比較

| 項目 | Streamable HTTP | SSE（旧来） | STDIO |
|------|-----------------|-------------|-------|
| エンドポイント | `/mcp` | `/sse` | stdin/stdout |
| メッセージ形式 | JSON-RPC（レスポンスはSSE形式） | SSE上のJSON-RPC | 改行区切りのJSON-RPC |
| プロセス管理 | サーバーを自分で起動 | サーバーを自分で起動 | クライアントがサーバーを起動 |
| 複数クライアントでの共有 | 可 | 可 | 不可 |

## `examples/` のサンプルサーバー（メンテナンス対象外）

`examples/` には古いデモサーバーがあります。メンテナンスしていないので、使わないでください。

- `examples/simple_pure_http_server.py`（`scripts/start_pure_http_server.sh` が起動するもの）は**モック結果を返します**（"Mock crawl result"）。ツールも2つしか公開せず、実際のクロールは行いません。
- `examples/pure_streamable_http_server.py` と `examples/run_http_server.py` は独自の非標準エンドポイントを使っています。

代わりに、組み込みの `--transport streamable-http` サーバーを使ってください。

## セキュリティ

このサーバーはCORSにも認証にも対応していません。デフォルトでは `127.0.0.1` にバインドします。localhost以外に公開する場合は、TLS、認証、CORSを受け持つリバースプロキシ（nginxやCaddyなど）の背後に置いてください。

## トラブルシューティング

ポートが使用中のときは、使っているプロセスを止めます。
```bash
lsof -i :8000
kill <PID>
```

HTTP 406 Not Acceptable が返るのは、`Accept: application/json, text/event-stream` ヘッダーが付いていないときです。

「Missing session ID」や HTTP 400 が返る場合は、`initialize` で返された `mcp-session-id` ヘッダーを以降のすべてのリクエストに付けてください。

接続を拒否される場合は、次の点を確認します。
- サーバーが起動しているか、ホストとポートが合っているか
- Docker内では `--host 0.0.0.0` でサーバーを起動しているか
- ファイアウォールで止められていないか

その他のトラブルシューティングは [インストールガイド](INSTALLATION.md#トラブルシューティング) を参照してください。

## 関連ドキュメント

- [設定例](CONFIGURATION_EXAMPLES.md)
- [APIリファレンス](API_REFERENCE.md)
- [インストールガイド](INSTALLATION.md)
