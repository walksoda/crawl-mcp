# 設定例

インストール方法やプラットフォーム別のMCPクライアント設定と、サーバーが読み込む環境変数をまとめています。

## UVX（推奨）

Claude Desktop（`claude_desktop_config.json`）の場合は次のとおりです。

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

Claude Code ならコマンドラインから追加できます。

```bash
claude mcp add crawl-mcp -- uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

プロジェクトの `.mcp.json` に書く場合、Claude Code は `type` フィールドを必要とします。

```json
{
  "mcpServers": {
    "crawl-mcp": {
      "type": "stdio",
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

## ローカルのチェックアウト

クライアントには、チェックアウト内の仮想環境にあるPythonを指定します。その環境にパッケージをインストール済み（`pip install -e .`）なら、`PYTHONPATH` も作業ディレクトリも要りません。

Linux と macOS では次のように書きます。

```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"]
    }
  }
}
```

Windows では `python.exe` のパスを指定します。

```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "C:\\path\\to\\crawl-mcp\\venv\\Scripts\\python.exe",
      "args": ["-m", "crawl4ai_mcp.server"]
    }
  }
}
```

インストールせずにソースのまま動かすなら、`PYTHONPATH` にはリポジトリのルートを指定してください（`site-packages` ではありません）。

```json
{
  "mcpServers": {
    "crawl4ai-source": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/crawl-mcp"
      }
    }
  }
}
```

### WSL（Windows Subsystem for Linux）

Windows版のClaude Desktopから、WSL内のサーバーを起動する設定です。

```json
{
  "mcpServers": {
    "crawl4ai-wsl": {
      "command": "wsl",
      "args": [
        "/home/user/crawl-mcp/venv/bin/python",
        "-m",
        "crawl4ai_mcp.server"
      ]
    }
  }
}
```

`env` ブロックの変数はWindows側の `wsl.exe` に設定されるだけで、`WSLENV` に列挙しない限りLinux側のプロセスには届きません。APIキーはリポジトリのルートにある `.env` に書いておけば、サーバーが起動時に読み込みます。

## HTTPトランスポート

トランスポート、ホスト、ポートはコマンドラインのフラグで指定します。環境変数では設定できません。

```bash
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000
```

`--transport http` は `streamable-http` の別名で、エンドポイントは `http://127.0.0.1:8000/mcp` です。先にサーバーを起動してから、クライアントにURLを設定します。

```json
{
  "mcpServers": {
    "crawl4ai-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

ポートを変えるときは、`--port` とURLの両方を書き換えてください。古いクライアント向けには、`--transport sse` で `/sse` に旧来のSSEトランスポートを立てられます。HTTPサーバーは複数のクライアントで共有でき、1つのクライアントにSTDIOとHTTPのエントリを並べて書いてもかまいません。curlやPythonからの使い方は [HTTP統合ガイド](HTTP_INTEGRATION.md) にあります。

## 環境変数

サーバーはリポジトリのルート、またはカレントディレクトリにある `.env` を読み込みます。すでに設定済みの変数は上書きしません。`.env.example` がひな形です。MCPクライアント設定の `env` ブロックに書いても同じように効きます。

| 変数 | デフォルト | 用途 |
|------|------------|------|
| `OPENAI_API_KEY` | なし | OpenAIのAPIキー |
| `ANTHROPIC_API_KEY` | なし | AnthropicのAPIキー |
| `AZURE_OPENAI_API_KEY` | なし | Azure OpenAIのAPIキー |
| `AZURE_OPENAI_ENDPOINT` | なし | Azure OpenAIのエンドポイントURL |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | 要約クライアントがAzure OpenAIで使うAPIバージョン |
| `MCP_LLM_CONFIG` | なし | LLM設定一式のJSON（後述）。組み込みのプロバイダー一覧を置き換える |
| `SEARCH_ENGINE` | `auto` | 既定の検索エンジン。`auto`、`google_custom`、`brave`、`exa` のいずれか。リクエストの `engine` キーで1回ごとに上書きできる |
| `SEARCH_ENGINE_ORDER` | `google_custom,brave,exa` | `auto` が、APIキーのあるエンジンを試す順番。未知の名前は無視する |
| `GOOGLE_SEARCH_API_KEY` | なし | Google Custom Search APIのキー。新規受付は終了しており、2027-01-01 に提供が終わる |
| `GOOGLE_SEARCH_ENGINE_ID` | なし | カスタム検索エンジンのID |
| `BRAVE_SEARCH_API_KEY` | なし | Brave Search APIのキー |
| `EXA_API_KEY` | なし | ExaのAPIキー |
| `CUSTOM_SEARCH_API_RPM` | `100` | Custom Search APIの1分あたりのリクエスト数。サーバー側で制限する。Brave と Exa は各サービスが制限し、HTTP 429 を返す。そのとき `auto` は次のエンジンへ進む |
| `GOOGLE_SEARCH_MODE` | なし | `SEARCH_ENGINE` が未設定のときだけ読む。`custom_search_only` は `google_custom` として扱う。`hybrid` と `googlesearch_only` は `auto` として扱う（googlesearch-python は削除した） |
| `CRAWL4AI_BROWSER_TYPE` | 未設定 | `chromium`、`firefox`、`webkit` のどれかを指定するとそのブラウザだけを使う。未設定ならWebKit、Chromiumの順に試す。検出回避ブラウザは常にChromium |
| `CRAWL4AI_CACHE_TTL_SECONDS` | `3600` | 取得したページを新しいとみなす秒数。過ぎると、次にそのURLをクロールしたときにキャッシュを使わず取り直す。`0` なら期限なしで、キャッシュを使い続ける |
| `CRAWL4AI_MCP_LOG_LEVEL` | 未設定（ログなし） | `DEBUG`、`INFO`、`WARNING`、`ERROR` のどれかでログをstderrに出す。`.env` を読み込む前に参照するので、プロセスの環境変数かクライアントの `env` ブロックで設定する |

`FASTMCP_LOG_LEVEL` はこのサーバーでは効かないので、ログを出すときは `CRAWL4AI_MCP_LOG_LEVEL` を使ってください。診断はstderrに出ます。

### LLMの設定

何も設定しなければ、デフォルトのプロバイダーは `openai`、モデルは `gpt-4.1` です。APIキーの変数が設定されたプロバイダーが使える状態になります。指定したプロバイダーにキーがない場合は、`openai`、`aoai`、`anthropic`、`ollama` の順に試します。Ollamaはキー不要で、`http://localhost:11434` に接続します。

デフォルトのプロバイダーやモデルを変えたいときは、`MCP_LLM_CONFIG` にJSON文字列を渡します。

```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/walksoda/crawl-mcp", "crawl-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MCP_LLM_CONFIG": "{\"default_provider\": \"anthropic\", \"default_model\": \"claude-3-5-sonnet-20241022\", \"providers\": {\"anthropic\": {\"api_key_env\": \"ANTHROPIC_API_KEY\", \"models\": [\"claude-3-5-sonnet-20241022\"]}}}"
      }
    }
  }
}
```

各プロバイダーのエントリに書けるのは `api_key` または `api_key_env`、`base_url` または `base_url_env`、`api_version`、`models`、`extra_headers` です。プロバイダー名は `openai`、`anthropic`、`ollama`、`aoai`（Azure OpenAI）の4つです。

`MCP_LLM_CONFIG` が未設定の場合、サーバーはリポジトリのルートかカレントディレクトリにある `claude_desktop_config.json` を探し、`mcpServers.crawl4ai.llm_config` を読みます。形式は `configs/claude_desktop_config.json` が参考になります。ただし中のパスは例なので、使う前に書き換えてください。

APIキーだけを設定する例です。

```json
{
  "mcpServers": {
    "crawl4ai-llm": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here",
        "ANTHROPIC_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## 設定ファイルの場所

Claude Desktopは次のファイルから設定を読みます。

- Windows は `%APPDATA%\Claude\claude_desktop_config.json`
- macOS は `~/Library/Application Support/Claude/claude_desktop_config.json`

ディレクトリがなければ先に作り、ファイルにサーバーのエントリを追加します。Windows PowerShellでは次のように操作します。

```powershell
New-Item -ItemType Directory -Force -Path "$env:APPDATA\Claude"
notepad "$env:APPDATA\Claude\claude_desktop_config.json"
```

macOSでは次のとおりです。

```bash
mkdir -p ~/Library/Application\ Support/Claude
open -e ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

編集後はClaude Desktopを再起動してください。

## ツールパラメータの例

1回のツール呼び出しに渡す `arguments` オブジェクトの例です。全パラメータは [APIリファレンス](API_REFERENCE.md) で確認できます。

基本的なページを `crawl_url` で取得する例です。

```json
{
  "url": "https://example.com",
  "wait_for_js": true,
  "timeout": 30,
  "generate_markdown": true
}
```

JavaScriptを多用するページでは、待機の設定を加えます。

```json
{
  "url": "https://spa.example.com",
  "wait_for_js": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded",
  "generate_markdown": true
}
```

`deep_crawl_site` の `max_depth` は1〜2、`max_pages` は1〜10です。範囲外の値はエラーになります。

```json
{
  "url": "https://example.com",
  "max_depth": 2,
  "max_pages": 10,
  "crawl_strategy": "bfs"
}
```

`intelligent_extract` でBM25による絞り込みを使う場合はこうなります。

```json
{
  "url": "https://example.com/article",
  "extraction_goal": "主な論点をまとめる",
  "content_filter": "bm25",
  "filter_query": "重要なコンテンツキーワード",
  "chunk_content": true
}
```

`process_file` で要約を有効にする例も挙げておきます。

```json
{
  "url": "https://example.com/report.pdf",
  "auto_summarize": true,
  "summary_length": "medium"
}
```

## 設定の確認

```bash
# JSONの構文を確認
python -m json.tool claude_desktop_config.json

# サーバーが起動するか確認
python -m crawl4ai_mcp.server --help

# UVXでのインストールを確認
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --help
```

`--help` には `--transport`（`stdio`、`streamable-http`、`http`、`sse`）、`--host`、`--port` が表示されます。引数が間違っていると、エラーをstderrに出して終了コード2で終わります。

ヘルスチェック用のエンドポイントはありません。起動中のHTTPサーバーを確かめるには、`get_supported_file_formats` のような軽いツールを呼びます。やり方は [HTTP統合ガイド](HTTP_INTEGRATION.md) を参照してください。

## ヒント

1. たいていの場合、UVXがいちばん手軽です。
2. クライアント設定のパスは絶対パスで書きます。
3. APIキーは `env` ブロックか `.env` に置き、共有するファイルには書かないでください。
4. クライアントを再起動する前に、JSONの構文を確認しておきます。
5. STDIOでは、ラッパースクリプトがstdoutに何か出すとMCPの通信が壊れます。そうした出力はstderrに回してください。

## 関連ドキュメント

- [インストールガイド](INSTALLATION.md)
- [HTTP統合ガイド](HTTP_INTEGRATION.md)
- [APIリファレンス](API_REFERENCE.md)
- [使い方の応用](ADVANCED_USAGE.md)
- [開発ガイド](DEVELOPMENT.md)
