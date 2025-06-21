# Crawl4AI MCP サーバー

crawl4aiライブラリの機能をModel Context Protocol (MCP)仕様に準拠してラップしたサーバーです。fastmcpフレームワークを使用して実装されています。

## 主な機能

- JavaScriptサポート付きWebクローリング
- 複数の抽出戦略（CSSセレクター、XPath、LLMベース）
- カスタムスキーマによる構造化データ抽出
- コンテンツフィルタリング付きMarkdown生成
- **📄 ファイル処理機能（MarkItDown統合）**
  - PDF、Microsoft Office、ZIP等の各種ファイル形式サポート
  - 自動ファイル形式検出と最適な変換処理
- **📺 YouTube トランスクリプト抽出（youtube-transcript-api v1.1.0+）**
  - 認証不要のシンプルで安定した字幕抽出
  - 自動生成・手動字幕の両方に対応
  - 多言語サポートと優先言語設定
  - タイムスタンプ付きセグメント情報
- **🔍 Google検索統合機能**
  - 31種類の検索ジャンル指定（学術、プログラミング、ニュース等）
  - 検索結果からのタイトル・スニペット自動抽出
  - セーフサーチ原則有効化でセキュリティ確保
  - バッチ検索と結果分析機能
  - 検索+クローリング統合処理
- **📺 YouTube動画バッチ処理機能**
  - 複数動画の一括トランスクリプト抽出
  - 同時処理数制御による効率的な処理
  - 詳細なエラーハンドリングと結果レポート
  - 翻訳機能（利用可能な場合）
- スクリーンショット撮影
- メディア抽出（画像、音声、動画）
- バッチクローリング機能

## セットアップ

### 1. 仮想環境の作成・セットアップ

**Linux/macOS:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup_windows.bat
```

### 2. 手動セットアップ

```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化
source venv/bin/activate  # Linux/macOS
# または
venv\Scripts\activate.bat  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

## 使用方法

### サーバーの起動

```bash
# STDIOトランスポート（デフォルト）
python -m crawl4ai_mcp.server

# HTTPトランスポート
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000

# SSEトランスポート
python -m crawl4ai_mcp.server --transport sse --host 127.0.0.1 --port 8001
```

## 🌐 HTTP API アクセス

このMCPサーバーは複数のHTTPプロトコルをサポートしており、用途に応じて最適な実装を選択できます。

### 🎯 Pure StreamableHTTP（推奨）

**Server-Sent Events (SSE) を使用しない純粋なJSON HTTPプロトコル**

#### サーバー起動
```bash
# 方法1: 起動スクリプト使用
./scripts/start_pure_http_server.sh

# 方法2: 直接起動
python examples/simple_pure_http_server.py --host 127.0.0.1 --port 8000

# 方法3: バックグラウンド起動
nohup python examples/simple_pure_http_server.py --port 8000 > server.log 2>&1 &
```

#### Claude Desktop設定
```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

#### 使用手順
1. **サーバー起動**: `./scripts/start_pure_http_server.sh`
2. **設定適用**: `configs/claude_desktop_config_pure_http.json` を使用
3. **Claude Desktop再起動**: 設定を適用

#### 動作確認
```bash
# ヘルスチェック
curl http://127.0.0.1:8000/health

# 完全テスト
python examples/pure_http_test.py
```

### 🔄 Legacy HTTP（SSE実装）

**従来のFastMCP StreamableHTTPプロトコル（SSE使用）**

#### サーバー起動
```bash
# 方法1: コマンドライン
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8001

# 方法2: 環境変数
export MCP_TRANSPORT=http
export MCP_HOST=127.0.0.1
export MCP_PORT=8001
python -m crawl4ai_mcp.server
```

#### Claude Desktop設定
```json
{
  "mcpServers": {
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### 📊 プロトコル比較

| 特徴 | Pure StreamableHTTP | Legacy HTTP (SSE) | STDIO |
|------|---------------------|-------------------|-------|
| レスポンス形式 | プレーンJSON | Server-Sent Events | バイナリ |
| 設定複雑度 | 低 (URLのみ) | 低 (URLのみ) | 高 (プロセス管理) |
| デバッグ容易性 | 高 (curl可能) | 中 (SSEパーサー必要) | 低 |
| 独立性 | 高 | 高 | 低 |
| パフォーマンス | 高 | 中 | 高 |

### 🚀 HTTP使用例

#### Pure StreamableHTTP
```bash
# 初期化
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

#### Legacy HTTP
```bash
curl -X POST "http://127.0.0.1:8001/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "generate_markdown": true}'
```

### 📚 詳細ドキュメント

- **Pure StreamableHTTP**: [PURE_STREAMABLE_HTTP.md](PURE_STREAMABLE_HTTP.md)
- **HTTP サーバー使用方法**: [HTTP_SERVER_USAGE.md](HTTP_SERVER_USAGE.md)
- **Legacy HTTP API**: [HTTP_API_GUIDE.md](HTTP_API_GUIDE.md)

### Claude Desktop での使用

#### 🎯 Pure StreamableHTTP使用（推奨）

1. **サーバー起動**:
   ```bash
   ./scripts/start_pure_http_server.sh
   ```

2. **設定ファイル適用**:
   - `configs/claude_desktop_config_pure_http.json` をClaude Desktopの設定ディレクトリにコピー
   - または既存の設定に以下を追加:
   ```json
   {
     "mcpServers": {
       "crawl4ai-pure-http": {
         "url": "http://127.0.0.1:8000/mcp"
       }
     }
   }
   ```

3. **Claude Desktop再起動**: 設定を適用

4. **利用開始**: チャットでcrawl4aiツールが利用可能になります

#### 🔄 従来のSTDIO使用

1. `configs/claude_desktop_config.json` をClaude Desktopの設定ディレクトリにコピー
2. Claude Desktopを再起動
3. チャットでcrawl4aiツールが利用可能になります

#### 📂 設定ファイルの場所

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/claude-desktop/claude_desktop_config.json
```

### LLM設定の管理

MCP設定ファイル内でLLMプロバイダーとモデルの設定を管理できます。`claude_desktop_config.json`に以下の設定を追加：

```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "PYTHONPATH": "/path/to/crawl/venv/lib/python3.10/site-packages"
      },
      "llm_config": {
        "default_provider": "openai",
        "default_model": "gpt-4.1",
        "providers": {
          "openai": {
            "api_key": null,
            "api_key_env": "OPENAI_API_KEY",
            "base_url": null,
            "models": ["gpt-4.1", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
          },
          "aoai": {
            "api_key": null,
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "base_url": null,
            "base_url_env": "AZURE_OPENAI_ENDPOINT",
            "api_version": "2025-04-01-preview",
            "models": ["gpt-4.1", "gpt-4.1-nano", "o4-mini", "o3", "o3-mini", "o1", "o1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-35-turbo"]
          },
          "anthropic": {
            "api_key": null,
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": null,
            "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
          },
          "ollama": {
            "api_key": null,
            "api_key_env": null,
            "base_url": "http://localhost:11434",
            "models": ["llama3.3", "qwen2.5"]
          }
        }
      }
    }
  }
}
```

**LLM設定のパラメータ：**

- `default_provider`: デフォルトのLLMプロバイダー
- `default_model`: デフォルトのLLMモデル（2025年対応：GPT-4.1推奨）
- `providers`: 利用可能なプロバイダーの設定
  - `api_key`: APIキーを直接記載（推奨）
  - `api_key_env`: APIキーを格納する環境変数名（オプション）
  - `base_url`: カスタムAPIエンドポイント（Ollamaなど）
  - `base_url_env`: ベースURLの環境変数名（Azure OpenAI用）
  - `api_version`: APIバージョン（Azure OpenAI用：2025-04-01-preview）
  - `models`: 利用可能なモデルのリスト

**プロバイダー別特徴：**

- **openai**: OpenAI公式API（GPT-4.1、GPT-4o対応）
- **aoai**: Azure OpenAI Service（企業向け、最新O-seriesモデル対応）
- **anthropic**: Anthropic Claude（高品質な推論）
- **ollama**: ローカル実行（プライバシー重視）

**APIキー設定方法：**

**方法1: 環境変数を使用（推奨・安全）**
```bash
# OpenAI API キー
export OPENAI_API_KEY="sk-proj-your-actual-openai-api-key-here"

# Anthropic API キー  
export ANTHROPIC_API_KEY="sk-ant-your-actual-anthropic-api-key-here"
```

**方法2: 設定ファイルに直接記載**
```json
{
  "providers": {
    "openai": {
      "api_key": "sk-proj-your-actual-openai-api-key-here",
      "api_key_env": "OPENAI_API_KEY"
    },
    "aoai": {
      "api_key": "your-azure-openai-api-key",
      "api_key_env": "AZURE_OPENAI_API_KEY",
      "base_url": "https://your-resource.openai.azure.com",
      "base_url_env": "AZURE_OPENAI_ENDPOINT"
    }
  }
}
```

**Azure OpenAI環境変数設定例：**
```bash
# Azure OpenAI API キー
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key-here"

# Azure OpenAI エンドポイント
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

### 📄 .envファイルによるAPIキー管理（推奨）

**🔒 セキュリティ強化：環境変数の.envファイル管理**

より安全で管理しやすいAPIキー設定のため、`.env`ファイルを使用することを強く推奨します。

**1. .envファイルの作成**
```bash
# プロジェクトのルートディレクトリに.envファイルを作成
cp .env.example .env
```

**2. .envファイルの編集**
`.env`ファイルを開いて、以下のように実際のAPIキーを設定：

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-proj-your-actual-openai-api-key-here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-api-key-here

# YouTube Transcript API (youtube-transcript-api v1.1.0+)
# No configuration required - works out of the box!

# Google Search API Configuration (if needed)
GOOGLE_SEARCH_API_KEY=your-google-search-api-key-here
GOOGLE_SEARCH_ENGINE_ID=your-custom-search-engine-id
```

**3. .envファイルの自動読み込み**
システムは自動的に以下の場所から.envファイルを検索・読み込みします：
- プロジェクトルートの`.env`
- 現在のディレクトリの`.env`
- 実行時ディレクトリの`.env`

**4. セキュリティ確保**
```bash
# .envファイルのパーミッションを制限
chmod 600 .env

# .gitignoreに.envファイルを追加（既に設定済み）
echo ".env" >> .gitignore
```

**5. APIキー取得方法**

**OpenAI API キー:**
1. https://platform.openai.com/api-keys にアクセス
2. 新しいAPIキーを作成
3. `sk-proj-`で始まるキーをコピー

**Azure OpenAI API キー:**
1. Azure Portalにログイン
2. Azure OpenAI リソースを選択
3. 「キーとエンドポイント」からキーとエンドポイントをコピー

**Anthropic API キー:**
1. https://console.anthropic.com/ にアクセス
2. APIキーを作成
3. `sk-ant-`で始まるキーをコピー

**YouTube動画処理:**
- youtube-transcript-api v1.1.0+により認証不要
- APIキーやOAuth設定は必要ありません

**6. 設定確認**
`get_llm_config_info`ツールを使用して、APIキーが正しく読み込まれているか確認できます。

**⚠️ 重要なセキュリティ注意事項:**
- `.env`ファイルは絶対にGitリポジトリにコミットしないでください
- 実際のAPIキーに置き換えてください（プレースホルダーのままでは認証エラー）
- APIキーは他人と共有しないでください
- 定期的にAPIキーをローテーションしてください

**Windows（WSL）での環境変数設定:**
```bash
# ~/.bashrc または ~/.profile に追加
echo 'export OPENAI_API_KEY="sk-proj-your-actual-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**インテリジェントプロバイダー選択：**

設定されたプロバイダーの中から、有効なAPIキーを持つものが自動選択されます：

1. 指定プロバイダーが有効な場合 → そのプロバイダーを使用
2. デフォルトプロバイダーが有効な場合 → デフォルトを使用
3. どちらも無効な場合 → フォールバック順序で自動選択
   - OpenAI → Azure OpenAI → Anthropic → Ollama

**設定の確認：**

`get_llm_config_info` ツールを使用して現在の設定状況を確認できます：

- 設定されているプロバイダーとモデル
- APIキーの設定状況と有効性
- 利用可能なモデル一覧
- 自動フォールバック状況

**2025年対応モデル：**

- **GPT-4.1**: 最新のGPT-4シリーズ（1Mトークンコンテキスト）
- **O-seriesモデル**: 推論特化型（o4-mini、o3、o1など）
- **GPT-4o**: マルチモーダル対応（テキスト・画像・音声）

### 使用例

```python
# 自動プロバイダー選択でのクロール
result = await intelligent_extract(
    url="https://example.com",
    extraction_goal="技術仕様の抽出",
    # llm_providerとllm_modelを指定しない場合、設定から自動選択
)

# 特定プロバイダーを指定（APIキーが無効な場合は自動フォールバック）
result = await intelligent_extract(
    url="https://example.com",
    extraction_goal="記事の要約",
    llm_provider="aoai",  # Azure OpenAIを指定
    llm_model="gpt-4.1"
)
```

## MCPコンポーネント

### ツール

#### `crawl_url`
基本的なWebクローリングとコンテンツ抽出（**深度クローリング対応**）

**基本パラメータ:**
- `url`: クローリング対象のURL
- `css_selector`: コンテンツ抽出用CSSセレクター（オプション）
- `xpath`: XPath セレクター（オプション）
- `extract_media`: メディアファイル抽出の有無
- `take_screenshot`: スクリーンショット撮影の有無
- `generate_markdown`: Markdown生成の有無
- `wait_for_selector`: 待機する特定要素（オプション）
- `timeout`: リクエストタイムアウト（秒）

**深度クローリングパラメータ:**
- `max_depth`: 最大クローリング深度（Noneで単一ページ）
- `max_pages`: 最大ページ数制限（デフォルト：10）
- `include_external`: 外部ドメインリンクを辿るか
- `crawl_strategy`: クローリング戦略（'bfs', 'dfs', 'best_first'）
- `url_pattern`: URLパターンフィルター（例：'*docs*'）
- `score_threshold`: URL選択の最小スコア閾値

**🚀 高度な機能パラメータ:**
- `content_filter`: コンテンツフィルター（'bm25', 'pruning', 'llm'）
- `filter_query`: フィルタリング用クエリ
- `chunk_content`: 大きなコンテンツの分割
- `chunk_strategy`: 分割戦略（'topic', 'regex', 'sentence'）
- `chunk_size`: 分割サイズ（トークン数）
- `overlap_rate`: 分割間の重複率
- `user_agent`: カスタムユーザーエージェント
- `headers`: カスタムHTTPヘッダー
- `enable_caching`: キャッシュの有効化
- `cache_mode`: キャッシュモード（'enabled', 'disabled', 'bypass'）
- `execute_js`: 実行するJavaScriptコード
- `wait_for_js`: JavaScript完了待機
- `simulate_user`: 人間らしいブラウジング動作の模擬
- `auth_token`: 認証トークン
- `cookies`: カスタムクッキー

#### `extract_structured_data` 🔄
各種戦略を使用した構造化データ抽出（自動プロバイダー選択対応）

**パラメータ:**
- `url`: クローリング対象のURL
- `schema`: 抽出用JSONスキーマ
- `extraction_type`: 抽出タイプ（'css'または'llm'）
- `css_selectors`: 各フィールド用CSSセレクター（オプション）
- `llm_provider`: LLMプロバイダー（自動選択対応）
- `llm_model`: LLMモデル名（自動選択対応）
- `instruction`: カスタム抽出指示 🆕（LLM抽出時の追加指示）

#### `batch_crawl`
複数URLの一括クローリング

**パラメータ:**
- `urls`: クローリング対象URLのリスト
- `config`: 設定パラメータ（オプション）

#### `crawl_url_with_fallback`
複数の戦略を試行するフォールバック付きクローリング

**パラメータ:**
- `url`: クローリング対象のURL
- その他のパラメータは `crawl_url` と同様

#### `deep_crawl_site`
**✨ 新機能**: 指定した深度でサイト全体を再帰的にクローリング

**パラメータ:**
- `url`: 開始URL
- `max_depth`: 最大クローリング深度（推奨：1-3）
- `max_pages`: 最大ページ数制限
- `crawl_strategy`: クローリング戦略（'bfs', 'dfs', 'best_first'）
- `include_external`: 外部ドメインリンクを辿るか
- `url_pattern`: URLパターンフィルター（例：'*docs*', '*blog*'）
- `score_threshold`: URL選択の最小スコア閾値（0.0-1.0）
- `extract_media`: メディアファイル抽出の有無

#### `intelligent_extract`
**🤖 AI機能**: 高度なフィルタリングとAI分析による知的コンテンツ抽出

**パラメータ:**
- `url`: 抽出対象のURL
- `extraction_goal`: 抽出目標の説明（例：「商品情報」、「記事要約」）
- `content_filter`: フィルタータイプ（'bm25', 'pruning', 'llm', 'none'）
- `filter_query`: フィルタリング用クエリ（BM25で必須）
- `chunk_content`: 大きなコンテンツを分割するか
- `use_llm`: LLMによる知的抽出を使用するか
- `llm_provider`: LLMプロバイダー（openai, claude等）
- `llm_model`: 使用する特定のモデル
- `custom_instructions`: カスタム抽出指示

#### `extract_entities`
**📋 エンティティ抽出**: 正規表現パターンによる高速エンティティ抽出

**パラメータ:**
- `url`: 抽出対象のURL
- `entity_types`: 抽出するエンティティタイプのリスト
- `custom_patterns`: カスタム正規表現パターン（オプション）
- `include_context`: 各エンティティの周辺文脈を含めるか
- `deduplicate`: 重複エンティティを除去するか

**利用可能なエンティティタイプ:**
- `emails`: メールアドレス
- `phones`: 電話番号
- `urls`: URL
- `dates`: 日付
- `ips`: IPアドレス
- `social_media`: ソーシャルメディアハンドル（@username, #hashtag）
- `prices`: 価格情報
- `credit_cards`: クレジットカード番号
- `coordinates`: 地理座標

#### `process_file` 🔄
**📄 ファイル処理**: Microsoft MarkItDownを使用した各種ファイル形式の処理・変換（拡張子推論機能付き）

**ファイル処理改善:**
- **拡張子なしファイル対応**: URLパターンからファイルタイプを推論
- **HTML文書認識**: `/html`パターンや`html`キーワードの検出
- **README自動認識**: 拡張子なしREADMEファイルの自動検出
- **エラー処理強化**: ファイルタイプ取得エラーの回避

**パラメータ:**
- `url`: 処理対象ファイルのURL（PDF、Office、ZIP等）
- `max_size_mb`: 最大ファイルサイズ（MB、デフォルト：100MB）
- `extract_all_from_zip`: ZIPアーカイブの全ファイル抽出（デフォルト：True）
- `include_metadata`: メタデータの取得（デフォルト：True）

**サポートファイル形式:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **アーカイブ**: .zip
- **Web/テキスト**: .html, .htm, .txt, .md, .csv, .rtf + **拡張子なし推論対応**
- **eBook**: .epub
- **README等の拡張子なしファイル** 🆕（自動推論）
- **URLパターンベース推論** 🆕（GitHub API等対応）

**対応改善例:**
- `https://example.com/docs/README` → Text File として処理
- `https://example.com/page.html` → HTML Document として処理
- `https://api.github.com/repos/user/repo/contents/file` → 推論による処理

#### `get_llm_config_info` 🆕
**🤖 LLM設定状況確認**: 現在のLLM設定状況の確認とプロバイダー状態の診断

**機能:**
- 設定されているプロバイダーとモデルの一覧
- APIキーの設定状況と有効性チェック
- 利用可能なモデル一覧の表示
- 自動フォールバック機能の動作状況
- 設定の問題診断とトラブルシューティング

**戻り値例:**
```json
{
  "success": true,
  "default_provider": "openai",
  "default_model": "gpt-4.1",
  "providers": {
    "openai": {
      "api_key_available": true,
      "models": ["gpt-4.1", "gpt-4o"]
    },
    "aoai": {
      "api_key_available": false,
      "base_url_required": true
    }
  }
}
```

#### `get_supported_file_formats`
**📋 サポート形式一覧**: ファイル処理でサポートされている形式とその詳細を取得

**戻り値:**
- サポートされている全ファイル形式のリスト
- 各形式の機能と特徴
- 最大ファイルサイズ制限
- 追加機能の説明

#### `extract_youtube_transcript` 🔄
**📺 YouTube動画処理**: youtube-transcript-api v1.1.0+を使用したシンプル字幕抽出

**✅ 2025年対応: 認証不要のyoutube-transcript-api v1.1.0+に移行済み**

**主な特徴:**
- **認証不要**: APIキーやOAuth設定は一切不要
- **安定性向上**: 公式ライブラリによる信頼性の高い字幕取得
- **シンプル設定**: インストール後すぐに使用可能
- **多言語対応**: 自動生成・手動字幕の両方をサポート
- **複数形式対応**: プレーンテキストとタイムスタンプ付きテキスト

**セットアップ要件:**
- **なし**: 追加設定は一切必要ありません
- **自動インストール**: `pip install -r requirements.txt`で自動的にインストール済み

**パラメータ:**
- `url`: YouTube動画のURL
- `languages`: 優先言語リスト（デフォルト：["ja", "en"]）
- `translate_to`: 翻訳先言語（将来実装予定）
- `include_timestamps`: タイムスタンプを含めるか

**戻り値:**
- 字幕テキスト（タイムスタンプ付き/なし）
- セグメント情報（開始・終了時間、継続時間）
- 言語情報（ソース言語、字幕種別）
- 動画メタデータ（タイトル、再生回数、投稿日等）

#### `batch_extract_youtube_transcripts` 🔄
**📺 一括YouTube処理**: youtube-transcript-api v1.1.0+による複数動画の並行字幕抽出

**✅ 安定性向上: 認証不要ライブラリによる信頼性の高いバッチ処理**

**改善点:**
- **認証不要**: APIクォータやレート制限の心配なし
- **エラー処理強化**: 個別動画ごとの詳細なエラー情報
- **処理統計**: 成功/失敗件数、処理時間等の詳細統計
- **シンプル設定**: 追加設定なしですぐに使用可能

**パラメータ:**
- `urls`: YouTube動画URLのリスト
- `languages`: 優先言語リスト
- `translate_to`: 翻訳先言語（将来実装予定）
- `include_timestamps`: タイムスタンプを含めるか
- `max_concurrent`: 最大並行処理数（1-3、APIクォータ考慮）

**戻り値:**
- 個別動画の処理結果リスト
- バッチ処理統計（成功率、エラー詳細）
- 処理方法: `youtube_transcript_api_batch`

#### `get_youtube_video_info` 🔄
**📋 YouTube情報取得**: youtube-transcript-api v1.1.0+による基本動画情報取得

**✅ 機能改善: 認証不要での基本情報取得**

**利用可能情報:**
- **基本情報**: 動画タイトル、投稿者
- **字幕情報**: 利用可能な字幕言語、手動/自動判別
- **言語設定**: 優先言語と利用可能言語
- **アクセス状況**: 字幕取得可能性の確認

**パラメータ:**
- `video_url`: YouTube動画のURL

**戻り値:**
- 基本動画情報（タイトル、投稿者）
- 字幕可用性と言語情報
- 利用可能な字幕言語リスト
- 取得可能性の確認結果

#### `get_youtube_api_setup_guide` 🆕
**🔧 利用ガイド**: youtube-transcript-api v1.1.0+の使用方法

**機能:**
- **使用方法説明**: 認証不要の簡単使用方法
- **機能説明**: 利用可能な機能とパラメータ
- **制限事項**: ライブラリの制限と対処法
- **トラブルシューティング**: よくある問題の解決方法

**戻り値:**
- 現在のライブラリ状況
- 詳細な使用ガイド
- 利用可能機能の説明
- トラブルシューティング情報

#### `search_google`
**🔍 Google検索**: ジャンル指定とメタデータ抽出付きGoogle検索

**パラメータ:**
- `query`: 検索クエリ文字列
- `num_results`: 取得結果数 (1-100、デフォルト: 10)
- `language`: 検索言語 (デフォルト: "en")
- `region`: 検索地域 (デフォルト: "us")
- `search_genre`: コンテンツジャンルフィルター（オプション）
- `safe_search`: セーフサーチ有効（セキュリティのため常にTrue）

**機能:**
- 検索結果からのタイトル・スニペット自動抽出
- 31種類の検索ジャンルによるコンテンツフィルタリング
- URL分類とドメイン分析
- セーフサーチをデフォルトで強制有効

#### `batch_search_google`
**🔍 バッチGoogle検索**: 複数Google検索と包括的分析

**パラメータ:**
- `queries`: 検索クエリのリスト
- `num_results_per_query`: クエリ毎の結果数 (1-100、デフォルト: 10)
- `max_concurrent`: 最大並行検索数 (1-5、デフォルト: 3)
- `language`: 検索言語 (デフォルト: "en")
- `region`: 検索地域 (デフォルト: "us")
- `search_genre`: コンテンツジャンルフィルター（オプション）

**戻り値:**
- 各クエリの個別検索結果
- クエリ横断分析と統計
- ドメイン分布と結果タイプ分析

#### `search_and_crawl`
**🔍 検索+クローリング統合**: Google検索後に上位結果を自動クローリング

**パラメータ:**
- `search_query`: Google検索クエリ
- `num_search_results`: 検索結果数 (1-20、デフォルト: 5)
- `crawl_top_results`: クローリング対象の上位結果数 (1-10、デフォルト: 3)
- `extract_media`: クローリングページからのメディア抽出
- `generate_markdown`: Markdownコンテンツ生成
- `search_genre`: コンテンツジャンルフィルター（オプション）

**戻り値:**
- 完全な検索メタデータとクローリングコンテンツ
- 成功率と処理統計
- 検索・クローリング結果の統合分析

#### `get_search_genres`
**📋 検索ジャンル一覧**: 利用可能な検索ジャンルとその詳細説明を取得

**戻り値:**
- 31種類の検索ジャンルと詳細説明
- カテゴリ別ジャンルリスト（学術、技術、ニュース等）
- 各ジャンルタイプの使用例

### リソース

#### `uri://crawl4ai/config`
デフォルトのクローラー設定オプションを提供

#### `uri://crawl4ai/examples`
使用例とサンプルリクエストを提供

### プロンプト

#### `crawl_website_prompt`
ウェブサイトクローリング用のガイド付きプロンプト

#### `analyze_crawl_results_prompt`
クローリング結果の分析用プロンプト

#### `batch_crawl_setup_prompt`
バッチクローリングのセットアップ用プロンプト

## 抽出戦略

### CSSセレクター
```python
{
    "extraction_type": "css",
    "css_selectors": {
        "title": "h1.title",
        "price": ".price",
        "description": ".description"
    }
}
```

### XPath
```python
{
    "extraction_type": "xpath",
    "xpath_expressions": {
        "title": "//h1[@class='title']/text()",
        "links": "//a/@href"
    }
}
```

### LLMベース抽出
```python
{
    "extraction_type": "llm",
    "schema": {
        "product_name": "商品名",
        "price": "価格",
        "availability": "在庫状況"
    },
    "llm_provider": "openai",
    "llm_model": "gpt-3.5-turbo"
}
```

## 深度クローリング使用例

### 基本的な深度クローリング
```json
{
    "url": "https://docs.example.com",
    "max_depth": 2,
    "max_pages": 20,
    "crawl_strategy": "bfs",
    "include_external": false
}
```

### 特定パターンでフィルタリング
```json
{
    "url": "https://blog.example.com",
    "max_depth": 3,
    "max_pages": 50,
    "crawl_strategy": "best_first",
    "url_pattern": "*2024*",
    "score_threshold": 0.5
}
```

### クローリング戦略の説明
- **BFS (幅優先探索)**: 全体的な網羅性重視
- **DFS (深度優先探索)**: 特定パスの深掘り重視
- **Best First**: スコアリング based で関連性の高いページ優先

## 🚀 高度な機能の使用例

### 🔍 Google検索機能の使用例

#### 基本Google検索
```json
{
    "query": "python 機械学習 チュートリアル",
    "num_results": 10,
    "language": "ja",
    "region": "jp"
}
```

#### ジャンル指定検索
```json
{
    "query": "機械学習 研究論文",
    "num_results": 15,
    "search_genre": "academic",
    "language": "ja"
}
```

#### バッチ検索と分析
```json
{
    "queries": [
        "Pythonプログラミング チュートリアル",
        "ウェブ開発 ガイド",
        "データサイエンス 入門"
    ],
    "num_results_per_query": 5,
    "max_concurrent": 3,
    "search_genre": "education"
}
```

#### 検索+クローリング統合
```json
{
    "search_query": "Python 公式ドキュメント",
    "num_search_results": 10,
    "crawl_top_results": 5,
    "extract_media": false,
    "generate_markdown": true,
    "search_genre": "documentation"
}
```

### AI駆動コンテンツ抽出
```json
{
    "url": "https://example-store.com/product/123",
    "extraction_goal": "商品の詳細情報と価格",
    "content_filter": "bm25",
    "filter_query": "商品 価格 仕様 レビュー",
    "use_llm": true,
    "llm_provider": "openai",
    "custom_instructions": "商品名、価格、主要な仕様、ユーザーレビューの要約を抽出してください"
}
```

### エンティティ抽出
```json
{
    "url": "https://company.com/contact",
    "entity_types": ["emails", "phones", "social_media"],
    "include_context": true,
    "deduplicate": true
}
```

### 高度なコンテンツフィルタリング
```json
{
    "url": "https://news.example.com",
    "content_filter": "pruning",
    "chunk_content": true,
    "chunk_strategy": "topic",
    "chunk_size": 800,
    "overlap_rate": 0.15
}
```

### 認証が必要なサイトのクローリング
```json
{
    "url": "https://private.example.com/data",
    "auth_token": "Bearer your-token-here",
    "cookies": {"session_id": "abc123"},
    "headers": {"X-API-Key": "your-api-key"}
}
```

### JavaScript実行とカスタムブラウザ設定
```json
{
    "url": "https://spa.example.com",
    "execute_js": "document.querySelector('#load-more').click();",
    "wait_for_js": true,
    "user_agent": "Mozilla/5.0 (custom agent)",
    "simulate_user": true
}
```

## 📄 ファイル処理機能の使用例

### PDF文書の処理
```json
{
    "url": "https://example.com/document.pdf",
    "max_size_mb": 50,
    "include_metadata": true
}
```

### Microsoft Office文書の処理
```json
{
    "url": "https://example.com/report.docx",
    "max_size_mb": 25,
    "include_metadata": true
}
```

### ZIPアーカイブの一括処理
```json
{
    "url": "https://example.com/documents.zip",
    "max_size_mb": 100,
    "extract_all_from_zip": true,
    "include_metadata": true
}
```

### 自動ファイル検出とクローリング統合
crawl_urlツールは自動的にファイル形式を検出し、適切な処理方法を選択します：
```json
{
    "url": "https://example.com/mixed-content.pdf",
    "generate_markdown": true
}
```

## 📺 YouTube動画処理機能の使用例

**✅ 2025年対応: youtube-transcript-api v1.1.0+による安定した字幕抽出**

### 基本的なトランスクリプト抽出
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["ja", "en"],
    "include_timestamps": true,
    "include_metadata": true
}
```

### 自動翻訳機能
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["en"],
    "translate_to": "ja",
    "include_timestamps": false
}
```

### 複数動画の一括処理
```json
{
    "urls": [
        "https://www.youtube.com/watch?v=VIDEO_ID1",
        "https://www.youtube.com/watch?v=VIDEO_ID2",
        "https://youtu.be/VIDEO_ID3"
    ],
    "languages": ["ja", "en"],
    "max_concurrent": 3
}
```

### 自動YouTube検出とクローリング統合
crawl_urlツールは自動的にYouTube URLを検出し、トランスクリプト抽出を実行：
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "generate_markdown": true
}
```

### 動画情報の事前確認
```json
{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

## 🎯 コンテンツフィルターの説明

### BM25フィルター
- **用途**: 検索クエリに基づく関連性でコンテンツをランク付け
- **設定**: `filter_query` で検索語句を指定
- **利点**: 高速で正確な関連コンテンツ抽出

### プルーニングフィルター
- **用途**: 低品質や繰り返しコンテンツの除去
- **設定**: 自動的に閾値ベースでフィルタリング
- **利点**: ノイズの多いサイトでもクリーンな結果

### LLMフィルター
- **用途**: AI による知的コンテンツ選別
- **設定**: `custom_instructions` で詳細な指示
- **利点**: 最も高度で柔軟なフィルタリング

## プロジェクト構造

```
crawl/
├── crawl4ai_mcp/
│   ├── __init__.py              # パッケージ初期化
│   ├── server.py                # メインMCPサーバー
│   ├── strategies.py            # 追加の抽出戦略
│   ├── file_processor.py        # MarkItDownファイル処理モジュール
│   ├── google_search_processor.py  # Google検索処理モジュール
│   ├── youtube_processor.py     # YouTubeトランスクリプト処理モジュール（認証不要）
│   └── suppress_output.py       # 出力抑制ユーティリティ
├── requirements.txt             # Python依存関係
├── setup.sh                     # Linux/macOSセットアップスクリプト
├── setup_windows.bat            # Windowsセットアップスクリプト
├── run_server.sh                # サーバー起動スクリプト
├── claude_desktop_config_windows.json  # Claude Desktop設定（Windows用）
├── claude_desktop_config_script.json   # Claude Desktop設定（スクリプト用）
├── claude_desktop_config.json          # Claude Desktop設定（基本）
├── setup_instructions_ja.md     # セットアップ手順（詳細）
├── troubleshooting_ja.md        # トラブルシューティング
├── README.md                    # 英語版README
└── README_ja.md                 # 日本語版README
```

## 依存関係

- `crawl4ai>=0.3.0` - Webクローリングライブラリ
- `fastmcp>=0.1.0` - MCPサーバーフレームワーク
- `pydantic>=2.0.0` - データ検証
- `markitdown>=0.0.1a2` - ファイル処理・変換（Microsoft製）
- `googlesearch-python>=1.3.0` - Google検索機能
- `aiohttp>=3.8.0` - メタデータ抽出用非同期HTTPクライアント
- `beautifulsoup4>=4.12.0` - タイトル・スニペット抽出用HTMLパーサー
- `youtube-transcript-api>=1.1.0` - YouTubeトランスクリプト抽出（認証不要）
- `asyncio` - 非同期処理
- `typing-extensions` - 型ヒント拡張

**✅ YouTube機能の改善点:**
- youtube-transcript-api v1.1.0+により認証不要で安定動作
- APIキーやOAuth設定は一切不要
- すぐに使用開始可能

## エラーハンドリング

サーバーは適切なエラーハンドリングと入力検証を実装しており、以下の場合にエラーレスポンスを返します：

- 無効なURL
- ネットワークエラー
- タイムアウト
- 抽出戦略の設定エラー
- スキーマ検証エラー

## 最新の改善点

### JSON出力の問題解決
crawl4aiの冗長な出力がMCPのJSON通信を妨害する問題を解決しました：

- **出力抑制機能**: `suppress_stdout_stderr` コンテキストマネージャーでcrawl4ai操作を完全にラップ
- **ログ設定**: crawl4ai関連の全ログレベルをCRITICALに設定し、プロパゲーションを無効化
- **起動スクリプト改良**: stderrを `/dev/null` にリダイレクトして残りの出力を抑制
- **設定の統一**: 全クローリング関数で `verbose=False` と `log_console=False` を適用

これにより、Claude DesktopでのJSON解析エラー（`Unexpected token '|'` など）が解決されます。

## トラブルシューティング

詳細なトラブルシューティング情報は `troubleshooting_ja.md` を参照してください。

## ライセンス

MIT License

## 開発者向け情報

このMCPサーバーはModel Context Protocol仕様に準拠しており、互換性のあるMCPクライアントから使用できます。FastMCPフレームワークを使用して構築されているため、簡単に拡張や変更が可能です。