# Crawl-MCP: crawl4ai用非公式MCPサーバー

> **⚠️ 重要**: これは優れた[crawl4ai](https://github.com/unclecode/crawl4ai)ライブラリの**非公式**MCPサーバー実装です。  
> **元のcrawl4aiプロジェクトとは無関係**です。

強力なcrawl4aiライブラリを高度なAI機能でラップする包括的なModel Context Protocol（MCP）サーバーです。**あらゆるソース**からコンテンツを抽出・分析：Webページ、PDF、Officeドキュメント、YouTube動画など。任意で有効にできるLLM要約（`auto_summarize`）とディスク保存（`output_path`）により、トークン使用量を抑えます。

## 🌟 主要機能

- **🔍 Google検索連携**: Google演算子を使用した6つの検索ジャンル（ファイル形式と言語のフィルタ）
- **🔍 高度なWebクローリング**: JavaScript対応、深度サイトマッピング、エンティティ抽出
- **🌐 汎用コンテンツ抽出**: Webページ、PDF、Word文書、Excel、PowerPoint、ZIPアーカイブ
- **🤖 AI搭載要約機能（オプトイン）**: `auto_summarize=true` 指定時にスマートなトークン削減（最大88.5%）。指定しない場合、大きなレスポンスは約25000トークンで切り詰められ、全文は `output_path` で保存可能
- **💾 ディスク永続化（token-saver）**: 全文をディスクに保存しメタデータのみ返却。エージェントは必要時にファイルから読み込み、コンテキストを節約
- **🎬 YouTube連携**: APIキー不要で動画内容・要約を抽出
- **⚡ 本格運用対応**: 包括的エラーハンドリング付き19の専門ツール

## 🚀 クイックスタート

### 前提条件（最初に必須）

- Python 3.11 以上（FastMCP が Python 3.11+ を要求）

**Playwright用システム依存関係のインストール:**

**Ubuntu 24.04 LTS（手動セットアップが必要）:**
```bash
# t64ライブラリ移行のため手動セットアップが必要
sudo apt update && sudo apt install -y \
  libnss3 libatk-bridge2.0-0 libxss1 libasound2t64 \
  libgbm1 libgtk-3-0t64 libxshmfence-dev libxrandr2 \
  libxcomposite1 libxcursor1 libxdamage1 libxi6 \
  fonts-noto-color-emoji fonts-unifont python3-venv python3-pip

python3 -m venv venv && source venv/bin/activate
pip install playwright==1.55.0 && playwright install chromium
sudo playwright install-deps
```

**その他のLinux（Chromiumとシステムライブラリをインストール）:**
```bash
uvx --from playwright==1.55.0 playwright install --with-deps chromium
```

**macOS / Windows:**
```bash
uvx --from playwright==1.55.0 playwright install chromium
```

Python環境にPlaywrightがインストール済みの場合は、`python -m playwright install --with-deps chromium`（Linux）または `playwright install chromium`（macOS/Windows）でも構いません。

### インストール

**UVX（推奨 - 最も簡単）:**
```bash
# 上記のシステム準備後 - これだけ！
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

**Docker（本番運用向け）:**
```bash
# リポジトリをクローン
git clone https://github.com/walksoda/crawl-mcp
cd crawl-mcp

# Docker Composeでビルド・実行（STDIOモード）
docker-compose up --build

# またはHTTPモードをポート8000で実行
docker-compose --profile http up --build crawl4ai-mcp-http

# または手動でビルド
docker build -t crawl4ai-mcp .
docker run -it crawl4ai-mcp
```

**Dockerの特徴:**
- 🔧 **マルチブラウザ対応**: Chromium、Firefox、Webkitのヘッドレスブラウザ
- 🐧 **Google Chrome**: 互換性のためChrome Stableを追加同梱
- ⚡ **最適化された性能**: Docker向けに事前設定されたブラウザフラグ
- 🔒 **セキュリティ**: 非rootユーザーで実行
- 📦 **完全な依存関係**: 必要なライブラリをすべて同梱

### Claude Desktopセットアップ

**UVXインストールの場合:**
`claude_desktop_config.json`に追加：

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
      ]
    }
  }
}
```

**Docker HTTPモード:**
```json
{
  "mcpServers": {
    "crawl-mcp": {
      "transport": "http",
      "baseUrl": "http://localhost:8000"
    }
  }
}
```

## 📖 ドキュメント

| トピック | 説明 |
|---------|------|
| **[インストールガイド](docs/ja/INSTALLATION.md)** | 全プラットフォーム向け完全インストール手順 |
| **[APIリファレンス](docs/ja/API_REFERENCE.md)** | 完全ツールドキュメントと使用例 |
| **[設定例](docs/ja/CONFIGURATION_EXAMPLES.md)** | プラットフォーム固有セットアップ設定 |
| **[HTTP統合](docs/ja/HTTP_INTEGRATION.md)** | HTTP APIアクセスと統合方法 |
| **[高度な使用法](docs/ja/ADVANCED_USAGE.md)** | パワーユーザー技術とワークフロー |
| **[開発ガイド](docs/ja/DEVELOPMENT.md)** | 貢献と開発セットアップ |

### 言語別ドキュメント

- **English**: [docs/](docs/) ディレクトリ
- **日本語**: [docs/ja/](docs/ja/) ディレクトリ

## 🛠️ ツール概要

### Webクローリング (3)
- `crawl_url` - JavaScript対応のWebページコンテンツ抽出（リモートの `.html`/`.htm` URLはブラウザでレンダリングし、ドキュメントURLはファイルとして変換）
- `deep_crawl_site` - 深度設定可能なサイト複数ページクローリング
- `crawl_url_with_fallback` - アンチボットサイト向けフォールバック戦略付きクローリング

### データ抽出 (3)
- `intelligent_extract` - LLMを使用したWebページからの特定データ抽出
- `extract_entities` - Webページからエンティティ（メール、電話番号等）を抽出
- `extract_structured_data` - CSSセレクターまたはLLMを使用した構造化データ抽出

### YouTube (4)
- `extract_youtube_transcript` - タイムスタンプ付きYouTubeトランスクリプト抽出（公開日時を `timezone` で変換）
- `batch_extract_youtube_transcripts` - 複数YouTube動画のトランスクリプト抽出（最大3件）
- `get_youtube_video_info` - YouTube動画メタデータ（指定 `timezone` での `published_at` を含む）とトランスクリプト可用性の取得
- `extract_youtube_comments` - ページネーション付きYouTube動画コメント抽出

### 検索 (4)
- `search_google` - ジャンルフィルタ付きGoogle検索（pdf, documents, presentations, spreadsheets, japanese, english）
- `batch_search_google` - 複数Google検索の実行（最大3件）
- `search_and_crawl` - Google検索と上位結果のクローリング
- `get_search_genres` - 利用可能な検索ジャンルの取得

### ファイル処理 (3)
- `process_file` - PDF、Word、Excel、PowerPoint、ZIPをMarkdownに変換
- `get_supported_file_formats` - サポートされているファイル形式と機能の取得
- `enhanced_process_large_content` - チャンキングとBM25フィルタリングによる大容量コンテンツ処理

### バッチ操作 (2)
- `batch_crawl` - フォールバック付きで複数URLを順次クローリング（最大3件）
- `multi_url_crawl` - パターンベース設定による複数URLクローリング（最大5URLパターン）

## 💾 大容量結果のディスク保存（token-saver）

情報収集系のすべてのツールは、取得した全文を直接ディスクへ書き出し、メタデータのみの軽量レスポンスを返す任意パラメータ `output_path` を受け付けます。これにより、巨大なページ・長尺のYouTubeトランスクリプト・バッチ全体を、コンテキスト予算を圧迫せずに取得できます。必要なときだけ保存ファイルから読み出してください。

**動作の仕組み:**
- 単一ファイル系ツール（例: `crawl_url`、`extract_youtube_transcript`）は1つの `.md`（JSON種別のツールは `.json`）を書き出します。絶対ファイルパスを渡してください。拡張子は省略すると自動付与されます。同じパスに通常ファイルが既に存在する場合、`overwrite=true` でない限り拒否されます。
- バッチ系ツール（`batch_crawl`、`multi_url_crawl`、`deep_crawl_site`、`search_and_crawl`、`batch_extract_youtube_transcripts`）は絶対**ディレクトリ**パスを期待し、URLごとに1つの `.md` と `index.json` を書き出します。存在しないパスはディレクトリとして扱われ作成されます（`/tmp/run.v1` のようにドットを含む名前も可）。既に通常ファイルとして存在する場合は拒否されます。`batch_crawl` / `multi_url_crawl` は従来の `list` 戻り値形状を保ち、各成功項目に `output_file` キーを埋め込みます。
- リクエスト辞書系ツール（`search_google`、`batch_search_google`、`search_and_crawl`、`batch_extract_youtube_transcripts`）は永続化キーをリクエスト辞書から直接読み取ります。
- 共通パラメータ: `output_path`（絶対パス。デフォルトは `None`、`""` でも永続化をスキップ）、`include_content_in_response`（既定 `false`。`true` の場合はレスポンスにも内容を含めるが、**`content_limit` / `content_offset` / `max_content_per_page` によるスライスは引き続き適用**）、`overwrite`（既定 `false`）。
- 書き込みはファイルごとにアトミック（一時ファイル + `os.replace`）で、親ディレクトリは自動作成されます。スライスやツール内部の切り詰めの**前**にスライスなしの全文を永続化するため、レスポンスがスライスされてもディスク上のコピーは常に完全です。
- バッチ辞書系ツール（`deep_crawl_site`、`search_and_crawl`、`batch_extract_youtube_transcripts`）は `success=false` の項目について項目単位の永続化をスキップします。これらの項目も `index.json` には `file: null` として現れるため、呼び出し側は試行リストを把握できます（`batch_crawl` / `multi_url_crawl` も失敗項目は保存せず、`index.json` に `output_file: null` として記録します）。

**Markdown単一ファイルの例:**
```json
{
  "tool": "crawl_url",
  "arguments": {
    "url": "https://example.com/long-article",
    "output_path": "/tmp/crawl_out/article.md"
  }
}
```

**JSON構造化抽出（拡張子は自動付与）:**
```json
{
  "tool": "extract_structured_data",
  "arguments": {
    "url": "https://example.com/products",
    "extraction_type": "css",
    "css_selectors": {"price": ".price", "name": "h1"},
    "output_path": "/tmp/crawl_out/products"
  }
}
```

**バッチ・ディレクトリモード:**
```json
{
  "tool": "batch_crawl",
  "arguments": {
    "urls": ["https://a.example", "https://b.example"],
    "output_path": "/tmp/crawl_out/batch_run1"
  }
}
```

永続化された各Markdownファイルは、`url`、`title`、`fetched_at`、`source_tool` を含むYAMLフロントマターブロックで始まるため、成果物は自己記述的です。

## 🎯 一般的な使用例

**コンテンツ研究:**
```bash
search_and_crawl → extract_structured_data → 構造化分析
```

**ドキュメントマイニング:**
```bash
deep_crawl_site → バッチ処理 → 包括的抽出
```

**メディア分析:**
```bash
extract_youtube_transcript → 要約ワークフロー
```

**サイトマッピング:**
```bash
batch_crawl → multi_url_crawl → 包括的データ
```

## 🚨 クイックトラブルシューティング

**インストールの問題:**
1. 上記のPlaywrightインストールコマンドを適切な権限で再実行
2. 開発インストール手法を試行
3. ブラウザ依存関係がインストールされているか確認

**パフォーマンスの問題:**
- JavaScript重要サイトには`wait_for_js: true`を使用
- 読み込みが遅いページではタイムアウトを増加
- 対象を絞った抽出には`extract_structured_data`を使用

**設定の問題:**
- `claude_desktop_config.json`のJSON構文をチェック
- ファイルパスが絶対パスであることを確認
- 設定変更後はClaude Desktopを再起動

## 🏗️ プロジェクト構造

- **元ライブラリ**: [crawl4ai](https://github.com/unclecode/crawl4ai) by unclecode
- **MCPラッパー**: このリポジトリ（walksoda）
- **実装**: 非公式サードパーティ統合

## 📄 ライセンス

このプロジェクトはcrawl4aiライブラリの非公式ラッパーです。基盤機能については元の[crawl4aiライセンス](https://github.com/unclecode/crawl4ai)をご参照ください。

## 🤝 コントリビューション

貢献ガイドラインと開発セットアップ手順については、[開発ガイド](docs/ja/DEVELOPMENT.md)をご覧ください。

## 🔗 関連プロジェクト

- [crawl4ai](https://github.com/unclecode/crawl4ai) - 基盤となるWebクローリングライブラリ
- [Model Context Protocol](https://modelcontextprotocol.io/) - このサーバーが実装する標準
- [Claude Desktop](https://docs.anthropic.com/claude/docs/claude-desktop) - MCPサーバーの主要クライアント
