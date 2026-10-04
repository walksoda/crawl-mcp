# Crawl-MCP: crawl4ai用非公式MCPサーバー

> [crawl4ai](https://github.com/unclecode/crawl4ai)ライブラリを使った**非公式**のMCPサーバーです。元のcrawl4aiプロジェクトとは関係ありません。

crawl4aiをラップしたModel Context Protocol（MCP）サーバーで、Webページ、PDF、Officeドキュメント、YouTube動画などからコンテンツを取り出します。レスポンスのトークン数は、任意で有効にするLLM要約（`auto_summarize`）とディスク保存（`output_path`）で抑えられます。

## 主な機能

- Google Custom Search、Brave、Exa でWeb検索でき、エンジン間で自動的にフォールバックします。結果は6つのジャンル（ファイル形式と言語）で絞り込めます
- JavaScriptを実行するWebクローリング、指定した深さでのサイト巡回、エンティティ抽出
- Webページ、PDF、Word、Excel、PowerPoint、ZIPアーカイブからのコンテンツ抽出
- LLMによる要約は `auto_summarize=true` を指定したときだけ動きます。指定しない場合、大きなレスポンスは約25000トークンで切り詰められ、全文は `output_path` で保存できます
- ディスク保存（token-saver）。`output_path` を指定すると全文をディスクに書き出し、レスポンスはメタデータだけになります。エージェントは必要になったときにファイルを読めばよく、コンテキストを消費しません
- YouTube動画のトランスクリプトや情報を、APIキーなしで取得します
- ツールは全部で21個あります

## クイックスタート

### 前提条件

- Python 3.11 以上（FastMCP が Python 3.11+ を要求）

先にPlaywright用のシステム依存パッケージを入れます。

Ubuntu 24.04 LTSでは手動のセットアップが必要です。
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

その他のLinuxでは、Chromiumとシステムライブラリをまとめて入れます。
```bash
uvx --from playwright==1.55.0 playwright install --with-deps chromium
```

macOSとWindowsの場合:
```bash
uvx --from playwright==1.55.0 playwright install chromium
```

Python環境にPlaywrightがインストール済みの場合は、`python -m playwright install --with-deps chromium`（Linux）または `playwright install chromium`（macOS/Windows）でもかまいません。

### インストール

UVXを使う方法が一番簡単で、推奨です。
```bash
# 上記のシステム準備後 - これだけ！
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

本番運用ではDockerも使えます。
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

Dockerイメージの内容は次のとおりです。
- Chromium、Firefox、Webkitのヘッドレスブラウザ
- 互換性のためにGoogle Chrome Stableも同梱
- Docker向けのブラウザフラグを設定済み
- 非rootユーザーで実行
- 必要なライブラリはすべてイメージに含まれる

### Claude Desktopセットアップ

UVXでインストールした場合は、`claude_desktop_config.json` に次を追加します。

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

DockerのHTTPモードを使う場合:
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

## ドキュメント

| トピック | 説明 |
|---------|------|
| [インストールガイド](docs/ja/INSTALLATION.md) | 各プラットフォームでのインストール手順 |
| [APIリファレンス](docs/ja/API_REFERENCE.md) | 全ツールのパラメータと使用例 |
| [設定例](docs/ja/CONFIGURATION_EXAMPLES.md) | プラットフォーム別の設定例 |
| [HTTP統合](docs/ja/HTTP_INTEGRATION.md) | HTTP経由での利用方法 |
| [使い方の応用](docs/ja/ADVANCED_USAGE.md) | 応用的な使い方とワークフロー |
| [開発ガイド](docs/ja/DEVELOPMENT.md) | 開発環境の準備とコントリビューション |

### 言語別ドキュメント

- English: [docs/](docs/)
- 日本語: [docs/ja/](docs/ja/)

## ツール一覧

### Webクローリング (3)
- `crawl_url` - JavaScript対応のWebページコンテンツ抽出（リモートの `.html`/`.htm` URLはブラウザでレンダリングし、ドキュメントURLはファイルとして変換）
- `deep_crawl_site` - 深度設定可能なサイト複数ページクローリング（`max_depth` 1〜2、`max_pages` 1〜10）
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

### 検索 (6)
- `search_web` - Web検索（ジャンルでの絞り込みは任意）
- `batch_search_web` - 複数のWeb検索の実行（最大3件）
- `search_and_crawl` - Web検索と上位結果のクローリング
- `get_search_genres` - 利用可能な検索ジャンルとエンジン設定の取得
- `search_google`、`batch_search_google` - `search_web` と `batch_search_web` の非推奨エイリアス

検索ツールを使うには、少なくとも1つのエンジンのAPIキーが要ります。`GOOGLE_SEARCH_API_KEY` と `GOOGLE_SEARCH_ENGINE_ID`（Google Custom Search。新規受付は終了しており、2027-01-01 に提供が終わる）、`BRAVE_SEARCH_API_KEY`、`EXA_API_KEY` のいずれかです。既定のエンジンは `SEARCH_ENGINE` で決めます。`auto` にすると、設定済みのエンジンを `SEARCH_ENGINE_ORDER`（デフォルト `google_custom,brave,exa`）の順に試します。リクエストの `engine` キーを指定すれば、その呼び出しだけエンジンを変えられます。詳しくは[APIリファレンス](docs/ja/API_REFERENCE.md#検索ツール)を参照してください。

`search_web`、`batch_search_web`、`search_and_crawl` の `search_genre` に指定できる値は共通で、`pdf`、`documents`、`presentations`、`spreadsheets`、`japanese`、`english` です。大文字と小文字は区別しません。それ以外の値を渡すと、検索を始める前に `invalid_search_genre` エラーが返ります。Exa はファイル形式で絞り込めないため、`auto` ではファイル形式のジャンルのとき Exa を飛ばします。

`result_type` を使うと、検索に Brave の結果枠を加えられます。`discussions` は掲示板の投稿（掲示板名、回答数、スコア付き）、`faq` はページ内の Q&A の抜粋で、ほかに `videos` と `news` があります。`["web", "discussions", "faq"]` のように複数を指定でき、追加の枠は `extra_results` に入ります。`brave` エンジンが必要です。`discussions`、`faq`、`videos` は Brave への1回のリクエストでまとめて取り、返るのは一部のクエリだけです。`news` は Brave の News Search へ別にリクエストを送ります。

### ファイル処理 (3)
- `process_file` - PDF、Word、Excel、PowerPoint、ZIPをMarkdownに変換
- `get_supported_file_formats` - サポートされているファイル形式と機能の取得
- `enhanced_process_large_content` - チャンキングとBM25フィルタリングによる大容量コンテンツ処理

### バッチ操作 (2)
- `batch_crawl` - フォールバック付きで複数URLを順次クローリング（最大3件）
- `multi_url_crawl` - パターンベース設定による複数URLクローリング（最大5URLパターン、1件ずつ順に処理）

## 大きな結果のディスク保存（token-saver）

情報を取得するツールはすべて、任意パラメータ `output_path` を受け付けます。指定すると取得した全文をディスクに書き出し、レスポンスはメタデータだけになります。巨大なページや長いYouTubeトランスクリプト、バッチ全体の結果も、コンテキストを圧迫せずに取得できます。中身は必要になったときに保存ファイルから読んでください。

- 単一ファイル系のツール（`crawl_url`、`extract_youtube_transcript` など）は `.md` を1つ書き出します（JSONを返すツールは `.json`）。絶対ファイルパスを渡してください。拡張子は省略すると自動で付きます。同じパスに通常ファイルがあると、`overwrite=true` でない限りエラーになります。
- バッチ系のツール（`batch_crawl`、`multi_url_crawl`、`deep_crawl_site`、`search_and_crawl`、`batch_extract_youtube_transcripts`）には絶対**ディレクトリ**パスを渡します。URLごとの `.md` と `index.json` がそこに書き出されます。存在しないパスはディレクトリとして作成します（`/tmp/run.v1` のようにドットを含む名前でもかまいません）。同名の通常ファイルがあるとエラーです。`batch_crawl` と `multi_url_crawl` の戻り値は従来どおり `list` で、成功した項目に `output_file` キーが入ります。
- リクエスト辞書を受け取るツール（`search_google`、`batch_search_google`、`search_and_crawl`、`batch_extract_youtube_transcripts`）では、保存用のキーもリクエスト辞書に入れます。
- 共通のパラメータは3つです。`output_path` は絶対パスで、デフォルトは `None` です（`""` でも保存しません）。`include_content_in_response`（デフォルト `false`）を `true` にするとレスポンスにも内容が入りますが、`content_limit`、`content_offset`、`max_content_per_page` によるスライスはそのままかかります。`overwrite` のデフォルトは `false` です。
- 書き込みはファイル単位でアトミックです（一時ファイルに書いてから `os.replace`）。親ディレクトリは自動で作ります。保存はスライスやツール内部の切り詰めより**前**に行うので、レスポンスが切り詰められてもディスク上のファイルは全文のままです。
- 辞書で結果を返すバッチ系のツール（`deep_crawl_site`、`search_and_crawl`、`batch_extract_youtube_transcripts`）は、`success=false` の項目をファイルに保存しません。それでも `index.json` には `file: null` として載るので、何を試したかは呼び出し側で分かります。`batch_crawl` と `multi_url_crawl` も同じく失敗項目は保存せず、`index.json` に `output_file: null` と記録します。

Markdownを1ファイルに保存する例:
```json
{
  "tool": "crawl_url",
  "arguments": {
    "url": "https://example.com/long-article",
    "output_path": "/tmp/crawl_out/article.md"
  }
}
```

構造化データをJSONで保存する例（拡張子は自動で付きます）:
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

バッチ結果をディレクトリに保存する例:
```json
{
  "tool": "batch_crawl",
  "arguments": {
    "urls": ["https://a.example", "https://b.example"],
    "output_path": "/tmp/crawl_out/batch_run1"
  }
}
```

保存されたMarkdownファイルの先頭には、`url`、`title`、`fetched_at`、`source_tool` を含むYAMLフロントマターが付きます。ファイル単体で取得元が分かります。

## 使い方の例

コンテンツの調査:
```bash
search_and_crawl → extract_structured_data → 構造化分析
```

ドキュメントの収集:
```bash
deep_crawl_site → バッチ処理 → 抽出
```

動画の分析:
```bash
extract_youtube_transcript → 要約ワークフロー
```

サイトの把握:
```bash
batch_crawl → multi_url_crawl → データ集約
```

## トラブルシューティング

インストールがうまくいかない場合:
1. 上のPlaywrightのインストールコマンドを、必要な権限で再実行する
2. 開発用のインストール手順を試す
3. ブラウザの依存パッケージが入っているか確認する

遅い、または内容が取れない場合:
- JavaScriptで描画するサイトには `wait_for_js: true` を指定する
- 読み込みが遅いページではタイムアウトを延ばす
- 必要な部分だけ取るなら `extract_structured_data` を使う

設定の問題:
- `claude_desktop_config.json` のJSON構文を確認する
- ファイルパスが絶対パスになっているか確認する
- 設定を変えたらClaude Desktopを再起動する

## プロジェクトの位置づけ

元のライブラリはunclecodeによる[crawl4ai](https://github.com/unclecode/crawl4ai)です。このリポジトリ（walksoda）はそのMCPラッパーで、サードパーティによる非公式の実装です。

## ライセンス

このプロジェクトはcrawl4aiライブラリの非公式ラッパーです。crawl4ai本体の機能については、元の[crawl4aiのライセンス](https://github.com/unclecode/crawl4ai)を参照してください。

## コントリビューション

開発環境の準備とコントリビューションの手順は[開発ガイド](docs/ja/DEVELOPMENT.md)にあります。

## 関連プロジェクト

- [crawl4ai](https://github.com/unclecode/crawl4ai) - 基盤となるWebクローリングライブラリ
- [Model Context Protocol](https://modelcontextprotocol.io/) - このサーバーが実装する標準
- [Claude Desktop](https://docs.anthropic.com/claude/docs/claude-desktop) - MCPクライアントの一つ
