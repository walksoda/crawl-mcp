# 使い方の応用

複数のツールを組み合わせる使い方とワークフローをまとめています。

各例は1回のツール呼び出しに渡す `arguments` オブジェクトです。キーはどれも、記載したツールに実在するパラメータです。全パラメータは [APIリファレンス](API_REFERENCE.md) に載っています。

## Webクロール

### サイトのクロール

`deep_crawl_site` は開始ページからリンクをたどります。

```json
{
  "url": "https://docs.example.com",
  "max_depth": 2,
  "max_pages": 10,
  "crawl_strategy": "best_first",
  "url_pattern": "*docs*",
  "score_threshold": 0.3
}
```

`max_depth` は1か2、`max_pages` は1〜10を指定できます。範囲外の値を渡すと、ページを取得する前にエラーが返ります。`crawl_strategy` には `bfs`、`dfs`、`best_first` のいずれかを使います。`url_pattern` に渡せるパターンは1つだけで、サーバーはそれを1つのフィルタとして扱うため、カンマ区切りで並べても分割されません。

レスポンスの `markdown` には全ページがつながって入り、`crawled_pages` にはページごとの `url`、`title`、`success`、`error` が並びます。`output_path` を指定すると、成功したページが1ページずつ `.md` ファイルになり、`index.json` と一緒に保存されます。このとき `include_content_in_response` を `true` にしなければ、レスポンスにはメタデータだけが残ります。

### JavaScriptを多用するページ

シングルページアプリケーションでは `crawl_url` に `wait_for_js` を付けます。目印になる要素があれば `wait_for_selector` も指定してください。

```json
{
  "url": "https://spa-application.com",
  "wait_for_js": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded"
}
```

### HTMLページとドキュメントURL

`crawl_url` はリモートの `.html` / `.htm` URLを、ほかのWebページと同じくブラウザでレンダリングします。PDF、DOCX、XLSX、PPTX、ZIPなどのドキュメントURLはダウンロードしてファイルとして変換し、ブラウザのフォールバックは通りません。これらのURLは `process_file` に直接渡してもかまいません。

### ボット対策のあるサイト

まず `crawl_url` で検出回避ブラウザを試します。

```json
{
  "url": "https://protected-site.com",
  "use_undetected_browser": true,
  "wait_for_js": true,
  "timeout": 90
}
```

うまくいかなければ、同じURLで `crawl_url_with_fallback` を呼びます。

```json
{
  "url": "https://protected-site.com",
  "wait_for_js": true,
  "timeout": 90
}
```

`crawl_url_with_fallback` は最大7段階を順に試します。静的な高速パス、通常のヘッドレスブラウザ、Chromiumのステルスモード、ユーザー操作のシミュレーション、モバイルのユーザーエージェント、AMP版やRSS版の取得、静的なHTML/JSON抽出の順です。成功した段階はドメインごとにキャッシュされ、次回はその段階から始まります。どの段階で取得できたかは、レスポンスの `extracted_data.fallback_strategy_used` と `fallback_stage` でわかります。HTTP 4xx/5xxが返った段階は失敗として扱います。AMPの段階では、エラーページに見えるページも採用しません。

## LLMによる抽出

### 必要な情報だけを抜き出す

`intelligent_extract` には抽出の目的と、必要なら追加の指示を渡します。

```json
{
  "url": "https://research-paper.com",
  "extraction_goal": "学術論文から手法、結果、結論を抽出",
  "content_filter": "llm",
  "custom_instructions": "定量的結果、統計的有意性、実用上の含意に絞る。参考文献と著者情報は無視する。",
  "use_llm": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4.1"
}
```

`content_filter` は `bm25`、`pruning`、`llm` から選びます。先にBM25でページを絞り込んでからLLMにかけたい場合は、`bm25` と `filter_query` を組み合わせます。

```json
{
  "url": "https://docs.example.com/reference",
  "extraction_goal": "実際に使える技術情報を抽出",
  "content_filter": "bm25",
  "filter_query": "技術ドキュメント APIリファレンス",
  "use_llm": true
}
```

### LLMプロバイダー

使うプロバイダーのAPIキーを、MCPクライアントの設定に書きます。

```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/walksoda/crawl-mcp", "crawl-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "AZURE_OPENAI_API_KEY": "your-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com"
      }
    }
  }
}
```

デフォルトのプロバイダーは `openai`、モデルは `gpt-4.1` です。指定したプロバイダーにAPIキーがなければ、`openai`、`aoai`、`anthropic`、`ollama` の順に切り替えます。`MCP_LLM_CONFIG` の書き方は [設定例](CONFIGURATION_EXAMPLES.md#環境変数) を参照してください。

## 大きなコンテンツ

サーバーは、頼まれない限り大きなページを要約しません。`auto_summarize` を付けずに約25,000トークンを超えた場合、レスポンスは切り詰められ、`recommendations` フィールドが付きます。全文が必要なら方法は2つあります。

1つ目は `content_limit` と `content_offset` による分割取得です。`content_offset` を 20000、40000 と進めながら繰り返し呼び出します。

```json
{
  "url": "https://example.com/very-long-article",
  "content_limit": 20000,
  "content_offset": 0
}
```

2つ目は `output_path` で全文をファイルに書き出す方法です。レスポンスにはメタデータとファイルパスだけが返ります。

```json
{
  "url": "https://example.com/very-long-article",
  "output_path": "/absolute/path/article.md"
}
```

ファイルには常にスライス前の全文が入ります。レスポンスにも本文を残したいときは `include_content_in_response` を、既存ファイルを上書きするときは `overwrite` を指定してください。

### 要約（オプトイン）

`auto_summarize: true` を付けると、`crawl_url` は約15,000トークンを超えるページをデフォルトのLLMプロバイダーで要約します。`content_limit` か `content_offset` を指定したときは要約しません。

```json
{
  "url": "https://example.com/very-long-article",
  "auto_summarize": true
}
```

`process_file` と `extract_youtube_transcript` では、しきい値、長さ、プロバイダーも指定できます。

```json
{
  "url": "https://example.com/report.pdf",
  "auto_summarize": true,
  "max_content_tokens": 15000,
  "summary_length": "long",
  "llm_provider": "anthropic"
}
```

| `summary_length` | 目安 | 内容 |
|------------------|------|------|
| `short` | 2〜3段落、約400トークン | 要点と主な結論 |
| `medium`（デフォルト） | 4〜6段落、約1000トークン | 重要な詳細を含む概要 |
| `long` | 8〜12段落、約2000トークン | 例や背景を含む分析 |

### 長い文書のチャンク処理

`enhanced_process_large_content` はページをチャンクに分け、関連の高いものを残し、必要なら要約もします。

```json
{
  "url": "https://example.com/long-report",
  "chunking_strategy": "topic",
  "filtering_strategy": "bm25",
  "filter_query": "results methodology conclusions",
  "max_chunk_tokens": 2000,
  "chunk_overlap": 200,
  "extract_top_chunks": 10,
  "similarity_threshold": 0.5,
  "summarize_chunks": true,
  "merge_strategy": "hierarchical",
  "final_summary_length": "medium"
}
```

`chunking_strategy` は `topic`、`sentence`、`overlap`、`regex`、`filtering_strategy` は `bm25`、`pruning`、`llm`、`merge_strategy` は `hierarchical` か `linear` です。

## バッチ処理

### 1回あたりの上限

| ツール | 1回あたりの上限 |
|--------|------------------|
| `batch_crawl` | URL 3件（1件ずつ順に処理） |
| `multi_url_crawl` | URL設定 5件（1件ずつ順に処理。`max_concurrent` は効かない） |
| `batch_extract_youtube_transcripts` | URL 3件。`max_concurrent` は任意（デフォルト3、上限5） |
| `batch_search_google` | クエリ 3件。`max_concurrent` は任意（デフォルト3、上限5） |
| `search_and_crawl` | `crawl_top_results` は3まで |
| `deep_crawl_site` | `max_depth` 1〜2、`max_pages` 1〜10 |

上限を超える処理は、何回かの呼び出しに分けてください。

### 複数URLのクロール

`batch_crawl` には3件までのURLと、オプションをそのまま並べて渡します。

```json
{
  "urls": [
    "https://site1.com/page1",
    "https://site1.com/page2",
    "https://site2.com/api-docs"
  ],
  "generate_markdown": true,
  "extract_media": false,
  "wait_for_js": false,
  "base_timeout": 60,
  "output_path": "/absolute/path/batch-run"
}
```

この場合の `output_path` はディレクトリです。成功したURLごとにMarkdownファイルができ、全結果の一覧は `index.json` に入ります。

URLごとに設定を変えたいときは `multi_url_crawl` を使います。

```json
{
  "url_configurations": {
    "https://spa.example.com/dashboard": {"wait_for_js": true, "timeout": 60},
    "https://docs.example.com/guide": {"css_selector": "main"},
    "https://blog.example.com/post": {"extract_media": true}
  },
  "base_timeout": 30
}
```

キーにはURLをそのまま書きます。値で設定できるのは `generate_markdown`、`extract_media`、`wait_for_js`、`timeout`、`use_undetected_browser`、`css_selector` です。

### YouTubeトランスクリプト

```json
{
  "request": {
    "urls": [
      "https://youtube.com/watch?v=VIDEO1",
      "https://youtube.com/watch?v=VIDEO2",
      "https://youtube.com/watch?v=VIDEO3"
    ],
    "languages": ["ja", "en"],
    "include_timestamps": true,
    "translate_to": "ja",
    "timezone": "Asia/Tokyo"
  }
}
```

`timezone` にはIANAのタイムゾーン名を指定し、省略時は `UTC` です。各動画の公開日時をこのタイムゾーンに変換し、`metadata.published_at` に入れて返します。`extract_youtube_transcript` と `get_youtube_video_info` も同じパラメータに対応しています。

### 検索とクロール

検索系のツールは `request` オブジェクトを1つ受け取ります。`search_and_crawl` は検索したうえで上位の結果をクロールします。

```json
{
  "request": {
    "search_query": "企業API ドキュメント ベストプラクティス",
    "crawl_top_results": 3,
    "generate_markdown": true,
    "max_content_per_page": 5000
  }
}
```

`batch_search_google` は3件までのクエリをまとめて検索します。

```json
{
  "request": {
    "queries": [
      "機械学習トレンド 2024",
      "AI導入 企業調査",
      "深層学習フレームワーク比較"
    ],
    "num_results_per_query": 10,
    "search_genre": "pdf",
    "recent_days": 365
  }
}
```

`search_genre` に指定できるのは `pdf`、`documents`、`presentations`、`spreadsheets`、`japanese`、`english` で、大文字と小文字は区別しません。`search_google`、`batch_search_google`、`search_and_crawl` は、ほかの値を受け取ると検索を始める前に `invalid_search_genre` エラーを返し、`valid_genres` に指定できる値を並べます。各ジャンルがクエリに付ける検索演算子は `get_search_genres` で確認できます。

## ワークフロー

### 複数の情報源を調べる

1. `search_google` でページを探す。
2. `search_and_crawl`（3件まで）か `batch_crawl`（3 URLまで）で集める。`output_path` を付ければ全文がディスクに残る。
3. `extract_structured_data` や `intelligent_extract` で必要な項目を取り出す。
4. 保存したMarkdownファイルを見比べる。長い文書には `enhanced_process_large_content` が向いている。

### ドキュメントサイトの構成を把握する

まず `deep_crawl_site` でページを集めます。

```json
{
  "url": "https://api-docs.example.com",
  "max_depth": 2,
  "max_pages": 10,
  "crawl_strategy": "bfs",
  "url_pattern": "*docs*",
  "extract_media": true,
  "output_path": "/absolute/path/api-docs"
}
```

次に、主要なページから `intelligent_extract` で詳細を取り出します。

```json
{
  "url": "https://api-docs.example.com/reference",
  "extraction_goal": "APIエンドポイントとパラメータ、使用例を一覧にする",
  "content_filter": "bm25",
  "filter_query": "API エンドポイント パラメータ 例 チュートリアル"
}
```

### 連絡先の抽出

`extract_entities` は、標準では正規表現で抽出します。

```json
{
  "url": "https://company-directory.com",
  "entity_types": ["emails", "phones", "urls"],
  "deduplicate": true,
  "include_context": true,
  "custom_patterns": {
    "linkedin": "linkedin\\.com/in/[\\w-]+",
    "github": "github\\.com/[\\w-]+"
  }
}
```

組み込みのタイプは `emails`、`phones`、`urls`、`dates`、`ips`、`prices`、`credit_cards`、`coordinates` の8つです。`email` や `phone` のような単数形でも指定でき、大文字と小文字は区別しません。`custom_patterns` のエントリは新しいタイプとして扱われ、その名前を `entity_types` に入れたときだけ抽出されます。`custom_patterns` が使われるのは正規表現モードのときだけです。`people` や `social_media` のように組み込みにないタイプを使うには `"use_llm": true` が要ります。付けずに指定したタイプは `unsupported_entity_types` に入り、レスポンスに警告が付きます。

## タイムアウト

| ツール | パラメータ | デフォルト |
|--------|------------|------------|
| `crawl_url`、`crawl_url_with_fallback` | `timeout`（秒） | 60 |
| `extract_structured_data` | `timeout`（秒） | 30 |
| `batch_crawl`、`multi_url_crawl` | URLごとの `base_timeout` | 30 |
| `deep_crawl_site` | ページごとの `base_timeout` | 60 |
| `extract_youtube_transcript` | ページクロールで代替するときの `fallback_timeout` | 60 |

## エラーと診断

どのツールも `success` を返し、失敗時は `error` にメッセージが入ります。`error_code` や `suggestions` が付くことも少なくありません。

クロールに失敗したら、次の順に試してください。

1. デフォルト設定の `crawl_url`
2. `use_undetected_browser: true` を付けた `crawl_url`、または `crawl_url_with_fallback`
3. `wait_for_js` を外し、`timeout` を短くした設定
4. PDFやOfficeのURLなら `process_file`

STDIOで動かしている間、stdoutにはMCPメッセージしか流れず、診断はstderrに出ます。Pythonのloggingはデフォルトでは出力されません。`CRAWL4AI_MCP_LOG_LEVEL` に `DEBUG`、`INFO`、`WARNING`、`ERROR` のどれかを設定すると、ログがstderrに出ます。この変数は起動時、`.env` を読み込む前に参照されます。そのため `.env` ではなく、MCPクライアント設定の `env` ブロックなど、プロセスの環境変数として渡してください。`FASTMCP_LOG_LEVEL` は効きません。問題を調べるときは、MCPクライアント側に残るサーバーログ（stderr）を見てください。

ヘルスチェック用のエンドポイントはありません。HTTPサーバーが動いているかどうかは、`get_supported_file_formats` のような軽いツールを呼んで確かめます。手順は [HTTP統合ガイド](HTTP_INTEGRATION.md) にあります。

## リトライ付きのPythonクライアント

`--transport streamable-http` で起動したサーバーに、このプロジェクトの依存関係である `fastmcp` のクライアントで接続する例です。

```python
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"


async def crawl_with_retries(url: str) -> Optional[Dict[str, Any]]:
    """crawl_url を設定を変えて試し、最後に crawl_url_with_fallback を使う"""
    strategies = [
        {"wait_for_js": True, "timeout": 60},
        {"wait_for_js": False, "timeout": 30},
        {"use_undetected_browser": True, "timeout": 60},
    ]

    async with Client(SERVER_URL) as client:
        for attempt, strategy in enumerate(strategies):
            result = await client.call_tool("crawl_url", {"url": url, **strategy})
            data = json.loads(result.content[0].text)
            if data.get("success"):
                return data
            logging.warning("試行 %d 失敗: %s", attempt + 1, data.get("error"))
            await asyncio.sleep(2 ** attempt)  # 指数バックオフ

        result = await client.call_tool("crawl_url_with_fallback", {"url": url})
        data = json.loads(result.content[0].text)
        return data if data.get("success") else None


print(asyncio.run(crawl_with_retries("https://example.com")))
```

同じ呼び出しをcurlで行う方法は [HTTP統合ガイド](HTTP_INTEGRATION.md) で説明しています。

## ヒント

1. まずはオプションなしの `crawl_url` から始め、ページに応じて設定を足す。
2. 長いページは `output_path` で保存するか、`content_limit` と `content_offset` で分割して読む。
3. バッチは1回あたり3 URL、3クエリまでに収める。
4. ボット対策のあるサイトでは `crawl_url_with_fallback` を使う。
5. PDFやOfficeファイルは `process_file` に渡す。
6. `auto_summarize` は必要なときだけ有効にする。LLMのAPIキーが要り、トークンも消費する。
7. 外部サービスのレート制限とクォータを守る。

## 関連ドキュメント

- [設定例](CONFIGURATION_EXAMPLES.md)
- [APIリファレンス](API_REFERENCE.md)
- [HTTP統合ガイド](HTTP_INTEGRATION.md)
- [開発ガイド](DEVELOPMENT.md)
- [インストールガイド](INSTALLATION.md)
