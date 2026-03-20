# APIリファレンス

Crawl4AI MCPサーバーで利用可能な全19ツールの完全リファレンスです。

## ツール選択ガイド

### 用途に応じた適切なツール選択

| **用途** | **推奨ツール** | **主要機能** |
|-------------|---------------------|------------------|
| 単一ページ | `crawl_url` | 基本クローリング、JS対応 |
| 複数ページ | `deep_crawl_site` | サイトマッピング、リンクフォロー |
| 検索+クローリング | `search_and_crawl` | Google検索+自動クロール |
| 困難サイト | `crawl_url_with_fallback` | フォールバック戦略 |
| 構造化データ | `extract_structured_data` | CSS/LLMスキーマ |
| LLMデータ抽出 | `intelligent_extract` | LLMベースの目標指向抽出 |
| エンティティ抽出 | `extract_entities` | メール、電話等の抽出 |
| ファイル処理 | `process_file` | PDF、Office、ZIP変換 |
| 大容量コンテンツ | `enhanced_process_large_content` | チャンキングとフィルタリング |
| YouTube字幕 | `extract_youtube_transcript` | 字幕抽出 |
| YouTubeコメント | `extract_youtube_comments` | コメント抽出 |
| 動画メタデータ | `get_youtube_video_info` | タイトル、説明等 |
| バッチクロール | `batch_crawl` | 複数URL同時処理（最大3件） |
| バッチ検索 | `batch_search_google` | 複数クエリ検索（最大3件） |

### パフォーマンス指針

- **深度クローリング**: 最大10ページ制限（安定性重視）
- **バッチ処理**: batch_crawl最大3 URL、batch_search_google最大3クエリ
- **タイムアウト計算**: `ページ数 x base_timeout` 推奨
- **大容量ファイル**: 最大100MB制限
- **コンテンツページネーション**: content_offset/content_limitで分割取得可能

### ベストプラクティス

**JavaScript重要サイト向け:**
- 常に `wait_for_js: true` を使用
- ボット検出回避には `use_undetected_browser: true` を使用
- タイムアウトを30-60秒に増加
- 特定要素には `wait_for_selector` を使用

**AI機能使用時:**
- 意味的抽出には `intelligent_extract` を使用
- テーブルデータには `extract_structured_data` のLLMテーブルモードを使用
- LLM利用不可時はCSS抽出にフォールバック

---

## Webクローリングツール

### `crawl_url`

JavaScript対応のWebページコンテンツ抽出。SPA向けにwait_for_js=trueを使用。content_offset/content_limitでページネーション対応。

**パラメータ:**
- `url` (str, 必須): クロール対象URL
- `css_selector` (str, オプション): 抽出用CSSセレクター（デフォルト: None）
- `extract_media` (bool): 画像/動画を抽出（デフォルト: False）
- `take_screenshot` (bool): スクリーンショットを撮影（デフォルト: False）
- `generate_markdown` (bool): Markdownを生成（デフォルト: True）
- `include_cleaned_html` (bool): クリーンなHTMLを含める（デフォルト: False）
- `wait_for_selector` (str, オプション): 要素の読み込みを待機（デフォルト: None）
- `timeout` (int): タイムアウト秒数（デフォルト: 60）
- `wait_for_js` (bool): JavaScript完了を待機（デフォルト: False）
- `auto_summarize` (bool): 大容量コンテンツの自動要約（デフォルト: False）
- `use_undetected_browser` (bool): ボット検出回避（デフォルト: False）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）

**レスポンス動作:**
- デフォルトではMarkdownコンテンツのみ返却（トークン削減）
- `include_cleaned_html=True`でクリーンなHTMLも取得可能
- トークン制限: 25000トークン（超過時は推奨事項付きで自動切り詰め）
- 初回クロール失敗時はundetectedブラウザでの自動フォールバック

### `deep_crawl_site`

設定可能な深度でサイトの複数ページをクロール。

**パラメータ:**
- `url` (str, 必須): 開始URL
- `max_depth` (int): リンク深度、1-2推奨（デフォルト: 2）
- `max_pages` (int): 最大ページ数、最大10（デフォルト: 5）
- `crawl_strategy` (str): 'bfs'|'dfs'|'best_first'（デフォルト: 'bfs'）
- `include_external` (bool): 外部リンクをフォロー（デフォルト: False）
- `url_pattern` (str, オプション): URLフィルタパターン（例: '*docs*'）
- `score_threshold` (float): 最小関連性スコア 0-1（デフォルト: 0.0）
- `extract_media` (bool): メディアを抽出（デフォルト: False）
- `base_timeout` (int): ページごとのタイムアウト（デフォルト: 60）

### `crawl_url_with_fallback`

アンチボットサイト向けフォールバック戦略付きクローリング。content_offset/content_limitでページネーション対応。

**パラメータ:**
- `url` (str, 必須): クロール対象URL
- `css_selector` (str, オプション): CSSセレクター
- `extract_media` (bool): メディアを抽出（デフォルト: False）
- `take_screenshot` (bool): スクリーンショット（デフォルト: False）
- `generate_markdown` (bool): Markdown生成（デフォルト: True）
- `wait_for_selector` (str, オプション): 待機要素
- `timeout` (int): タイムアウト秒数（デフォルト: 60）
- `wait_for_js` (bool): JavaScript待機（デフォルト: False）
- `auto_summarize` (bool): 自動要約（デフォルト: False）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置（デフォルト: 0）

---

## データ抽出ツール

### `intelligent_extract`

LLMを使用してWebページから特定のデータを抽出。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `extraction_goal` (str, 必須): 抽出対象データの説明
- `content_filter` (str): 'bm25'|'pruning'|'llm'（デフォルト: 'bm25'）
- `filter_query` (str, オプション): BM25フィルタキーワード
- `chunk_content` (bool): コンテンツ分割（デフォルト: False）
- `use_llm` (bool): LLM有効化（デフォルト: True）
- `llm_provider` (str, オプション): LLMプロバイダー
- `llm_model` (str, オプション): LLMモデル
- `custom_instructions` (str, オプション): LLMへのカスタム指示

### `extract_entities`

Webページからエンティティ（メール、電話番号等）を抽出。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `entity_types` (List[str], 必須): 抽出タイプ: email, phone, url, date, ip, price
- `custom_patterns` (Dict[str, str], オプション): カスタム正規表現パターン
- `include_context` (bool): 周囲のコンテキストを含める（デフォルト: True）
- `deduplicate` (bool): 重複を除去（デフォルト: True）
- `use_llm` (bool): NERにLLMを使用（デフォルト: False）
- `llm_provider` (str, オプション): LLMプロバイダー
- `llm_model` (str, オプション): LLMモデル

### `extract_structured_data`

CSSセレクターまたはLLMを使用した構造化データ抽出。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `extraction_type` (str): 'css'|'llm'|'table'（デフォルト: 'css'）
- `css_selectors` (Dict[str, str], オプション): CSSセレクターマッピング
- `extraction_schema` (Dict[str, str], オプション): スキーマ定義
- `generate_markdown` (bool): Markdown生成（デフォルト: False）
- `wait_for_js` (bool): JavaScript待機（デフォルト: False）
- `timeout` (int): タイムアウト秒数（デフォルト: 30）
- `use_llm_table_extraction` (bool): LLMテーブル抽出を使用（デフォルト: False）
- `table_chunking_strategy` (str): 'intelligent'|'fixed'|'semantic'（デフォルト: 'intelligent'）

---

## ファイル処理ツール

### `process_file`

PDF、Word、Excel、PowerPoint、ZIPをMarkdownに変換。

**パラメータ:**
- `url` (str, 必須): ファイルURL（PDF、Office、ZIP等）
- `max_size_mb` (int): 最大ファイルサイズMB（デフォルト: 100）
- `extract_all_from_zip` (bool): ZIPから全ファイル抽出（デフォルト: True）
- `include_metadata` (bool): メタデータを含める（デフォルト: True）
- `auto_summarize` (bool): 大容量コンテンツの自動要約（デフォルト: False）
- `max_content_tokens` (int): 要約トリガーの最大トークン数（デフォルト: 15000）
- `summary_length` (str): 'short'|'medium'|'long'（デフォルト: 'medium'）
- `llm_provider` (str, オプション): LLMプロバイダー
- `llm_model` (str, オプション): LLMモデル
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置（デフォルト: 0）

**対応形式:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **アーカイブ**: .zip
- **Web/テキスト**: .html, .htm, .txt, .md, .csv, .rtf
- **電子書籍**: .epub

### `get_supported_file_formats`

対応ファイル形式と機能の一覧を取得。

**パラメータ:** なし

**戻り値:** 対応形式のリストと各形式の機能情報

### `enhanced_process_large_content`

チャンキングとBM25フィルタリングによる大容量コンテンツ処理。

**パラメータ:**
- `url` (str, 必須): 処理対象URL
- `chunking_strategy` (str): 'topic'|'sentence'|'overlap'|'regex'（デフォルト: 'sentence'）
- `filtering_strategy` (str): 'bm25'|'pruning'|'llm'（デフォルト: 'bm25'）
- `filter_query` (str, オプション): BM25フィルタリング用キーワード
- `max_chunk_tokens` (int): チャンクごとの最大トークン数（デフォルト: 2000）
- `chunk_overlap` (int): オーバーラップトークン数（デフォルト: 200）
- `extract_top_chunks` (int): 抽出する上位チャンク数（デフォルト: 5）
- `similarity_threshold` (float): 最小類似度 0-1（デフォルト: 0.5）
- `summarize_chunks` (bool): チャンクを要約（デフォルト: False）
- `merge_strategy` (str): 'hierarchical'|'linear'（デフォルト: 'linear'）
- `final_summary_length` (str): 'short'|'medium'|'long'（デフォルト: 'short'）

---

## YouTube処理ツール

### `extract_youtube_transcript`

YouTubeトランスクリプト抽出。公開キャプション付き動画で動作。ページクロールへのフォールバック対応。

**パラメータ:**
- `url` (str, 必須): YouTube動画URL
- `languages` (List[str]|str, オプション): 優先言語コード順（デフォルト: ["ja", "en"]）
- `translate_to` (str, オプション): 翻訳対象言語
- `include_timestamps` (bool): タイムスタンプを含める（デフォルト: False）
- `preserve_formatting` (bool): 書式を保持（デフォルト: True）
- `include_metadata` (bool): 動画メタデータを含める（デフォルト: True）
- `auto_summarize` (bool): 大容量コンテンツの自動要約（デフォルト: False）
- `max_content_tokens` (int): 要約トリガーの最大トークン数（デフォルト: 15000）
- `summary_length` (str): 'short'|'medium'|'long'（デフォルト: 'medium'）
- `llm_provider` (str, オプション): LLMプロバイダー
- `llm_model` (str, オプション): LLMモデル
- `enable_crawl_fallback` (bool): API失敗時のページクロールフォールバック（デフォルト: True）
- `fallback_timeout` (int): フォールバッククロールのタイムアウト秒数（デフォルト: 60）
- `enrich_metadata` (bool): ページクロールでメタデータを充実化（デフォルト: True）
- `content_offset` (int): コンテンツ開始位置（デフォルト: 0）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）

### `batch_extract_youtube_transcripts`

複数YouTube動画からトランスクリプトを一括抽出。1回の呼び出しで最大3 URL。

**パラメータ:**
- `request` (Dict, 必須): urls（最大3件）、languages、include_timestampsを含む辞書

### `get_youtube_video_info`

YouTube動画のメタデータとトランスクリプト利用可能情報を取得。

**パラメータ:**
- `video_url` (str, 必須): YouTube動画URL
- `summarize_transcript` (bool): トランスクリプトを要約（デフォルト: False）
- `max_tokens` (int): 要約のトークン制限（デフォルト: 25000）
- `llm_provider` (str, オプション): LLMプロバイダー
- `llm_model` (str, オプション): LLMモデル
- `summary_length` (str): 'short'|'medium'|'long'（デフォルト: 'medium'）
- `include_timestamps` (bool): タイムスタンプを含める（デフォルト: True）

### `extract_youtube_comments`

YouTube動画のコメントを抽出。comment_offsetによるページネーション対応。

**パラメータ:**
- `url` (str, 必須): YouTube動画URL
- `sort_by` (str): 'popular'|'recent'（デフォルト: 'popular'）
- `max_comments` (int): 最大取得コメント数 1-1000（デフォルト: 300）
- `comment_offset` (int): スキップするコメント数（ページネーション用）（デフォルト: 0）
- `include_replies` (bool): 返信コメントを含める（デフォルト: True）
- `content_offset` (int): コンテンツ開始位置（デフォルト: 0）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）

---

## Google検索ツール

### `search_google`

ジャンルフィルタリング付きGoogle検索実行。

**パラメータ（request辞書）:**
- `query` (str, 必須): 検索クエリ文字列
- `num_results` (int): 返す結果数（デフォルト: 10）
- `search_genre` (str, オプション): コンテンツジャンルフィルタ
- `language` (str): 検索言語（デフォルト: "en"）
- `region` (str): 検索地域（デフォルト: "us"）
- `recent_days` (int, オプション): 直近N日間に限定
- `content_limit` (int): 最大返却文字数（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置（デフォルト: 0）

**機能:**
- 検索結果からの自動タイトル・スニペット抽出
- Google公式オペレーターを使用した最適化された検索ジャンル
- URL分類とドメイン分析

### `batch_search_google`

複数Google検索の一括実行。1回の呼び出しで最大3クエリ。

**パラメータ（request辞書）:**
- `queries` (List[str], 必須): 検索クエリリスト（最大3件）
- `num_results_per_query` (int, オプション): クエリごとの結果数
- `search_genre` (str, オプション): コンテンツジャンルフィルタ
- `recent_days` (int, オプション): 直近N日間に限定

### `search_and_crawl`

Google検索と上位結果の自動クロール実行。

**パラメータ（request辞書）:**
- `search_query` (str, 必須): Google検索クエリ
- `crawl_top_results` (int): クロール対象上位結果数（デフォルト: 2、最大3）
- `search_genre` (str, オプション): コンテンツジャンルフィルタ
- `recent_days` (int, オプション): 直近N日間に限定
- `generate_markdown` (bool): Markdown生成（デフォルト: True）
- `max_content_per_page` (int): ページごとの最大コンテンツ長（デフォルト: 5000）

### `get_search_genres`

ターゲット検索で利用可能な検索ジャンル一覧を取得。

**パラメータ:** なし

**戻り値:** 利用可能なジャンルとその説明

---

## バッチ操作ツール

### `batch_crawl`

フォールバック付き複数URLクロール。1回の呼び出しで最大3 URL。

**パラメータ:**
- `urls` (List[str], 必須): クロール対象URL（最大3件）
- `base_timeout` (int): URLごとのタイムアウト（デフォルト: 30）
- `generate_markdown` (bool): Markdown生成（デフォルト: True）
- `extract_media` (bool): メディア抽出（デフォルト: False）
- `wait_for_js` (bool): JavaScript待機（デフォルト: False）

### `multi_url_crawl`

パターンベース設定による複数URLクロール。1回の呼び出しで最大5 URLパターン。

**パラメータ:**
- `url_configurations` (Dict[str, Dict], 必須): URLと設定のマッピング（最大5件）
- `pattern_matching` (str): 'wildcard'|'regex'（デフォルト: 'wildcard'）
- `default_config` (Dict, オプション): デフォルト設定
- `base_timeout` (int): URLごとのタイムアウト（デフォルト: 30）
- `max_concurrent` (int): 最大同時実行数（デフォルト: 3）

---

## ツール分類

### カテゴリ別

| カテゴリ | ツール | ツール数 |
|---------|--------|---------|
| Webクローリング | `crawl_url`, `deep_crawl_site`, `crawl_url_with_fallback` | 3 |
| データ抽出 | `intelligent_extract`, `extract_entities`, `extract_structured_data` | 3 |
| YouTube | `extract_youtube_transcript`, `batch_extract_youtube_transcripts`, `get_youtube_video_info`, `extract_youtube_comments` | 4 |
| 検索 | `search_google`, `batch_search_google`, `search_and_crawl`, `get_search_genres` | 4 |
| ファイル処理 | `process_file`, `get_supported_file_formats`, `enhanced_process_large_content` | 3 |
| バッチ操作 | `batch_crawl`, `multi_url_crawl` | 2 |
| **合計** | | **19** |

### 用途別
- **コンテンツ発見**: `search_google`, `search_and_crawl`, `batch_search_google`
- **データ抽出**: `crawl_url`, `intelligent_extract`, `extract_entities`, `extract_structured_data`
- **メディア処理**: `extract_youtube_transcript`, `extract_youtube_comments`, `process_file`
- **サイト分析**: `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`, `multi_url_crawl`
- **ユーティリティ**: `get_youtube_video_info`, `get_search_genres`, `get_supported_file_formats`

---

## 統合例

詳細な設定例については、[設定例](CONFIGURATION_EXAMPLES.md)をご覧ください。

HTTP API統合については、[HTTP統合ガイド](HTTP_INTEGRATION.md)をご覧ください。

高度な使用パターンについては、[高度な使用ガイド](ADVANCED_USAGE.md)をご覧ください。
