# APIリファレンス

Crawl4AI MCPサーバーで利用可能な全19ツールの完全リファレンスです。

## ツール選択ガイド

### 用途に応じた適切なツール選択

| **用途** | **推奨ツール** | **主要機能** |
|-------------|---------------------|------------------|
| 単一ページ | `crawl_url` | 基本クローリング、JS対応 |
| 複数ページ（最大10） | `deep_crawl_site` | サイトマッピング、リンクフォロー |
| 検索+クローリング | `search_and_crawl` | Google検索+自動クロール |
| 困難サイト | `crawl_url_with_fallback` | 複数のリトライ戦略 |
| 構造化データ | `extract_structured_data` | CSS/LLMスキーマ |
| LLMデータ抽出 | `intelligent_extract` | 目標指向・フィルタ対応の抽出 |
| エンティティ抽出 | `extract_entities` | メール、電話番号、URL、日付、IP、価格 |
| ファイル処理 | `process_file` | PDF、Office、ZIP変換 |
| 大容量コンテンツ | `enhanced_process_large_content` | チャンキングとフィルタリング |
| YouTube字幕 | `extract_youtube_transcript` | 字幕抽出 |
| YouTubeコメント | `extract_youtube_comments` | 並べ替え付きコメント抽出 |
| YouTube字幕（バッチ） | `batch_extract_youtube_transcripts` | 最大3本の動画を一括処理 |
| 動画メタデータ | `get_youtube_video_info` | タイトル、説明、公開日時、要約 |
| バッチクロール | `batch_crawl` | 最大3 URLを順次処理 |
| 設定付き複数URL | `multi_url_crawl` | 最大5 URL、URLごとの設定 |
| バッチ検索 | `batch_search_google` | 最大3クエリを同時実行 |
| 検索ジャンル | `get_search_genres` | 利用可能なジャンルフィルタ一覧 |
| 対応ファイル形式 | `get_supported_file_formats` | 対応形式の一覧 |

### パフォーマンス指針

- **深度クローリング**: 最大10ページ制限（安定性重視）
- **バッチ処理**: 1回の呼び出しあたりの上限あり（URL 3件 / URL設定 5件 / クエリ 3件 / 動画 3本）
- **タイムアウト計算**: `ページ数 x base_timeout` 推奨
- **大容量ファイル**: 最大100MB制限
- **リトライ戦略**: 初回失敗時は手動でのリトライを推奨
- **コンテンツページネーション**: content_offset/content_limitで分割取得可能

### ベストプラクティス

**JavaScript重要サイト向け:**
- 常に `wait_for_js: true` を使用
- タイムアウトを30-60秒に増加
- 特定要素には `wait_for_selector` を使用
- ボット検出回避には `use_undetected_browser: true` を使用（`crawl_url`）

**AI機能使用時:**
- 意味的抽出には `extract_structured_data` のLLMモードまたは `intelligent_extract` を使用
- テーブルデータには `extract_structured_data` のLLMテーブルモードを使用
- LLM利用不可時はCSS抽出にフォールバック
- 抽出スキーマは目的に合わせてカスタマイズ

## 💾 結果のディスク保存（output_path）

以下で💾が付いた情報収集系ツールは、取得した全文をディスクへ書き出し、メタデータのみの軽量レスポンスを返す3つの任意パラメータを受け付けます。これにより、エージェントは巨大なページ・長尺のトランスクリプト・バッチ全体を取得しても、生のデータでコンテキストを消費せず、必要なときだけファイルから読み込めます。

**パラメータ:**
- `output_path` (Optional[str]): 結果の書き込み先絶対パス。`None`（デフォルト）または `""` で永続化をスキップ。単一ファイル系ツールは**ファイル**パス（拡張子は省略時に自動付与）、バッチ系ツールは**ディレクトリ**パスを期待し、項目ごとに1ファイルと `index.json` を書き出します。
- `include_content_in_response` (bool): `true` の場合は内容もレスポンスに含めます（`content_limit`/`content_offset`/`max_content_per_page` によるスライスは引き続き適用）。`false`（デフォルト）の場合はメタデータとファイルパスのみ返却します。
- `overwrite` (bool): 既存の出力ファイルを上書きします（デフォルト: `false`）。単一ファイル系では `output_path` のファイルが対象。バッチ系ではディレクトリ内に生成される項目ごとのファイルと `index.json` が対象です（バッチの `output_path` 自体が通常ファイルの場合は overwrite に関わらず拒否されます）。

リクエスト辞書系ツール（`search_google`、`batch_search_google`、`search_and_crawl`、`batch_extract_youtube_transcripts`）では、これら3つはトップレベルのパラメータではなく `request` 辞書のキーとして指定します。

**動作:**
- 書き込みはアトミック（一時ファイル + `os.replace`）で、親ディレクトリは自動作成されます。
- スライスや切り詰めの**前**にスライスなしの全文を永続化するため、レスポンスがスライスされてもディスク上のコピーは常に完全です。
- 各Markdownファイルは `url`、`title`、`fetched_at`、`source_tool` を含むYAMLフロントマターで始まり、自己記述的です。
- バッチ（ディレクトリ）系ツールは `success=false` の項目の永続化をスキップしますが、その項目も `index.json` に出力パスが null として現れます（`deep_crawl_site`/`search_and_crawl`/`batch_extract_youtube_transcripts` は `file: null`、`batch_crawl`/`multi_url_crawl` は `output_file: null`）。

**永続化に対応するツール:**
- 単一ファイル（`output_path` はファイル）: `crawl_url`, `crawl_url_with_fallback`, `extract_structured_data`, `intelligent_extract`, `extract_entities`, `process_file`, `enhanced_process_large_content`, `extract_youtube_transcript`, `extract_youtube_comments`, `get_youtube_video_info`, `search_google`, `batch_search_google`
- ディレクトリ/バッチ（`output_path` はディレクトリ）: `deep_crawl_site`, `search_and_crawl`, `batch_crawl`, `multi_url_crawl`, `batch_extract_youtube_transcripts`

## Webクローリングツール

### `crawl_url` 💾

JavaScript対応のWebページコンテンツ抽出。スクリーンショット取得と任意のコンテンツ要約に対応。SPA向けにwait_for_js=trueを使用。content_offset/content_limitでページネーション対応。

**パラメータ:**
- `url` (str, 必須): クロール対象URL
- `css_selector` (Optional[str]): 抽出用CSSセレクター（デフォルト: None）
- `extract_media` (bool): 画像/動画を抽出（デフォルト: False）
- `take_screenshot` (bool): スクリーンショットを撮影（デフォルト: False）
- `generate_markdown` (bool): Markdownを生成（デフォルト: True）
- `include_cleaned_html` (bool): クリーンなHTMLを含める（デフォルト: False）
- `wait_for_selector` (Optional[str]): 要素の読み込みを待機（デフォルト: None）
- `timeout` (int): タイムアウト秒数（デフォルト: 60）
- `wait_for_js` (bool): JavaScript完了を待機（デフォルト: False）
- `auto_summarize` (bool): 大容量コンテンツをLLMで要約。`true` を明示した場合のみ実行（デフォルト: False）
- `use_undetected_browser` (bool): ボット検出回避用のundetectedブラウザを使用（デフォルト: False）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**レスポンス動作:**
- デフォルトではMarkdownコンテンツのみ返却（トークン削減）
- `include_cleaned_html=True`でクリーンなHTMLも取得可能
- トークン制限: 25000トークン（超過時は推奨事項付きで自動切り詰め）
- 要約は `auto_summarize=true` の場合のみ実行されます（`content_limit`/`content_offset` によるページング中は要約をスキップ）。指定しない場合、大きなページもそのまま返され、レスポンスは約25000トークンで切り詰められます。要約されていない全文をディスクに保存するには `output_path` を使用してください
- リモートの `.html`/`.htm` URLは、通常のWebページと同様にブラウザでレンダリングされます。ドキュメントURL（PDF、Office、ZIP、テキスト、arXivの `/pdf/` リンク等）はファイルとして変換され、ブラウザでのフォールバックは行いません
- 初回クロール失敗時はundetectedブラウザでの自動フォールバック（ドキュメントURLを除く）
- `take_screenshot`: `output_path` 指定時はメインファイルの隣にPNG（`<stem>_screenshot.png`）を保存し、レスポンスに `screenshot_path` を含めます。`output_path` 未指定時はスクリーンショットをレスポンスから除外し、警告を返します

### `deep_crawl_site` 💾

設定可能な深度でサイトの複数ページを再帰的にクロールし、サイトをマッピング。

**パラメータ:**
- `url` (str, 必須): 開始URL
- `max_depth` (int): リンク深度、対応範囲1-2（デフォルト: 2）
- `max_pages` (int): 最大ページ数、対応上限10（デフォルト: 5）
- `crawl_strategy` (str): クロール戦略、'bfs'、'dfs'、'best_first' のいずれか（デフォルト: 'bfs'）
- `include_external` (bool): 外部リンクをフォロー（デフォルト: False）
- `url_pattern` (Optional[str]): URLフィルタパターン（例: '*docs*'、'*blog*'）（デフォルト: None）
- `score_threshold` (float): 最小関連性スコア 0.0-1.0（デフォルト: 0.0）
- `extract_media` (bool): ページから画像/動画を抽出（デフォルト: False）
- `base_timeout` (int): ページごとのタイムアウト秒数（デフォルト: 60）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（URLごとの `.md` ファイルと `index.json` をディレクトリに保存）

### `crawl_url_with_fallback` 💾

アンチボットサイト向けフォールバック戦略付きクローリング。content_offset/content_limitでページネーション対応。

**パラメータ:**
- `url` (str, 必須): クロール対象URL
- `css_selector` (Optional[str]): CSSセレクター（デフォルト: None）
- `extract_media` (bool): メディアを抽出（デフォルト: False）
- `take_screenshot` (bool): スクリーンショット（デフォルト: False）
- `generate_markdown` (bool): Markdown生成（デフォルト: True）
- `wait_for_selector` (Optional[str]): 待機要素（デフォルト: None）
- `timeout` (int): タイムアウト秒数（デフォルト: 60）
- `wait_for_js` (bool): JavaScript待機（デフォルト: False）
- `auto_summarize` (bool): コンテンツをLLMで要約。`true` を明示した場合のみ実行（デフォルト: False）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**レスポンス動作:**
- トークン制限: `crawl_url` と同じく25000トークン（超過時は推奨事項付きで自動切り詰め）
- 要約は `auto_summarize=true` の場合のみ実行されます。指定しない場合、大きなページもそのまま返され、レスポンスは切り詰められます。要約されていない全文をディスクに保存するには `output_path` を使用してください
- リモートの `.html`/`.htm` URLはブラウザでレンダリングされます。ドキュメントURL（PDF、Office、ZIP、テキスト等）はブラウザのフォールバック段階を経ず、ファイルとして1回だけ変換されます
- `take_screenshot`: スクリーンショットは `output_path` 指定時のみ保存されます（`<stem>_screenshot.png` として保存し、`screenshot_path` で参照）

## データ抽出ツール

### `extract_structured_data` 💾

CSSセレクターまたはLLMスキーマを使用した構造化データ抽出。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `extraction_type` (str): 抽出方式、'css'、'llm'、'table' のいずれか（デフォルト: 'css'）
- `css_selectors` (Optional[Dict]): 抽出用CSSセレクターマッピング（デフォルト: None）
- `extraction_schema` (Optional[Dict]): LLM抽出用スキーマ定義（デフォルト: None）
- `generate_markdown` (bool): Markdown生成（デフォルト: False）
- `wait_for_js` (bool): 抽出前にJavaScriptを待機（デフォルト: False）
- `timeout` (int): タイムアウト秒数（デフォルト: 30）
- `use_llm_table_extraction` (bool): LLMテーブル抽出を使用（デフォルト: False）
- `table_chunking_strategy` (str): テーブル分割方式、'intelligent'、'fixed'、'semantic' のいずれか（デフォルト: 'intelligent'）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（`table_data` を含む単一の `.json` ファイルを保存）

### `intelligent_extract` 💾

設定可能なフィルタリングと任意のLLM活用による、目標指向のコンテンツ抽出。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `extraction_goal` (str, 必須): 抽出対象データの自然言語による説明
- `content_filter` (str): フィルタ戦略、'bm25'、'pruning'、'llm' のいずれか（デフォルト: 'bm25'）
- `filter_query` (Optional[str]): フィルタ用キーワード（デフォルト: None）
- `chunk_content` (bool): 大きな文書をチャンクに分割（デフォルト: False）
- `use_llm` (bool): 意味的抽出にLLMを使用（デフォルト: True）
- `llm_provider` (Optional[str]): LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): LLMモデル、未指定時は自動検出（デフォルト: None）
- `custom_instructions` (Optional[str]): 抽出処理への追加指示（デフォルト: None）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（単一の `.json` ファイルを保存）

### `extract_entities` 💾

パターンマッチングまたはLLMを使用して、Webページから特定種別のエンティティ（メール、電話番号等）を抽出。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `entity_types` (List[str], 必須): 抽出するエンティティ種別。正規表現で対応する値: 'emails'、'phones'、'urls'、'dates'、'ips'、'prices'、'credit_cards'、'coordinates'。それ以外の種別（例: 'social_media'）は `use_llm=true` の場合のみ抽出
- `custom_patterns` (Optional[Dict]): カスタム正規表現パターン（デフォルト: None）
- `include_context` (bool): 各エンティティの周囲のコンテキストを含める（デフォルト: True）
- `deduplicate` (bool): 重複を除去（デフォルト: True）
- `use_llm` (bool): 固有表現抽出（NER）にLLMを使用（デフォルト: False）
- `llm_provider` (Optional[str]): LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): LLMモデル、未指定時は自動検出（デフォルト: None）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（単一の `.json` ファイルを保存）

## ファイル処理ツール

### `process_file` 💾

Microsoft MarkItDownを使用して、PDF、Word、Excel、PowerPoint、ZIPなどをMarkdownに変換。

**パラメータ:**
- `url` (str, 必須): ファイルURLまたはローカルパス（PDF、Office、ZIP等）。http/https URL、`file://` URI、絶対パスに対応
- `max_size_mb` (int): 最大ファイルサイズMB（デフォルト: 100）
- `extract_all_from_zip` (bool): ZIPから全ファイル抽出（デフォルト: True）
- `include_metadata` (bool): メタデータを含める（デフォルト: True）
- `auto_summarize` (bool): 大容量コンテンツをLLMで要約。`true` を明示した場合のみ実行（デフォルト: False）
- `max_content_tokens` (int): `auto_summarize=true` 時に要約を行うしきい値トークン数（デフォルト: 15000）
- `summary_length` (str): 要約の長さ、'short'、'medium'、'long' のいずれか（デフォルト: 'medium'）
- `llm_provider` (Optional[str]): 要約用LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): 要約用LLMモデル、未指定時は自動検出（デフォルト: None）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**対応形式:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **アーカイブ**: .zip
- **Web/テキスト**: .html, .htm, .txt, .md, .csv, .rtf
- **電子書籍**: .epub

注: `process_file` は `.html`/`.htm` をファイルとして変換しますが、`crawl_url` と `crawl_url_with_fallback` はリモートの `.html`/`.htm` URLをブラウザでレンダリングします。

### `get_supported_file_formats`

対応ファイル形式と機能の一覧を取得。

**パラメータ:** なし

**戻り値:** 対応形式のリストと各形式の機能情報

### `enhanced_process_large_content` 💾

高度なチャンキングとフィルタリング戦略による大容量コンテンツ処理。

**パラメータ:**
- `url` (str, 必須): 処理対象URL
- `chunking_strategy` (str): 分割方式、'topic'、'sentence'、'overlap'、'regex' のいずれか（デフォルト: 'sentence'）
- `filtering_strategy` (str): チャンクのフィルタ方式、'bm25'、'pruning'、'llm' のいずれか（デフォルト: 'bm25'）
- `filter_query` (Optional[str]): BM25フィルタリング用キーワード（デフォルト: None）
- `max_chunk_tokens` (int): チャンクごとの最大トークン数（デフォルト: 2000）
- `chunk_overlap` (int): 連続するチャンク間のオーバーラップトークン数（デフォルト: 200）
- `extract_top_chunks` (int): 返却する上位チャンク数（デフォルト: 5）
- `similarity_threshold` (float): チャンク採用の最小類似度 0-1（デフォルト: 0.5）
- `summarize_chunks` (bool): 統合前に各チャンクを要約（デフォルト: False）
- `merge_strategy` (str): チャンクの統合方式、'hierarchical' または 'linear'（デフォルト: 'linear'）
- `final_summary_length` (str): 最終要約の長さ、'short'、'medium'、'long' のいずれか（デフォルト: 'short'）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（チャンクと要約を含む単一の `.json` ファイルを保存）

## YouTube処理ツール

### `extract_youtube_transcript` 💾

言語の優先順位指定と任意の翻訳に対応したYouTubeトランスクリプト抽出。公開キャプション付き動画で動作し、ページクロールへのフォールバックに対応。

**パラメータ:**
- `url` (str, 必須): YouTube動画URL
- `languages` (Optional[List[str] | str]): 優先言語コード順（デフォルト: ["ja", "en"]）
- `translate_to` (Optional[str]): 翻訳対象言語（デフォルト: None）
- `include_timestamps` (bool): タイムスタンプを含める（デフォルト: False）
- `preserve_formatting` (bool): 書式を保持（デフォルト: True）
- `include_metadata` (bool): 動画メタデータを含める（デフォルト: True）
- `auto_summarize` (bool): 大容量トランスクリプトをLLMで要約。`true` を明示した場合のみ実行（デフォルト: False）
- `max_content_tokens` (int): `auto_summarize=true` 時に要約を行うしきい値トークン数（デフォルト: 15000）
- `summary_length` (str): 要約の長さ、'short'、'medium'、'long' のいずれか（デフォルト: 'medium'）
- `llm_provider` (Optional[str]): 要約用LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): 要約用LLMモデル、未指定時は自動検出（デフォルト: None）
- `enable_crawl_fallback` (bool): API失敗時のページクロールフォールバック（デフォルト: True）
- `fallback_timeout` (int): フォールバッククロールのタイムアウト秒数（デフォルト: 60）
- `enrich_metadata` (bool): ページクロールでメタデータ（upload_date、view_count）を補完（デフォルト: True）
- `timezone` (str): 公開日時を `metadata.published_at` に変換する際のIANAタイムゾーン名（例: 'Asia/Tokyo'、'America/New_York'）。`enrich_metadata=True` が必要（デフォルト: 'UTC'）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**公開日時:** `metadata.published_at` には `timezone` に変換した公開日時が入ります。`metadata.published_at.original` には常にYouTubeが返したオフセット付きの元の値が保持されます。

### `extract_youtube_comments` 💾

YouTube動画のコメントを抽出・並べ替え。comment_offsetによるページネーション対応。

**パラメータ:**
- `url` (str, 必須): YouTube動画URL
- `sort_by` (str): 並び順、'popular' または 'recent'（デフォルト: 'popular'）
- `max_comments` (int): 最大取得コメント数 1-1000（デフォルト: 300）
- `comment_offset` (int): スキップするコメント数（ページネーション用）（デフォルト: 0）
- `include_replies` (bool): 返信コメントを含める（デフォルト: True）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `content_limit` (int): 最大返却文字数、0=無制限（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（全コメントを含む単一の `.json` ファイルを、トークン制限によるコメント削減の前に保存）

### `batch_extract_youtube_transcripts` 💾

複数YouTube動画からトランスクリプトを一括抽出。1回の呼び出しで最大3本。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `urls` (List[str], 必須): YouTube動画URLのリスト（最大3件）
  - `languages` (List[str]): 優先言語コード順（デフォルト: ["ja", "en"]）
  - `include_timestamps` (bool): タイムスタンプを含める（デフォルト: False）
  - `timezone` (str): 各動画の公開日時を `metadata.published_at` に変換する際のIANAタイムゾーン名。`metadata.published_at.original` にはYouTubeのオフセットが保持されます（デフォルト: 'UTC'）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（動画ごとの `.md` ファイルと `index.json` をディレクトリに保存）
  - その他に受け付けるキー: `translate_to` (Optional[str]、デフォルト: None)、`preserve_formatting` (bool、デフォルト: True)、`include_metadata` (bool、デフォルト: True)、`max_concurrent` (int、デフォルト: 3、最大5)

### `get_youtube_video_info` 💾

YouTube動画のメタデータとトランスクリプト利用可能情報を取得。

**パラメータ:**
- `video_url` (str, 必須): YouTube動画URL
- `summarize_transcript` (bool): トランスクリプトを要約（デフォルト: False）
- `max_tokens` (int): 要約のトークン制限（デフォルト: 25000）
- `llm_provider` (Optional[str]): 要約用LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): 要約用LLMモデル、未指定時は自動検出（デフォルト: None）
- `summary_length` (str): 要約の長さ、'short'、'medium'、'long' のいずれか（デフォルト: 'medium'）
- `include_timestamps` (bool): タイムスタンプを含める（デフォルト: True）
- `timezone` (str): `published_at` として返す公開日時の変換に使うIANAタイムゾーン名（例: 'Asia/Tokyo'、'America/New_York'）（デフォルト: 'UTC'）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（動画情報とトランスクリプトを含む単一の `.md` ファイルを保存）

**戻り値:**
- 利用可能なトランスクリプト言語
- 手動作成/自動生成の区別
- 翻訳可能な言語の情報
- 公開日時 `published_at`（`timezone` に変換済み。`published_at.original` にはYouTubeのオフセットを保持）

## Google検索ツール

### `search_google` 💾

ジャンルフィルタリングとメタデータ抽出付きのGoogle検索。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `query` (str, 必須): 検索クエリ文字列
  - `num_results` (int): 返す結果数 1-100（デフォルト: 10）
  - `search_genre` (Optional[str]): コンテンツジャンルフィルタ（デフォルト: None）
  - `language` (str): 検索言語（デフォルト: "en"）
  - `region` (str): 検索地域（デフォルト: "us"）
  - `recent_days` (Optional[int]): 直近N日間に限定（デフォルト: None）
  - `content_limit` (int): 結果ごとの最大返却文字数、0=無制限（デフォルト: 0）
  - `content_offset` (int): コンテンツ開始位置（デフォルト: 0）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（スライス前の全結果を単一の `.json` ファイルに保存）
  - その他に受け付けるキー: `safe_search` (bool、デフォルト: True)

**機能:**
- 検索結果からの自動タイトル・スニペット抽出
- クエリにGoogle演算子を付加する6つの検索ジャンル: `pdf`、`documents`、`presentations`、`spreadsheets`（ファイル形式フィルタ）、`japanese`、`english`（言語フィルタ）
- URL分類とドメイン分析
- セーフサーチがデフォルトで有効

### `search_and_crawl` 💾

Google検索と上位結果の自動クロール実行。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `search_query` (str, 必須): Google検索クエリ
  - `crawl_top_results` (int): クロール対象の上位結果数、最大3（デフォルト: 2）
  - `search_genre` (Optional[str]): コンテンツジャンルフィルタ（デフォルト: None）
  - `recent_days` (Optional[int]): 直近N日間に限定（デフォルト: None）
  - `generate_markdown` (bool): クロールしたページからMarkdownを生成（デフォルト: True）
  - `max_content_per_page` (int): レスポンス内のページごとの最大文字数（デフォルト: 5000）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（ページごとの `.md` ファイルと `index.json` をディレクトリに保存。`max_content_per_page` による切り詰めの前に保存）

**戻り値:**
- 検索メタデータとクロールしたコンテンツ一式
- 成功率と処理統計
- 検索結果とクロール結果の統合分析

### `batch_search_google` 💾

複数のGoogle検索を同時実行。1回の呼び出しで最大3クエリ。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `queries` (List[str], 必須): 検索クエリのリスト（最大3件）
  - `num_results_per_query` (int): クエリごとの結果数 1-100（デフォルト: 10）
  - `search_genre` (Optional[str]): 全クエリに適用するコンテンツジャンルフィルタ（デフォルト: None）
  - `recent_days` (Optional[int]): 直近N日間に限定（デフォルト: None）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（全結果を単一の `.json` ファイルに保存）
  - その他に受け付けるキー: `language` (str、デフォルト: "en")、`region` (str、デフォルト: "us")、`max_concurrent` (int、デフォルト: 3、最大5)

### `get_search_genres`

ターゲット検索で利用可能な検索ジャンル一覧を取得。

**パラメータ:** なし

**戻り値:** 利用可能なジャンルとその説明

## バッチ操作ツール

### `batch_crawl` 💾

1回のリクエストで複数URL（最大3件）を、フォールバック付きで順次クロール。

**パラメータ:**
- `urls` (List[str], 必須): クロール対象URLのリスト（最大3件）
- `base_timeout` (int): URLごとのタイムアウト秒数（デフォルト: 30）
- `generate_markdown` (bool): Markdown生成（デフォルト: True）
- `extract_media` (bool): ページから画像/動画を抽出（デフォルト: False）
- `wait_for_js` (bool): JavaScript完了を待機（デフォルト: False）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（URLごとの `.md` ファイルと `index.json` をディレクトリに保存。戻り値はリストのままで、成功項目に `output_file` が追加される）

### `multi_url_crawl` 💾

URLごとの設定オプション付きで複数URL（最大5件）をクロール。

**パラメータ:**
- `url_configurations` (Dict, 必須): URLごとの設定マップ、最大5件（例: `{"https://site1.com": {"wait_for_js": true}}`）
- `pattern_matching` (str): URLパターンの照合方式、'wildcard' または 'regex'（デフォルト: 'wildcard'）
- `default_config` (Optional[Dict]): 全URLに適用するデフォルト設定（デフォルト: None）
- `base_timeout` (int): URLごとのタイムアウト秒数（デフォルト: 30）
- `max_concurrent` (int): 最大同時実行数（デフォルト: 3）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#-結果のディスク保存output_path)を参照（URLごとの `.md` ファイルと `index.json` をディレクトリに保存。戻り値はリストのままで、成功項目に `output_file` が追加される）

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

### 複雑さ別
- **シンプル**: `crawl_url`, `process_file`, `get_youtube_video_info`, `get_search_genres`, `get_supported_file_formats`
- **中程度**: `deep_crawl_site`, `search_google`, `extract_youtube_transcript`, `extract_youtube_comments`, `batch_crawl`
- **高度**: `extract_structured_data`, `intelligent_extract`, `extract_entities`, `search_and_crawl`, `batch_search_google`, `multi_url_crawl`, `batch_extract_youtube_transcripts`, `enhanced_process_large_content`

### 用途別
- **コンテンツ発見**: `search_google`, `search_and_crawl`, `batch_search_google`, `get_search_genres`
- **データ抽出**: `crawl_url`, `extract_structured_data`, `intelligent_extract`, `extract_entities`
- **メディア処理**: `extract_youtube_transcript`, `extract_youtube_comments`, `batch_extract_youtube_transcripts`, `process_file`, `get_supported_file_formats`
- **サイト分析**: `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`, `multi_url_crawl`
- **大容量コンテンツ**: `enhanced_process_large_content`, `get_youtube_video_info`

## 統合例

詳細な設定例については、[設定例](CONFIGURATION_EXAMPLES.md)をご覧ください。

HTTP API統合については、[HTTP統合ガイド](HTTP_INTEGRATION.md)をご覧ください。

高度な使用パターンについては、[高度な使用ガイド](ADVANCED_USAGE.md)をご覧ください。
