# APIリファレンス

Crawl4AI MCPサーバーの全19ツールのパラメータと動作をまとめています。

## ツール選択ガイド

### 用途別のツール

| 用途 | ツール | 主な機能 |
|-------------|---------------------|------------------|
| 単一ページ | `crawl_url` | 基本クローリング、JS対応 |
| 複数ページ（最大10） | `deep_crawl_site` | サイトマッピング、リンクフォロー |
| Web検索 | `search_web` | Google Custom Search、Brave、Exa で検索 |
| 検索+クローリング | `search_and_crawl` | Web検索+自動クロール |
| 取得しにくいサイト | `crawl_url_with_fallback` | 複数のリトライ戦略 |
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
| バッチ検索 | `batch_search_web` | 最大3クエリを同時実行 |
| 検索ジャンル | `get_search_genres` | 利用可能なジャンルフィルタ一覧 |
| 対応ファイル形式 | `get_supported_file_formats` | 対応形式の一覧 |

### 上限と目安

- `deep_crawl_site` で取得できるのは最大10ページです。安定して動かすための制限です
- バッチ系のツールは1回の呼び出しで扱える件数に上限があります（URL 3件、URL設定 5件、クエリ 3件、動画 3本）
- タイムアウトは `ページ数 x base_timeout` を目安に設定してください
- 扱えるファイルサイズは最大100MBです
- 初回で失敗したときは、手動でリトライしてください
- 長いコンテンツは content_offset と content_limit で分けて取得できます

### 使い方のこつ

JavaScriptで描画するサイトでは、次の設定が効きます。
- `wait_for_js: true` を指定する
- タイムアウトを30〜60秒に延ばす
- 特定の要素を待つなら `wait_for_selector` を使う
- ボット検出を避けるには `use_undetected_browser: true` を指定する（`crawl_url`）

LLMを使う抽出では、次のように使い分けます。
- 意味に基づく抽出は `extract_structured_data` のLLMモードか `intelligent_extract` で行う
- 表のデータは `extract_structured_data` のLLMテーブルモードで取る
- LLMが使えないときはCSS抽出に切り替える
- 抽出スキーマは目的に合わせて書く

## 結果のディスク保存（output_path）

情報を取得するツール（この節の末尾に一覧があります）は、次の3つの任意パラメータを受け付けます。指定すると取得した全文をディスクに書き出し、レスポンスはメタデータだけになります。巨大なページや長いトランスクリプト、バッチ全体を取得してもコンテキストを消費せず、中身は必要なときにファイルから読めます。

**パラメータ:**
- `output_path` (Optional[str]): 結果の書き込み先絶対パス。`None`（デフォルト）か `""` なら保存しません。単一ファイル系のツールには**ファイル**パスを渡します（拡張子は省略すると自動で付きます）。バッチ系のツールには**ディレクトリ**パスを渡し、項目ごとのファイルと `index.json` がそこに書き出されます。
- `include_content_in_response` (bool): `true` にすると内容もレスポンスに入ります（`content_limit`、`content_offset`、`max_content_per_page` によるスライスはそのままかかります）。`false`（デフォルト）ではメタデータとファイルパスだけを返します。
- `overwrite` (bool): 既存の出力ファイルを上書きします（デフォルト: `false`）。単一ファイル系では `output_path` のファイル、バッチ系ではディレクトリ内に作る項目ごとのファイルと `index.json` が対象です。バッチの `output_path` 自体が通常ファイルだった場合は、overwrite の値にかかわらずエラーになります。

リクエスト辞書系ツール（`search_web`、`batch_search_web`、`search_and_crawl`、`batch_extract_youtube_transcripts`、エイリアスの `search_google`/`batch_search_google`）では、これら3つはトップレベルのパラメータではなく `request` 辞書のキーとして指定します。

**動作:**
- 書き込みはアトミックです（一時ファイルに書いてから `os.replace`）。親ディレクトリは自動で作ります。
- 保存はスライスや切り詰めより**前**に行うので、レスポンスが切り詰められてもディスク上のファイルは全文のままです。
- Markdownファイルの先頭には `url`、`title`、`fetched_at`、`source_tool` を含むYAMLフロントマターが付き、ファイル単体で取得元が分かります。
- バッチ（ディレクトリ）系のツールは `success=false` の項目をファイルに保存しません。それでも `index.json` には出力パスが null の項目として載ります（`deep_crawl_site`/`search_and_crawl`/`batch_extract_youtube_transcripts` は `file: null`、`batch_crawl`/`multi_url_crawl` は `output_file: null`）。

**`output_path` に対応するツール:**
- `output_path` にファイルを指定するもの: `crawl_url`, `crawl_url_with_fallback`, `extract_structured_data`, `intelligent_extract`, `extract_entities`, `process_file`, `enhanced_process_large_content`, `extract_youtube_transcript`, `extract_youtube_comments`, `get_youtube_video_info`, `search_web`, `batch_search_web`（エイリアスの `search_google`/`batch_search_google` も同じ）
- `output_path` にディレクトリを指定するもの: `deep_crawl_site`, `search_and_crawl`, `batch_crawl`, `multi_url_crawl`, `batch_extract_youtube_transcripts`

## Webクローリングツール

### `crawl_url`

Webページの内容をJavaScriptを実行したうえで取得します。スクリーンショットと、指定したときの要約にも対応します。SPAではwait_for_js=trueを指定してください。content_offsetとcontent_limitで分割取得できます。

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
- `content_limit` (int): 返す最大文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**レスポンス動作:**
- デフォルトではMarkdownだけを返す（トークン節約のため）
- `include_cleaned_html=True` で整形済みHTMLも返す
- 上限は25000トークンで、超えると対処方法の案内を付けて切り詰める
- 要約は `auto_summarize=true` の場合のみ実行されます（`content_limit`/`content_offset` によるページング中は要約をスキップ）。指定しない場合、大きなページもそのまま返され、レスポンスは約25000トークンで切り詰められます。要約していない全文をディスクに残すには `output_path` を指定します
- リモートの `.html`/`.htm` URLは、通常のWebページと同じくブラウザでレンダリングします。ドキュメントURL（PDF、Office、ZIP、テキスト、arXivの `/pdf/` リンク等）はファイルとして変換し、ブラウザでのフォールバックは行いません
- 初回のクロールに失敗すると、undetectedブラウザで自動的に再試行する（ドキュメントURLを除く）
- `take_screenshot`: `output_path` を指定したときはメインファイルの隣にPNG（`<stem>_screenshot.png`）を保存し、レスポンスに `screenshot_path` を入れます。`output_path` がなければスクリーンショットはレスポンスに含めず、警告を返します

### `deep_crawl_site`

指定した深さまでリンクをたどり、サイト内の複数ページをクロールします。

**パラメータ:**
- `url` (str, 必須): 開始URL
- `max_depth` (int): リンク深度、1-2（デフォルト: 2）
- `max_pages` (int): 最大ページ数、1-10（デフォルト: 5）
- `crawl_strategy` (str): クロール戦略、'bfs'、'dfs'、'best_first' のいずれか（デフォルト: 'bfs'）
- `include_external` (bool): 外部リンクをフォロー（デフォルト: False）
- `url_pattern` (Optional[str]): URLフィルタパターン（例: '*docs*'、'*blog*'）（デフォルト: None）
- `score_threshold` (float): 最小関連性スコア 0.0-1.0（デフォルト: 0.0）
- `extract_media` (bool): ページから画像/動画を抽出（デフォルト: False）
- `base_timeout` (int): ページごとのタイムアウト秒数（デフォルト: 60）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（URLごとの `.md` ファイルと `index.json` をディレクトリに保存）

**レスポンス動作:**
- `max_depth` か `max_pages` が範囲外なら、ページを取得する前にエラーを返す。`error_code` は、`max_depth` なら `invalid_depth` か `depth_too_large`、`max_pages` なら `invalid_max_pages` か `max_pages_too_large`
- `markdown` と `content` には、クロールした全ページをつなげた1つの文書が入る
- `crawled_pages` はページごとの `url`、`title`、`success`、`error` の一覧
- `output_path` を指定すると、成功したページを1ページ1つの `.md` ファイルに保存し、`index.json` に載せる。`include_content_in_response=true` でなければ、つなげた `markdown`/`content` と各ページの本文はレスポンスから外す

### `crawl_url_with_fallback`

ボット対策のあるサイト向けに、複数のフォールバック戦略を順に試してクロールします。content_offsetとcontent_limitで分割取得できます。

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
- `content_limit` (int): 返す最大文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**レスポンス動作:**
- 上限は `crawl_url` と同じ25000トークンで、超えると対処方法の案内を付けて切り詰める
- 要約は `auto_summarize=true` の場合だけ実行されます。指定しなければ大きなページもそのまま返り、レスポンスは切り詰められます。要約していない全文をディスクに残すには `output_path` を指定します
- リモートの `.html`/`.htm` URLはブラウザでレンダリングします。ドキュメントURL（PDF、Office、ZIP、テキスト等）はブラウザのフォールバック段階を通らず、ファイルとして1回だけ変換します
- `take_screenshot` のスクリーンショットは `output_path` を指定したときだけ保存されます（`<stem>_screenshot.png` として保存し、`screenshot_path` で参照）
- フォールバックの段階でHTTP 4xx/5xxのステータスが返ったら、次の段階に進む。AMPの段階ではエラーページに見えるページも採用しないので、存在しない `<url>/amp` のページが本文として返ることはない

## データ抽出ツール

### `extract_structured_data`

CSSセレクターかLLMスキーマで構造化データを抽出します。

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
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（`table_data` を含む単一の `.json` ファイルを保存）

### `intelligent_extract`

自然言語で書いた目的に沿ってコンテンツを抽出します。フィルタの種類を選べ、LLMを使うかどうかも指定できます。

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
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（単一の `.json` ファイルを保存）

### `extract_entities`

Webページからメールアドレスや電話番号などのエンティティを、正規表現かLLMで抽出します。

**パラメータ:**
- `url` (str, 必須): 対象URL
- `entity_types` (List[str], 必須): 抽出するエンティティ種別。正規表現で対応する値: 'emails'、'phones'、'urls'、'dates'、'ips'、'prices'、'credit_cards'、'coordinates'。それ以外の種別（例: 'people'、'organizations'、'social_media'）は `use_llm=true` の場合のみ抽出
- `custom_patterns` (Optional[Dict]): `{type_name: regex}` 形式のカスタム正規表現パターン。`type_name` を `entity_types` に入れると抽出される。正規表現モードでのみ使う（デフォルト: None）
- `include_context` (bool): 各エンティティの周囲のコンテキストを含める（デフォルト: True）
- `deduplicate` (bool): 重複を除去（デフォルト: True）
- `use_llm` (bool): 固有表現抽出（NER）にLLMを使用（デフォルト: False）
- `llm_provider` (Optional[str]): LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): LLMモデル、未指定時は自動検出（デフォルト: None）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（単一の `.json` ファイルを保存）

**レスポンス動作:**
- 種別名の大文字と小文字は区別しない。単数形（'email'、'phone'、'url'、'date'、'ip'、'price'、'credit_card'、'coordinate'、'person'、'organization'、'location'、'product'、'event'）は複数形の名前に読み替える。'person' は 'people' になる。`entity_types_requested` には読み替えた後の名前が入る
- 正規表現モードでパターンのない種別を指定すると、その種別は `unsupported_entity_types` に入り、`warnings` にメッセージが付く

## ファイル処理ツール

### `process_file`

Microsoft MarkItDownでPDF、Word、Excel、PowerPoint、ZIPなどをMarkdownに変換します。

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
- `content_limit` (int): 返す最大文字数、0=無制限（デフォルト: 0）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

**対応形式:**

| 種類 | 拡張子 |
|------|--------|
| PDF | .pdf |
| Microsoft Office | .docx, .pptx, .xlsx, .xls |
| アーカイブ | .zip |
| Web、テキスト | .html, .htm, .txt, .md, .csv, .rtf |
| 電子書籍 | .epub |

`process_file` は `.html`/`.htm` をファイルとして変換しますが、`crawl_url` と `crawl_url_with_fallback` はリモートの `.html`/`.htm` URLをブラウザでレンダリングします。

### `get_supported_file_formats`

対応しているファイル形式と、形式ごとの機能を返します。パラメータはありません。

### `enhanced_process_large_content`

大きなコンテンツをチャンクに分け、フィルタで絞り込んでから処理します。

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
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（チャンクと要約を含む単一の `.json` ファイルを保存）

## YouTube処理ツール

### `extract_youtube_transcript`

YouTubeのトランスクリプトを取得します。言語の優先順位と翻訳先を指定できます。対象は公開キャプションのある動画で、取得に失敗したときはページのクロールに切り替えます。

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
- `content_limit` (int): 返す最大文字数、0=無制限（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（単一の `.md` ファイルを保存）

`metadata.published_at` には `timezone` に変換した公開日時が入ります。`metadata.published_at.original` には、YouTubeが返したオフセット付きの元の値が常に残ります。

### `extract_youtube_comments`

YouTube動画のコメントを並べ替えて取得します。comment_offsetで続きを取得できます。

**パラメータ:**
- `url` (str, 必須): YouTube動画URL
- `sort_by` (str): 並び順、'popular' または 'recent'（デフォルト: 'popular'）
- `max_comments` (int): 最大取得コメント数 1-1000（デフォルト: 300）
- `comment_offset` (int): スキップするコメント数（ページネーション用）（デフォルト: 0）
- `include_replies` (bool): 返信コメントを含める（デフォルト: True）
- `content_offset` (int): コンテンツ開始位置、0始まり（デフォルト: 0）
- `content_limit` (int): 返す最大文字数、0=無制限（デフォルト: 0）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（全コメントを含む単一の `.json` ファイルを、トークン制限によるコメント削減の前に保存）

### `batch_extract_youtube_transcripts`

複数のYouTube動画のトランスクリプトをまとめて取得します。1回の呼び出しで3本までです。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `urls` (List[str], 必須): YouTube動画URLのリスト（最大3件）
  - `languages` (List[str]): 優先言語コード順（デフォルト: ["ja", "en"]）
  - `include_timestamps` (bool): タイムスタンプを含める（デフォルト: False）
  - `timezone` (str): 各動画の公開日時を `metadata.published_at` に変換する際のIANAタイムゾーン名。`metadata.published_at.original` にはYouTubeのオフセットが保持されます（デフォルト: 'UTC'）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（動画ごとの `.md` ファイルと `index.json` をディレクトリに保存）
  - その他に受け付けるキー: `translate_to` (Optional[str]、デフォルト: None)、`preserve_formatting` (bool、デフォルト: True)、`include_metadata` (bool、デフォルト: True)、`max_concurrent` (int、デフォルト: 3、最大5)

### `get_youtube_video_info`

YouTube動画のメタデータと、トランスクリプトがどの言語で取れるかを返します。

**パラメータ:**
- `video_url` (str, 必須): YouTube動画URL
- `summarize_transcript` (bool): トランスクリプトを要約（デフォルト: False）
- `max_tokens` (int): 要約のトークン制限（デフォルト: 25000）
- `llm_provider` (Optional[str]): 要約用LLMプロバイダー、未指定時は自動検出（デフォルト: None）
- `llm_model` (Optional[str]): 要約用LLMモデル、未指定時は自動検出（デフォルト: None）
- `summary_length` (str): 要約の長さ、'short'、'medium'、'long' のいずれか（デフォルト: 'medium'）
- `include_timestamps` (bool): タイムスタンプを含める（デフォルト: True）
- `timezone` (str): `published_at` として返す公開日時の変換に使うIANAタイムゾーン名（例: 'Asia/Tokyo'、'America/New_York'）（デフォルト: 'UTC'）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（動画情報とトランスクリプトを含む単一の `.md` ファイルを保存）

**戻り値:**
- 利用可能なトランスクリプト言語
- 手動作成/自動生成の区別
- 翻訳可能な言語の情報
- 公開日時 `published_at`（`timezone` に変換済み。`published_at.original` にはYouTubeのオフセットを保持）

## 検索ツール

検索ツールは検索エンジンのAPIを呼び出します。対応しているエンジンは3つです。

| エンジン | `engine` の値 | APIキー | 備考 |
|---|---|---|---|
| Google Custom Search JSON API | `google_custom` | `GOOGLE_SEARCH_API_KEY` と `GOOGLE_SEARCH_ENGINE_ID` | 新規受付は終了しており、2027-01-01 に提供が終わる |
| Brave Search API | `brave` | `BRAVE_SEARCH_API_KEY` | クエリに検索演算子（`site:`、`filetype:`、`lang:`）を書ける。1リクエストで返るのは20件まで。`num_results` がそれより多いと、20件ごとに1リクエストを送る。リクエストは1本ずつ送り、HTTP 429 の `X-RateLimit-Reset` が2秒以下ならその秒数待って再試行する（2回まで） |
| Exa | `exa` | `EXA_API_KEY` | クエリを自然文として読む。`site:` と `-site:` はドメインの絞り込みとして送り、それ以外の演算子はクエリの文字列に残る。Exa が返せば、結果に `published_date` と `author` が付く |

エンジンの選び方:
- リクエストの `engine` キーで、その呼び出しのエンジンを選ぶ。指定がなければ環境変数 `SEARCH_ENGINE` に従う（デフォルト: `auto`）
- `auto` は、APIキーが設定されたエンジンを `SEARCH_ENGINE_ORDER` の順（デフォルト: `google_custom,brave,exa`）に試す。レート制限、キーの拒否、クレジット切れ、通信エラー、5xx で失敗したら次のエンジンへ進む。0件は正しい応答として扱い、次へは進まない。試した経過はレスポンスの `fallback_info.attempts` に入る
- APIキーのないエンジンを名指しすると、通信の前に `error_code: search_engine_not_configured` を返す。`engine` に未知の値を渡すと `error_code: invalid_search_engine` を返し、`valid_engines` に指定できる値を並べる
- レスポンスの `engine` キーに、実際に応答したエンジンが入る

共通のキーとエンジンごとのパラメータの対応:

| キー | `google_custom` | `brave` | `exa` |
|---|---|---|---|
| `language` | `hl` | `search_lang` | 使わない |
| `region` | `gl` | `country`（`uk` は `GB` に変える。Brave の一覧にないコードは `ALL`） | `userLocation` |
| `recent_days` | `dateRestrict` | `freshness` の日付範囲 | `startPublishedDate` |
| `safe_search` | `safe=active` | `safesearch=moderate` | `moderation` |

### `search_web`

Web検索を実行し、タイトル、スニペット、URL、メタデータを付けた結果を返します。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `query` (str, 必須): 検索クエリ文字列
  - `num_results` (int): 返す結果数 1-100（デフォルト: 10）
  - `search_genre` (Optional[str]): ジャンルフィルタ。`pdf`、`documents`、`presentations`、`spreadsheets`、`japanese`、`english` のいずれか（デフォルト: None）
  - `engine` (Optional[str]): `auto`、`google_custom`、`brave`、`exa` のいずれか（デフォルト: `SEARCH_ENGINE` の設定）
  - `result_type` (str または List[str]): `web`、`discussions`、`faq`、`videos`、`news` のいずれか。複数をリストかカンマ区切りの文字列で指定できる（デフォルト: "web"）。下記を参照
  - `language` (str): 検索言語（デフォルト: "en"）
  - `region` (str): 検索地域（デフォルト: "us"）
  - `recent_days` (Optional[int]): 直近N日間に限定（デフォルト: None）
  - `content_limit` (int): 結果ごとに返す最大文字数、0=無制限（デフォルト: 0）
  - `content_offset` (int): コンテンツ開始位置（デフォルト: 0）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（スライス前の全結果を単一の `.json` ファイルに保存）
  - その他に受け付けるキー: `safe_search` (bool、デフォルト: True)

**機能:**
- 各結果には `rank`、`url`、`domain`、`title`、`snippet` と、URLの種類を示す `type`（`academic`、`documentation`、`video` など）が入る
- 検索ジャンルは6つあり、`pdf`、`documents`、`presentations`、`spreadsheets` はファイル形式で、`japanese` と `english` は言語で絞る。`google_custom` と `brave` ではジャンルの検索演算子をクエリに付ける。`exa` にはファイル形式の絞り込みがないので、`exa` を名指ししてファイル形式のジャンルを指定すると `error_code: unsupported_search_genre` を返し、`auto` ではそのジャンルのとき `exa` を飛ばす。`exa` で `japanese` を指定すると地域を JP にし、`english` は何もしない
- `search_genre` は大文字と小文字を区別せずに照合する。一覧にない値を渡すと、通信を始める前に `success: false` と `error_code: invalid_search_genre` を返し、`valid_genres` に指定できる値を並べる。`batch_search_web` と `search_and_crawl` も `search_genre` と `engine` を同じように検査する
- エンジンが0件を返したときは、短くしたクエリの候補を `alternative_queries` に入れる
- セーフサーチはデフォルトで有効
- `result_type` で、Web の結果のほかに Brave の結果枠を受け取れる。`discussions` は Reddit のスレッドや GitHub Issues などの掲示板の投稿で、`forum_name`、`num_answers`、`score`、`question`、`top_comment` が付く。`faq` はページ内の Q&A の抜粋で、`question` と、回答を入れた `snippet` が付く。`videos` には `duration`、`creator`、`publisher`、`news` には `source`、`age`、`breaking` が付く。`web`、`discussions`、`faq`、`videos` は Brave の Web 検索1回でまとめて取る。`news` だけは Brave の News Search へ別にリクエストを送る（1回で50件まで）。Web 検索の news 枠は、具体的な出来事や概念を書いたクエリではほとんど空になるためである。News Search は日付のある記事全般を返すので、ブログやドキュメントのページも混ざる。日本のニュースを取るときは `region: jp` を指定する。`results` には、`web` を指定していれば Web の結果、指定していなければ先頭に書いた枠が入り、ほかの枠は `extra_results` に枠の名前ごとに入る（Brave が返さなかった枠は空のリスト）。掲示板の投稿と回答の本文は1,000文字で切る。`news` を追加の枠として指定し、そのリクエストだけが失敗したときは、ほかの結果を返し、エラーを `extra_errors` に入れる
- `web` 以外の枠を返せるのは `brave` だけなので、`auto` では Brave だけを使い、`google_custom` か `exa` を名指しすると `error_code: unsupported_result_type` を返す。Brave が `discussions`、`faq`、`videos` を返すのは一部のクエリだけで、`num_results` では減らすことしかできず、20件を超えるページ送りで増えるのは Web の結果だけになる。指定した枠がどれも空なら `success: false` になり、言い換えるか `web` を使うよう案内する。未知の枠を渡すと、通信の前に `error_code: invalid_result_type` を返す

### `search_google`

`search_web` の非推奨エイリアスです。`request` 辞書で受け付けるキー（`query`、`num_results`、`search_genre`、`engine`、`result_type`、`language`、`region`、`recent_days`、`content_limit`、`content_offset`、`output_path`、`include_content_in_response`、`overwrite`）も結果も `search_web` と同じです。エンジンも `search_web` と同じ規則で選ぶため、Google に限りません。既存のクライアントが動き続けるように残しています。

### `search_and_crawl`

Web検索を実行し、上位の結果をそのままクロールします。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `search_query` (str, 必須): 検索クエリ。書いたとおりにエンジンへ送る
  - `crawl_top_results` (int): クロール対象の上位結果数、最大3（デフォルト: 2）
  - `search_genre` (Optional[str]): ジャンルフィルタ。値は `search_web` と同じ（デフォルト: None）
  - `engine` (Optional[str]): 検索エンジン。値は `search_web` と同じ（デフォルト: `SEARCH_ENGINE` の設定）
  - `result_type` (str または List[str]): 結果の枠。扱いは `search_web` と同じ（デフォルト: "web"）。クロールするのは `results` のページで（`discussions` だけを指定すれば掲示板のスレッド）、`extra_results` はそのまま返す
  - `recent_days` (Optional[int]): 直近N日間に限定（デフォルト: None）
  - `generate_markdown` (bool): クロールしたページからMarkdownを生成（デフォルト: True）
  - `max_content_per_page` (int): レスポンス内のページごとの最大文字数（デフォルト: 5000）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（ページごとの `.md` ファイルと `index.json` をディレクトリに保存。`max_content_per_page` による切り詰めの前に保存）

**戻り値:**
- `engine`、`result_type`、検索のメタデータ、`search_results`、`extra_results`（`auto` で次のエンジンへ進んだときは `search_fallback_info` も入る）
- `crawled_pages`: クロールしたページごとの要素で、`search_rank`、`url`、`title`、`success`、`markdown`（`generate_markdown` が false なら `content`）、`content_length`、失敗時の `error`、元の検索結果の `search_result` を持つ。失敗したページや本文が空のページは、フォールバック用のクローラーで取り直し、`fallback_used` を付ける
- 成功率などの処理統計
- 検索結果とクロール結果をまとめた分析

### `batch_search_web`

複数のWeb検索を並行して実行します。1回の呼び出しで3クエリまでです。

**パラメータ:**
- `request` (Dict, 必須): 以下のキーを持つリクエスト辞書
  - `queries` (List[str], 必須): 検索クエリのリスト（最大3件）
  - `num_results_per_query` (int): クエリごとの結果数 1-100（デフォルト: 10）
  - `search_genre` (Optional[str]): 全クエリに適用するジャンルフィルタ。値は `search_web` と同じ（デフォルト: None）
  - `engine` (Optional[str]): 全クエリに使う検索エンジン。値は `search_web` と同じ（デフォルト: `SEARCH_ENGINE` の設定）
  - `result_type` (str または List[str]): 全クエリに使う結果の枠。扱いは `search_web` と同じ（デフォルト: "web"）。`results` の各要素に、それぞれの `extra_results` が入る
  - `recent_days` (Optional[int]): 直近N日間に限定（デフォルト: None）
  - `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（全結果を単一の `.json` ファイルに保存）
  - その他に受け付けるキー: `language` (str、デフォルト: "en")、`region` (str、デフォルト: "us")、`max_concurrent` (int、デフォルト: 3、最大5)

`results` の各要素に `engine` が入り、`auto` で次のエンジンへ進んだときは `fallback_info` も入ります。`auto` では、クエリごとに応答したエンジンが異なることがあります。

### `batch_search_google`

`batch_search_web` の非推奨エイリアスです。`request` 辞書で受け付けるキー（`queries`、`num_results_per_query`、`search_genre`、`engine`、`result_type`、`recent_days`、`output_path`、`include_content_in_response`、`overwrite`）も結果も `batch_search_web` と同じです。

### `get_search_genres`

`search_genre` に指定できるジャンルと、検索エンジンの設定を返します。パラメータはありません。

**レスポンスのキー:**
- `genres`: ジャンル名と説明の対応
- `query_operators`: ジャンル名と、クエリに付ける検索演算子の対応
- `configuration`: `default_engine`、`auto_order`、`configured_engines`（APIキーのあるエンジン）、`engines`、`result_types`、`genre_support`（エンジンごとのジャンルの扱い）、対応する言語と地域
- `total_genres`、`usage_example`、`notes`

## バッチ操作ツール

### `batch_crawl`

複数のURL（最大3件）を1件ずつ順にクロールします。失敗したURLにはフォールバックを試します。

**パラメータ:**
- `urls` (List[str], 必須): クロール対象URLのリスト（最大3件）
- `base_timeout` (int): URLごとのタイムアウト秒数（デフォルト: 30）
- `generate_markdown` (bool): Markdown生成（デフォルト: True）
- `extract_media` (bool): ページから画像/動画を抽出（デフォルト: False）
- `wait_for_js` (bool): JavaScript完了を待機（デフォルト: False）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（URLごとの `.md` ファイルと `index.json` をディレクトリに保存。戻り値はリストのままで、成功項目に `output_file` が追加される）

### `multi_url_crawl`

複数のURL（最大5件）を、URLごとに設定を変えてクロールします。

**パラメータ:**
- `url_configurations` (Dict, 必須): URLごとの設定マップ、最大5件（例: `{"https://site1.com": {"wait_for_js": true}}`）
- `pattern_matching` (str): URLパターンの照合方式、'wildcard' または 'regex'（デフォルト: 'wildcard'）
- `default_config` (Optional[Dict]): 全URLに適用するデフォルト設定（デフォルト: None）
- `base_timeout` (int): URLごとのタイムアウト秒数（デフォルト: 30）
- `max_concurrent` (int): 互換性のために残しているだけで効果はなく、URLは1件ずつ順に処理する（デフォルト: 3）
- `output_path`、`include_content_in_response`、`overwrite`: [結果のディスク保存](#結果のディスク保存output_path)を参照（URLごとの `.md` ファイルと `index.json` をディレクトリに保存。戻り値はリストのままで、成功項目に `output_file` が追加される）

## ツール分類

### カテゴリ別

| カテゴリ | ツール | ツール数 |
|---------|--------|---------|
| Webクローリング | `crawl_url`, `deep_crawl_site`, `crawl_url_with_fallback` | 3 |
| データ抽出 | `intelligent_extract`, `extract_entities`, `extract_structured_data` | 3 |
| YouTube | `extract_youtube_transcript`, `batch_extract_youtube_transcripts`, `get_youtube_video_info`, `extract_youtube_comments` | 4 |
| 検索 | `search_web`, `batch_search_web`, `search_and_crawl`, `get_search_genres`, `search_google`（非推奨）, `batch_search_google`（非推奨） | 6 |
| ファイル処理 | `process_file`, `get_supported_file_formats`, `enhanced_process_large_content` | 3 |
| バッチ操作 | `batch_crawl`, `multi_url_crawl` | 2 |
| **合計** | | **21** |

### 複雑さ別

| 複雑さ | ツール |
|------|--------|
| 単純 | `crawl_url`, `process_file`, `get_youtube_video_info`, `get_search_genres`, `get_supported_file_formats` |
| 中程度 | `deep_crawl_site`, `search_web`, `extract_youtube_transcript`, `extract_youtube_comments`, `batch_crawl` |
| 複雑 | `extract_structured_data`, `intelligent_extract`, `extract_entities`, `search_and_crawl`, `batch_search_web`, `multi_url_crawl`, `batch_extract_youtube_transcripts`, `enhanced_process_large_content` |

### 用途別

| 用途 | ツール |
|------|--------|
| コンテンツを探す | `search_web`, `search_and_crawl`, `batch_search_web`, `get_search_genres` |
| データを抽出する | `crawl_url`, `extract_structured_data`, `intelligent_extract`, `extract_entities` |
| 動画やファイルを処理する | `extract_youtube_transcript`, `extract_youtube_comments`, `batch_extract_youtube_transcripts`, `process_file`, `get_supported_file_formats` |
| サイトを調べる | `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`, `multi_url_crawl` |
| 大きなコンテンツを扱う | `enhanced_process_large_content`, `get_youtube_video_info` |

## 統合例

- クライアントごとの設定は[設定例](CONFIGURATION_EXAMPLES.md)
- HTTP経由での呼び出し方は[HTTP統合ガイド](HTTP_INTEGRATION.md)
- 応用的な使い方は[使い方の応用](ADVANCED_USAGE.md)
