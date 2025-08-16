# APIリファレンス

Crawl4AI MCPサーバーで利用可能なすべてのMCPツールの完全リファレンスです。

## 🛠️ ツール選択ガイド

### 📋 **用途に応じた適切なツール選択**

| **用途** | **推奨ツール** | **主要機能** |
|-------------|---------------------|------------------|
| 単一ページ | `crawl_url` | 基本クローリング、JS対応 |
| 複数ページ (最大5) | `deep_crawl_site` | サイトマッピング、リンクフォロー |
| 検索+クローリング | `search_and_crawl` | Google検索+自動クロール |
| 困難サイト | `crawl_url_with_fallback` | 複数リトライ戦略 |
| 特定データ抽出 | `intelligent_extract` | AI搭載抽出 |
| パターン検索 | `extract_entities` | メール、電話、URL等 |
| 構造化データ | `extract_structured_data` | CSS/XPath/LLMスキーマ |
| ファイル処理 | `process_file` | PDF、Office、ZIP変換 |
| YouTube処理 | `extract_youtube_transcript` | 字幕抽出 |

### ⚡ **パフォーマンス指針**

- **深度クローリング**: 最大5ページ制限（安定性重視）
- **バッチ処理**: 同時実行制限あり
- **タイムアウト計算**: `ページ数 × base_timeout` 推奨
- **大容量ファイル**: 最大100MB制限
- **リトライ戦略**: 初回失敗時は手動リトライ推奨

### 🎯 **ベストプラクティス**

**JavaScript重要サイト向け:**
- 常に `wait_for_js: true` を使用
- `simulate_user: true` で互換性向上
- タイムアウトを30-60秒に増加
- 特定要素には `wait_for_selector` を使用

**AI機能使用時:**
- `get_llm_config_info` でLLM設定確認
- 大容量ドキュメントには `auto_summarize: true` で自動要約使用
- LLM利用不可時は非AI機能にフォールバック
- 意味的理解には `intelligent_extract` を使用
- 用途に応じて要約の長さをカスタマイズ（'short'は概要用、'long'は詳細分析用）

## 🔧 Webクローリングツール

### `crawl_url`

高度なWebクローリング機能。深度クローリング対応、インテリジェントフィルタリング、大容量コンテンツの自動要約機能付き。

**主要パラメータ:**
- `url`: クロール対象URL
- `max_depth`: 最大クローリング深度（単一ページの場合はNone）
- `crawl_strategy`: 戦略タイプ（'bfs', 'dfs', 'best_first'）
- `content_filter`: フィルタタイプ（'bm25', 'pruning', 'llm'）
- `chunk_content`: 大容量ドキュメントのコンテンツ分割有効化
- `execute_js`: カスタムJavaScriptコード実行
- `user_agent`: カスタムユーザーエージェント文字列
- `headers`: カスタムHTTPヘッダー
- `cookies`: 認証用クッキー
- `auto_summarize`: LLMを使用した大容量コンテンツの自動要約
- `max_content_tokens`: 自動要約トリガーの最大トークン数（デフォルト: 15000）
- `summary_length`: 要約の長さ設定（'short', 'medium', 'long'）
- `llm_provider`: 要約用LLMプロバイダー（未指定時は自動検出）
- `llm_model`: 要約用特定LLMモデル（未指定時は自動検出）

### `deep_crawl_site`

包括的なサイトマッピングと再帰的クローリング専用ツール。

**パラメータ:**
- `url`: 開始URL
- `max_depth`: 最大クローリング深度（推奨: 1-3）
- `max_pages`: クロール対象最大ページ数
- `crawl_strategy`: クローリング戦略（'bfs', 'dfs', 'best_first'）
- `url_pattern`: URLフィルタパターン（例：'*docs*', '*blog*'）
- `score_threshold`: 最小関連性スコア（0.0-1.0）

### `crawl_url_with_fallback`

最大信頼性のための複数フォールバック戦略を持つ堅牢なクローリング。

### `batch_crawl`

統一レポート付きの複数URL並列処理。

## 🧠 AI搭載抽出ツール

### `intelligent_extract`

高度フィルタリングと分析機能付きのAI搭載コンテンツ抽出。

**パラメータ:**
- `url`: 対象URL
- `extraction_goal`: 抽出対象の説明
- `content_filter`: コンテンツ品質のフィルタタイプ
- `use_llm`: LLMベースのインテリジェント抽出有効化
- `llm_provider`: LLMプロバイダー（openai, claude等）
- `custom_instructions`: 詳細抽出指示

### `extract_entities`

正規表現パターンを使用した高速エンティティ抽出。

**内蔵エンティティタイプ:**
- `emails`: メールアドレス
- `phones`: 電話番号
- `urls`: URLとリンク
- `dates`: 日付形式
- `ips`: IPアドレス
- `social_media`: ソーシャルメディアハンドル（@username, #hashtag）
- `prices`: 価格情報
- `credit_cards`: クレジットカード番号
- `coordinates`: 地理座標

### `extract_structured_data`

CSS/XPathセレクターまたはLLMスキーマを使用した従来型構造化データ抽出。

## 📄 ファイル処理ツール

### `process_file`

**📄 ファイル処理**: Microsoft MarkItDownを使用した様々なファイル形式のMarkdown変換。

**パラメータ:**
- `url`: ファイルURL（PDF、Office、ZIP等）
- `max_size_mb`: 最大ファイルサイズ制限（デフォルト: 100MB）
- `extract_all_from_zip`: ZIPアーカイブからの全ファイル抽出
- `include_metadata`: レスポンスにファイルメタデータを含める

**対応形式:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **アーカイブ**: .zip
- **Web/テキスト**: .html, .htm, .txt, .md, .csv, .rtf
- **電子書籍**: .epub

### `get_supported_file_formats`

**📋 形式情報**: 対応ファイル形式と機能の包括的リストを取得。

## 📺 YouTube処理ツール

### `extract_youtube_transcript`

**📺 YouTube処理**: youtube-transcript-api v1.1.0+を使用した言語設定と翻訳機能付きYouTube動画トランスクリプト抽出。

**✅ 安定して信頼性が高い - 認証不要！**

**パラメータ:**
- `url`: YouTube動画URL
- `languages`: 優先言語の順序リスト（デフォルト: ["ja", "en"]）
- `translate_to`: 翻訳対象言語（オプション）
- `include_timestamps`: トランスクリプトにタイムスタンプを含める
- `preserve_formatting`: 元の書式を保持
- `include_metadata`: 動画メタデータを含める

### `batch_extract_youtube_transcripts`

**📺 YouTubeバッチ処理**: 複数YouTube動画のトランスクリプトを並列抽出。

**✅ 安定したバッチ処理のための制御された同時実行による性能向上。**

**パラメータ:**
- `urls`: YouTube動画URLのリスト
- `languages`: 優先言語リスト
- `translate_to`: 翻訳対象言語（オプション）
- `include_timestamps`: トランスクリプトにタイムスタンプを含める
- `max_concurrent`: 最大同時リクエスト数（1-5、デフォルト: 3）

### `get_youtube_video_info`

**📋 YouTube情報**: 完全なトランスクリプト抽出なしでYouTube動画の利用可能トランスクリプト情報を取得。

**パラメータ:**
- `video_url`: YouTube動画URL

**戻り値:**
- 利用可能トランスクリプト言語
- 手動/自動生成の区別
- 翻訳可能言語情報

## 🔍 Google検索ツール

### `search_google`

**🔍 Google検索**: ジャンルフィルタリングとメタデータ抽出機能付きGoogle検索実行。

**パラメータ:**
- `query`: 検索クエリ文字列
- `num_results`: 返す結果数（1-100、デフォルト: 10）
- `language`: 検索言語（デフォルト: "en"）
- `region`: 検索地域（デフォルト: "us"）
- `search_genre`: コンテンツジャンルフィルタ（オプション）
- `safe_search`: セーフサーチ有効（セキュリティのため常にTrue）

**機能:**
- 検索結果からの自動タイトル・スニペット抽出
- Google公式オペレーターを使用した7つの最適化された検索ジャンル
- URL分類とドメイン分析
- デフォルトでセーフサーチ強制

### `batch_search_google`

**🔍 Google一括検索**: 包括的分析付き複数Google検索実行。

**パラメータ:**
- `queries`: 検索クエリのリスト
- `num_results_per_query`: クエリあたりの結果数（1-100、デフォルト: 10）
- `max_concurrent`: 最大同時検索数（1-5、デフォルト: 3）
- `language`: 検索言語（デフォルト: "en"）
- `region`: 検索地域（デフォルト: "us"）
- `search_genre`: コンテンツジャンルフィルタ（オプション）

**戻り値:**
- 各クエリの個別検索結果
- クエリ間分析と統計
- ドメイン分布と結果タイプ分析

### `search_and_crawl`

**🔍 統合検索+クロール**: Google検索と上位結果の自動クロール実行。

**パラメータ:**
- `search_query`: Google検索クエリ
- `num_search_results`: 検索結果数（1-20、デフォルト: 5）
- `crawl_top_results`: クロール対象上位結果数（1-10、デフォルト: 3）
- `extract_media`: クロールページからのメディア抽出
- `generate_markdown`: マークダウンコンテンツ生成
- `search_genre`: コンテンツジャンルフィルタ（オプション）

**戻り値:**
- 完全な検索メタデータとクロールコンテンツ
- 成功率と処理統計
- 検索・クロール結果の統合分析

### `get_search_genres`

**📋 検索ジャンル**: 利用可能検索ジャンルと説明の包括的リストを取得。

**戻り値:**
- 説明付き7つの最適化された検索ジャンル
- 分類されたジャンルリスト（ファイル形式、時間基準、言語・地域）
- 各ジャンルタイプの使用例

## 📚 MCPリソース

### 利用可能リソース

- `uri://crawl4ai/config`: デフォルトクローラー設定オプション
- `uri://crawl4ai/examples`: 使用例とサンプルリクエスト

## 🎯 MCPプロンプト

### 利用可能プロンプト

- `crawl_website_prompt`: ガイド付きWebサイトクローリングワークフロー
- `analyze_crawl_results_prompt`: クロール結果分析
- `batch_crawl_setup_prompt`: バッチクローリングセットアップ

## 📊 ツール分類

### 複雑さ別
- **簡単**: `crawl_url`, `extract_entities`, `process_file`
- **中程度**: `deep_crawl_site`, `search_google`, `extract_youtube_transcript`
- **高度**: `intelligent_extract`, `search_and_crawl`, `batch_crawl`

### 用途別
- **コンテンツ発見**: `search_google`, `search_and_crawl`
- **データ抽出**: `crawl_url`, `intelligent_extract`, `extract_entities`
- **バッチ処理**: `batch_crawl`, `batch_search_google`, `batch_extract_youtube_transcripts`
- **メディア処理**: `extract_youtube_transcript`, `process_file`
- **サイト分析**: `deep_crawl_site`, `crawl_url_with_fallback`

## 🔧 統合例

詳細な設定例については、[設定例](CONFIGURATION_EXAMPLES.md)をご覧ください。

HTTP API統合については、[HTTP統合ガイド](HTTP_INTEGRATION.md)をご覧ください。

高度な使用パターンについては、[高度な使用ガイド](ADVANCED_USAGE.md)をご覧ください。