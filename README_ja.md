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
  - ZIPアーカイブ内の複数ファイル一括処理
- **🔍 Google検索統合機能**
  - 31種類の検索ジャンル指定（学術、プログラミング、ニュース等）
  - 検索結果からのタイトル・スニペット自動抽出
  - セーフサーチ原則有効化でセキュリティ確保
  - バッチ検索と結果分析機能
  - 検索+クローリング統合処理
- **📺 YouTube動画処理機能 [非推奨]**
  - ⚠️ 現在、YouTubeのAPI仕様変更により非推奨
  - YouTube動画のトランスクリプト（字幕）自動抽出（不安定）
  - 多言語サポートと自動翻訳機能（制限あり）
  - タイムスタンプ付きまたはクリーンテキスト出力
  - バッチ処理による複数動画の一括処理（非推奨）
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

### Claude Desktop での使用

Windows環境でClaude Desktopから使用する場合：

1. `claude_desktop_config_windows.json` をClaude Desktopの設定ディレクトリにコピー
2. Claude Desktopを再起動
3. チャットでcrawl4aiツールが利用可能になります

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

#### `extract_structured_data`
各種戦略を使用した構造化データ抽出

**パラメータ:**
- `url`: クローリング対象のURL
- `schema`: 抽出用JSONスキーマ
- `extraction_type`: 抽出タイプ（'css'または'llm'）
- `css_selectors`: 各フィールド用CSSセレクター（オプション）
- `llm_provider`: LLMプロバイダー（オプション）
- `llm_model`: LLMモデル名（オプション）

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

#### `process_file`
**📄 ファイル処理**: Microsoft MarkItDownを使用した各種ファイル形式の処理・変換

**パラメータ:**
- `url`: 処理対象ファイルのURL（PDF、Office、ZIP等）
- `max_size_mb`: 最大ファイルサイズ（MB、デフォルト：100MB）
- `extract_all_from_zip`: ZIPアーカイブの全ファイル抽出（デフォルト：True）
- `include_metadata`: メタデータの取得（デフォルト：True）

**サポートファイル形式:**
- **PDF**: .pdf
- **Microsoft Office**: .docx, .pptx, .xlsx, .xls
- **アーカイブ**: .zip
- **Web/テキスト**: .html, .htm, .txt, .md, .csv, .rtf
- **eBook**: .epub

#### `get_supported_file_formats`
**📋 サポート形式一覧**: ファイル処理でサポートされている形式とその詳細を取得

**戻り値:**
- サポートされている全ファイル形式のリスト
- 各形式の機能と特徴
- 最大ファイルサイズ制限
- 追加機能の説明

#### `extract_youtube_transcript` [非推奨]
**📺 YouTube動画処理**: YouTube動画からトランスクリプト（字幕）を抽出・変換

**⚠️ 注意: YouTubeのAPI仕様変更により現在非推奨。使用は推奨されません。**

**パラメータ:**
- `url`: YouTube動画のURL
- `languages`: 優先言語リスト（デフォルト：["ja", "en"]）
- `translate_to`: 翻訳先言語（オプション）
- `include_timestamps`: タイムスタンプを含めるか
- `preserve_formatting`: 元の書式を保持するか
- `include_metadata`: 動画メタデータを含めるか

#### `batch_extract_youtube_transcripts` [非推奨]
**📺 一括YouTube処理**: 複数のYouTube動画を並行処理でトランスクリプト抽出

**⚠️ 注意: YouTubeのAPI仕様変更により現在非推奨。バッチ処理は特に不安定です。**

**パラメータ:**
- `urls`: YouTube動画URLのリスト
- `languages`: 優先言語リスト
- `translate_to`: 翻訳先言語（オプション）
- `include_timestamps`: タイムスタンプを含めるか
- `max_concurrent`: 最大並行処理数（1-10）

#### `get_youtube_video_info`
**📋 YouTube情報取得**: 動画の利用可能なトランスクリプト言語等の情報を取得

**パラメータ:**
- `video_url`: YouTube動画のURL

**戻り値:**
- 利用可能なトランスクリプト言語
- 手動/自動生成の区別
- 翻訳可能言語の情報

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

## 📺 YouTube動画処理機能の使用例 [非推奨]

**⚠️ 重要: 以下のYouTube機能は現在非推奨です。YouTubeのAPI仕様変更により不安定な状態です。**

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
│   ├── youtube_processor.py     # YouTubeトランスクリプト処理モジュール [非推奨]
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
- `youtube-transcript-api>=1.0.3` - YouTubeトランスクリプト抽出 [非推奨]
- `asyncio` - 非同期処理
- `typing-extensions` - 型ヒント拡張

**⚠️ YouTube機能に関する注意事項:**
- YouTubeのAPI仕様変更により、一時的にトランスクリプト抽出が失敗する場合があります
- その場合は別の動画で試すか、時間をおいて再試行してください
- 動画情報の取得は通常通り機能します

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