# Project Structure

このプロジェクトの構造と各ディレクトリの役割について説明します。

## 📁 ディレクトリ構造

```
crawl/
├── README.md                    # プロジェクトのメイン説明（英語）
├── README_ja.md                # プロジェクトのメイン説明（日本語）
├── requirements.txt            # Python依存関係
├── crawl4ai_mcp/              # メインのMCPサーバーコード
│   ├── __init__.py
│   ├── config.py              # 設定管理
│   ├── file_processor.py      # ファイル処理機能
│   ├── google_search_processor.py  # Google検索機能
│   ├── server.py              # メインMCPサーバー
│   ├── strategies.py          # 抽出戦略
│   ├── suppress_output.py     # 出力抑制ユーティリティ
│   └── youtube_processor.py   # YouTube処理機能
├── configs/                   # 設定ファイル
│   ├── claude_desktop_config.json                # STDIO設定
│   ├── claude_desktop_config_pure_http.json     # Pure HTTP設定
│   ├── claude_desktop_config_script.json        # スクリプト設定
│   └── claude_desktop_config_windows.json       # Windows設定
├── docs/                      # ドキュメント
│   ├── CHANGELOG.md           # 変更履歴
│   ├── HTTP_API_GUIDE.md      # HTTP API ガイド
│   ├── HTTP_SERVER_USAGE.md   # HTTPサーバー使用方法
│   ├── PURE_STREAMABLE_HTTP.md # Pure StreamableHTTP実装ガイド
│   ├── YOUTUBE_SETUP_2025.md  # YouTube API セットアップ
│   ├── setup_instructions_ja.md # セットアップ手順（日本語）
│   └── troubleshooting_ja.md  # トラブルシューティング（日本語）
├── examples/                  # サンプルコードとデバッグツール
│   ├── check_api_keys.py      # APIキー確認ツール
│   ├── debug_extraction.py   # 抽出デバッグツール
│   ├── pure_http_test.py      # Pure HTTPテストクライアント
│   ├── pure_streamable_http_server.py  # Pure HTTPサーバー（フル機能）
│   ├── run_http_server.py     # Legacy HTTPサーバー起動
│   ├── simple_http_test.py    # シンプルHTTPテスト
│   ├── simple_pure_http_server.py     # Pure HTTPサーバー（シンプル）
│   └── working_http_test.py   # 動作確認HTTPテスト
├── scripts/                   # 実行スクリプト
│   ├── run_http_server.sh     # HTTPサーバー起動スクリプト
│   ├── run_server.sh          # 汎用サーバー起動スクリプト
│   ├── setup.sh               # Linux/macOS セットアップスクリプト
│   ├── setup_windows.bat      # Windows セットアップスクリプト
│   └── start_pure_http_server.sh  # Pure HTTPサーバー起動スクリプト
├── tests/                     # テストファイル
│   ├── test_*.py              # 各種機能テスト
│   └── transcript_*.json      # テスト用データ
└── venv/                      # Python仮想環境
```

## 🎯 主要ファイルの説明

### 📋 メインファイル
- **README.md/README_ja.md**: プロジェクトの概要と使用方法
- **requirements.txt**: Python依存関係リスト

### 🏗️ コアモジュール (crawl4ai_mcp/)
- **server.py**: MCPサーバーのメイン実装
- **config.py**: 設定管理とLLMプロバイダー設定
- **youtube_processor.py**: YouTube字幕抽出（youtube-transcript-api使用）
- **google_search_processor.py**: Google検索統合機能
- **file_processor.py**: ファイル処理（MarkItDown統合）

### ⚙️ 設定ファイル (configs/)
- **claude_desktop_config_pure_http.json**: **推奨** - Pure StreamableHTTP用
- **claude_desktop_config.json**: 従来のSTDIO設定
- **claude_desktop_config_windows.json**: Windows専用設定
- **claude_desktop_config_script.json**: スクリプト実行用設定

### 📚 ドキュメント (docs/)
- **PURE_STREAMABLE_HTTP.md**: Pure StreamableHTTP実装の詳細ガイド
- **HTTP_SERVER_USAGE.md**: HTTPサーバーの使用方法
- **CHANGELOG.md**: バージョン履歴と変更点
- **YOUTUBE_SETUP_2025.md**: YouTube API設定（非推奨）

### 🧪 サンプル・デバッグ (examples/)
- **simple_pure_http_server.py**: **推奨** - Pure StreamableHTTPサーバー
- **pure_http_test.py**: Pure HTTPテストクライアント
- **check_api_keys.py**: API設定確認ツール

### 🔧 スクリプト (scripts/)
- **start_pure_http_server.sh**: **推奨** - Pure HTTPサーバー起動
- **setup.sh**: 自動セットアップスクリプト

### 🧪 テスト (tests/)
- **test_*.py**: 各種機能の単体・統合テスト
- 機能別にテストファイルが整理されています

## 🚀 推奨使用方法

### 1. Pure StreamableHTTP方式（推奨）
```bash
# サーバー起動
./scripts/start_pure_http_server.sh

# 設定適用
cp configs/claude_desktop_config_pure_http.json ~/.config/claude-desktop/claude_desktop_config.json
```

### 2. 従来のSTDIO方式
```bash
# 設定適用のみ
cp configs/claude_desktop_config.json ~/.config/claude-desktop/claude_desktop_config.json
```

## 📝 開発・デバッグ

### テスト実行
```bash
# Pure HTTPテスト
python examples/pure_http_test.py

# 機能別テスト
python tests/test_youtube_transcript.py
python tests/test_google_search_integration.py
```

### API確認
```bash
# API設定確認
python examples/check_api_keys.py

# 抽出デバッグ
python examples/debug_extraction.py
```

この構造により、プロジェクトがより整理され、使いやすくなりました。各ファイルの役割が明確になり、新規ユーザーも簡単に必要なファイルを見つけることができます。