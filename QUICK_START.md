# Quick Start Guide

## 🚀 最速セットアップ (Pure StreamableHTTP)

### 1. サーバー起動
```bash
./scripts/start_pure_http_server.sh
```

### 2. Claude Desktop設定
```bash
# Linux/macOS
cp configs/claude_desktop_config_pure_http.json ~/.config/claude-desktop/claude_desktop_config.json

# Windows
cp configs/claude_desktop_config_pure_http.json %APPDATA%\Claude\claude_desktop_config.json
```

### 3. Claude Desktop再起動

### 4. 動作確認
```bash
curl http://127.0.0.1:8000/health
python examples/pure_http_test.py
```

## 📂 プロジェクト構造

```
crawl/
├── README.md                   # プロジェクト概要
├── configs/                    # 設定ファイル
│   └── claude_desktop_config_pure_http.json  # Pure HTTP設定
├── scripts/                    # 実行スクリプト
│   └── start_pure_http_server.sh             # サーバー起動
├── examples/                   # サンプル・テスト
│   ├── simple_pure_http_server.py            # HTTPサーバー
│   └── pure_http_test.py                     # テストクライアント
├── docs/                       # 詳細ドキュメント
└── tests/                      # 機能テスト
```

## 🔧 開発・デバッグ

### テスト実行
```bash
python examples/pure_http_test.py           # HTTP APIテスト
python tests/test_youtube_transcript.py     # YouTube機能テスト
python tests/test_google_search_integration.py  # Google検索テスト
```

### API確認
```bash
python examples/check_api_keys.py           # API設定確認
```

## 📚 詳細情報

- **完全ガイド**: [README.md](README.md)
- **HTTP実装**: [docs/PURE_STREAMABLE_HTTP.md](docs/PURE_STREAMABLE_HTTP.md)
- **構造説明**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)