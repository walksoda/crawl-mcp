# Crawl-MCP: crawl4ai用非公式MCPサーバー

> **⚠️ 重要**: これは優れた[crawl4ai](https://github.com/unclecode/crawl4ai)ライブラリの**非公式**MCPサーバー実装です。  
> **元のcrawl4aiプロジェクトとは無関係**です。

強力なcrawl4aiライブラリをラップする包括的なModel Context Protocol（MCP）サーバーです。このサーバーは、標準化されたMCPインターフェイスを通じて高度なWebクローリング、コンテンツ抽出、AI搭載の分析機能を提供します。

## 🚀 クイックスタート

### 前提条件（最初に必須）

**Playwright用システム依存関係のインストール:**

**Linux/macOS:**
```bash
sudo bash scripts/prepare_for_uvx_playwright.sh
```

**Windows（管理者として）:**
```powershell
scripts/prepare_for_uvx_playwright.ps1
```

### インストール

**UVX（推奨 - 最も簡単）:**
```bash
# 上記のシステム準備後 - これだけ！
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

### Claude Desktopセットアップ

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
      ],
      "env": {
        "CRAWL4AI_LANG": "ja"
      }
    }
  }
}
```

**英語インターフェイス用:**
```json
"env": {
  "CRAWL4AI_LANG": "en"
}
```

## ✨ 主要機能

- **🌐 高度なWebクローリング** - JavaScript重要サイト、SPA、動的コンテンツ
- **🧠 AI搭載抽出** - LLMベースのコンテンツ分析と要約
- **📄 ファイル処理** - PDF、Office文書、ZIPアーカイブ（Microsoft MarkItDown）
- **📺 YouTubeトランスクリプト** - 多言語、タイムスタンプ付き抽出
- **🔍 Google検索統合** - 31の検索ジャンルとメタデータ抽出
- **🔄 バッチ処理** - 複数URL、検索クエリ、動画トランスクリプト
- **🌍 多言語** - 英語と日本語インターフェイスサポート

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

### Webクローリング
- `crawl_url` - JavaScript対応の単一ページクローリング
- `deep_crawl_site` - 複数ページのサイトマッピングと探索
- `crawl_url_with_fallback` - リトライ戦略付き堅牢なクローリング
- `batch_crawl` - 複数URLの同時処理

### AI搭載分析
- `intelligent_extract` - カスタム指示付き意味的コンテンツ抽出
- `auto_summarize` - 大容量コンテンツのLLMベース要約
- `extract_entities` - パターンベースエンティティ抽出（メール、電話、URL等）

### メディア処理
- `process_file` - PDF、Office文書、ZIPアーカイブのMarkdown変換
- `extract_youtube_transcript` - 多言語トランスクリプト抽出
- `batch_extract_youtube_transcripts` - 複数動画の処理

### 検索統合
- `search_google` - ジャンルフィルタ付きGoogle検索とメタデータ
- `search_and_crawl` - 検索とコンテンツ抽出の組み合わせ
- `batch_search_google` - 複数検索クエリと分析

## 🎯 一般的な使用例

**コンテンツ研究:**
```bash
search_and_crawl → intelligent_extract → 構造化分析
```

**ドキュメントマイニング:**
```bash
deep_crawl_site → バッチ処理 → 包括的抽出
```

**メディア分析:**
```bash
extract_youtube_transcript → auto_summarize → 洞察生成
```

**競合インテリジェンス:**
```bash
batch_crawl → extract_entities → 比較分析
```

## 🚨 クイックトラブルシューティング

**インストールの問題:**
1. システム診断を実行: `get_system_diagnostics`ツールを使用
2. 適切な権限でセットアップスクリプトを再実行
3. 開発インストール手法を試行

**パフォーマンスの問題:**
- JavaScript重要サイトには`wait_for_js: true`を使用
- 読み込みが遅いページではタイムアウトを増加
- 大容量コンテンツには`auto_summarize`を有効化

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