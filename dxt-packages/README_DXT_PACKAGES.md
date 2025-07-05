# DXTパッケージディレクトリ

このディレクトリには、Claude Desktop用のDXT (Desktop Extensions) パッケージが格納されています。

## 📦 DXTパッケージとは

DXT (Desktop Extensions) は、Claude Desktop用の拡張機能パッケージ形式です。
`.dxt` ファイルをClaude Desktopにインストールすることで、ワンクリックでMCPサーバーの機能を追加できます。

## 📁 ディレクトリ構成

```
dxt-packages/
├── README_DXT_PACKAGES.md       # このファイル（DXT説明）
├── FINAL_RELEASE_NOTES.md       # リリースノート
└── crawl4ai-dxt-correct/        # Crawl4AI DXTパッケージ v1.0.7
    ├── crawl4ai-dxt-correct.dxt  # 📦 インストールファイル（Claude Desktopにドラッグ&ドロップ）
    ├── manifest.json             # DXT設定ファイル
    ├── README.md                 # ユーザー向け使用説明書
    ├── requirements.txt          # Python依存関係
    └── server/                   # MCPサーバー実装
        ├── main.py               # エントリーポイント
        └── crawl4ai_mcp/         # MCPサーバーモジュール
```

## 🚀 インストール方法

### 1. DXTファイルのインストール
1. `crawl4ai-dxt-correct/crawl4ai-dxt-correct.dxt` をダウンロード
2. Claude Desktop を開く
3. 設定 → 拡張機能 を開く
4. DXTファイルをドラッグ&ドロップ
5. 自動インストール完了！

### 2. 使用開始
Claude Desktopのチャットで以下のような指示が可能になります：
- "このウェブサイトの内容を抽出して: https://example.com"
- "YouTube動画の字幕を取得して: https://youtu.be/VIDEO_ID"
- "AI論文を検索して上位結果を分析して"

## 🛠️ 含まれる機能

### 🚀 **完全JavaScript対応**
- **React、Vue、Angular SPA**完全対応
- **動的コンテンツローディング**自動待機
- **カスタムJavaScript実行**
- **人間的ブラウジングシミュレーション**

### Webクローリング
- 高度なクローリング（完全JavaScript実行サポート）
- 深度クローリング（サイトマップ作成、最大5ページ）
- AI搭載コンテンツ抽出
- 複数リトライ戦略

### メディア処理
- YouTube字幕抽出（認証不要、安定版）
- PDF/Office文書のMarkdown変換（自動復旧機能付き）
- Google検索との統合（31ジャンル対応）

### データ抽出
- エンティティ抽出（メール、電話、URL等）
- 構造化データ抽出（CSS/XPath/LLM）
- バッチ処理機能（同時実行制御）

## 📋 システム要件

- **Claude Desktop**: v0.10.0以上
- **Python**: 3.8.0以上
- **プラットフォーム**: Windows、macOS、Linux
- **メモリ**: 512MB以上
- **ディスク容量**: 1GB以上

## 🔧 技術仕様

- **DXT仕様**: v0.1（Anthropic公式）
- **MCPフレームワーク**: FastMCP
- **パッケージサイズ**: 119.8kB
- **ツール数**: 19個
- **プロンプト数**: 4個
- **リソース数**: 3個

## 📚 参考資料

- **使用方法**: `crawl4ai-dxt-correct/README.md`
- **技術詳細**: `FINAL_RELEASE_NOTES.md`
- **作成ガイド**: `../DXT_CREATION_GUIDE.md`
- **トラブルシューティング**: `../DXT_TROUBLESHOOTING_GUIDE.md`

## 🏷️ バージョン情報

- **現在のバージョン**: v1.0.7
- **リリース日**: 2025年6月29日
- **ステータス**: 本番環境対応・安定版

このDXTパッケージは完全にテスト済みで、本番環境での使用に適しています。