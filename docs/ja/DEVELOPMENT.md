# 開発ガイド

Crawl4AI MCPサーバープロジェクトでの開発と貢献のための完全ガイドです。

## 🏗️ プロジェクトアーキテクチャ

### リポジトリ構造

```
crawl-mcp/
├── crawl4ai_mcp/              # メインサーバー実装
│   ├── server.py              # 主要MCPサーバー
│   ├── config.py              # 設定管理
│   ├── tools/                 # MCPツール実装
│   ├── prompts/               # MCPプロンプト定義
│   ├── models/                # データモデルとスキーマ
│   ├── file_processor.py      # ファイル処理ロジック
│   ├── youtube_processor.py   # YouTube統合
│   ├── google_search_processor.py # Google検索統合
│   └── strategies.py          # クローリング戦略
├── Dockerfile                 # Docker設定ファイル
├── docker-compose.yml         # Docker Compose設定
├── .dockerignore              # Docker除外設定
├── scripts/                   # セットアップとユーティリティスクリプト
├── configs/                   # 設定例
├── examples/                  # 使用例とテスト
├── docs/                      # ドキュメント
├── requirements.txt           # Python依存関係
├── pyproject.toml            # UVXパッケージング設定
└── CLAUDE.md                 # 開発者指示書
```

### 主要コンポーネント

**サーバー実装:**
- `server.py` - すべてのMCPツールを含むメインFastMCPサーバー
- `config.py` - 環境と設定管理
- `suppress_output.py` - クリーンな実行のための出力抑制

**コアプロセッサー:**
- `file_processor.py` - Microsoft MarkItDown統合
- `youtube_processor.py` - YouTubeトランスクリプト抽出
- `google_search_processor.py` - Google検索統合
- `strategies.py` - クローリング戦略実装

**MCPフレームワーク:**
- `tools/` - 個別ツール実装
- `prompts/` - ワークフロープロンプト定義
- `models/` - リクエスト/レスポンススキーマ

## 🔄 開発・配布方式

このプロジェクトは**複数の配布方法**をサポートしています：

1. **開発**: `crawl4ai_mcp/server.py`
   - 開発用の直接Python実行
   - 仮想環境セットアップ
   - デバッグと修正が容易

2. **UVX配布**: PyPIパッケージ
   - GitHub Releases経由で配布
   - 自動UVX対応
   - エンドユーザーの簡単インストール

3. **Docker配布**: コンテナデプロイメント
   - 本番対応のコンテナイメージ
   - マルチブラウザヘッドレスサポート
   - 簡単なスケーリングとデプロイメント

### 開発ワークフロー

**標準的な開発プロセス**：

```bash
# 1. 開発環境のセットアップ
source ./venv/bin/activate

# 2. 開発・テスト
vim crawl4ai_mcp/server.py
python -m crawl4ai_mcp.server

# 3. Dockerテスト（オプション）
docker-compose up --build

# 4. バージョン更新
vim pyproject.toml  # バージョン番号を更新

# 5. コミット・タグ付け
git add .
git commit -m "feat: 新機能を追加"
git tag -a v0.1.4 -m "Release v0.1.4"

# 6. プッシュ（自動配布トリガー）
git push origin main --tags
```

### 同期が必要な変更

- **ツールの説明** - LLMが適切にツールを選択できるよう確保
- **新機能** - 新しいMCPツールやパラメータ
- **バグ修正** - セキュリティ修正と動作改善
- **依存関係** - requirements.txtの更新
- **設定** - デフォルト値とタイムアウト調整
- **パッケージング** - UVX互換性のためのpyproject.toml

## 🛠️ 開発環境セットアップ

### 前提条件

**システム依存関係:**
```bash
# Linux/macOS
sudo bash scripts/prepare_for_uvx_playwright.sh

# Windows（管理者として）
powershell -ExecutionPolicy Bypass -File scripts/prepare_for_uvx_playwright.ps1
```

### ローカル開発セットアップ

**方法1: UVパッケージマネージャー（推奨）**
```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
uv sync
source .venv/bin/activate  # Linux/macOS
# または .venv\Scripts\activate  # Windows
```

**方法2: 従来の仮想環境**
```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
python -m venv venv
source venv/bin/activate  # Linux/macOS
# または venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
python -m playwright install chromium
```

### 必須仮想環境

**⚠️ 重要**: 開発には常に`./venv`仮想環境を使用してください：

```bash
# 仮想環境をアクティベート
source venv/bin/activate

# 正しい環境を確認
which python  # ./venv/bin/pythonを指すべき
python -c "import sys; print(sys.prefix)"  # venvパスを表示すべき
```

## 🧪 テストと品質保証

### テストコマンド

**基本サーバーテスト:**
```bash
# メインサーバー起動テスト
python -m crawl4ai_mcp.server

# HTTPサーバーテスト
python -m crawl4ai_mcp.server --transport http --host 127.0.0.1 --port 8000

# MCPクライアントでテスト
claude mcp test crawl4ai
```

**HTTP APIテスト:**
```bash
# Pure StreamableHTTPテスト
python examples/pure_http_test.py

# Legacy HTTPテスト
curl -X POST "http://127.0.0.1:8000/tools/crawl_url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# ヘルスチェック
curl http://127.0.0.1:8000/health
```

**YouTube統合テスト:**
```bash
# 直接APIテスト
python test_youtube_transcript_direct.py "https://www.youtube.com/watch?v=VIDEO_ID"

# MCP統合テスト
python test_mcp_youtube.py
```

### プリコミット品質チェック

**コミット前必須チェック:**
```bash
# 1. textlintでコミットメッセージをリント
npx textlint commit-message.txt

# 2. サーバー同期をチェック
diff crawl4ai_mcp/server.py dxt-packages/crawl4ai-dxt-correct/server/crawl4ai_mcp/server.py

# 3. 差分が存在しないことを確認（空の出力は同期を意味）
echo "サーバー同期状況: $([ -z "$(diff crawl4ai_mcp/server.py dxt-packages/crawl4ai-dxt-correct/server/crawl4ai_mcp/server.py)" ] && echo "✅" || echo "❌")"

# 4. 両方の実装をテスト
python -m crawl4ai_mcp.server --help
uvx --from ./dxt-packages/crawl4ai-dxt-correct/crawl4ai-dxt-correct.dxt crawl-mcp --help
```

## 🔧 開発ツールと設定

### 環境変数

```bash
# 開発ログ
export FASTMCP_LOG_LEVEL=DEBUG

# 言語設定
export CRAWL4AI_LANG=ja  # または英語の場合は 'en'

# APIキー（開発）
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://resource.openai.azure.com"

# Python環境
export PYTHONPATH="/path/to/crawl:$PYTHONPATH"
export PYTHONUNBUFFERED=1
```

### デバッグ設定

**高詳細デバッグ:**
```json
{
  "mcpServers": {
    "crawl4ai-debug": {
      "command": "/path/to/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG",
        "PYTHONPATH": "/path/to/crawl",
        "DEBUG": "1",
        "PYTHONUNBUFFERED": "1",
        "PLAYWRIGHT_DEBUG": "1"
      }
    }
  }
}
```

### コードスタイルと標準

**Pythonコード標準:**
- PEP 8スタイルガイドラインに従う
- 適切な場所で型ヒントを使用
- docstringで関数を文書化
- try/catchで優雅にエラーを処理
- I/O操作にはasync/awaitを使用

**コミットメッセージ標準:**
- 英語のみ
- コミットメッセージに絵文字は使用しない
- 形式: `type: 簡潔な説明`
- 例:
  - `feat: YouTubeバッチ処理を追加`
  - `fix: 大容量ファイル処理のメモリリークを解決`
  - `docs: APIリファレンスドキュメントを更新`

## 📦 ビルドとパッケージング

### DXTパッケージ管理

**DXTパッケージのビルド:**
```bash
cd dxt-packages/crawl4ai-dxt-correct/

# サーバーが同期されていることを確認
cp ../../crawl4ai_mcp/server.py server/crawl4ai_mcp/server.py

# パッケージをビルド
python -m zipfile -c crawl4ai-dxt-correct.dxt manifest.json README.md requirements.txt server/

# パッケージ内容を確認
python -m zipfile -l crawl4ai-dxt-correct.dxt
```

**バージョン管理:**
```bash
# manifest.jsonでバージョンを更新
vim dxt-packages/crawl4ai-dxt-correct/manifest.json

# バージョン履歴を更新
vim FINAL_RELEASE_NOTES.md

# リリースをタグ
git tag -a v1.x.x -m "バージョン1.x.xリリース"
git push origin v1.x.x
```

### UVX互換性

**pyproject.toml設定:**
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "crawl-mcp"
version = "1.0.0"
dependencies = [
    "crawl4ai>=0.7.2",
    "playwright>=1.54.0",
    "fastmcp>=0.2.0"
]

[project.scripts]
crawl-mcp = "crawl4ai_mcp.server:main"
```

## 🔍 デバッグとトラブルシューティング

### 一般的な開発問題

**インポートエラー:**
```bash
# 仮想環境がアクティブであることを確認
source venv/bin/activate

# Pythonパスをチェック
python -c "import sys; print('\\n'.join(sys.path))"

# 不足している依存関係をインストール
pip install -r requirements.txt
```

**ブラウザーの問題:**
```bash 
# Playwrightブラウザーを再インストール
python -m playwright install chromium

# システム依存関係（Linux）
sudo apt-get install libnss3 libnspr4 libasound2 libatk-bridge2.0-0

# ブラウザーインストールをチェック
python -c "from playwright.sync_api import sync_playwright; print('ブラウザーチェック: OK')"
```

**サーバー起動問題:**
```bash
# ポートの可用性をチェック
lsof -i :8000

# 既存のプロセスを終了
kill -9 $(lsof -t -i:8000)

# サーバーログをチェック
export FASTMCP_LOG_LEVEL=DEBUG
python -m crawl4ai_mcp.server
```

### パフォーマンスプロファイリング

**メモリとCPU監視:**
```bash
# メモリ使用量を監視
python -m memory_profiler crawl4ai_mcp/server.py

# CPUプロファイリング
python -m cProfile -o profile.prof -m crawl4ai_mcp.server

# プロファイルを分析
python -m pstats profile.prof
```

**リクエストトレーシング:**
```bash
# リクエストトレーシングを有効化
export CRAWL4AI_TRACE=1
export FASTMCP_LOG_LEVEL=DEBUG

# リクエストパターンを監視
tail -f server.log | grep -E "(REQUEST|RESPONSE|ERROR)"
```

## 🤝 貢献ガイドライン

### 開発ワークフロー

1. **フォークとクローン**
   ```bash
   git clone https://github.com/your-username/crawl-mcp.git
   cd crawl-mcp
   git remote add upstream https://github.com/walksoda/crawl-mcp.git
   ```

2. **機能ブランチを作成**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **開発とテスト**
   ```bash
   # crawl4ai_mcp/server.pyに変更を加える
   # 徹底的にテスト
   python -m crawl4ai_mcp.server
   ```

4. **同期とビルド**
   ```bash
   # DXTパッケージに同期
   cp crawl4ai_mcp/server.py dxt-packages/crawl4ai-dxt-correct/server/crawl4ai_mcp/server.py
   
   # DXTパッケージをリビルド
   cd dxt-packages/crawl4ai-dxt-correct/
   python -m zipfile -c crawl4ai-dxt-correct.dxt manifest.json README.md requirements.txt server/
   ```

5. **品質チェック**
   ```bash
   # テストを実行
   python examples/pure_http_test.py
   
   # 同期をチェック
   diff crawl4ai_mcp/server.py dxt-packages/crawl4ai-dxt-correct/server/crawl4ai_mcp/server.py
   
   # コミットメッセージをリント
   npx textlint commit-message.txt
   ```

6. **プルリクエストを提出**
   ```bash
   git add .
   git commit -m "feat: 新機能の説明を追加"
   git push origin feature/your-feature-name
   ```

### コードレビュープロセス

**提出前のチェック:**
- [ ] 開発とDXTパッケージの両方で変更をテスト
- [ ] サーバー同期を確認
- [ ] 必要に応じてドキュメントを更新
- [ ] コミットメッセージが標準に従う
- [ ] 議論なしに破壊的変更はしない

**レビュー基準:**
- コード品質とスタイル
- テストカバレッジ
- ドキュメントの完全性
- パフォーマンスへの影響
- セキュリティ考慮事項
- 後方互換性

## 📊 監視と分析

### 開発メトリクス

**サーバーパフォーマンス:**
```bash
# リクエスト処理時間
grep "処理時間" server.log | awk '{print $4}' | sort -n

# メモリ使用パターン
ps aux | grep crawl4ai_mcp | awk '{print $6}' # RSS メモリ

# 成功/失敗率
grep -c "SUCCESS" server.log
grep -c "ERROR" server.log
```

**ツール使用統計:**
```bash
# 最も使用されるツール
grep "ツール呼び出し:" server.log | awk '{print $3}' | sort | uniq -c | sort -nr

# 平均レスポンスサイズ
grep "レスポンスサイズ:" server.log | awk '{sum+=$3; count++} END {print sum/count}'
```

## 🔗 関連リソース

- **APIリファレンス**: [API_REFERENCE.md](API_REFERENCE.md)
- **設定例**: [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
- **高度な使用法**: [ADVANCED_USAGE.md](ADVANCED_USAGE.md)
- **インストールガイド**: [INSTALLATION.md](INSTALLATION.md)
- **HTTP統合**: [HTTP_INTEGRATION.md](HTTP_INTEGRATION.md)

## 💡 開発ベストプラクティス

1. **常に仮想環境を使用** - 依存関係の分離に必須
2. **両方のサーバー実装をテスト** - メインとDXTパッケージは同じように動作する必要がある
3. **コミット前に同期** - デプロイメントの不整合を防ぐ
4. **コミットメッセージ標準に従う** - 英語のみ、絵文字なし、明確な説明
5. **定期的にパフォーマンスをプロファイル** - メモリとCPU使用量を監視
6. **徹底的に文書化** - 新機能でドキュメントを更新
7. **開発中はデバッグログを使用** - `FASTMCP_LOG_LEVEL=DEBUG`
8. **エッジケースをテスト** - 大容量ファイル、ネットワーク問題、レート制限
9. **リソース使用量を監視** - メモリリークとパフォーマンス劣化
10. **依存関係を慎重にバージョン管理** - 安定性のためにバージョンを固定