# 開発ガイド

Crawl4AI MCPサーバーの開発に参加する人向けのガイドです。

## アーキテクチャ

### リポジトリ構造

```
crawl-mcp/
├── crawl4ai_mcp/
│   ├── __init__.py              # バージョン検出。import時に crawl4ai_guard を適用
│   ├── server.py                # FastMCPエントリポイント（main(): --transport/--host/--port）
│   ├── config.py                # .env の読み込みとLLMプロバイダー設定
│   ├── constants.py             # 共通の上限値や要約長の設定
│   ├── validators.py            # 入力検証（validate_*_params、validate_output_path）
│   ├── crawl4ai_guard.py        # crawl4ai の旧形式キャッシュ移行処理を無効化
│   ├── suppress_output.py       # ライブラリの出力をMCPのstdoutに混ぜないための処理
│   ├── server_tools/            # @mcp.tool() アダプタ（クライアントに見えるスキーマ）
│   │   ├── __init__.py          # register_all_tools()
│   │   ├── _shared.py           # 共通import、readOnlyHintアノテーション、modules_unavailable_error()
│   │   ├── crawl_tools.py       # crawl_url、crawl_url_with_fallback、deep_crawl_site
│   │   ├── extraction_tools.py  # intelligent_extract、extract_entities、extract_structured_data
│   │   ├── youtube_tools.py     # YouTube のトランスクリプト・情報・コメント系ツール
│   │   ├── search_tools.py      # search_google、batch_search_google、search_and_crawl
│   │   ├── file_tools.py        # process_file、enhanced_process_large_content など
│   │   └── batch_tools.py       # batch_crawl、multi_url_crawl
│   ├── tools/                   # 後方互換のための再エクスポートfacade（実装なし）
│   ├── core/                    # ビジネスロジック（クロール、フォールバック、抽出、YouTube、検索、ファイル、バッチ）
│   ├── processors/              # YouTube・Google検索・ファイル変換のプロセッサー
│   ├── infra/                   # ブラウザ設定、フォールバック戦略、fingerprint、セッション、戦略/コンテンツキャッシュ
│   ├── utils/                   # LLMクライアント、トークン計算、時刻パース
│   ├── models/                  # Pydanticのリクエスト/レスポンスモデル
│   ├── middleware/              # レスポンス用ヘルパーと output_path 保存（下記の注記を参照）
│   ├── server_helpers.py        # server_tools が使うヘルパーの再エクスポート（スライス、変換、検索キャッシュ）
│   ├── prompts/                 # MCPプロンプト
│   ├── youtube_processor.py     # processors/ を再エクスポートする互換シム
│   ├── file_processor.py        #   （古いimportのために残している）
│   └── google_search_processor.py
├── tests/
│   ├── unit/                    # 高速な単体テスト（MCP接続不要）
│   └── mcp/                     # ツールテスト（stdioでサーバーを起動し、実ネットワークにアクセス）
├── scripts/run_tests.sh         # テスト実行用ラッパー
├── Dockerfile / docker-compose.yml
├── pyproject.toml               # パッケージ設定（依存関係とバージョンの唯一の情報源）
└── requirements.txt             # 依存関係の写し。Dockerビルドで使用
```

`middleware/` について補足します。`middleware/pipeline.py` の `ToolPipeline` クラス群（およびパイプラインのステップ用モジュール）は定義されていますが、どのツールにも組み込まれていません。ツールハンドラーは個別のヘルパー関数を直接呼びます。実際に使われているのは、`middleware/file_persistence.py` の `finalize_tool_response`（`output_path` への保存）と、`server_helpers.py` 経由で `server_tools/_shared.py` に渡される `_convert_result_to_dict` / `_process_content_fields`（`response_transform.py`）、`_apply_content_slicing`（`content_slicing.py`）、`_apply_search_content_slicing`（`search_slicing.py`）、検索結果キャッシュ（`cache.py`）です。

### リクエストの流れ

ツール呼び出しは次の層を順に通ります。

1. `server.py` が `FastMCP("Crawl4AI")` を生成し、`register_all_tools(mcp, _get_modules)` を呼びます。import前に logging・warnings・FastMCPのバナーを抑制します。
2. `server_tools/*_tools.py` は薄い `@mcp.tool()` アダプタです。パラメータは `Annotated[..., Field(description=...)]` で宣言し、これがMCPクライアントに見えるスキーマになります。`get_modules()` で `tools/` を遅延ロードし、失敗時は `modules_unavailable_error()` を返します。
3. `tools/` は後方互換のために `core/` の関数を再エクスポートします。
4. `core/` がロジックを実装し、`processors/`、`infra/`、`utils/`、`models/` を使います。

### ツールハンドラーの定型パターン

ツールを追加・変更するときは、`server_tools/crawl_tools.py` の `crawl_url` と同じ順序で処理します。

1. 入力を検証します（`validators.py`）。`validate_*_params`、`validate_content_slicing_params`、`validate_output_path(output_path, overwrite)` は外部アクセスより*前に*実行します。
2. `get_modules()` 経由で core の関数を呼びます。
3. `_convert_result_to_dict` で結果を正規化し、続けて `_process_content_fields` を通します。
4. 成功時の return の直前、スライスや切り詰めより*前に* `finalize_tool_response(..., tool_kind=KIND_*)` を呼んで保存します。この順序なので、ディスクにはスライス前の全文が残ります。`KIND_*` 定数は `middleware/file_persistence.py` の冒頭で定義されています。
5. 最後に `_apply_content_slicing`（`content_limit` / `content_offset`）と `apply_token_limit`（約25,000トークン）を適用します。

`output_path` に対応するツールは、`output_path`、`include_content_in_response`、`overwrite` の3パラメータをそろえて公開します。

### STDIO出力のルール

STDIOトランスポートでは、stdoutはMCPのJSON-RPCメッセージ専用です。サーバーのコードから**stdoutには出力しないでください**。診断はstderrに書きます（`print(..., file=sys.stderr)`）。STDIOで動いている間、`server.py` は誤ってstdoutに書かれた出力をstderrへ回し、`suppress_output.py` はライブラリが出す余分な出力を抑えます。loggingはimport時に無効化されている（`logging.disable(logging.CRITICAL)`）ため、`FASTMCP_LOG_LEVEL` を設定してもログは出力されません。

## 開発と配布

ロジックは `crawl4ai_mcp/core/`、ツールの登録は `crawl4ai_mcp/server_tools/` で編集します。配布方法は2つあります。

- UVXでGitHubリポジトリから直接入れる（`uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp`）
- `Dockerfile` からコンテナイメージをビルドする

### 開発ワークフロー

```bash
# 1. ブランチを作成
git switch -c feat/your-feature

# 2. 開発・テスト
venv/bin/pytest tests/unit/
venv/bin/pytest tests/mcp/ -m "not slow"

# 3. Dockerでテスト（任意）
docker-compose up --build

# 4. コミット（.env や venv/ が混入しないよう、git add . ではなくファイル名を指定する）
git add crawl4ai_mcp/core/your_module.py tests/unit/test_your_module.py
git commit -m "feat: add new functionality"
```

### リリース

```bash
# 1. pyproject.toml の version を更新（バージョンの唯一の情報源）
# 2. コミットとタグ作成
git add pyproject.toml
git commit -m "release: bump version to X.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# 3. ブランチとタグをまとめてプッシュ
git push --atomic origin main vX.Y.Z
```

## 開発環境のセットアップ

### ローカル開発環境

リポジトリのルートに `venv` という名前の仮想環境を作ってください。`tests/conftest.py` のMCPテストクライアントは `./venv/bin/python` でサーバーを起動します。

```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
python -m venv venv            # または: uv venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -e ".[test]"       # uv の場合: uv pip install --python venv/bin/python -e ".[test]"
python -m playwright install chromium
```

入ったら環境を確認します。

```bash
which python                                          # ./venv/bin/python を指すこと
python -c "import crawl4ai_mcp; print(crawl4ai_mcp.__file__)"   # このチェックアウト内を指すこと
```

## テスト

```bash
venv/bin/pytest tests/unit/                          # 単体テスト（高速、MCP接続不要）
venv/bin/pytest tests/mcp/ -m "not slow"             # MCPツールテスト（stdioサーバー、実ネットワーク）
venv/bin/pytest tests/unit/test_file_persistence.py::<test_name>   # 単一テスト

./scripts/run_tests.sh quick                         # 単体テスト
./scripts/run_tests.sh mcp                           # MCPツールテスト（slowを除く）
./scripts/run_tests.sh full cov                      # 全テスト＋カバレッジ
./scripts/run_tests.sh youtube|crawl|search|file|utility   # ツールグループ単位
```

- `pytest.ini` では `asyncio_mode = auto`、デフォルトのタイムアウト120秒、マーカー `slow`、`mcp`、`youtube`、`crawl`、`search`、`file`、`utility` を設定しています。
- CI（Python 3.11 / 3.12）は `tests/unit/` → `tests/mcp/ -m "not slow"` → `-m slow`（失敗許容）の順に実行します。

サーバーを手で起動して確かめる場合は、次のコマンドを使います。
```bash
python -m crawl4ai_mcp.server --help
python -m crawl4ai_mcp.server                                               # STDIO
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000   # HTTP（/mcp）
```

HTTPサーバーを curl やMCPクライアントから呼び出す方法は [HTTP統合ガイド](HTTP_INTEGRATION.md) を参照してください。

### ドキュメントの検査

`scripts/check_docs.py` はドキュメントとコードを突き合わせます。検査するのは次の5点です。全ツールの引数（`request` 辞書を取るツールはそのキーも）が `docs/API_REFERENCE.md` と `docs/ja/API_REFERENCE.md` に載っているか、依存バージョンの記載が `pyproject.toml` と一致するか、リポジトリ内のパス・リンク・アンカーが実在するか、例で設定している環境変数をサーバーが読んでいるか、本文に絵文字や誇張語がないか。CI では `tests/unit/test_docs_sync.py` から実行されます。

```bash
venv/bin/python scripts/check_docs.py              # ドキュメントとコードの突き合わせ
venv/bin/python scripts/check_docs.py --release    # 前回のタグ以降のコード変更が CHANGELOG にあるかも見る
git config core.hooksPath .githooks                # push の前に --release の検査を走らせる
```

ツールの引数や挙動を変えたときは、同じ変更の中で API リファレンス（英日）、README（英日）、`CHANGELOG.md` の `[Unreleased]` を更新してください。

## 開発用の設定

### 環境変数

`config.py` はプロジェクトルートの `.env` を読み込みます（`.env.example` を参照）。

```bash
# LLMプロバイダー（プロバイダーごとのキー、または MCP_LLM_CONFIG のJSONで指定）
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://resource.openai.azure.com"

# Google検索
export GOOGLE_SEARCH_MODE=hybrid            # googlesearch_only | custom_search_only | hybrid（デフォルト）
export GOOGLE_SEARCH_API_KEY="..."
export GOOGLE_SEARCH_ENGINE_ID="..."
```

### ローカルMCPクライアントの設定

```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl-mcp"
    }
  }
}
```

### コードスタイルと標準

- PEP 8 に従う
- 型ヒントを付ける
- 関数には docstring を書く
- エラーを処理し、ツールからは構造化されたエラーdictを返す
- I/O処理には async/await を使う

コミットメッセージは英語で短く書き、絵文字は使いません。形式は `type: 簡潔な説明` で、たとえば次のように書きます。
- `feat: add YouTube batch processing`
- `fix: resolve memory leak in large file processing`
- `docs: update API reference`

## ビルドとパッケージング

### Docker

```bash
# イメージをビルド
docker build -t crawl4ai-mcp:latest .

# STDIOモード（デフォルト）
docker-compose up --build
docker run -i crawl4ai-mcp:latest

# HTTPモード
docker-compose --profile http up --build crawl4ai-mcp-http
docker run -p 8000:8000 crawl4ai-mcp:latest \
  python -m crawl4ai_mcp.server --transport http --host 0.0.0.0 --port 8000
```

Dockerビルドでは `requirements.txt` から依存関係をインストールします。

### パッケージングと依存関係

`pyproject.toml` の該当部分を抜粋します。
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "crawl-mcp"
requires-python = ">=3.11"
dependencies = [
    "crawl4ai>=0.9.4,<0.10",
    "fastmcp>=2.14.2,<3",
    "playwright==1.55.0",
    # ...
]

[project.optional-dependencies]
test = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0", "pytest-timeout>=2.3.1", "pytest-cov>=5.0.0"]

[project.scripts]
crawl-mcp = "crawl4ai_mcp.server:main"
```

`pyproject.toml` が依存関係とバージョンの唯一の情報源です。`requirements.txt` はDockerビルド用に依存関係を写したもので、`pyproject.toml` を変更したときは合わせて更新してください。

## デバッグとトラブルシューティング

importエラーが出るときは、仮想環境を有効にしてパッケージを入れ直し、import元を確認します。
```bash
source venv/bin/activate
pip install -e ".[test]"
python -c "import crawl4ai_mcp; print(crawl4ai_mcp.__file__)"
```

ブラウザが起動しないときは、Chromiumを入れ直します。
```bash
python -m playwright install chromium
python -m playwright install --with-deps chromium   # Linux: システムライブラリも導入（sudo が必要）
```

ポートが使われていてサーバーが起動しないときは、使っているプロセスを止めます。
```bash
lsof -i :8000               # ポートの使用状況を確認
kill <PID>
```

サーバーは診断をstderrに出力し、Pythonのloggingはimport時に無効化されています。調査するときは、ターミナルでサーバーを起動してstderrを確認するか、一時的に `print(..., file=sys.stderr)` を追加してください。stdoutには出力しないでください。

CPUプロファイルを取るには次のようにします。
```bash
python -m cProfile -o profile.prof -m crawl4ai_mcp.server
python -m pstats profile.prof
```

## コントリビューションの手順

1. フォークしてクローンする
   ```bash
   git clone https://github.com/your-username/crawl-mcp.git
   cd crawl-mcp
   git remote add upstream https://github.com/walksoda/crawl-mcp.git
   ```

2. 機能ブランチを作る
   ```bash
   git switch -c feat/your-feature-name
   ```

3. `crawl4ai_mcp/core/`（ロジック）と `crawl4ai_mcp/server_tools/`（ツールのパラメータと登録）を変更し、`tests/unit/` にテストを追加します。

4. テストを実行する
   ```bash
   venv/bin/pytest tests/unit/
   venv/bin/pytest tests/mcp/ -m "not slow"
   docker build -t crawl4ai-mcp:test .     # 任意
   ```

5. コミットしてプルリクエストを作る
   ```bash
   git add <変更したファイル>
   git commit -m "feat: add new feature description"
   git push origin feat/your-feature-name
   ```

### プルリクエストのチェックリスト

- [ ] 単体テストとMCPツールテストが通る
- [ ] `scripts/check_docs.py` が通り、挙動の変更をドキュメントと `CHANGELOG.md` に書いた
- [ ] 依存関係を変更した場合は `requirements.txt` も更新した
- [ ] コミットメッセージが上記の形式に沿っている
- [ ] 議論なしに破壊的変更を加えていない

## 関連ドキュメント

- [APIリファレンス](API_REFERENCE.md)
- [設定例](CONFIGURATION_EXAMPLES.md)
- [使い方の応用](ADVANCED_USAGE.md)
- [インストールガイド](INSTALLATION.md)
- [HTTP統合](HTTP_INTEGRATION.md)
