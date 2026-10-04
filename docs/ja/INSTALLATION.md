# インストールガイド

Crawl4AI MCPサーバーのインストール手順を、方法ごとに説明します。

## 前提条件

- Python 3.11 以上（`requires-python = ">=3.11"`、3.11 と 3.12 で動作確認）
- Playwright の Chromium とシステムライブラリ。このプロジェクトは `playwright==1.55.0` に固定しています。初回利用の前に、このバージョンに対応するブラウザをインストールしてください。

### Playwright 用 Chromium のインストール

Linux ではブラウザとシステムライブラリをまとめて入れます（sudo が必要です）。
```bash
uvx --from playwright==1.55.0 playwright install --with-deps chromium
```

macOS と Windows ではブラウザだけを入れます。
```bash
uvx --from playwright==1.55.0 playwright install chromium
```

パッケージをインストール済みの環境がある場合（方法2・方法3）は、その環境内で同じ処理を実行してもかまいません。Linux では `python -m playwright install --with-deps chromium`、それ以外では `python -m playwright install chromium` です。

#### システムライブラリの手動インストール（Ubuntu 24.04 LTS）

`--with-deps` が使えない場合（Playwright の手順で sudo を使えない場合など）は、ライブラリを手動でインストールします。Ubuntu 24.04 では `t64` 付きのパッケージ名を使います。

```bash
sudo apt update && sudo apt install -y \
  libnss3 libnspr4 libatk-bridge2.0-0 libxss1 libasound2t64 \
  libgbm1 libgtk-3-0t64 libxshmfence-dev libxrandr2 \
  libxcomposite1 libxcursor1 libxdamage1 libxi6 \
  fonts-noto-color-emoji fonts-unifont
```

それより古い Debian / Ubuntu では、`t64` の付かない名前（`libasound2`、`libgtk-3-0`）を使ってください。

## インストール方法

### 方法1: UVX（推奨）

```bash
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

環境を手で用意する必要がありません。依存関係は自動で解決され、ほかの環境とも分離されます。

### 方法2: 開発環境

```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
python3 -m venv venv && source venv/bin/activate
pip install -e .
python -m playwright install chromium
python -m crawl4ai_mcp.server
```

ローカルで開発やカスタマイズをするときの方法です。テスト用の依存関係は [開発ガイド](DEVELOPMENT.md) を参照してください。

### 方法3: pip によるインストール

```bash
pip install git+https://github.com/walksoda/crawl-mcp
# クローン済みリポジトリからの場合: pip install .
python -m playwright install chromium
crawl-mcp
```

既存の Python 環境に入れるときの方法です。

依存関係は `pyproject.toml` でバージョン範囲を制約して定義しています（`crawl4ai>=0.9.4,<0.10`、`fastmcp>=2.14.2,<3`、`playwright==1.55.0` など）。`requirements.txt` は Docker ビルド用に同じ内容を写したものです。Anthropic の SDK（`anthropic>=0.40.0`）も通常の依存関係に入っているため、UVX でインストールした環境でも Anthropic による要約が使えます。

### セットアップスクリプト（従来方式）

リポジトリには簡易セットアップスクリプトもあります。`venv` を作り、`requirements.txt` と Playwright の Chromium をインストールして、サーバーの起動コマンドを表示します。`requirements.txt` ではパッケージ本体は入らないので、サーバーはリポジトリのルートで起動してください。Linux のシステムライブラリはスクリプトでは入りません。Chromium が起動しないときは前提条件の手順を見てください。

Linux と macOS の場合:
```bash
bash scripts/setup.sh
```

Windows の場合:
```cmd
scripts\setup_windows.bat
```

## Claude Desktop との連携

### 基本設定

`claude_desktop_config.json` に以下を追加します。

```json
{
  "mcpServers": {
    "crawl-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ]
    }
  }
}
```

### 設定ファイルの場所

Windows:
```
%APPDATA%\Claude\claude_desktop_config.json
```

macOS:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

## トラブルシューティング

### よくあるインストールの問題

1. ブラウザや実行ファイルが見つからないと言われたら、上の Chromium のインストールコマンドを `playwright==1.55.0` で再実行してください。Playwright のバージョンが違うと、ダウンロードされる Chromium のビルドも変わります。
2. Linux でシステムライブラリが足りない場合は、`--with-deps` を使うか、前提条件の手順で手動インストールします。
3. Linux で `--with-deps` を使うには sudo が必要です。
4. UVX でうまくいかなければ、方法2か方法3を試してください。

### エラー別の対処法

`ModuleNotFoundError` が出る場合:
- 仮想環境が有効になっているか確認する
- パッケージを `pip install -e .`（または `pip install .`）でインストールする
- インストールせずにソースから実行する場合は、リポジトリのルートで実行する

Linux や WSL で Playwright のブラウザエラーが出る場合:
- Ubuntu 24.04 では `t64` 付きのライブラリ名（`libasound2t64`、`libgtk-3-0t64`）を使う
- サーバーは Chromium をヘッドレスで動かすため、WSL でも X サーバーは不要

STDIO で MCP クライアントが JSON のパースエラーを出す場合:
- STDIO では、サーバーは stdout を MCP メッセージ専用にしています。`.env` 読み込み時の通知を含め、診断出力はすべて stderr に出ます。
- それでもクライアントが不正な JSON を報告する場合は、サーバー起動前に stdout へ出力するラッパースクリプトやシェルのプロファイルがないか確認してください。

サーバーのログを見たい場合:
- ログはデフォルトでは出ません。クライアント設定の `env` ブロックで `CRAWL4AI_MCP_LOG_LEVEL` に `DEBUG`、`INFO`、`WARNING`、`ERROR` のどれかを設定すると、ログが stderr に出ます。
- サーバーはこの変数を `.env` より先に読むため、`.env` に書いても効きません。

キャッシュから返る結果が16文字のハッシュ文字列になる場合:
- crawl4ai の旧形式キャッシュ移行処理は、既存のキャッシュデータベースを壊すことがありました。サーバーは起動時にこの移行処理を無効にするため、crawl4ai をアップグレードしても `~/.crawl4ai/crawl4ai.db` は書き換えられません。
- 以前の実行でキャッシュがすでに壊れているなら、crawl4ai のキャッシュ（`~/.crawl4ai/`）は削除してかまいません。次のクロールで作り直されます。

## 次のステップ

インストールが済んだら、次の順に進めてください。

1. `python -m crawl4ai_mcp.server --help`（または `crawl-mcp --help`）でインストールを確認する
2. MCP クライアントに上記のサーバー設定を追加する
3. 簡単な `crawl_url` 呼び出しで動作を試す
4. 使えるツールを [APIリファレンス](API_REFERENCE.md) で確認する

詳しい使い方は、メインの [README](../../README_ja.md) または [使い方の応用](ADVANCED_USAGE.md) を参照してください。
