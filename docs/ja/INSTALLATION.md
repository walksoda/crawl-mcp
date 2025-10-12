# インストールガイド

このガイドでは、Crawl4AI MCPサーバーの詳細なインストール手順を提供します。

## 🔧 前提条件セットアップ（必須）

**Python 3.11 以上が必要です（FastMCP のサポート要件に準拠）**

**どのインストール方法を選択する場合でも、Playwrightのシステム依存関係を準備する必要があります：**

### 🐧 Linux/macOS

#### Ubuntu 24.04 LTSユーザー（手動インストールが必要）

**⚠️ 重要:** Ubuntu 24.04のt64ライブラリ移行により、自動セットアップスクリプトは廃止されました。手動インストールを使用してください：

```bash
# Ubuntu 24.04 LTS 手動セットアップ（t64移行により必要）
sudo apt update && sudo apt install -y \
  libnss3 \
  libatk-bridge2.0-0 \
  libxss1 \
  libasound2t64 \
  libgbm1 \
  libgtk-3-0t64 \
  libxshmfence-dev \
  libxrandr2 \
  libxcomposite1 \
  libxcursor1 \
  libxdamage1 \
  libxi6 \
  fonts-noto-color-emoji \
  fonts-unifont \
  python3-venv \
  python3-pip

# Playwrightとブラウザをインストール
python3 -m venv venv
source venv/bin/activate
pip install playwright==1.54.0
playwright install chromium
sudo playwright install-deps
```

#### その他のLinuxディストリビューション/macOS（自動スクリプト）

```bash
# Playwrightのシステム依存関係をインストール（全方法で必要）
sudo bash scripts/prepare_for_uvx_playwright.sh

# 日本語対応（オプション）
export CRAWL4AI_LANG=ja
sudo bash scripts/prepare_for_uvx_playwright.sh
```

### 🪟 Windows

```powershell
# PowerShellを管理者権限で実行（全方法で必要）
scripts/prepare_for_uvx_playwright.ps1

# 実行ポリシーでスクリプトがブロックされる場合：
# powershell -ExecutionPolicy Bypass -File "scripts/prepare_for_uvx_playwright.ps1"
```

### システム準備機能

- **クロスプラットフォーム**: Linux（apt/yum/pacman/apk）+ Windows
- **最小依存関係**: Playwrightに必要なシステムライブラリのみインストール
- **UVX最適化**: UVX実行環境専用設計
- **多言語対応**: 英語（デフォルト）+ 日本語（`CRAWL4AI_LANG=ja`）
- **バージョン同期**: requirements.txtからPlaywrightバージョンを自動読み取り
- **スマートインストール**: 手動インストール手順で正しい固定バージョンを使用
- **強化されたエラー処理**: MCPクライアント向けChromiumバージョン互換性メッセージの改善

## 🚀 インストール方法

### 方法1: UVX（推奨 - 最も簡単で本番向け）⭐

**最も便利なワンコマンドインストール：**

```bash
# 上のシステム準備後 - これだけです！
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp
```

**✅ 利점：** ゼロ設定、自動依存関係管理、分離環境

### 方法2: 開発環境

```bash
# 上のシステム準備後、開発環境を作成
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # 安定性のため固定バージョンを使用
python -m playwright install chromium
python -m crawl4ai_mcp.server
```

**用途：** ローカル開発とカスタマイズ

### 方法3: 直接インストール

```bash
# 上のシステム準備後
pip install -r requirements.txt  # 推奨：固定バージョンを使用
# 代替手段： pip install crawl4ai==0.7.2 playwright==1.54.0
python -m playwright install chromium
python -m crawl4ai_mcp.server
```

**用途：** グローバルインストールまたはシステム全体での展開

## 🔧 開発セットアップ

### ローカル開発セットアップ

```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
uv sync
```

### クイックセットアップ（従来方式）

**Linux/macOS:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup_windows.bat
```

### 手動セットアップ

1. **仮想環境の作成と有効化:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# または
venv\Scripts\activate.bat  # Windows
```

2. **Python依存関係のインストール:**
```bash
pip install -r requirements.txt  # 安定性のため固定バージョンをインストール
```

3. **Playwrightブラウザ依存関係のインストール（Linux/WSL）:**

**Ubuntu 24.04 LTS:**
```bash
sudo apt update && sudo apt install -y \
  libnss3 libatk-bridge2.0-0 libxss1 libasound2t64 \
  libgbm1 libgtk-3-0t64 libxshmfence-dev libxrandr2 \
  libxcomposite1 libxcursor1 libxdamage1 libxi6 \
  fonts-noto-color-emoji fonts-unifont
```

**その他のLinuxディストリビューション:**
```bash
sudo apt-get update
sudo apt-get install libnss3 libnspr4 libasound2 libatk-bridge2.0-0 libdrm2 libgtk-3-0 libgbm1
```

## ⚙️ Claude Desktop統合

### 基本設定

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

### 設定ファイルの場所

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/claude-desktop/claude_desktop_config.json
```

## 🔍 トラブルシューティング

### 一般的なインストール問題

インストールに失敗した場合：

1. **Chromiumチェック**: `get_system_diagnostics`ツールで診断実行
2. **ブラウザの問題**: 上のChromiumセットアップスクリプトを再実行
3. **権限**: スクリプトが適切な権限（sudo/管理者）で実行されていることを確認
4. **代替方法**: UVXが失敗した場合は方法2（開発）または方法3（直接）を試行
5. **UVX成功**: システム準備後、UVX（方法1）は通常確実に動作します

### 特定のエラー解決方法

**ModuleNotFoundError:**
- 仮想環境が有効化されていることを確認
- PYTHONPATHが正しく設定されていることを確認
- 依存関係をインストール: `pip install -r requirements.txt`

**Playwrightブラウザエラー（Ubuntu 24.04 LTS）:**
- t64ライブラリ名を使用: `sudo apt-get install libnss3 libnspr4 libasound2t64 libgtk-3-0t64`
- t64移行により手動インストールが必要（前提条件セクションを参照）
- WSLの場合: X11転送またはヘッドレスモードを確認

**Playwrightブラウザエラー（その他のLinux）:**
- システム依存関係をインストール: `sudo apt-get install libnss3 libnspr4 libasound2`
- WSLの場合: X11転送またはヘッドレスモードを確認

**JSON解析エラー:**
- **解決済み**: 最新バージョンで出力抑制機能が実装されました
- crawl4aiの冗長出力は適切に抑制されています

### PowerShell実行ポリシー（Windows）

Windowsで実行ポリシーエラーが発生した場合：

```powershell
# オプション1: このスクリプトのみ実行ポリシーをバイパス
powershell -ExecutionPolicy Bypass -File "scripts/prepare_for_uvx_playwright.ps1"

# オプション2: 実行ポリシーを一時的に変更（管理者として実行）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# セットアップ後に元に戻す
Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser
```

## 📚 次のステップ

インストール成功後：

1. **インストール確認**: `python -m crawl4ai_mcp.server --help`を実行
2. **基本機能テスト**: 簡単なクロール操作を試行
3. **Claude Desktop設定**: MCPサーバー設定を追加
4. **ドキュメント探索**: 利用可能なツールは[APIリファレンス](API_REFERENCE.md)を確認

詳細な使用方法については、メインの[README](../README_ja.md)または[高度な使用ガイド](ADVANCED_USAGE.md)をご覧ください。
