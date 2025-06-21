# Crawl4AI MCP トラブルシューティング

このファイルには、Crawl4AI MCPサーバーの一般的な問題と解決方法をまとめています。

## 🔧 基本セットアップ

### 1. Claude Desktop 設定ファイル

**Windows での設定ファイル場所:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**推奨設定 (claude_desktop_config_windows.json):**
```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "wsl",
      "args": [
        "-e",
        "bash",
        "-c",
        "cd /home/user/prj/crawl && source venv/bin/activate && PYTHONPATH=/home/user/prj/crawl:$PYTHONPATH python -m crawl4ai_mcp.server"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### 2. 手動テスト

WSLで直接サーバーが起動するか確認：

```bash
cd /home/user/prj/crawl
source venv/bin/activate
python -m crawl4ai_mcp.server --help
```

## 🚨 よくある問題と解決方法

### 問題1: ModuleNotFoundError

**エラー:**
```
ModuleNotFoundError: No module named 'crawl4ai_mcp'
```

**解決方法:**
1. PYTHONPATHが正しく設定されているか確認
2. 仮想環境が有効化されているか確認
3. 依存関係が正しくインストールされているか確認

```bash
cd /home/user/prj/crawl
source venv/bin/activate
pip install -r requirements.txt
```

### 問題2: Playwright ブラウザエラー

**エラー:**
```
playwright._impl._api_types.Error: Browser has been closed
```

**解決方法:**
WSLでブラウザ依存関係をインストール：

```bash
sudo apt-get update
sudo apt-get install libnss3 libnspr4 libasound2 libatk-bridge2.0-0 libdrm2 libgtk-3-0 libgbm1
```

### 問題3: JSON解析エラー (解決済み)

**エラー:**
```
Unexpected token '|', "| ✓ | ⏱: 0.79s " is not valid JSON
```

**解決状況:**
この問題は最新版で修正されています：
- 出力抑制機能を実装
- `suppress_stdout_stderr` コンテキストマネージャーを使用
- ログレベルをCRITICALに設定
- 起動スクリプトでstderr抑制

### 問題4: 接続確認

**期待されるツール:**
- `crawl_url` - 基本的なWebクローリング
- `extract_structured_data` - 構造化データ抽出
- `batch_crawl` - バッチクローリング
- `crawl_url_with_fallback` - フォールバック付きクローリング

**期待されるプロンプト:**
- `crawl_website_prompt` - ウェブサイトクローリング用
- `analyze_crawl_results_prompt` - 結果分析用
- `batch_crawl_setup_prompt` - バッチクローリング用

**期待されるリソース:**
- `uri://crawl4ai/config` - 設定オプション
- `uri://crawl4ai/examples` - 使用例

## ⚠️ 重要なポイント

1. **仮想環境を必ず有効化** - `source venv/bin/activate`
2. **PYTHONPATHを正しく設定** - モジュール検索パスの確保
3. **Claude Desktop を完全に再起動** - 設定変更後は必須
4. **ブラウザ依存関係をインストール** - WSL環境でのPlaywright動作に必要

## 🔄 サーバー再起動

設定を変更した後は、必ずClaude Desktopを再起動してください：

1. Claude Desktopを完全に終了
2. タスクマネージャーで関連プロセスがないことを確認
3. Claude Desktopを再起動
4. 新しいチャットを開始

## 📊 デバッグ

問題が発生した場合の確認手順：

1. **WSLでの手動テスト:**
   ```bash
   cd /home/user/prj/crawl
   source venv/bin/activate
   python -c "from crawl4ai_mcp.server import mcp; print('OK')"
   ```

2. **依存関係の確認:**
   ```bash
   pip list | grep -E "(crawl4ai|fastmcp|pydantic)"
   ```

3. **設定ファイルの確認:**
   ```bash
   cat claude_desktop_config_windows.json
   ```