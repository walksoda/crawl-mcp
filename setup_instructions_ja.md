# Claude Desktop での MCP サーバー設定方法

## 1. Claude Desktop アプリケーションの設定

### macOS の場合
Claude Desktop の設定ファイルの場所：
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### Windows の場合
Claude Desktop の設定ファイルの場所：
```
%APPDATA%\Claude\claude_desktop_config.json
```

### Linux の場合
Claude Desktop の設定ファイルの場所：
```
~/.config/Claude/claude_desktop_config.json
```

## 2. 設定ファイルの編集

### Linux/macOS 用設定
```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {}
    }
  }
}
```

### Windows (WSL) 用設定
```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "wsl",
      "args": [
        "/home/user/prj/crawl/venv/bin/python",
        "-m",
        "crawl4ai_mcp.server"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

### Windows (ネイティブ) 用設定
```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "C:\\path\\to\\your\\crawl\\venv\\Scripts\\python.exe",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "C:\\path\\to\\your\\crawl",
      "env": {}
    }
  }
}
```

## 3. パスの確認

現在のプロジェクトパスを確認：
```bash
pwd
# 出力例: /home/user/prj/crawl
```

Python実行ファイルのパスを確認：
```bash
source venv/bin/activate
which python
# 出力例: /home/user/prj/crawl/venv/bin/python
```

## 4. 設定例（このプロジェクト用）

```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {}
    }
  }
}
```

## 5. 設定後の手順

1. **Claude Desktop を再起動**
2. **接続確認**：新しいチャットで以下が利用可能か確認
   - `crawl_url` ツール
   - `extract_structured_data` ツール
   - `batch_crawl` ツール

## 6. トラブルシューティング

### 接続できない場合
1. パスが正しいか確認
2. 仮想環境が正しく設定されているか確認
3. 依存関係がインストールされているか確認

### ログの確認
Claude Desktop のログでエラーメッセージを確認してください。

### 手動テスト
設定前に手動でサーバーが起動するか確認：
```bash
cd /home/user/prj/crawl
source venv/bin/activate
python -m crawl4ai_mcp.server
```

## 7. 使用例

Claude Desktop で以下のようにリクエストできます：

```
https://example.com のコンテンツをクローリングして、記事のタイトルと概要を抽出してください。
```

```
https://news.ycombinator.com から最新ニュースのタイトル一覧を取得してください。
```

## 8. セキュリティ注意事項

- 信頼できるサイトのみクローリングしてください
- 大量のリクエストは避けてください（レート制限）
- 個人情報を含むサイトのクローリングは注意してください