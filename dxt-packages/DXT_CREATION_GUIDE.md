# DXT パッケージ作成ガイド

このドキュメントは、Crawl4AI MCP サーバーをDXTパッケージ化した際の経験をまとめたものです。

## 概要

DXT (Desktop Extensions) は、Claude Desktop 用の拡張機能パッケージ形式です。MCPサーバーをDXTパッケージ化することで、ワンクリックインストールが可能になります。

## 前提条件

### 必要なツール
```bash
npm install -g @anthropic-ai/dxt
```

### プロジェクト構造
```
project/
├── crawl4ai_mcp/           # 既存のMCPサーバーコード
├── requirements.txt        # Python依存関係
└── ...
```

## DXT パッケージ作成手順

### 1. 正しいディレクトリ構造の作成

DXT公式仕様に従った構造：
```
crawl4ai-dxt/
├── manifest.json           # DXT設定ファイル（必須）
├── README.md               # ユーザー向けドキュメント
├── requirements.txt        # Python依存関係
└── server/                 # サーバーファイル
    ├── main.py            # DXTエントリーポイント
    └── crawl4ai_mcp/      # MCPサーバーコード
```

### 2. manifest.json の作成

**重要**: DXT公式仕様 v0.1 に準拠する必要があります。

```json
{
  "dxt_version": "0.1",
  "name": "crawl4ai-extension",
  "display_name": "Crawl4AI Web Crawler MCP",
  "version": "1.0.7",
  "description": "...",
  "author": {
    "name": "作者名",
    "email": "メールアドレス",
    "url": "GitHubリポジトリURL"
  },
  "server": {
    "type": "python",
    "entry_point": "server/main.py",
    "mcp_config": {
      "command": "python",
      "args": ["${__dirname}/server/main.py"],
      "env": {
        "PYTHONPATH": "${__dirname}/server",
        "OPENAI_API_KEY": "${user_config.openai_api_key}"
      }
    }
  },
  "user_config": {
    "openai_api_key": {
      "type": "string",
      "title": "OpenAI API Key",
      "description": "...",
      "required": false,
      "sensitive": true
    }
  },
  "compatibility": {
    "platforms": ["darwin", "win32", "linux"],
    "apps": {
      "claude-desktop": ">=0.10.0"
    },
    "runtimes": {
      "python": ">=3.8.0 <4.0.0"
    }
  }
}
```

**重要なポイント**:
- `${__dirname}` 変数を使用してパス解決
- `user_config` でセキュアなAPI key管理
- `sensitive: true` で機密情報を保護

### 3. エントリーポイント (server/main.py) の作成

```python
#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# PYTHONPATHの設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def install_dependencies():
    """自動依存関係インストール"""
    import subprocess
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_path), "--quiet"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    try:
        from crawl4ai_mcp.server import main as server_main
        server_main()
    except ImportError as e:
        if install_dependencies():
            from crawl4ai_mcp.server import main as server_main
            server_main()
        else:
            print(f"Failed to install dependencies: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
```

### 4. FastMCP 互換性の確保

**重要**: FastMCP の最新版では、デコレータに括弧が必要です。

```python
# ❌ 古い書き方（エラーになる）
@mcp.tool
@mcp.prompt

# ✅ 新しい書き方（必須）
@mcp.tool()
@mcp.prompt()

# ✅ 引数ありは元々正しい
@mcp.resource("uri://example")
```

### 5. パッケージのビルド

```bash
cd crawl4ai-dxt
dxt pack
```

成功すると `crawl4ai-extension-1.0.7.dxt` が生成されます。

## よくある問題と解決方法

### パスエラー
```
can't open file 'C:\...\server\main.py': [Errno 2] No such file or directory
```

**解決策**: manifest.json で `${__dirname}` 変数を正しく使用
```json
"args": ["${__dirname}/server/main.py"]
```

### モジュールインポートエラー
```
ModuleNotFoundError: No module named 'crawl4ai_mcp'
```

**解決策**: PYTHONPATHの設定
```python
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
```

### 依存関係エラー
```
No module named 'youtube_transcript_api'
```

**解決策**: 自動インストール機能の実装（上記 main.py 参照）

### FastMCP デコレータエラー
```
The @tool decorator was used incorrectly. Use @tool() instead of @tool
```

**解決策**: すべてのデコレータに括弧を追加
```bash
# 一括置換
sed -i 's/@mcp\.tool/@mcp.tool()/g' server.py
sed -i 's/@mcp\.prompt/@mcp.prompt()/g' server.py
```

## ベストプラクティス

### 1. セキュリティ
- API キーは `user_config` で `sensitive: true` に設定
- 環境変数経由でサーバーに渡す

### 2. エラーハンドリング
- 依存関係の自動インストール機能
- 詳細なログ出力
- プラットフォーム固有の警告

### 3. ユーザビリティ
- 分かりやすい設定項目
- 包括的なREADME
- バージョン履歴の記載

### 4. 互換性
- 複数プラットフォーム対応
- Python バージョン範囲の指定
- Claude Desktop バージョン要件

## 参考リンク

- [DXT 公式リポジトリ](https://github.com/anthropics/dxt)
- [MANIFEST.md 仕様](https://github.com/anthropics/dxt/blob/main/MANIFEST.md)
- [Python サンプル](https://github.com/anthropics/dxt/tree/main/examples/file-manager-python)
- [FastMCP ドキュメント](https://github.com/jlowin/fastmcp)

## 開発フロー

1. ローカルでMCPサーバーの動作確認
2. DXT構造への変換
3. manifest.json の作成
4. エントリーポイントの実装
5. FastMCP互換性の確保
6. パッケージのビルドとテスト
7. Claude Desktop での動作確認

このガイドに従うことで、MCPサーバーを効率的にDXTパッケージ化できます。