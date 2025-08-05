# 設定例

異なるインストール方法、プラットフォーム、使用ケースに対応する包括的な設定例です。

## 🚀 UVXインストール設定

### 基本UVX設定（推奨）

**英語環境用:**
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
        "CRAWL4AI_LANG": "en"
      }
    }
  }
}
```

**日本語環境用:**
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

### デバッグログ付きUVX

```json
{
  "mcpServers": {
    "crawl-mcp-debug": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "ja",
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

## 🖥️ 開発環境設定

### ローカル開発セットアップ

**Linux/macOS:**
```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "C:\\path\\to\\your\\crawl\\venv\\Scripts\\python.exe",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "C:\\path\\to\\your\\crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### カスタムPythonパス付き開発環境

```json
{
  "mcpServers": {
    "crawl4ai-custom": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "PYTHONPATH": "/path/to/crawl/venv/lib/python3.10/site-packages"
      }
    }
  }
}
```

## 🌐 HTTPトランスポート設定

### HTTP STDIO設定

```json
{
  "mcpServers": {
    "crawl4ai-stdio": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "PYTHONPATH": "/home/user/prj/crawl/venv/lib/python3.10/site-packages"
      }
    }
  }
}
```

### Legacy HTTP設定

```json
{
  "mcpServers": {
    "crawl4ai-legacy-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

### Pure StreamableHTTP設定

```json
{
  "mcpServers": {
    "crawl4ai-pure-http": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### カスタムポート付きHTTP

```json
{
  "mcpServers": {
    "crawl4ai-http-custom": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server", "--transport", "http", "--port", "8080"],
      "cwd": "/path/to/project",
      "env": {
        "PYTHONPATH": "/path/to/venv/lib/python3.10/site-packages"
      }
    }
  }
}
```

## 🖥️ プラットフォーム固有設定

### WSL（Windows Subsystem for Linux）

```json
{
  "mcpServers": {
    "crawl4ai-wsl": {
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

### シンプルなWSLコマンド

```json
{
  "mcpServers": {
    "crawl4ai-wsl-simple": {
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

### macOS設定

```json
{
  "mcpServers": {
    "crawl4ai-macos": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {}
    }
  }
}
```

## 🔧 環境変数

### 利用可能な環境変数

```json
{
  "mcpServers": {
    "crawl4ai-full-env": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "ja",
        "FASTMCP_LOG_LEVEL": "INFO",
        "MCP_TRANSPORT": "stdio",
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": "8000"
      }
    }
  }
}
```

### 本番環境

```json
{
  "mcpServers": {
    "crawl4ai-production": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "ja",
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

## 🛠️ 高度な設定例

### LLM設定付き構成

```json
{
  "mcpServers": {
    "crawl4ai-llm": {
      "command": "python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl",
      "env": {
        "PYTHONPATH": "/path/to/crawl/venv/lib/python3.10/site-packages",
        "OPENAI_API_KEY": "your-api-key-here",
        "ANTHROPIC_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### マルチインスタンス設定

```json
{
  "mcpServers": {
    "crawl4ai-primary": {
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
    },
    "crawl4ai-secondary": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### テスト設定

```json
{
  "mcpServers": {
    "crawl4ai-test": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG",
        "PYTEST_CURRENT_TEST": "true"
      }
    }
  }
}
```

## 📂 設定ファイルの場所

### プラットフォーム固有のパス

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

### 設定ファイルセットアップコマンド

**Linux/macOS:**
```bash
# 設定ファイルをコピー
cp configs/claude_desktop_config.json ~/.config/claude-desktop/claude_desktop_config.json

# ディレクトリが存在しない場合は作成
mkdir -p ~/.config/claude-desktop/
```

**Windows PowerShell:**
```powershell
# 設定ファイルをコピー
Copy-Item "configs\claude_desktop_config.json" "$env:APPDATA\Claude\claude_desktop_config.json"

# ディレクトリが存在しない場合は作成
New-Item -ItemType Directory -Force -Path "$env:APPDATA\Claude"
```

## 🔍 ツールパラメータ設定例

### 基本的なWebクローリング設定

```json
{
  "wait_for_js": true,
  "simulate_user": true, 
  "timeout": 30,
  "generate_markdown": true
}
```

### JavaScript重要サイト用設定

```json
{
  "wait_for_js": true,
  "simulate_user": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded",
  "execute_js": "window.scrollTo(0, document.body.scrollHeight);",
  "generate_markdown": true
}
```

### 高度なクローリング設定

```json
{
  "url": "https://example.com",
  "max_depth": 2,
  "crawl_strategy": "bfs",
  "content_filter": "bm25",
  "filter_query": "重要なコンテンツキーワード",
  "chunk_content": true,
  "auto_summarize": true,
  "summary_length": "medium"
}
```

## 🚨 トラブルシューティング設定

### デバッグ設定

```json
{
  "mcpServers": {
    "crawl4ai-debug": {
      "command": "/home/user/prj/crawl/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/home/user/prj/crawl",
      "env": {
        "FASTMCP_LOG_LEVEL": "DEBUG",
        "PYTHONPATH": "/home/user/prj/crawl",
        "DEBUG": "1"
      }
    }
  }
}
```

### エラー分離設定

```json
{
  "mcpServers": {
    "crawl4ai-isolated": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "ja",
        "FASTMCP_LOG_LEVEL": "ERROR",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## 📊 設定の検証

### 検証コマンド

```bash
# 設定構文をテスト
python -m json.tool claude_desktop_config.json

# サーバー起動をテスト
python -m crawl4ai_mcp.server --help

# UVXインストールをテスト
uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp --help
```

### ヘルスチェック設定

```json
{
  "mcpServers": {
    "crawl4ai-health": {
      "url": "http://127.0.0.1:8000/mcp",
      "timeout": 30000
    }
  }
}
```

## 🔗 関連ドキュメント

- **インストールガイド**: [INSTALLATION.md](INSTALLATION.md)
- **HTTP統合**: [HTTP_INTEGRATION.md](HTTP_INTEGRATION.md)
- **APIリファレンス**: [API_REFERENCE.md](API_REFERENCE.md)
- **高度な使用法**: [ADVANCED_USAGE.md](ADVANCED_USAGE.md)
- **開発ガイド**: [DEVELOPMENT.md](DEVELOPMENT.md)

## 💡 設定のコツ

1. **UVXを簡単に使用**: ほとんどのユーザーに推奨
2. **まずローカルでデバッグ**: 開発環境から始める
3. **ファイルパスをチェック**: すべてのパスが絶対パスで正しいことを確認
4. **環境変数**: 機密データには環境変数を使用
5. **設定をテスト**: 使用前にJSON構文を検証
6. **プラットフォーム考慮**: OSごとに異なるパス
7. **ログレベル**: トラブルシューティングにはDEBUG、本番にはERROR