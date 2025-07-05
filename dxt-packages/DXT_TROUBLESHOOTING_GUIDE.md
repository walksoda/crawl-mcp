# DXT トラブルシューティングガイド

このドキュメントは、DXTパッケージ作成・動作時に発生する問題とその解決方法をまとめたものです。

## 共通エラーパターン

### 1. ファイルパス関連エラー

#### エラー例
```
can't open file 'C:\Users\...\app-0.11.4\server\main.py': [Errno 2] No such file or directory
```

#### 原因
- DXTパッケージ展開後のパス解決が正しくない
- `${__dirname}` 変数が使用されていない

#### 解決方法
```json
// manifest.json - 修正前
"args": ["server/main.py"]

// manifest.json - 修正後
"args": ["${__dirname}/server/main.py"]
```

### 2. Python モジュールインポートエラー

#### エラー例
```
ModuleNotFoundError: No module named 'crawl4ai_mcp'
ModuleNotFoundError: No module named 'youtube_transcript_api'
```

#### 原因と解決方法

**ケース1: 相対インポートの問題**
```python
# 解決方法: PYTHONPATHの設定
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
```

**ケース2: 依存関係未インストール**
```python
# 解決方法: 自動インストール機能
def install_dependencies():
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

# メイン処理での使用
try:
    from crawl4ai_mcp.server import main as server_main
    server_main()
except ImportError:
    if install_dependencies():
        from crawl4ai_mcp.server import main as server_main
        server_main()
```

### 3. FastMCP デコレータエラー

#### エラー例
```
The @tool decorator was used incorrectly. Did you forget to call it? Use @tool() instead of @tool
The @prompt decorator was used incorrectly. Use @prompt() instead of @prompt
```

#### 原因
FastMCP の新しいバージョンでは、引数なしデコレータにも括弧が必要

#### 解決方法
```python
# ❌ 古い書き方
@mcp.tool
async def crawl_url(request):
    pass

@mcp.prompt
async def analyze_content(message):
    pass

# ✅ 新しい書き方
@mcp.tool()
async def crawl_url(request):
    pass

@mcp.prompt()
async def analyze_content(message):
    pass

# ✅ 引数ありは変更不要
@mcp.resource("uri://example")
async def get_config():
    pass
```

**一括修正スクリプト**:
```bash
sed -i 's/@mcp\.tool/@mcp.tool()/g' server.py
sed -i 's/@mcp\.prompt/@mcp.prompt()/g' server.py
```

### 4. DXT パッケージビルドエラー

#### エラー例
```
ERROR: Cannot pack extension with invalid manifest
Manifest validation failed:
- server.mcp_config.command: Required
```

#### 解決方法
manifest.json の必須フィールドを確認：

```json
{
  "dxt_version": "0.1",           // 必須
  "name": "extension-name",       // 必須
  "version": "1.0.0",            // 必須
  "description": "...",          // 必須
  "author": {                    // 必須
    "name": "作者名",
    "email": "email@example.com"
  },
  "server": {                    // 必須
    "type": "python",
    "entry_point": "server/main.py",
    "mcp_config": {              // 必須
      "command": "python",       // 必須
      "args": ["${__dirname}/server/main.py"]
    }
  }
}
```

### 5. Claude Desktop での認識エラー

#### 症状
DXTファイルをインストールしても Claude Desktop で認識されない

#### 解決方法

1. **Claude Desktop の再起動**
2. **ログの確認**
   - Windows: `%APPDATA%\Claude\logs`
   - macOS: `~/Library/Logs/Claude`
   - Linux: `~/.config/claude/logs`

3. **設定の確認**
   ```json
   // 正しい user_config 設定
   "user_config": {
     "api_key": {
       "type": "string",
       "title": "API Key",
       "required": false,
       "sensitive": true
     }
   }
   ```

### 6. プラットフォーム固有の問題

#### Windows
```
Visual C++ Build Tools may be required for some dependencies
```
**解決方法**: Visual Studio Build Tools のインストール

#### Linux
```
ImportError: libGL.so.1: cannot open shared object file
```
**解決方法**: システム依存関係のインストール
```bash
sudo apt-get install libnss3 libnspr4 libasound2 libdrm2 libxcomposite1
```

#### macOS
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```
**解決方法**: 証明書の更新
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

## デバッグ手順

### 1. ログの確認
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Working directory: {os.getcwd()}")
logger.debug(f"Python path: {sys.path}")
logger.debug(f"Environment variables: {dict(os.environ)}")
```

### 2. ステップバイステップ確認
1. `manifest.json` の検証: `dxt pack --dry-run`
2. Python インポートの確認: `python -c "from crawl4ai_mcp.server import main"`
3. 依存関係の確認: `pip list`
4. パス解決の確認: `echo ${__dirname}`

### 3. 最小限の動作確認
```python
# test_minimal.py
try:
    import sys
    print(f"Python version: {sys.version}")
    
    from crawl4ai_mcp.server import main
    print("Import successful")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

## バージョン履歴と対応

| バージョン | 主な問題 | 解決内容 |
|-----------|----------|----------|
| v1.0.1-1.0.3 | ファイルパス解決 | `${__dirname}` 変数の正しい使用 |
| v1.0.4 | DXT仕様準拠 | 公式仕様に基づく構造修正 |
| v1.0.5 | 依存関係エラー | 自動インストール機能追加 |
| v1.0.6 | @tool デコレータ | FastMCP 互換性修正 |
| v1.0.7 | @prompt デコレータ | 完全なFastMCP対応 |

## 予防策

### 1. 開発時
- ローカルでの十分なテスト
- 複数環境での動作確認
- 依存関係の明確な管理

### 2. パッケージング時
- manifest.json の検証
- ファイル構造の確認
- デコレータ構文の確認

### 3. リリース時
- バージョン番号の一貫性
- ドキュメントの更新
- 既知の問題の明記

## 参考情報

### 有用なコマンド
```bash
# DXT パッケージ検証
dxt pack --dry-run

# Python モジュール確認
python -c "import sys; print('\n'.join(sys.path))"

# 依存関係確認
pip check

# FastMCP バージョン確認
pip show fastmcp
```

### ログ出力の活用
```python
# 詳細なエラー情報
import traceback
try:
    # 処理
    pass
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
```

このガイドを参考に、問題を迅速に特定・解決できます。