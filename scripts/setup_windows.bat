@echo off
REM Crawl4AI MCP サーバーのセットアップスクリプト (Windows)

echo Crawl4AI MCP サーバーのセットアップを開始します...

REM Python仮想環境の作成
echo Python仮想環境を作成中...
python -m venv venv

REM 仮想環境の有効化
echo 仮想環境を有効化中...
call venv\Scripts\activate.bat

REM 依存関係のインストール
echo 依存関係をインストール中...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo セットアップが完了しました！
echo.
echo 使用方法:
echo 1. 仮想環境を有効化: venv\Scripts\activate.bat
echo 2. サーバーを起動: python -m crawl4ai_mcp.server
echo 3. 例の実行: python example_usage.py
echo.
echo サーバーを停止するには Ctrl+C を押してください
pause