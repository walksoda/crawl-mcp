"""
MCP Prompt definitions for Crawl4AI MCP Server.

This module contains all prompt functions that provide structured guidance
for various crawling and processing operations.
"""

from typing import List, Dict, Any


def crawl_website_prompt(url: str, extraction_type: str = "basic") -> List[Dict[str, Any]]:
    """
    Create a prompt for crawling a website with specific instructions.
    
    Args:
        url: The URL to crawl
        extraction_type: Type of extraction (basic, structured, content)
    
    Returns:
        List of prompt messages for crawling
    """
    if extraction_type == "basic":
        content = f"""URLをクローリングしてコンテンツを取得してください: {url}

基本的なクローリングを実行し、以下の情報を取得してください：
- ページタイトル
- メインコンテンツ
- Markdown形式のテキスト

crawl_urlツールを使用してください。"""

    elif extraction_type == "structured":
        content = f"""構造化データ抽出のためにURLをクローリングしてください: {url}

以下の手順で実行してください：
1. ページの構造を分析
2. 適切なCSSセレクターまたはXPathを特定
3. extract_structured_dataツールを使用して構造化データを抽出

抽出したいデータの種類を指定してください（例：記事タイトル、価格、説明文など）。"""

    elif extraction_type == "content":
        content = f"""コンテンツ分析のためにURLをクローリングしてください: {url}

以下の分析を行ってください：
1. ページの主要コンテンツを抽出
2. 重要な見出しやセクションを特定
3. メディアファイル（画像、動画など）があれば一覧化
4. ページの構造と内容を要約

crawl_urlツールを使用し、extract_media=trueに設定してください。"""

    else:
        content = f"""URLをクローリングしてください: {url}

利用可能な抽出タイプ：
- basic: 基本的なコンテンツ取得
- structured: 構造化データ抽出
- content: 詳細なコンテンツ分析

適切なツールを選択して実行してください。"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


def analyze_crawl_results_prompt(crawl_data: str) -> List[Dict[str, Any]]:
    """
    Create a prompt for analyzing crawl results.
    
    Args:
        crawl_data: JSON string of crawl results
    
    Returns:
        List of prompt messages for analysis
    """
    content = f"""以下のクローリング結果を分析してください：

{crawl_data}

分析項目：
1. 取得したコンテンツの概要
2. 主要な情報やキーポイント
3. データの構造と品質
4. 追加で抽出すべき情報があるか
5. 結果の有用性と改善点

詳細な分析レポートを提供してください。"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


def batch_crawl_setup_prompt(urls: str) -> List[Dict[str, Any]]:
    """
    Create a prompt for setting up batch crawling.
    
    Args:
        urls: Comma-separated list of URLs
    
    Returns:
        List of prompt messages for batch crawling
    """
    url_list = [url.strip() for url in urls.split(",")]
    
    content = f"""複数のURLを一括でクローリングしてください：

対象URL（{len(url_list)}件）：
{chr(10).join(f"- {url}" for url in url_list)}

batch_crawlツールを使用して以下を実行してください：
1. 全URLのコンテンツを取得
2. 各ページの基本情報を収集
3. 結果を比較・分析
4. 共通点や相違点を特定
5. 統合レポートを作成

効率的な一括処理を行い、結果をまとめて報告してください。"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]


def process_file_prompt(file_url: str, file_type: str = "auto") -> List[Dict[str, Any]]:
    """
    Create a prompt for processing files with MarkItDown.
    
    Args:
        file_url: URL of the file to process
        file_type: Type of file processing (auto, pdf, office, zip)
    
    Returns:
        List of prompt messages for file processing
    """
    if file_type == "pdf":
        content = f"""PDFファイルを処理してMarkdown形式に変換してください: {file_url}

以下の機能を使用してPDFを処理してください：
1. process_fileツールを使用してPDFをダウンロード・処理
2. テキスト内容をMarkdown形式で抽出
3. 文書の構造（見出し、段落、リストなど）を保持
4. メタデータ（タイトル、作成者、作成日など）を取得
5. 処理結果を分析・要約

PDFの内容を理解しやすい形式で提示してください。"""

    elif file_type == "office":
        content = f"""Microsoft Officeファイルを処理してMarkdown形式に変換してください: {file_url}

以下の手順で処理してください：
1. process_fileツールを使用してOfficeファイルを処理
2. 文書の種類に応じた適切な抽出を実行
   - Word: テキスト、見出し、表、画像キャプション
   - Excel: シート名、セルデータ、表形式
   - PowerPoint: スライドタイトル、コンテンツ、ノート
3. 元の構造とフォーマットを可能な限り保持
4. メタデータと追加情報を含める

処理した内容を構造化された形式で提示してください。"""

    elif file_type == "zip":
        content = f"""ZIPアーカイブを処理して中身のファイルを分析してください: {file_url}

以下の処理を実行してください：
1. process_fileツールを使用してZIPファイルを処理
2. アーカイブ内の全ファイルを抽出・分析
3. 各ファイルの種類と内容を特定
4. サポートされているファイル形式をMarkdownに変換
5. ファイル構造とディレクトリ階層を可視化
6. 処理できなかったファイルがあれば理由を説明

全体的な分析結果と各ファイルの詳細を報告してください。"""

    else:  # auto
        content = f"""ファイルを自動検出して適切に処理してください: {file_url}

以下の手順で処理してください：
1. get_supported_file_formatsツールでサポート形式を確認
2. process_fileツールを使用してファイルを処理
3. ファイル形式に応じた最適な抽出を実行
4. 内容をMarkdown形式で構造化
5. メタデータと追加情報を取得
6. 処理結果を分析・要約

ファイルの種類と内容に応じた詳細な分析を提供してください。

利用可能なツール：
- process_file: ファイル処理とMarkdown変換
- get_supported_file_formats: サポート形式の確認"""

    return [{"role": "user", "content": {"type": "text", "text": content}}]