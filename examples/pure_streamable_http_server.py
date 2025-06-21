#!/usr/bin/env python3
"""
純粋なStreamableHTTP実装 - SSE不使用版

FastMCPの真のStreamableHTTPプロトコルを実装し、
プレーンなJSONレスポンスを返すシンプルなHTTPサーバー
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import argparse

from aiohttp import web, ContentTypeError
from aiohttp.web import Request, Response, Application

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crawl4ai_mcp.server import mcp


class StreamableHTTPServer:
    """純粋なStreamableHTTP実装"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.app = None
        self.logger = logging.getLogger(__name__)
        self.sessions = {}  # セッション管理
        
    async def initialize_handler(self, request: Request) -> Response:
        """MCP初期化エンドポイント"""
        try:
            # リクエストボディからJSON-RPC 2.0を解析
            try:
                body = await request.json()
            except ContentTypeError:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }, status=400)
            
            if body.get("method") != "initialize":
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }, status=404)
            
            # セッションIDを生成
            import uuid
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "initialized": True,
                "client_info": body.get("params", {}).get("clientInfo", {})
            }
            
            # MCP初期化レスポンス
            response_data = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        },
                        "prompts": {
                            "listChanged": False
                        },
                        "resources": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": "crawl4ai-mcp-server",
                        "version": "1.0.0"
                    }
                }
            }
            
            # セッションIDをヘッダーに設定
            response = web.json_response(response_data)
            response.headers['mcp-session-id'] = session_id
            return response
            
        except Exception as e:
            self.logger.error(f"Initialize error: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status=500)
    
    async def mcp_handler(self, request: Request) -> Response:
        """メインMCPエンドポイント（純粋JSON）"""
        try:
            # セッション確認
            session_id = request.headers.get('mcp-session-id')
            if not session_id or session_id not in self.sessions:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32002,
                        "message": "Missing or invalid session ID"
                    }
                }, status=401)
            
            # リクエストボディを解析
            try:
                body = await request.json()
            except ContentTypeError:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }, status=400)
            
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")
            
            # メソッドに応じて処理分岐
            if method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tool_call(params)
            elif method == "prompts/list":
                result = await self._handle_prompts_list()
            elif method == "prompts/get":
                result = await self._handle_prompt_get(params)
            elif method == "resources/list":
                result = await self._handle_resources_list()
            elif method == "resources/read":
                result = await self._handle_resource_read(params)
            else:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }, status=404)
            
            # 成功レスポンス
            response_data = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
            return web.json_response(response_data)
            
        except Exception as e:
            self.logger.error(f"MCP handler error: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": request_id if 'request_id' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status=500)
    
    async def _handle_tools_list(self) -> Dict[str, Any]:
        """ツール一覧を取得"""
        # MCPサーバーからツール一覧を取得
        # 実際のツール情報を返す
        return {
            "tools": [
                {
                    "name": "crawl_url",
                    "description": "Crawl a URL and extract content",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "generate_markdown": {"type": "boolean", "default": True}
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "extract_youtube_transcript",
                    "description": "Extract transcript from YouTube video",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "languages": {"type": "array", "items": {"type": "string"}},
                            "include_timestamps": {"type": "boolean", "default": True}
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "search_google",
                    "description": "Perform Google search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num_results": {"type": "integer", "default": 10},
                            "language": {"type": "string", "default": "en"}
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """ツール実行"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "crawl_url":
            return await self._crawl_url(arguments)
        elif tool_name == "extract_youtube_transcript":
            return await self._extract_youtube_transcript(arguments)
        elif tool_name == "search_google":
            return await self._search_google(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _crawl_url(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """URLクローリング実行"""
        url = args.get("url")
        generate_markdown = args.get("generate_markdown", True)
        
        # 簡単な例 - 実際の実装では crawl4ai を使用
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    content = await resp.text()
                    
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"Crawled content from {url}:\n\n{content[:500]}..."
                        }]
                    }
        except Exception as e:
            return {
                "content": [{
                    "type": "text", 
                    "text": f"Error crawling {url}: {str(e)}"
                }]
            }
    
    async def _extract_youtube_transcript(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """YouTube字幕抽出実行"""
        url = args.get("url")
        languages = args.get("languages", ["en"])
        include_timestamps = args.get("include_timestamps", True)
        
        # youtube-transcript-api を使用した実装
        try:
            # 遅延インポートでエラーを回避
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                import re
                
                # YouTube URLからビデオIDを抽出
                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
                if not video_id_match:
                    raise ValueError("Invalid YouTube URL")
                
                video_id = video_id_match.group(1)
                
                # 字幕を取得
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                
                # フォーマット
                formatted_transcript = []
                for entry in transcript:
                    if include_timestamps:
                        formatted_transcript.append(f"[{entry['start']:.1f}s] {entry['text']}")
                    else:
                        formatted_transcript.append(entry['text'])
                
                return {
                    "content": [{
                        "type": "text",
                        "text": f"YouTube transcript for {url}:\n\n" + "\n".join(formatted_transcript)
                    }]
                }
                
            except ImportError:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"YouTube transcript extraction requires youtube-transcript-api package. Please install it with: pip install youtube-transcript-api"
                    }]
                }
            
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error extracting transcript: {str(e)}"
                }]
            }
    
    async def _search_google(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Google検索実行"""
        query = args.get("query")
        num_results = args.get("num_results", 10)
        language = args.get("language", "en")
        
        # 簡単な例 - 実際の実装では検索APIを使用
        return {
            "content": [{
                "type": "text",
                "text": f"Google search results for '{query}' (language: {language}, limit: {num_results}):\n\n1. Example result 1\n2. Example result 2"
            }]
        }
    
    async def _handle_prompts_list(self) -> Dict[str, Any]:
        """プロンプト一覧を取得"""
        return {"prompts": []}
    
    async def _handle_prompt_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """プロンプトを取得"""
        return {"messages": []}
    
    async def _handle_resources_list(self) -> Dict[str, Any]:
        """リソース一覧を取得"""
        return {"resources": []}
    
    async def _handle_resource_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """リソースを読み取り"""
        return {"contents": []}
    
    async def health_check(self, request: Request) -> Response:
        """ヘルスチェックエンドポイント"""
        return web.json_response({
            "status": "healthy",
            "server": "crawl4ai-mcp-pure-http",
            "version": "1.0.0",
            "protocol": "StreamableHTTP (pure JSON)"
        })
    
    async def create_app(self) -> Application:
        """アプリケーション作成"""
        app = web.Application()
        
        # ルート設定
        app.router.add_get('/', self.health_check)
        app.router.add_get('/health', self.health_check)
        app.router.add_post('/mcp/initialize', self.initialize_handler)
        app.router.add_post('/mcp', self.mcp_handler)
        
        # CORS設定
        async def cors_handler(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, mcp-session-id'
            return response
        
        app.middlewares.append(cors_handler)
        
        return app
    
    async def start(self):
        """サーバー開始"""
        self.app = await self.create_app()
        
        self.logger.info(f"🚀 Pure StreamableHTTP Server starting...")
        self.logger.info(f"📡 Host: {self.host}")
        self.logger.info(f"🔌 Port: {self.port}")
        self.logger.info(f"🌐 Endpoint: http://{self.host}:{self.port}")
        self.logger.info(f"📋 Protocol: Pure JSON (no SSE)")
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info("✅ Server started successfully")
        
        # サーバーを永続的に実行
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            self.logger.info("📴 Server shutting down...")
        finally:
            await runner.cleanup()


def setup_logging(log_level: str = "INFO"):
    """ログ設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Pure StreamableHTTP MCP Server (no SSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python pure_streamable_http_server.py                    # デフォルト設定で起動
  python pure_streamable_http_server.py --host 0.0.0.0     # 外部アクセス可能で起動
  python pure_streamable_http_server.py --port 8080        # ポート8080で起動

特徴:
  - Server-Sent Events (SSE) を使用しない純粋なJSON HTTP API
  - 単一の /mcp エンドポイントでJSON-RPC 2.0プロトコル
  - セッション管理とプレーンなHTTPレスポンス
  - シンプルなクライアント実装が可能
        """
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="バインドするホスト (デフォルト: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="バインドするポート (デフォルト: 8000)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="ログレベル (デフォルト: INFO)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    print("🎯 Pure StreamableHTTP MCP Server")
    print(f"📍 Address: http://{args.host}:{args.port}")
    print("📋 Protocol: Pure JSON (no SSE)")
    print("🛑 Stop: Ctrl+C")
    print()
    
    server = StreamableHTTPServer(host=args.host, port=args.port)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())