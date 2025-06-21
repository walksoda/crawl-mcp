#!/usr/bin/env python3
"""
シンプルなPure StreamableHTTP実装 - 最小限のテスト用
"""

import asyncio
import json
import logging
from aiohttp import web
import argparse
import uuid

class SimpleHTTPServer:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.sessions = {}
        
    async def health_check(self, request):
        """ヘルスチェック"""
        return web.json_response({
            "status": "healthy",
            "server": "simple-pure-http",
            "version": "1.0.0",
            "protocol": "StreamableHTTP (pure JSON)"
        })
    
    async def initialize(self, request):
        """MCP初期化"""
        try:
            body = await request.json()
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {"initialized": True}
            
            response_data = {
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "simple-mcp-server", "version": "1.0.0"}
                }
            }
            
            response = web.json_response(response_data)
            response.headers['mcp-session-id'] = session_id
            return response
            
        except Exception as e:
            return web.json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)}
            }, status=500)
    
    async def mcp_handler(self, request):
        """メインMCPエンドポイント"""
        try:
            session_id = request.headers.get('mcp-session-id')
            if not session_id or session_id not in self.sessions:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32002, "message": "Missing session ID"}
                }, status=401)
            
            body = await request.json()
            method = body.get("method")
            
            if method == "tools/list":
                result = {
                    "tools": [
                        {"name": "crawl_url", "description": "Simple URL crawler"},
                        {"name": "extract_youtube_transcript", "description": "YouTube transcript extractor"}
                    ]
                }
            elif method == "tools/call":
                tool_name = body.get("params", {}).get("name")
                if tool_name == "crawl_url":
                    result = {"content": [{"type": "text", "text": "Mock crawl result"}]}
                elif tool_name == "extract_youtube_transcript":
                    result = {"content": [{"type": "text", "text": "Mock YouTube transcript"}]}
                else:
                    result = {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}
            else:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }, status=404)
            
            return web.json_response({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": result
            })
            
        except Exception as e:
            return web.json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)}
            }, status=500)
    
    async def start(self):
        """サーバー開始"""
        app = web.Application()
        app.router.add_get('/health', self.health_check)
        app.router.add_post('/mcp/initialize', self.initialize)
        app.router.add_post('/mcp', self.mcp_handler)
        
        print(f"🚀 Simple Pure HTTP Server starting on {self.host}:{self.port}")
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("Server stopping...")
        finally:
            await runner.cleanup()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    server = SimpleHTTPServer(args.host, args.port)
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())