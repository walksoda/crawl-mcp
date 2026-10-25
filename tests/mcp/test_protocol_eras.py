"""
Protocol era tests.

The server must serve both MCP protocol eras over stdio and Streamable HTTP:
- 2026-07-28 clients (stateless per-request envelope, server/discover)
- older clients (initialize handshake, 2024-11-05 through 2025-11-25)
"""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from crawl4ai_mcp import __version__

ERAS = {
    "auto": "2026-07-28",
    "legacy": "2025-11-25",
}
HANDSHAKE_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def http_url(project_root, venv_python):
    """Run the server with --transport http and return its /mcp URL."""
    port = _free_port()
    proc = subprocess.Popen(
        [str(venv_python), "-m", "crawl4ai_mcp.server",
         "--transport", "http", "--port", str(port)],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except urllib.error.HTTPError:
                break  # any HTTP status means the server is listening
            except OSError:
                if proc.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("HTTP server did not start")
                time.sleep(0.3)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(params=[(t, m) for t in ("stdio", "http") for m in sorted(ERAS)],
                ids=lambda p: f"{p[0]}-{p[1]}")
async def era_client(request, project_root, venv_python):
    """Connect over the param's transport with the param's client mode."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    transport_name, mode = request.param
    if transport_name == "stdio":
        transport = StdioTransport(
            command=str(venv_python),
            args=["-m", "crawl4ai_mcp.server"],
            cwd=str(project_root),
        )
    else:
        transport = request.getfixturevalue("http_url")
    async with Client(transport, mode=mode) as client:
        yield mode, client


@pytest.mark.mcp
class TestProtocolEras:
    """Both eras negotiate, report the package version and serve tools."""

    async def test_negotiated_version(self, era_client):
        mode, client = era_client
        assert client.protocol_version == ERAS[mode]

    async def test_server_info_reports_package_version(self, era_client):
        _, client = era_client
        assert client.server_info.name == "Crawl4AI"
        assert client.server_info.version == __version__

    async def test_tools_listed_and_callable(self, era_client):
        _, client = era_client
        tools = await client.list_tools()
        assert len(tools) == 21

        result = await client.call_tool("get_supported_file_formats", {})
        data = json.loads(result.content[0].text)
        assert data["success"] is True
        assert ".pdf" in data["supported_formats"]


@pytest.mark.mcp
@pytest.mark.parametrize("version", HANDSHAKE_VERSIONS)
def test_initialize_accepts_older_versions(version, project_root, venv_python):
    """An initialize request for each older version gets that version back."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    proc = subprocess.run(
        [str(venv_python), "-m", "crawl4ai_mcp.server"],
        cwd=str(project_root),
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    response = json.loads(proc.stdout.splitlines()[0])
    assert response["result"]["protocolVersion"] == version
