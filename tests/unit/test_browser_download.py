"""browser_download against a local HTTP server and a real headless Chromium.

Checks the limits that a mock cannot: a file over max_bytes and a stalled
transfer are cancelled while they arrive, even when the download starts
before the page has finished loading.
"""

import time

import pytest

from crawl4ai_mcp.processors import document_download as dd
from tests.unit._local_http import MB, Server


@pytest.fixture(scope="module")
def server():
    srv = Server().start()
    yield srv
    srv.stop()


@pytest.fixture(autouse=True)
def short_limits(monkeypatch, server):
    server.reset()
    monkeypatch.setattr(dd, "BROWSER_DOWNLOAD_TIMEOUT_S", 10)
    monkeypatch.setattr(dd, "BROWSER_DOWNLOAD_GRACE_S", 2)
    monkeypatch.setattr(dd, "BROWSER_TRANSFER_TIMEOUT_S", 3)


async def test_small_file_is_returned(server):
    result = await dd.browser_download(f"{server.url}/small", 1 * MB)
    assert result.data == b"%PDF-1.4 small"
    assert result.filename == "f.pdf"


async def test_file_over_the_limit_is_cancelled_early(server):
    result = await dd.browser_download(f"{server.url}/big", 1 * MB)
    assert result.too_large is True
    assert server.disconnected["/big"].wait(5)
    assert server.sent["/big"] < 100 * MB  # stopped long before the 200 MB end


async def test_download_started_while_page_loads_is_size_checked(server):
    result = await dd.browser_download(f"{server.url}/script-download", 1 * MB)
    assert result.too_large is True
    assert server.disconnected["/big"].wait(5)
    assert server.sent["/big"] < 100 * MB


async def test_stalled_transfer_times_out(server):
    start = time.monotonic()
    result = await dd.browser_download(f"{server.url}/stall", 20 * MB)
    assert result.data is None
    assert "did not finish" in result.error
    assert time.monotonic() - start < 15


async def test_block_page_reports_title_and_html(server):
    result = await dd.browser_download(f"{server.url}/blocked", 1 * MB)
    assert result.data is None
    assert result.is_challenge is True
    assert "check" in result.html
