"""FileProcessor.download_file over real HTTP to a local server.

A refused download raises DownloadRefused, oversized and slow transfers end
with an error, and cancelling the caller closes the connection.
"""

import asyncio
import threading
from urllib.parse import urlparse

import pytest

from crawl4ai_mcp.processors import file_processor as fp_module
from crawl4ai_mcp.processors.document_download import DownloadRefused
from crawl4ai_mcp.processors.file_processor import FileProcessor
from tests.unit._local_http import Server


@pytest.fixture(scope="module")
def server():
    srv = Server().start()
    yield srv
    srv.stop()


@pytest.fixture(autouse=True)
def fresh(server):
    server.reset()


async def test_download_sends_browser_user_agent(server):
    content, content_type = await FileProcessor().download_file(f"{server.url}/small")
    assert content == b"%PDF-1.4 small"
    assert content_type == "application/octet-stream"
    assert server.headers["/small"]["User-Agent"].startswith("Mozilla/5.0")


async def test_refused_status_raises_download_refused(server):
    with pytest.raises(DownloadRefused) as info:
        await FileProcessor().download_file(f"{server.url}/refused")
    assert info.value.status == 403


async def test_oversized_stream_stops_early(server):
    with pytest.raises(ValueError, match="File too large"):
        await FileProcessor().download_file(f"{server.url}/big", max_size_mb=1)
    assert server.disconnected["/big"].wait(5)
    assert server.sent["/big"] < 100 * 1024 * 1024


async def test_trickle_hits_the_overall_deadline(server, monkeypatch):
    monkeypatch.setattr(fp_module, "HTTP_DOWNLOAD_TIMEOUT_S", 0.5)
    with pytest.raises(ValueError, match="did not finish"):
        await FileProcessor().download_file(f"{server.url}/trickle")
    assert server.disconnected["/trickle"].wait(5)


async def test_cancel_closes_the_connection(server):
    first_byte = server.started.setdefault("/trickle", threading.Event())
    task = asyncio.ensure_future(FileProcessor().download_file(f"{server.url}/trickle"))
    assert await asyncio.to_thread(first_byte.wait, 5)  # the transfer is under way
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.disconnected["/trickle"].wait(5)


async def test_follows_as_many_redirects_as_requests(server):
    # requests follows 30 redirects and fails on the 31st
    content, _ = await FileProcessor().download_file(f"{server.url}/redirect/30")
    assert content == b"%PDF-1.4 small"
    with pytest.raises(ValueError, match="more than 30 redirects"):
        await FileProcessor().download_file(f"{server.url}/redirect/31")


async def test_cookie_from_an_ip_host_is_kept_across_redirects(server):
    assert server.url.startswith("http://127.0.0.1")
    content, _ = await FileProcessor().download_file(f"{server.url}/cookie-start")
    assert content == b"%PDF-1.4 cookie"


async def test_read_silence_is_reported_as_such(server, monkeypatch):
    monkeypatch.setattr(fp_module, "HTTP_READ_TIMEOUT_S", 0.5)
    with pytest.raises(ValueError, match="no response or data for"):
        await FileProcessor().download_file(f"{server.url}/stall")


class TestRequestsCompatibleSettings:
    def test_all_proxy_and_cidr_no_proxy(self, monkeypatch):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
            monkeypatch.delenv(name.lower(), raising=False)
        monkeypatch.setenv("ALL_PROXY", "http://proxy.example:3128")
        monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")
        proxy, _ = fp_module._requests_compatible_settings("https://example.com/a.pdf")
        assert proxy == "http://proxy.example:3128"
        proxy, _ = fp_module._requests_compatible_settings("http://10.1.2.3/a.pdf")
        assert proxy is None

    def test_proxy_without_scheme(self, monkeypatch):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
            monkeypatch.delenv(name.lower(), raising=False)
        monkeypatch.setenv("HTTP_PROXY", "proxy.example:3128")
        proxy, _ = fp_module._requests_compatible_settings("http://example.com/a.pdf")
        assert proxy == "http://proxy.example:3128"

    def test_bad_ca_bundle_does_not_affect_plain_http(self, monkeypatch):
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/nonexistent/ca.pem")
        _, context = fp_module._requests_compatible_settings("http://example.com/a.pdf")
        assert context is None

    def test_requests_ca_bundle_is_used(self, monkeypatch):
        import requests.utils
        loaded = []
        real = fp_module.ssl.create_default_context

        def spy(**kwargs):
            loaded.append(kwargs)
            return real()

        monkeypatch.setattr(fp_module.ssl, "create_default_context", spy)
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", requests.utils.DEFAULT_CA_BUNDLE_PATH)
        fp_module._requests_compatible_settings("https://example.com/a.pdf")
        assert loaded == [{"cafile": requests.utils.DEFAULT_CA_BUNDLE_PATH}]


async def test_settings_are_chosen_for_each_redirect_target(server, monkeypatch):
    seen = []
    real = fp_module._requests_compatible_settings

    def spy(url):
        seen.append(urlparse(url).path)
        return real(url)

    monkeypatch.setattr(fp_module, "_requests_compatible_settings", spy)
    await FileProcessor().download_file(f"{server.url}/redirect/2")
    assert seen == ["/redirect/2", "/redirect/1", "/small"]


async def test_url_credentials_are_kept_on_the_same_origin(server):
    url = server.url.replace("http://", "http://user:pass@") + "/auth-start"
    content, _ = await FileProcessor().download_file(url)
    assert content == b"%PDF-1.4 auth"


async def test_final_url_is_returned(server):
    _, _, final_url = await FileProcessor()._download_http(f"{server.url}/redirect/2", 10)
    assert final_url == f"{server.url}/small"
