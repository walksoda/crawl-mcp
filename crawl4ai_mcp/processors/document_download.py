"""
Remote document download helpers for FileProcessor.

- Content sniffing: decide the real file type from the first bytes, since
  servers often send PDFs as application/octet-stream or send an HTML block
  page with a .pdf URL.
- Browser download: some sites reject plain HTTP clients (TLS fingerprint,
  JavaScript checks) but serve the file to a browser. Playwright, which
  crawl4ai already depends on, downloads the file as Chromium would.
  crawl4ai's own accept_downloads path cannot be used: it only tolerates
  net::ERR_ABORTED, while Chromium reports "Download is starting" when a
  navigation turns into a download.
- Slide sites: a deck page links its PDF; find that link.

Sites that answer with an interactive human check (Cloudflare Turnstile and
the like) are reported as blocked, not worked around.
"""

import asyncio
import html as html_lib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

# Statuses that mean "this client was refused", as opposed to "not found".
REFUSED_STATUSES = {403, 429, 503}

BROWSER_DOWNLOAD_TIMEOUT_S = 45
# How long to wait for a download after the page itself has loaded. A
# JavaScript check that passes without interaction redirects within this time.
BROWSER_DOWNLOAD_GRACE_S = 8
# Upper bound for receiving the file once the download has started.
BROWSER_TRANSFER_TIMEOUT_S = 120

_CHALLENGE_TITLES = (
    "just a moment",
    "verifying",
    "attention required",
    "security check",
    "access denied",
    "content blocked",
    "request rate threshold exceeded",
)


class DownloadRefused(Exception):
    """The server refused the HTTP client (403, 429 or 503)."""

    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} from {url}")
        self.status = status
        self.url = url


def sniff_extension(data: bytes) -> Optional[str]:
    """Return the file type implied by the first bytes, or None if unknown.

    ``.zip`` stands for every ZIP container (docx, xlsx, pptx, epub, zip).
    """
    head = data[:4096]
    if head.startswith(b"%PDF-"):
        return ".pdf"
    # local file header, empty archive, spanned archive
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return ".zip"
    if head.startswith(b"{\\rtf"):
        return ".rtf"
    # Only the start counts (a Markdown or text file may quote HTML later
    # on), after a BOM, whitespace, comments and an XML declaration.
    text = head.lower()
    while True:
        text = text.lstrip(b"\xef\xbb\xbf \t\r\n")
        if text.startswith(b"<!--") and b"-->" in text:
            text = text[text.index(b"-->") + 3:]
        elif text.startswith(b"<?xml") and b"?>" in text:
            text = text[text.index(b"?>") + 2:]
        else:
            break
    if text.startswith((b"<!doctype html", b"<html", b"<head")):
        return ".html"
    return None


@dataclass
class BrowserDownload:
    data: Optional[bytes] = None
    # The file name the browser proposed and the URL it downloaded from
    # (after redirects), used to tell the file type
    filename: Optional[str] = None
    url: Optional[str] = None
    status: Optional[int] = None
    title: Optional[str] = None
    # The rendered page when no download started (for slide deck links).
    html: Optional[str] = None
    error: Optional[str] = None
    too_large: bool = False

    @property
    def is_challenge(self) -> bool:
        title = (self.title or "").lower()
        return any(marker in title for marker in _CHALLENGE_TITLES)


async def browser_download(url: str, max_bytes: int) -> BrowserDownload:
    """Open ``url`` in headless Chromium and return the file it downloads.

    The partial file is watched while it arrives, so a file over
    ``max_bytes`` or a transfer that stalls is cancelled. When no download
    starts, the result carries the page's status, title and HTML so the
    caller can explain why (for example a human check) or follow a link.
    """
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    result = BrowserDownload()
    download_dir = tempfile.mkdtemp(prefix="crawl4ai_mcp_dl_")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, downloads_path=download_dir)
            try:
                context = await browser.new_context(accept_downloads=True)
                page = await context.new_page()
                downloads = []
                started = asyncio.Event()

                # Playwright tags the handler with an attribute, so it must be
                # a Python function, not a builtin such as list.append.
                def on_download(download):
                    downloads.append(download)
                    started.set()

                page.on("download", on_download)
                # The download can start while the page is still loading (a
                # script on a slow page), so watch for it alongside goto and
                # start the size checks as soon as it begins.
                goto = asyncio.ensure_future(
                    page.goto(url, timeout=BROWSER_DOWNLOAD_TIMEOUT_S * 1000)
                )
                waiter = asyncio.ensure_future(started.wait())
                try:
                    await asyncio.wait({goto, waiter}, return_when=asyncio.FIRST_COMPLETED)
                    if not started.is_set():
                        try:
                            response = goto.result()
                            result.status = response.status if response else None
                        except PlaywrightError as e:
                            if "Download is starting" not in str(e) and "ERR_ABORTED" not in str(e):
                                raise
                        try:
                            await asyncio.wait_for(started.wait(), BROWSER_DOWNLOAD_GRACE_S)
                        except asyncio.TimeoutError:
                            pass

                    if downloads:
                        await _receive(downloads[0], download_dir, max_bytes, result)
                    else:
                        try:
                            result.title = await page.title()
                            result.html = await page.content()
                        except PlaywrightError:
                            pass
                finally:
                    await _settle(goto, waiter)
            finally:
                await browser.close()
    except Exception as e:  # browser missing, launch failure, timeout
        result.error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)
    return result


async def _receive(download, download_dir: str, max_bytes: int, result: BrowserDownload) -> None:
    """Wait for ``download`` to finish, cancelling it when too large or stalled."""
    finished = asyncio.ensure_future(download.path())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BROWSER_TRANSFER_TIMEOUT_S
    try:
        while True:
            if _dir_size(download_dir) > max_bytes:
                result.too_large = True
                result.error = f"File too large: exceeds {max_bytes // (1024 * 1024)}MB limit"
                await download.cancel()
                return
            if loop.time() > deadline:
                result.error = f"Download did not finish within {BROWSER_TRANSFER_TIMEOUT_S}s"
                await download.cancel()
                return
            if finished.done():
                break
            await asyncio.wait({finished}, timeout=0.2)
        path = finished.result()
        if path is None or os.path.getsize(path) > max_bytes:
            result.too_large = path is not None
            result.error = await download.failure() or (
                f"File too large: exceeds {max_bytes // (1024 * 1024)}MB limit"
            )
            return
        with open(path, "rb") as f:
            result.data = f.read()
        result.filename = download.suggested_filename
        result.url = download.url
    finally:
        await _settle(finished)


async def _settle(*tasks: asyncio.Future) -> None:
    """Cancel unfinished child tasks and wait for them to end.

    The children's own results and exceptions are discarded, but a
    cancellation of the caller still propagates out of the await.
    """
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _dir_size(path: str) -> int:
    total = 0
    for entry in os.scandir(path):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except FileNotFoundError:
            # Chromium renamed or removed the partial file mid-scan
            continue
    return total


_SPEAKERDECK_PDF = re.compile(
    r"https://files\.speakerdeck\.com/presentations/[0-9a-f]+/[^\"'\s<>]+?\.pdf(?:\?[^\"'\s<>]*)?"
)
_DOCSWELL_DOWNLOAD = re.compile(r"(?:https://www\.docswell\.com)?/slide/[0-9A-Z]+/[0-9A-Z]+/download")
# The "View PDF" link of an arXiv abstract page (/abs/<id>)
_ARXIV_PDF = re.compile(r'href="(/pdf/[^"]+)"')


def find_document_link(page_url: str, page_html: bytes) -> Optional[str]:
    """Return the PDF link of a slide deck or abstract page.

    Covers Speaker Deck and Docswell deck pages and arXiv abstract pages.
    """
    host = (urlparse(page_url).hostname or "").lower()
    text = page_html.decode("utf-8", errors="replace")
    if _on_domain(host, "speakerdeck.com"):
        match = _SPEAKERDECK_PDF.search(text)
    elif _on_domain(host, "docswell.com"):
        match = _DOCSWELL_DOWNLOAD.search(text)
    elif _on_domain(host, "arxiv.org"):
        match = _ARXIV_PDF.search(text)
    else:
        return None
    if not match:
        return None
    return urljoin(page_url, html_lib.unescape(match.group(match.lastindex or 0)))


def _on_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)
