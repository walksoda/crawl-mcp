"""
File Processing Module for MarkItDown Integration
Handles conversion of PDF, Office, and ZIP files to Markdown format
"""

import asyncio
import io
import os
import tempfile
import zipfile
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
import yarl
import requests.utils
import ssl
from markitdown import MarkItDown, StreamInfo
import base64
from urllib.parse import urljoin, urlparse, unquote
import logging
import re

from ..validators import is_file_uri, is_local_path, file_uri_to_local_path
from .document_download import (
    REFUSED_STATUSES,
    DownloadRefused,
    browser_download,
    find_document_link,
    sniff_extension,
)

# Content-Type to file extension mapping for URLs without extension
CONTENT_TYPE_TO_EXT = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/zip': '.zip',
    'application/x-zip-compressed': '.zip',
    'text/html': '.html',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/csv': '.csv',
    'application/rtf': '.rtf',
    'application/epub+zip': '.epub',
}

# Browser-like headers for file downloads. Some sites reject the default
# User-Agent of HTTP client libraries with 403.
DOWNLOAD_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Upper bound for one HTTP download, so a server that sends a trickle of
# bytes cannot hold a worker thread indefinitely.
HTTP_DOWNLOAD_TIMEOUT_S = 300
# Connect timeout and the longest silence while reading.
HTTP_READ_TIMEOUT_S = 30
# requests' limit; aiohttp's default is 10.
HTTP_MAX_REDIRECTS = 30
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

# Extensions that denote web pages rather than downloadable documents.
# Crawl tools render these in a browser instead of converting them as files.
WEB_PAGE_EXTENSIONS = {'.html', '.htm'}

# Known URL patterns that serve files without standard extensions
_KNOWN_FILE_URL_PATTERNS = [
    # arXiv PDF: https://arxiv.org/pdf/XXXX.XXXXX or XXXX.XXXXXvN
    (re.compile(r'https?://arxiv\.org/pdf/'), '.pdf'),
    # Generic /pdf/ path heuristic (e.g., publisher sites)
    (re.compile(r'https?://[^/]+/.*/pdf/'), '.pdf'),
]


# Text formats that may begin with HTML-looking lines
_TEXT_EXTENSIONS = {'.md', '.txt', '.csv'}
_TEXT_CONTENT_TYPES = {'text/markdown', 'text/plain', 'text/csv', 'text/x-markdown'}

_BLOCKED_SUGGESTION = (
    "The site does not serve this file to automated clients. Look for another copy "
    "(an arXiv or author version), or read the abstract page with crawl_url_with_fallback."
)


def _zip_container_extension(data: bytes) -> str:
    """Tell docx, xlsx, pptx and epub apart from a plain ZIP by their entries."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            if 'word/document.xml' in names:
                return '.docx'
            if 'xl/workbook.xml' in names:
                return '.xlsx'
            if 'ppt/presentation.xml' in names:
                return '.pptx'
            if 'mimetype' in names and archive.read('mimetype').strip() == b'application/epub+zip':
                return '.epub'
    except (zipfile.BadZipFile, KeyError, OSError):
        pass
    return '.zip'


def _requests_compatible_settings(url: str) -> tuple:
    """Proxy and TLS settings read from the environment the way requests does.

    Downloads used requests before; aiohttp's own trust_env misses ALL_PROXY,
    CIDR entries in NO_PROXY and REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE. The CA
    bundle is only loaded for https URLs.
    """
    scheme = urlparse(url).scheme
    proxies = requests.utils.get_environ_proxies(url)
    proxy = proxies.get(scheme) or proxies.get('all')
    if proxy:
        # requests accepts "host:port"; aiohttp needs a scheme
        proxy = requests.utils.prepend_scheme_if_needed(proxy, 'http')
    if scheme != 'https':
        return proxy, None
    ca_bundle = (
        os.environ.get('REQUESTS_CA_BUNDLE')
        or os.environ.get('CURL_CA_BUNDLE')
        or requests.utils.DEFAULT_CA_BUNDLE_PATH
    )
    if os.path.isdir(ca_bundle):
        context = ssl.create_default_context(capath=ca_bundle)
    else:
        context = ssl.create_default_context(cafile=ca_bundle)
    return proxy, context


def _fetch_metadata(metadata: Optional[Dict[str, Any]], fetched: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Add how the file was fetched: the followed document URL and the browser."""
    metadata = dict(metadata or {})
    if fetched['source_url'] != url:
        metadata['document_url'] = fetched['source_url']
    if fetched['via_browser']:
        metadata['downloaded_with'] = 'browser'
    return metadata


def _failure(url: str, error_code: str, error: str, suggestion: str) -> Dict[str, Any]:
    return {
        'success': False,
        'url': url,
        'file_type': None,
        'error': error,
        'error_code': error_code,
        'suggestion': suggestion,
    }


class FileProcessor:
    """Process various file formats using MarkItDown"""

    def __init__(self):
        self.markitdown = MarkItDown()
        self.supported_extensions = {
            # PDF files
            '.pdf': 'PDF Document',
            # Microsoft Office files
            '.docx': 'Microsoft Word Document',
            '.pptx': 'Microsoft PowerPoint Presentation',
            '.xlsx': 'Microsoft Excel Spreadsheet',
            '.xls': 'Microsoft Excel Spreadsheet (Legacy)',
            # Archive files
            '.zip': 'ZIP Archive',
            # Other supported formats
            '.html': 'HTML Document',
            '.htm': 'HTML Document',
            '.txt': 'Text File',
            '.md': 'Markdown File',
            '.csv': 'CSV File',
            '.rtf': 'Rich Text Format',
            '.epub': 'EPUB eBook'
        }

    def _convert(self, path: str, data: bytes):
        """Convert a file with MarkItDown, declaring UTF-8 when the bytes are valid UTF-8.

        MarkItDown guesses the charset from the start of the file, so a text
        file whose first non-ASCII byte comes late is mis-detected as ASCII.
        """
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            result = self.markitdown.convert(path)
        else:
            result = self.markitdown.convert(path, stream_info=StreamInfo(charset='utf-8'))

        if path.lower().endswith('.pdf') and not (result.text_content or '').strip():
            raise ValueError(
                "PDF has no extractable text layer (likely scanned images). "
                "OCR is not supported."
            )
        return result

    def _is_valid_extension(self, ext: str) -> bool:
        """Check if extension looks like a real file extension (not numeric ID like arXiv)"""
        if not ext:
            return False
        # Extension without dot for checking
        ext_part = ext[1:] if ext.startswith('.') else ext
        # Numeric-only extensions (like .21796 from arXiv) are not valid file extensions
        if ext_part.isdigit():
            return False
        return True

    def _match_known_file_url_pattern(self, url: str) -> Optional[str]:
        """Check URL against known file hosting patterns.

        Returns expected extension (e.g., '.pdf') or None if no match.
        """
        for pattern, ext in _KNOWN_FILE_URL_PATTERNS:
            if pattern.search(url):
                return ext
        return None

    def _resolve_local_path(self, file_path_or_url: str) -> Optional[str]:
        if is_file_uri(file_path_or_url):
            return file_uri_to_local_path(file_path_or_url)
        if is_local_path(file_path_or_url):
            return file_path_or_url.strip()
        return None

    async def _read_local_file(self, file_path: str, max_size_mb: int = 100) -> tuple:
        path = Path(file_path).resolve()
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {file_path}")
        resolved_ext = path.suffix.lower()
        if resolved_ext not in self.supported_extensions:
            raise ValueError(
                f"Resolved file has unsupported extension: {resolved_ext}. "
                f"Supported: {', '.join(self.supported_extensions.keys())}"
            )
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")
        content = path.read_bytes()
        return content, None

    def is_supported_file(self, file_path_or_url: str, content_type: Optional[str] = None) -> bool:
        """Check if file format is supported

        HTTP URLs always pass: many documents are served from URLs without
        an extension (``/download``, ``?id=...``, ``.cfm``), so the type is
        decided after download from the content and the Content-Type.
        """
        try:
            local = self._resolve_local_path(file_path_or_url)
            if local is not None:
                ext = Path(local).suffix.lower()
                return ext in self.supported_extensions

            if file_path_or_url.startswith(('http://', 'https://')):
                return True
            else:
                # Local file: check extension
                ext = Path(file_path_or_url).suffix.lower()
                return ext in self.supported_extensions
        except Exception:
            return False

    def is_document_url(self, url: str) -> bool:
        """Check if a remote URL points to a document file rather than a web page.

        Narrower than is_supported_file: web page extensions (.html, .htm) and
        the substring heuristics are excluded so that crawl tools load those
        URLs in a browser.
        """
        try:
            ext = Path(unquote(urlparse(url).path)).suffix.lower()
            if self._is_valid_extension(ext) and ext in self.supported_extensions:
                return ext not in WEB_PAGE_EXTENSIONS
            known_ext = self._match_known_file_url_pattern(url)
            return known_ext is not None and known_ext in self.supported_extensions
        except Exception:
            return False

    def get_file_type(self, file_path_or_url: str) -> Optional[str]:
        """Get human-readable file type description"""
        try:
            local = self._resolve_local_path(file_path_or_url)
            if local is not None:
                path = local
            elif file_path_or_url.startswith('http'):
                parsed = urlparse(file_path_or_url)
                path = unquote(parsed.path)
            else:
                path = file_path_or_url

            ext = Path(path).suffix.lower()
            file_type = self.supported_extensions.get(ext)

            # If no extension found but it might be a known format, try to infer
            if not file_type and file_path_or_url.startswith('http'):
                # Check known URL patterns (e.g., arxiv.org/pdf/)
                known_ext = self._match_known_file_url_pattern(file_path_or_url)
                if known_ext:
                    return self.supported_extensions.get(known_ext)
                # Check for common patterns
                if '/html' in file_path_or_url or 'html' in file_path_or_url:
                    return 'HTML Document'
                elif 'README' in file_path_or_url.upper() and not ext:
                    return 'Text File'

            return file_type
        except Exception:
            return None

    def get_extension_for_url(self, url: str, content_type: Optional[str] = None) -> str:
        """Determine file extension from URL or Content-Type

        Priority:
        1. URL path extension (if present, valid, and supported)
        2. Content-Type header mapping (authoritative when available)
        3. Known URL patterns (heuristic fallback)
        4. Empty string (fallback)
        """
        try:
            local = self._resolve_local_path(url)
            if local is not None:
                ext = Path(local).suffix.lower()
                return ext if ext in self.supported_extensions else ''

            # 1. Try URL extension first
            parsed = urlparse(url)
            path = unquote(parsed.path)
            ext = Path(path).suffix.lower()
            # Only use extension if it's valid (not numeric like arXiv IDs)
            if self._is_valid_extension(ext) and ext in self.supported_extensions:
                return ext

            # 2. Content-Type (authoritative, takes priority over heuristics)
            if content_type:
                mapped = CONTENT_TYPE_TO_EXT.get(content_type, '')
                if mapped:
                    return mapped

            # 3. Known URL patterns (e.g., arxiv.org/pdf/)
            known_ext = self._match_known_file_url_pattern(url)
            if known_ext:
                return known_ext

            return ''
        except Exception:
            return ''

    async def download_file(self, url: str, max_size_mb: int = 100) -> tuple:
        """Download file from URL or read local file with size limit

        Returns:
            tuple: (content_bytes, content_type or None)
        """
        local_path = self._resolve_local_path(url)
        if local_path is not None:
            return await self._read_local_file(local_path, max_size_mb)

        content, content_type, _ = await self._download_http(url, max_size_mb)
        return content, content_type

    async def _download_http(self, url: str, max_size_mb: int) -> tuple:
        """Stream an HTTP download; raises DownloadRefused for 403/429/503.

        Returns ``(content, content_type, final_url)``; the final URL is the
        target of the last redirect. The whole transfer is limited to
        HTTP_DOWNLOAD_TIMEOUT_S, so a server that sends a trickle cannot hold
        the request, and cancelling the caller closes the connection.
        """
        timeout = aiohttp.ClientTimeout(
            total=None, connect=HTTP_READ_TIMEOUT_S, sock_read=HTTP_READ_TIMEOUT_S
        )
        deadline = asyncio.timeout(HTTP_DOWNLOAD_TIMEOUT_S)
        try:
            async with deadline:
                # unsafe=True keeps cookies from IP-address hosts, as requests does
                async with aiohttp.ClientSession(
                    headers=DOWNLOAD_HEADERS, timeout=timeout,
                    cookie_jar=aiohttp.CookieJar(unsafe=True),
                ) as session:
                    # Redirects are followed here, one hop at a time, so the
                    # proxy and TLS settings are chosen for each target as
                    # requests does.
                    # Credentials in the URL (user:pass@host) are sent as Basic
                    # auth, and only to the URL's own origin, as requests does.
                    start = yarl.URL(url)
                    auth = (
                        aiohttp.BasicAuth(start.user or '', start.password or '')
                        if start.user is not None else None
                    )
                    current = start.with_user(None)
                    origin = current.origin()
                    for _ in range(HTTP_MAX_REDIRECTS + 1):
                        proxy, ssl_context = _requests_compatible_settings(str(current))
                        async with session.get(
                            current, proxy=proxy, ssl=ssl_context, allow_redirects=False,
                            auth=auth if current.origin() == origin else None,
                        ) as response:
                            location = response.headers.get('Location')
                            if response.status in _REDIRECT_STATUSES and location:
                                current = yarl.URL(urljoin(str(response.url), location)).with_user(None)
                                continue
                            content, content_type = await self._read_response(response, url, max_size_mb)
                            return content, content_type, str(response.url)
                    raise ValueError(
                        f"Failed to download file: more than {HTTP_MAX_REDIRECTS} redirects"
                    )
        except TimeoutError:
            if deadline.expired():
                raise ValueError(f"Download did not finish within {HTTP_DOWNLOAD_TIMEOUT_S}s")
            raise ValueError(f"Download timed out: no response or data for {HTTP_READ_TIMEOUT_S}s")
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to download file: {type(e).__name__}: {e}")

    async def _read_response(self, response, url: str, max_size_mb: int) -> tuple:
        if response.status in REFUSED_STATUSES:
            raise DownloadRefused(response.status, url)
        if response.status >= 400:
            raise ValueError(f"Failed to download file: HTTP {response.status} {response.reason} for url: {url}")

        content_type_header = response.headers.get('content-type', '')
        content_type = content_type_header.split(';')[0].strip().lower() if content_type_header else None

        max_bytes = max_size_mb * 1024 * 1024
        if response.content_length is not None and response.content_length > max_bytes:
            size_mb = response.content_length / (1024 * 1024)
            raise ValueError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")

        content = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            content += chunk
            if len(content) > max_bytes:
                raise ValueError(f"File too large: exceeds {max_size_mb}MB limit")

        return bytes(content), content_type

    def extract_zip_contents(self, zip_data: bytes) -> Dict[str, Any]:
        """Extract and process contents of ZIP file"""
        extracted_files = []

        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                for file_name in file_list:
                    # Skip directories and hidden files
                    if file_name.endswith('/') or file_name.startswith('.'):
                        continue

                    try:
                        # Check if file is supported
                        if not self.is_supported_file(file_name):
                            extracted_files.append({
                                'name': file_name,
                                'type': 'unsupported',
                                'size': zip_ref.getinfo(file_name).file_size,
                                'content': None,
                                'error': 'Unsupported file format'
                            })
                            continue

                        # Extract file content
                        with zip_ref.open(file_name) as file:
                            file_content = file.read()

                            # Create temporary file for MarkItDown processing
                            with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as temp_file:
                                temp_file.write(file_content)
                                temp_file.flush()

                                try:
                                    # Convert to markdown
                                    result = self._convert(temp_file.name, file_content)
                                    extracted_files.append({
                                        'name': file_name,
                                        'type': self.get_file_type(file_name),
                                        'size': len(file_content),
                                        'content': result.text_content,
                                        'title': getattr(result, 'title', None),
                                        'metadata': getattr(result, 'metadata', {}),
                                        'error': None
                                    })
                                except Exception as e:
                                    extracted_files.append({
                                        'name': file_name,
                                        'type': self.get_file_type(file_name),
                                        'size': len(file_content),
                                        'content': None,
                                        'error': f"Conversion failed: {str(e)}"
                                    })
                                finally:
                                    # Clean up temp file
                                    try:
                                        os.unlink(temp_file.name)
                                    except:
                                        pass

                    except Exception as e:
                        extracted_files.append({
                            'name': file_name,
                            'type': 'error',
                            'size': 0,
                            'content': None,
                            'error': f"Extraction failed: {str(e)}"
                        })

                return {
                    'total_files': len(file_list),
                    'processed_files': len(extracted_files),
                    'files': extracted_files
                }

        except zipfile.BadZipFile:
            raise ValueError("Invalid ZIP file format")
        except Exception as e:
            raise ValueError(f"ZIP processing failed: {str(e)}")

    async def _fetch_remote(self, url: str, max_size_mb: int, _follow_links: bool) -> Dict[str, Any]:
        """Download a remote document, retrying in a browser when refused.

        Returns ``{'data', 'content_type', 'source_url', 'via_browser'}`` on
        success, or a failure dict with ``error_code`` and ``suggestion``.
        """
        refused = None
        final_url = url
        try:
            data, content_type, final_url = await self._download_http(url, max_size_mb)
        except DownloadRefused as e:
            data, content_type = None, None
            refused = f"the server refused the download (HTTP {e.status})"

        # Whether HTML was asked for depends on the URL the caller gave: a
        # document URL that redirects to an .html block page is still refused.
        if data is not None and self._is_web_page(data, content_type, [url, final_url]):
            # A web page came back instead of a file.
            link = find_document_link(final_url, data) if _follow_links else None
            if link:
                # source_url stays the linked file's final URL (after redirects)
                return await self._fetch_remote(link, max_size_mb, _follow_links=False)
            if not (self.is_document_url(url) or self.is_document_url(final_url)):
                return _failure(
                    url, 'not_a_document',
                    f"The URL returned a web page (Content-Type: {content_type}), not a document.",
                    "Use crawl_url for web pages. For a slide deck or paper page, pass the link of its PDF.",
                )
            refused = "the server returned a web page instead of the document (likely a bot check)"

        if refused is None:
            return {'data': data, 'content_type': content_type, 'source_url': final_url, 'via_browser': False}

        browser = await browser_download(url, max_size_mb * 1024 * 1024)
        if browser.data is not None and not self._is_web_page(
            browser.data, None, [url, browser.url, browser.filename]
        ):
            return {'data': browser.data, 'content_type': None,
                    'source_url': browser.url or url, 'via_browser': True,
                    'filename': browser.filename}
        if browser.data is not None:
            # The browser saved a web page (a bot check) under the file's name
            return _failure(
                url, 'download_blocked',
                f"Download blocked: {refused}, and a browser downloaded a web page instead of the file.",
                _BLOCKED_SUGGESTION,
            )
        if browser.too_large:
            return _failure(
                url, 'file_too_large', browser.error,
                "Raise max_size_mb if the file is expected to be this large.",
            )
        # A deck page the HTTP client was refused may still render in a browser.
        link = find_document_link(url, browser.html.encode()) if _follow_links and browser.html else None
        if link:
            # source_url stays the linked file's final URL (after redirects)
            return await self._fetch_remote(link, max_size_mb, _follow_links=False)

        if browser.is_challenge:
            detail = f"a browser got a bot check page instead (\"{browser.title}\")"
        elif browser.error:
            detail = f"a browser download also failed ({browser.error})"
        else:
            detail = f"a browser did not receive the file (HTTP {browser.status}, page \"{browser.title}\")"
        return _failure(url, 'download_blocked', f"Download blocked: {refused}, and {detail}.", _BLOCKED_SUGGESTION)

    def _detect_extension(self, data: bytes, content_type: Optional[str], names: List[Optional[str]]) -> str:
        """Decide the file type of downloaded or local ``data``.

        The content decides first: the PDF and RTF signatures, and for a ZIP
        the files inside it (docx, xlsx, pptx, epub or a plain archive).
        Text formats have no signature, so then the names are tried in
        order (``names``: the final URL, the requested URL, a download file
        name), then the Content-Type, then known URL patterns.
        """
        sniffed = sniff_extension(data)
        if sniffed in ('.pdf', '.rtf'):
            return sniffed
        if sniffed == '.zip':
            return _zip_container_extension(data)

        for name in names:
            ext = self._name_extension(name)
            if ext:
                return ext
        mapped = CONTENT_TYPE_TO_EXT.get(content_type or '', '')
        if mapped:
            return mapped
        for name in names:
            if name and name.startswith(('http://', 'https://')):
                known = self._match_known_file_url_pattern(name)
                if known:
                    return known
        return ''

    def _name_extension(self, name: Optional[str]) -> str:
        """The supported extension of a URL, local path or file name, or ''."""
        if not name:
            return ''
        local = self._resolve_local_path(name)
        path = local if local is not None else unquote(urlparse(name).path) if '://' in name else name
        ext = Path(path).suffix.lower()
        if self._is_valid_extension(ext) and ext in self.supported_extensions:
            return ext
        return ''

    def _is_web_page(self, data: bytes, content_type: Optional[str], names: List[Optional[str]]) -> bool:
        """Whether a download that was not asked for as HTML came back as a web page.

        A file declared as text (by Content-Type, or by a .md/.txt/.csv name)
        may begin with HTML and is not treated as a page.
        """
        if sniff_extension(data) != '.html' or self._url_expects_html(names[0]):
            return False
        if content_type in _TEXT_CONTENT_TYPES:
            return False
        return not any(self._name_extension(name) in _TEXT_EXTENSIONS for name in names)

    def _url_expects_html(self, url: str) -> bool:
        ext = Path(unquote(urlparse(url).path)).suffix.lower()
        return ext in WEB_PAGE_EXTENSIONS

    async def process_file_from_url(self, url: str, max_size_mb: int = 100) -> Dict[str, Any]:
        """Process file from URL"""
        try:
            if self._resolve_local_path(url) is not None:
                file_data, content_type = await self.download_file(url, max_size_mb)
                fetched = {'source_url': url, 'via_browser': False}
            else:
                fetched = await self._fetch_remote(url, max_size_mb, _follow_links=True)
                if fetched.get('data') is None:
                    return fetched
                file_data, content_type = fetched['data'], fetched['content_type']

            ext = self._detect_extension(
                file_data, content_type,
                names=[fetched['source_url'], url, fetched.get('filename')],
            )

            # Validate support after knowing actual type
            if not ext or ext not in self.supported_extensions:
                return {
                    'success': False,
                    'error': f"Unsupported file format. Content-Type: {content_type}. Supported: {', '.join(self.supported_extensions.keys())}",
                    'error_code': 'unsupported_format',
                    'file_type': content_type,
                    'url': url
                }

            file_type = self.supported_extensions.get(ext, 'Unknown')

            # Handle ZIP files specially
            if ext == '.zip':
                zip_contents = self.extract_zip_contents(file_data)
                return {
                    'success': True,
                    'url': url,
                    'file_type': file_type,
                    'size_bytes': len(file_data),
                    'is_archive': True,
                    'content': None,
                    'archive_contents': zip_contents,
                    'metadata': _fetch_metadata({}, fetched, url)
                }

            # Process single file with correct extension
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(file_data)
                temp_file.flush()

                try:
                    result = self._convert(temp_file.name, file_data)
                    metadata = _fetch_metadata(getattr(result, 'metadata', None), fetched, url)
                    return {
                        'success': True,
                        'url': url,
                        'file_type': file_type,
                        'size_bytes': len(file_data),
                        'is_archive': False,
                        'content': result.text_content,
                        'title': getattr(result, 'title', None),
                        'metadata': metadata
                    }
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'file_type': None
            }

    async def process_file_from_data(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Process file from binary data"""
        # Get file type early to avoid reference errors
        file_type = self.get_file_type(filename)

        if not self.is_supported_file(filename):
            return {
                'success': False,
                'error': f"Unsupported file format. Supported: {', '.join(self.supported_extensions.keys())}",
                'file_type': file_type
            }

        try:

            # Handle ZIP files specially
            if filename.lower().endswith('.zip'):
                zip_contents = self.extract_zip_contents(file_data)
                return {
                    'success': True,
                    'filename': filename,
                    'file_type': file_type,
                    'size_bytes': len(file_data),
                    'is_archive': True,
                    'content': None,
                    'archive_contents': zip_contents
                }

            # Process single file
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as temp_file:
                temp_file.write(file_data)
                temp_file.flush()

                try:
                    result = self._convert(temp_file.name, file_data)
                    return {
                        'success': True,
                        'filename': filename,
                        'file_type': file_type,
                        'size_bytes': len(file_data),
                        'is_archive': False,
                        'content': result.text_content,
                        'title': getattr(result, 'title', None),
                        'metadata': getattr(result, 'metadata', {})
                    }
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'file_type': file_type
            }
