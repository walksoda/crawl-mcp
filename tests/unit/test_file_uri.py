"""Unit tests for local file path handling (file:// URIs and absolute paths)."""

import pytest
from pathlib import Path

from crawl4ai_mcp.validators import (
    is_file_uri,
    is_local_path,
    file_uri_to_local_path,
    validate_url,
)
from crawl4ai_mcp.processors.file_processor import FileProcessor


class TestFileUriToLocalPath:
    def test_unix_path(self):
        assert file_uri_to_local_path("file:///home/user/doc.pdf") == "/home/user/doc.pdf"

    def test_windows_drive_letter(self):
        assert file_uri_to_local_path("file:///C:/Users/doc.pdf") == "C:/Users/doc.pdf"

    def test_url_encoded_spaces(self):
        assert file_uri_to_local_path("file:///path/my%20file.pdf") == "/path/my file.pdf"

    def test_windows_encoded(self):
        assert file_uri_to_local_path("file:///C:/My%20Docs/file.pdf") == "C:/My Docs/file.pdf"

    def test_localhost_authority(self):
        assert file_uri_to_local_path("file://localhost/C:/Users/doc.pdf") == "C:/Users/doc.pdf"

    def test_unc_authority_rejected(self):
        with pytest.raises(ValueError, match="UNC file URIs are not supported"):
            file_uri_to_local_path("file://server/share/a.pdf")

    def test_unc_in_path_rejected(self):
        with pytest.raises(ValueError, match="UNC file URIs are not supported"):
            file_uri_to_local_path("file:////server/share/a.pdf")

    def test_unc_localhost_path_rejected(self):
        with pytest.raises(ValueError, match="UNC file URIs are not supported"):
            file_uri_to_local_path("file://localhost//server/share/a.pdf")

    def test_empty_file_uri_rejected(self):
        with pytest.raises(ValueError, match="Empty file path"):
            file_uri_to_local_path("file://")

    def test_root_only_file_uri_rejected(self):
        with pytest.raises(ValueError, match="Empty file path"):
            file_uri_to_local_path("file:///")


class TestIsLocalPath:
    def test_unix_absolute(self):
        assert is_local_path("/home/user/doc.pdf") is True

    def test_windows_backslash(self):
        assert is_local_path("C:\\Users\\doc.pdf") is True

    def test_windows_forward_slash(self):
        assert is_local_path("C:/Users/doc.pdf") is True

    def test_http_url(self):
        assert is_local_path("http://example.com") is False

    def test_relative_path(self):
        assert is_local_path("./doc.pdf") is False

    def test_bare_filename(self):
        assert is_local_path("doc.pdf") is False

    def test_unc_backslash_rejected(self):
        assert is_local_path("\\\\server\\share\\file.pdf") is False

    def test_unc_slash_rejected(self):
        assert is_local_path("//server/share/file.pdf") is False


class TestValidateUrlLocalFiles:
    def test_file_uri_accepted(self):
        assert validate_url("file:///home/user/doc.pdf") is None

    def test_absolute_unix_path_accepted(self):
        assert validate_url("/home/user/doc.pdf") is None

    def test_absolute_windows_path_accepted(self):
        assert validate_url("C:/Users/doc.pdf") is None

    def test_http_unchanged(self):
        assert validate_url("http://example.com") is None

    def test_https_unchanged(self):
        assert validate_url("https://example.com") is None

    def test_ftp_rejected(self):
        result = validate_url("ftp://example.com")
        assert result is not None
        assert result["error_code"] == "invalid_url_scheme"

    def test_relative_path_rejected(self):
        result = validate_url("./doc.pdf")
        assert result is not None

    def test_unc_file_uri_rejected(self):
        result = validate_url("file://server/share/a.pdf")
        assert result is not None
        assert result["error_code"] == "unsupported_file_uri"

    def test_unc_in_path_rejected(self):
        result = validate_url("file:////server/share/a.pdf")
        assert result is not None
        assert result["error_code"] == "unsupported_file_uri"

    def test_empty_file_uri_rejected(self):
        result = validate_url("file://")
        assert result is not None
        assert result["error_code"] == "unsupported_file_uri"

    def test_root_file_uri_rejected(self):
        result = validate_url("file:///")
        assert result is not None
        assert result["error_code"] == "unsupported_file_uri"

    def test_empty_rejected(self):
        result = validate_url("")
        assert result is not None


class TestIsSupportedFileLocal:
    def setup_method(self):
        self.fp = FileProcessor()

    def test_file_uri_pdf(self):
        assert self.fp.is_supported_file("file:///home/user/doc.pdf") is True

    def test_absolute_path_pdf(self):
        assert self.fp.is_supported_file("/home/user/doc.pdf") is True

    def test_file_uri_exe_rejected(self):
        assert self.fp.is_supported_file("file:///home/user/doc.exe") is False

    def test_absolute_path_no_extension(self):
        assert self.fp.is_supported_file("/etc/passwd") is False

    def test_http_url_unchanged(self):
        assert self.fp.is_supported_file("https://example.com/doc.pdf") is True

    def test_http_url_no_ext_unchanged(self):
        assert self.fp.is_supported_file("https://example.com/page") is False


class TestReadLocalFile:
    @pytest.mark.asyncio
    async def test_read_via_file_uri(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        fp = FileProcessor()
        content, ct = await fp.download_file(f"file://{f}", max_size_mb=1)
        assert content == b"hello world"
        assert ct is None

    @pytest.mark.asyncio
    async def test_read_via_absolute_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        fp = FileProcessor()
        content, ct = await fp.download_file(str(f), max_size_mb=1)
        assert content == b"hello world"
        assert ct is None

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        fp = FileProcessor()
        with pytest.raises(ValueError, match="File not found"):
            await fp.download_file("file:///nonexistent/path/doc.pdf")

    @pytest.mark.asyncio
    async def test_directory_rejected(self, tmp_path):
        fp = FileProcessor()
        with pytest.raises(ValueError, match="Not a regular file"):
            await fp.download_file(f"file://{tmp_path}")

    @pytest.mark.asyncio
    async def test_size_limit(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (2 * 1024 * 1024))
        fp = FileProcessor()
        with pytest.raises(ValueError, match="File too large"):
            await fp.download_file(f"file://{f}", max_size_mb=1)


class TestSymlinkSecurity:
    @pytest.mark.asyncio
    async def test_symlink_to_unsupported_file_rejected(self, tmp_path):
        target = tmp_path / "secret.txt.bak"
        target.write_text("sensitive data")
        link = tmp_path / "report.pdf"
        link.symlink_to(target)
        fp = FileProcessor()
        with pytest.raises(ValueError, match="unsupported extension"):
            await fp._read_local_file(str(link))


class TestHttpRegression:
    """Ensure HTTP URL handling is not broken by local file support."""

    def setup_method(self):
        self.fp = FileProcessor()

    def test_is_supported_file_http_pdf(self):
        assert self.fp.is_supported_file("https://example.com/doc.pdf") is True

    def test_is_supported_file_http_no_ext(self):
        assert self.fp.is_supported_file("https://example.com/page") is False

    def test_get_extension_for_url_http(self):
        assert self.fp.get_extension_for_url("https://example.com/doc.pdf") == ".pdf"

    def test_get_extension_for_url_http_no_ext(self):
        assert self.fp.get_extension_for_url("https://example.com/page") == ""

    def test_get_file_type_http(self):
        assert self.fp.get_file_type("https://example.com/doc.pdf") == "PDF Document"
