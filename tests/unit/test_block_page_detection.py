"""Tests for block/challenge page detection used by the fallback crawler."""

from crawl4ai_mcp.infra.content_processors import is_block_page, is_block_result


ARTICLE_TEXT = "This article explains how robot arms are calibrated. " * 60

NORMAL_HTML = f"""
<html><head>
<meta name="robots" content="index,follow">
<script src="https://cdnjs.cloudflare.com/ajax/libs/x.js"></script>
<script>var captcha = "blocked"; var rateLimit = "rate limit";</script>
</head><body><article><p>{ARTICLE_TEXT}</p></article></body></html>
"""

CLOUDFLARE_CHALLENGE_HTML = """
<html><head><title>Just a moment...</title>
<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/jsch/v1"></script></head>
<body><noscript>Enable JavaScript and cookies to continue</noscript></body></html>
"""

SECURITY_CHECK_HTML = """
<html><head><title>Security check</title></head><body>
<h1>Verifying your access, please wait...</h1>
<p>Please confirm to continue.</p><label>I'm not a robot</label>
</body></html>
"""


class TestIsBlockPage:
    def test_markup_only_indicators_are_ignored(self):
        # "robots" meta, cloudflare CDN URL and words inside <script> are not visible text
        assert is_block_page(NORMAL_HTML) is False

    def test_long_article_mentioning_indicator_is_not_block(self):
        text = ARTICLE_TEXT + " Access denied errors are covered in the next section."
        assert is_block_page(text) is False

    def test_cloudflare_challenge_detected(self):
        assert is_block_page(CLOUDFLARE_CHALLENGE_HTML) is True

    def test_security_check_page_detected(self):
        assert is_block_page(SECURITY_CHECK_HTML) is True

    def test_short_markdown_block_page_detected(self):
        assert is_block_page("# Access Denied\n\nYou don't have permission.") is True

    def test_generic_words_alone_are_not_block(self):
        assert is_block_page("A short note about a robot hosted on Cloudflare.") is False

    def test_empty_content(self):
        assert is_block_page("") is False
        assert is_block_page(None) is False


class TestIsBlockResult:
    def test_uses_markdown_before_raw_html(self):
        result = {"markdown": ARTICLE_TEXT, "raw_content": CLOUDFLARE_CHALLENGE_HTML}
        assert is_block_result(result) is False

    def test_falls_back_to_raw_html(self):
        result = {"markdown": "", "content": None, "raw_content": CLOUDFLARE_CHALLENGE_HTML}
        assert is_block_result(result) is True

    def test_object_attributes(self):
        class R:
            markdown = "Just a moment... Enable JavaScript and cookies to continue"
            content = None
        assert is_block_result(R()) is True
