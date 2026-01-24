"""
Crawl MCP Tools Test Suite.

Tests for web crawling MCP tools:
- crawl_url
- crawl_url_with_fallback
- deep_crawl_site
- extract_structured_data
"""

import pytest

from tests.mcp.conftest import (
    extract_mcp_content,
    assert_tool_success,
    assert_content_contains,
    assert_content_length,
    assert_content_slicing,
    parse_mcp_json,
)


@pytest.mark.mcp
@pytest.mark.crawl
class TestCrawlUrl:
    """Tests for crawl_url tool."""

    @pytest.mark.asyncio
    async def test_basic_crawl(self, mcp_client, crawl_test_sites):
        """Test basic URL crawling."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {"url": site["url"]}
        )

        assert_tool_success(result)
        assert_content_length(result, site["min_content_length"])

    @pytest.mark.asyncio
    async def test_crawl_with_markdown(self, mcp_client, crawl_test_sites):
        """Test URL crawling with markdown generation."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "generate_markdown": True
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        assert len(content) >= site["min_content_length"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_crawl_wikipedia(self, mcp_client, crawl_test_sites):
        """Test crawling a Wikipedia page."""
        site = crawl_test_sites["wikipedia"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "generate_markdown": True
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 500)

    @pytest.mark.asyncio
    async def test_crawl_with_js_wait(self, mcp_client, crawl_test_sites):
        """Test crawling with JavaScript wait."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "wait_for_js": False
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    async def test_crawl_with_timeout(self, mcp_client, crawl_test_sites):
        """Test crawling with custom timeout."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "timeout": 30
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    async def test_crawl_with_content_limit(self, mcp_client, crawl_test_sites):
        """Test crawling with content limit."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "content_limit": 100,
                "generate_markdown": True
            }
        )

        assert_tool_success(result)
        # Verify slicing was applied correctly
        assert_content_slicing(result, expected_limit=100, expected_offset=0)

    @pytest.mark.asyncio
    async def test_crawl_with_content_slicing(self, mcp_client, crawl_test_sites):
        """Test crawling with content offset and limit."""
        site = crawl_test_sites["simple_html"]

        # First get full content for comparison
        full_result = await mcp_client.call_tool(
            "crawl_url",
            {"url": site["url"], "generate_markdown": True}
        )
        full_data = parse_mcp_json(full_result)
        full_markdown = full_data.get("markdown", "")

        # Get sliced content with offset=50, limit=100
        sliced_result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "content_offset": 50,
                "content_limit": 100,
                "generate_markdown": True
            }
        )

        assert_tool_success(sliced_result)
        assert_content_slicing(sliced_result, expected_limit=100, expected_offset=50)

        # Verify actual content is sliced correctly
        sliced_data = parse_mcp_json(sliced_result)
        sliced_markdown = sliced_data.get("markdown", "")

        # Sliced content should match substring of full content
        if len(full_markdown) > 50:
            expected_slice = full_markdown[50:150]
            assert sliced_markdown == expected_slice, \
                f"Sliced content mismatch: got '{sliced_markdown[:50]}...', expected '{expected_slice[:50]}...'"

    @pytest.mark.asyncio
    async def test_crawl_with_pagination(self, mcp_client, crawl_test_sites):
        """Test pagination by fetching content in chunks with offset."""
        site = crawl_test_sites["simple_html"]

        # Get first chunk (0-100)
        first_chunk = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "content_offset": 0,
                "content_limit": 100,
                "generate_markdown": True
            }
        )
        first_data = parse_mcp_json(first_chunk)
        first_markdown = first_data.get("markdown", "")

        # Get second chunk (100-200) - continuation
        second_chunk = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "content_offset": 100,
                "content_limit": 100,
                "generate_markdown": True
            }
        )
        second_data = parse_mcp_json(second_chunk)
        second_markdown = second_data.get("markdown", "")

        # Verify chunks are different (no overlap)
        assert first_markdown != second_markdown, \
            "First and second chunks should be different"

        # Verify continuation works - second chunk should not start with same content
        if first_markdown and second_markdown:
            assert not second_markdown.startswith(first_markdown[:20]), \
                "Second chunk should start from offset, not beginning"

        # Verify slicing info shows correct offsets
        assert first_data.get("slicing_info", {}).get("markdown", {}).get("offset") == 0
        assert second_data.get("slicing_info", {}).get("markdown", {}).get("offset") == 100

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_crawl_with_undetected_browser(self, mcp_client, crawl_test_sites):
        """Test crawling with undetected browser mode."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url",
            {
                "url": site["url"],
                "use_undetected_browser": True,
                "timeout": 60
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)


@pytest.mark.mcp
@pytest.mark.crawl
class TestCrawlUrlWithFallback:
    """Tests for crawl_url_with_fallback tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_fallback_crawl(self, mcp_client, crawl_test_sites):
        """Test crawl with fallback strategies."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url_with_fallback",
            {"url": site["url"]}
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    async def test_fallback_with_options(self, mcp_client, crawl_test_sites):
        """Test fallback crawl with custom options."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url_with_fallback",
            {
                "url": site["url"],
                "generate_markdown": True,
                "timeout": 60
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    async def test_fallback_with_content_limit(self, mcp_client, crawl_test_sites):
        """Test fallback crawl with content limit parameter acceptance."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url_with_fallback",
            {
                "url": site["url"],
                "content_limit": 100,
                "generate_markdown": True
            }
        )

        # Verify slicing_info is present in response (parameter was accepted)
        data = parse_mcp_json(result)
        assert "slicing_info" in data, "slicing_info should be present in response"
        slicing_info = data["slicing_info"]
        assert slicing_info.get("markdown", {}).get("limit") == 100

    @pytest.mark.asyncio
    async def test_fallback_with_content_slicing(self, mcp_client, crawl_test_sites):
        """Test fallback crawl with content offset and limit parameter acceptance."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "crawl_url_with_fallback",
            {
                "url": site["url"],
                "content_offset": 50,
                "content_limit": 100,
                "generate_markdown": True
            }
        )

        # Verify slicing_info is present in response (parameters were accepted)
        data = parse_mcp_json(result)
        assert "slicing_info" in data, "slicing_info should be present in response"
        slicing_info = data["slicing_info"]
        assert slicing_info.get("markdown", {}).get("limit") == 100
        assert slicing_info.get("markdown", {}).get("offset") == 50

    @pytest.mark.asyncio
    async def test_fallback_with_pagination(self, mcp_client, crawl_test_sites):
        """Test pagination with fallback crawl by fetching content in chunks."""
        site = crawl_test_sites["simple_html"]

        # Get first chunk (0-100)
        first_chunk = await mcp_client.call_tool(
            "crawl_url_with_fallback",
            {
                "url": site["url"],
                "content_offset": 0,
                "content_limit": 100,
                "generate_markdown": True
            }
        )
        first_data = parse_mcp_json(first_chunk)

        # Get second chunk (100-200) - continuation
        second_chunk = await mcp_client.call_tool(
            "crawl_url_with_fallback",
            {
                "url": site["url"],
                "content_offset": 100,
                "content_limit": 100,
                "generate_markdown": True
            }
        )
        second_data = parse_mcp_json(second_chunk)

        # Verify slicing info shows correct offsets
        assert first_data.get("slicing_info", {}).get("markdown", {}).get("offset") == 0
        assert second_data.get("slicing_info", {}).get("markdown", {}).get("offset") == 100

        # If both succeeded with content, verify they are different
        first_md = first_data.get("markdown", "")
        second_md = second_data.get("markdown", "")
        if first_md and second_md:
            assert first_md != second_md, "Chunks should be different"


@pytest.mark.mcp
@pytest.mark.crawl
class TestDeepCrawlSite:
    """Tests for deep_crawl_site tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_deep_crawl_basic(self, mcp_client, crawl_test_sites):
        """Test basic deep crawl."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "deep_crawl_site",
            {
                "url": site["url"],
                "max_pages": 2,
                "max_depth": 1
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_deep_crawl_with_strategy(self, mcp_client, crawl_test_sites):
        """Test deep crawl with BFS strategy."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "deep_crawl_site",
            {
                "url": site["url"],
                "max_pages": 2,
                "max_depth": 1,
                "crawl_strategy": "bfs"
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        assert len(content) > 0


@pytest.mark.mcp
@pytest.mark.crawl
class TestExtractStructuredData:
    """Tests for extract_structured_data tool."""

    @pytest.mark.asyncio
    async def test_css_extraction(self, mcp_client, crawl_test_sites):
        """Test CSS selector-based extraction."""
        site = crawl_test_sites["simple_html"]

        result = await mcp_client.call_tool(
            "extract_structured_data",
            {
                "url": site["url"],
                "extraction_type": "css"
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        assert len(content) > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_table_extraction(self, mcp_client):
        """Test table extraction."""
        # Use a page known to have tables
        result = await mcp_client.call_tool(
            "extract_structured_data",
            {
                "url": "https://en.wikipedia.org/wiki/List_of_programming_languages",
                "extraction_type": "table",
                "timeout": 60
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        assert len(content) > 0
