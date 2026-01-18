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
