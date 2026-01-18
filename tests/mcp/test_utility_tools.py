"""
Utility MCP Tools Test Suite.

Tests for utility MCP tools:
- batch_crawl
- multi_url_crawl
"""

import pytest

from tests.mcp.conftest import (
    extract_mcp_content,
    assert_tool_success,
    assert_content_contains,
    assert_content_length,
)


@pytest.mark.mcp
@pytest.mark.utility
class TestBatchCrawl:
    """Tests for batch_crawl tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_crawl_multiple_urls(self, mcp_client, crawl_test_sites):
        """Test batch crawling multiple URLs."""
        urls = [
            crawl_test_sites["simple_html"]["url"],
            "https://httpbin.org/html",
        ]

        result = await mcp_client.call_tool(
            "batch_crawl",
            {
                "urls": urls,
                "generate_markdown": True
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        # Should have content from multiple URLs
        assert len(content) > 100

    @pytest.mark.asyncio
    async def test_batch_crawl_single_url(self, mcp_client, crawl_test_sites):
        """Test batch crawl with single URL."""
        urls = [crawl_test_sites["simple_html"]["url"]]

        result = await mcp_client.call_tool(
            "batch_crawl",
            {
                "urls": urls,
                "generate_markdown": True
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_crawl_with_options(self, mcp_client, crawl_test_sites):
        """Test batch crawl with custom options."""
        urls = [
            crawl_test_sites["simple_html"]["url"],
        ]

        result = await mcp_client.call_tool(
            "batch_crawl",
            {
                "urls": urls,
                "generate_markdown": True,
                "base_timeout": 30,
                "max_concurrent": 2
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)


@pytest.mark.mcp
@pytest.mark.utility
class TestMultiUrlCrawl:
    """Tests for multi_url_crawl tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multi_url_crawl_basic(self, mcp_client, crawl_test_sites):
        """Test multi-URL crawl with pattern-based config."""
        url_configs = {
            crawl_test_sites["simple_html"]["url"]: {
                "generate_markdown": True
            }
        }

        result = await mcp_client.call_tool(
            "multi_url_crawl",
            {
                "url_configurations": url_configs
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multi_url_crawl_with_defaults(self, mcp_client, crawl_test_sites):
        """Test multi-URL crawl with default config."""
        url_configs = {
            crawl_test_sites["simple_html"]["url"]: {}
        }

        result = await mcp_client.call_tool(
            "multi_url_crawl",
            {
                "url_configurations": url_configs,
                "default_config": {
                    "generate_markdown": True,
                    "wait_for_js": False
                }
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)
