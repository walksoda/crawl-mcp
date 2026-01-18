"""
Search MCP Tools Test Suite.

Tests for search-related MCP tools:
- search_google
- batch_search_google
- search_and_crawl
"""

import pytest

from tests.mcp.conftest import (
    extract_mcp_content,
    assert_tool_success,
    assert_content_contains,
    assert_content_length,
)


@pytest.mark.mcp
@pytest.mark.search
class TestSearchGoogle:
    """Tests for search_google tool."""

    @pytest.mark.asyncio
    async def test_basic_search(self, mcp_client, search_test_queries):
        """Test basic Google search."""
        query_info = search_test_queries["technical"]

        result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3
                }
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        # Should return search results
        assert len(content) > 50

    @pytest.mark.asyncio
    async def test_search_with_genre(self, mcp_client, search_test_queries):
        """Test search with specific genre."""
        query_info = search_test_queries["technical"]

        result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 2,
                    "search_genre": "technical"
                }
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 30)

    @pytest.mark.asyncio
    async def test_search_with_language(self, mcp_client, search_test_queries):
        """Test search with language filter."""
        query_info = search_test_queries["technical"]

        result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 2,
                    "language": "en"
                }
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 30)

    @pytest.mark.asyncio
    async def test_search_news(self, mcp_client, search_test_queries):
        """Test news search."""
        query_info = search_test_queries["news"]

        result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 2,
                    "search_genre": "news"
                }
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 30)


@pytest.mark.mcp
@pytest.mark.search
class TestBatchSearchGoogle:
    """Tests for batch_search_google tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_search_multiple_queries(self, mcp_client, search_test_queries):
        """Test batch search with multiple queries."""
        queries = [
            search_test_queries["technical"]["query"],
            search_test_queries["news"]["query"],
        ]

        result = await mcp_client.call_tool(
            "batch_search_google",
            {
                "request": {
                    "queries": queries,
                    "num_results_per_query": 2
                }
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        # Should have results for multiple queries
        assert len(content) > 100

    @pytest.mark.asyncio
    async def test_batch_search_single_query(self, mcp_client, search_test_queries):
        """Test batch search with single query."""
        queries = [search_test_queries["technical"]["query"]]

        result = await mcp_client.call_tool(
            "batch_search_google",
            {
                "request": {
                    "queries": queries,
                    "num_results_per_query": 2
                }
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 30)


@pytest.mark.mcp
@pytest.mark.search
class TestSearchAndCrawl:
    """Tests for search_and_crawl tool."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_search_and_crawl_basic(self, mcp_client, search_test_queries):
        """Test search and crawl combined operation."""
        query_info = search_test_queries["technical"]

        result = await mcp_client.call_tool(
            "search_and_crawl",
            {
                "request": {
                    "search_query": query_info["query"],
                    "crawl_top_results": 1
                }
            }
        )

        assert_tool_success(result)
        content = extract_mcp_content(result)
        # Should have crawled content from search results
        assert len(content) > 100

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_search_and_crawl_with_genre(self, mcp_client, search_test_queries):
        """Test search and crawl with search genre."""
        query_info = search_test_queries["academic"]

        result = await mcp_client.call_tool(
            "search_and_crawl",
            {
                "request": {
                    "search_query": query_info["query"],
                    "crawl_top_results": 1,
                    "search_genre": "academic"
                }
            }
        )

        assert_tool_success(result)
        assert_content_length(result, 50)
