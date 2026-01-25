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
    parse_mcp_json,
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

    @pytest.mark.asyncio
    async def test_search_with_content_limit(self, mcp_client, search_test_queries):
        """Test search with content limit."""
        query_info = search_test_queries["technical"]

        result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3,
                    "content_limit": 200
                }
            }
        )

        assert_tool_success(result)
        # Verify slicing_info is present in response
        data = parse_mcp_json(result)
        assert "slicing_info" in data, "slicing_info should be present in response"
        slicing_info = data["slicing_info"]
        content_info = slicing_info.get("content", {})
        assert content_info.get("limit") == 200
        # Verify content field exists
        assert "content" in data, "content field should be present"

    @pytest.mark.asyncio
    async def test_search_with_content_slicing(self, mcp_client, search_test_queries):
        """Test search with content offset and limit."""
        query_info = search_test_queries["technical"]

        result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3,
                    "content_offset": 50,
                    "content_limit": 100
                }
            }
        )

        assert_tool_success(result)
        # Verify slicing_info is present in response
        data = parse_mcp_json(result)
        assert "slicing_info" in data, "slicing_info should be present in response"
        slicing_info = data["slicing_info"]
        content_info = slicing_info.get("content", {})
        assert content_info.get("limit") == 100
        assert content_info.get("offset") == 50
        assert content_info.get("source") == "combined_search_snippets"

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, mcp_client, search_test_queries):
        """Test pagination by fetching search content in chunks."""
        query_info = search_test_queries["technical"]

        # Get first chunk (0-100)
        first_chunk = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3,
                    "content_offset": 0,
                    "content_limit": 100
                }
            }
        )
        first_data = parse_mcp_json(first_chunk)

        # Get second chunk (100-200) - continuation
        second_chunk = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3,
                    "content_offset": 100,
                    "content_limit": 100
                }
            }
        )
        second_data = parse_mcp_json(second_chunk)

        # Verify slicing info shows correct offsets
        assert first_data.get("slicing_info", {}).get("content", {}).get("offset") == 0
        assert second_data.get("slicing_info", {}).get("content", {}).get("offset") == 100

        # If both succeeded with content, verify they are different
        first_content = first_data.get("content", "")
        second_content = second_data.get("content", "")
        if first_content and second_content:
            assert first_content != second_content, "Chunks should be different"

    @pytest.mark.asyncio
    async def test_search_cache_hit(self, mcp_client, search_test_queries):
        """Test cache hit for repeated search with slicing."""
        query_info = search_test_queries["technical"]

        # Clear cache before test
        from crawl4ai_mcp.server_helpers import clear_search_cache
        clear_search_cache()

        # First request - should be cache miss
        first_result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3,
                    "content_limit": 100
                }
            }
        )
        first_data = parse_mcp_json(first_result)
        assert first_data.get("cache_hit") is False, "First request should be cache miss"

        # Skip cache hit test if first search failed (no results to cache)
        if not first_data.get("success"):
            pytest.skip("First search failed, cannot test cache hit")

        # Second request with different slicing - should be cache hit
        second_result = await mcp_client.call_tool(
            "search_google",
            {
                "request": {
                    "query": query_info["query"],
                    "num_results": 3,
                    "content_offset": 50,
                    "content_limit": 100
                }
            }
        )
        second_data = parse_mcp_json(second_result)
        assert second_data.get("cache_hit") is True, "Second request should be cache hit"


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
