"""Every tool sets its display title explicitly.

FastMCP derives a title from the function name when none is given
("Crawl Url", "Extract Youtube Transcript"), so each tool passes title=.
The title only spells out the name, so it must map back to it.
"""

import asyncio

from crawl4ai_mcp.server import mcp


def _tools():
    return asyncio.run(mcp.list_tools())


def test_every_tool_has_an_explicit_title():
    missing = [tool.name for tool in _tools() if not tool.title]
    assert missing == []


def test_title_spells_out_the_name():
    mismatched = {
        tool.name: tool.title
        for tool in _tools()
        if (tool.title or "").lower().replace(" ", "_").replace("-", "_") != tool.name
    }
    assert mismatched == {}
