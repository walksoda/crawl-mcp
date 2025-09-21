<!-- Use this template to create PRs that add or modify MCP tools. -->

## Summary

Describe the change at a high-level. If this PR introduces a new MCP tool, include the tool name and a one-line purpose.

## Implementation checklist

- [ ] I added the tool implementation under `crawl4ai_mcp/tools/`.
- [ ] I added a thin adapter in `crawl4ai_mcp/server.py` that imports the tool lazily via `_load_tool_modules()`.
- [ ] The tool returns Pydantic models or dicts and the adapter calls `.model_dump()` or `.dict()` before returning.
- [ ] I added or updated unit tests for the new tool.
- [ ] I ran `flake8` and `pytest` locally and fixed any issues.
- [ ] I added any new env vars to `.env.example` and documented usage in `README.md`.

## Minimal PR template: add a new tool example

This shows the minimal files and snippets needed when adding a new MCP tool called `my_tool`.

1) New tool implementation: `crawl4ai_mcp/tools/my_tool.py`

```py
from pydantic import BaseModel
from typing import Any


class MyToolResult(BaseModel):
    ok: bool
    message: str


async def run(arg1: str) -> MyToolResult:
    # Implement the heavy logic here. Keep this file import-safe at top-level.
    return MyToolResult(ok=True, message=f"received {arg1}")
```

2) Adapter in `crawl4ai_mcp/server.py` (thin, lazy import):

```py
@mcp.tool()
async def my_tool_adapter(arg1: str):
    _load_tool_modules()  # ensures crawl4ai_mcp.tools.* are importable lazily
    from crawl4ai_mcp.tools import my_tool

    result = await my_tool.run(arg1)
    # return serializable object (dict or pydantic model)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return result
```

3) Tests: put a small unit test under `tests/test_my_tool.py`.

```py
import asyncio

from crawl4ai_mcp.tools import my_tool


def test_my_tool_run():
    res = asyncio.get_event_loop().run_until_complete(my_tool.run("hello"))
    assert res.ok is True
    assert "hello" in res.message
```

## Reviewer notes

- Keep adapter logic thin — heavy work belongs in `crawl4ai_mcp/tools/*`.
- Use `create_openai_client(...)` in `crawl4ai_mcp/tools/utilities.py` for any OpenAI interactions (avoid direct OpenAI SDK instantiation in tools).
- Large tool outputs (HTML, long text) should call `_apply_token_limit_fallback(result, max_tokens=20000)` before returning to prevent MCP token issues.

## How I tested

- Ran `flake8` and `pytest` locally (or: describe CI run results).

---

If you need a PR checklist variant for bugfixes, docs, or release-only changes, adapt the checklist above.
