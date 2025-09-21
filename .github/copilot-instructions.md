
## Copilot instructions — focused, actionable (short)

This repo is an MCP server that wraps the crawl4ai library. Key entry points
and conventions for an AI coding agent:

- Entry point: `crawl4ai_mcp/server.py` — registers tools via `@mcp.tool()` and
  does lazy heavy-imports (`_load_heavy_imports`, `_load_tool_modules`). Don't
  move the early environment variable assignments (they silence FastMCP banners
  and warnings on import).

- Tool implementation: put heavy logic in `crawl4ai_mcp/tools/*` and keep
  `server.py` adapters thin. Tools commonly return pydantic models; adapters
  call `.model_dump()` or `.dict()` before returning.

- Token safety: use `_apply_token_limit_fallback(result, max_tokens=20000)` on
  large responses (see `server.py`) to avoid MCP token overflow. When adding a
  new tool that may return large content (extracted HTML, markdown, tables),
  call this helper before returning.

- LLM clients: use the centralized factory in
  `crawl4ai_mcp/tools/utilities.py` (`create_openai_client(provider, cfg, purpose)`)
  — do not instantiate OpenAI clients ad-hoc. Respect `OPENAI_CHAT_URL` and
  `OPENAI_EMBEDDINGS_URL` Env vars documented in `.env.example`.

- Dev environment (macOS): prefer Homebrew Python or pyenv (Python 3.10+).
  Typical flow:

  ```bash
  /opt/homebrew/bin/python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  playwright install chromium
  pip install pytest flake8
  flake8 && pytest
  ```

- Docker & CI: `docker-compose.yml` supports `CRAWL_IMAGE` (fallback to GHCR).
  CI builds multi-arch images via `.github/workflows/ghcr-publish.yml` and runs
  a lint+pytest gate before pushing.

- Adding a new MCP tool (minimal pattern):

  - Create `crawl4ai_mcp/tools/my_tool.py` with async functions that return
    pydantic models or dicts.
  - Import lazily from `server.py`'s `_load_tool_modules()` (or add to that
    import list).
  - Add an adapter in `server.py`:

    ```py
    @mcp.tool()
    async def my_tool_adapter(...):
        _load_tool_modules()
        return await my_tool.run(...)
    ```

- Files to read first: `README.md`, `crawl4ai_mcp/server.py`,
  `crawl4ai_mcp/tools/utilities.py`, `crawl4ai_mcp/tools/web_crawling.py`,
  `crawl4ai_mcp/tools/file_processing.py`, `.github/workflows/ghcr-publish.yml`,
  `docker-compose.yml`, `.env.example`.

If you'd like I can expand this with one of the following (pick one):
1. A concrete code snippet PR template showing how to add a new tool + adapter.
2. A small smoke-test script (Python) that imports `crawl4ai_mcp` and runs a
   minimal `get_system_diagnostics()` call.
3. A PR reviewer checklist tailored to this repo (lint, token fallback, env
   changes, docker impact).

## PR reviewer checklist (quick)
Use this checklist when reviewing PRs to this repository. Copy into PR checks
and require green before merging.

- Run local lint + tests:

  ```bash
  source .venv/bin/activate  # or create venv with /opt/homebrew/bin/python3.12
  pip install -r requirements.txt
  pip install pytest flake8
  flake8
  pytest -q
  ```

- Token & payload safety:
  - If the PR returns large content from a tool, ensure `_apply_token_limit_fallback(result, max_tokens=20000)` is called before returning.
  - For LLM changes, confirm usage of `create_openai_client` in `crawl4ai_mcp/tools/utilities.py` instead of ad-hoc OpenAI client instantiation.

- New tool modules:
  - Implementation lives under `crawl4ai_mcp/tools/`.
  - `server.py` imports must remain lazy — add module to `_load_tool_modules()` if adding a new tool adapter.
  - Tool outputs must be serializable (dict or pydantic model). Adapters should call `.model_dump()` or `.dict()`.

- Docker & CI impact:
  - If you change `docker-compose.yml` (ports, profiles, env), update `README.md` and note the change in PR description.
  - CI workflow `.github/workflows/ghcr-publish.yml` runs lint/tests. Make sure new runtime deps don't break faster-than-ci installs.

- Secrets & environment variables:
  - Add new env var names to `.env.example` showing example values.
  - Never commit secrets. If new CI secrets are required, document them and ask repo admins to add them.

- Documentation & examples:
  - Update `README.md` or `docs/` for user-visible behavior changes and Docker run examples.

- Smoke test before merge (recommended):
  - Run `get_system_diagnostics()` via a small script or UVX to validate environment-level changes (Playwright, browsers, path issues).

If you'd like, I can convert this checklist into a GitHub PR template or a GitHub Actions job that blocks merging until checks pass.


