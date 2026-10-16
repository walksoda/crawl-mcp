# Development Guide

How to set up, test, and contribute to the Crawl4AI MCP Server.

## Project Architecture

### Repository Structure

```
crawl-mcp/
├── crawl4ai_mcp/
│   ├── __init__.py              # Version detection; installs crawl4ai_guard on import
│   ├── server.py                # FastMCP entry point (main(): --transport/--host/--port)
│   ├── config.py                # .env loading and LLM provider configuration
│   ├── constants.py             # Shared limits and summary-length settings
│   ├── validators.py            # Input validation (validate_*_params, validate_output_path)
│   ├── crawl4ai_guard.py        # Disables crawl4ai's legacy cache migration
│   ├── suppress_output.py       # Keeps library output off the MCP stdout channel
│   ├── server_tools/            # @mcp.tool() adapters (the schema clients see)
│   │   ├── __init__.py          # register_all_tools()
│   │   ├── _shared.py           # Shared imports, readOnlyHint annotations, modules_unavailable_error()
│   │   ├── crawl_tools.py       # crawl_url, crawl_url_with_fallback, deep_crawl_site
│   │   ├── extraction_tools.py  # intelligent_extract, extract_entities, extract_structured_data
│   │   ├── youtube_tools.py     # YouTube transcript/info/comments tools
│   │   ├── search_tools.py      # search_web, batch_search_web, search_and_crawl (+ search_google aliases)
│   │   ├── file_tools.py        # process_file, enhanced_process_large_content, ...
│   │   └── batch_tools.py       # batch_crawl, multi_url_crawl
│   ├── tools/                   # Backward-compatible re-export facade (no logic)
│   ├── core/                    # Business logic (crawl, fallback, extraction, YouTube, search, files, batch)
│   ├── processors/              # YouTube, web search (search_engines/: Google Custom Search, Brave, Exa, SearXNG), and file conversion processors
│   ├── infra/                   # Browser config, fallback strategies, fingerprint, sessions, strategy/content cache
│   ├── utils/                   # LLM client, token counting, time parsing
│   ├── models/                  # Pydantic request/response models
│   ├── middleware/              # Response helpers and output_path persistence; see note below
│   ├── server_helpers.py        # Re-exports helpers used by server_tools (slicing, transforms, search cache)
│   ├── prompts/                 # MCP prompts
│   ├── youtube_processor.py     # Compatibility shims re-exporting processors/
│   ├── file_processor.py        #   (kept for older imports)
│   └── google_search_processor.py
├── tests/
│   ├── unit/                    # Fast unit tests (no MCP connection)
│   └── mcp/                     # Tool tests: start the server over stdio and hit the network
├── scripts/run_tests.sh         # Test runner wrapper
├── Dockerfile / docker-compose.yml
├── pyproject.toml               # Package config (source of truth for dependencies and version)
└── requirements.txt             # Mirror of the dependencies, used by the Docker build
```

The `ToolPipeline` classes in `middleware/pipeline.py` (and the pipeline step modules) are defined but not wired into any tool. The tool handlers call plain helper functions directly instead: `finalize_tool_response` from `middleware/file_persistence.py` (writing results to `output_path`), plus `_convert_result_to_dict` / `_process_content_fields` (`response_transform.py`), `_apply_content_slicing` (`content_slicing.py`), `_apply_search_content_slicing` (`search_slicing.py`), and the search result cache (`cache.py`), which reach `server_tools/_shared.py` through `server_helpers.py`.

### Request Flow

A tool call passes through these layers:

1. `server.py` creates `FastMCP("Crawl4AI")` and calls `register_all_tools(mcp, _get_modules)`. Warnings and the FastMCP banner are suppressed before the imports, and logging is configured there too (off unless `CRAWL4AI_MCP_LOG_LEVEL` is set). `main()` parses `--transport`, `--host`, and `--port` with argparse.
2. `server_tools/*_tools.py` are thin `@mcp.tool()` adapters. Parameters are declared with `Annotated[..., Field(description=...)]`, which becomes the schema MCP clients see. `get_modules()` lazily loads `tools/`; on failure the handler returns `modules_unavailable_error()`.
3. `tools/` re-exports functions from `core/` for backward compatibility.
4. `core/` implements the logic and uses `processors/`, `infra/`, `utils/`, and `models/`.

### Tool Handler Pattern

When adding or changing a tool, follow the order used by `crawl_url` in `server_tools/crawl_tools.py`:

1. Validate input with `validators.py`: `validate_*_params`, `validate_content_slicing_params`, and `validate_output_path(output_path, overwrite)`. Do this before any network access.
2. Call the core function through `get_modules()`.
3. Normalize the result with `_convert_result_to_dict`, then `_process_content_fields`.
4. Persist the result with `finalize_tool_response(..., tool_kind=KIND_*)` right before the successful return and before any slicing or truncation, so the file on disk holds the full, unsliced content. `KIND_*` constants are defined at the top of `middleware/file_persistence.py`.
5. Slice and limit the response with `_apply_content_slicing` (`content_limit` / `content_offset`) and `apply_token_limit` (about 25,000 tokens).

Tools that support `output_path` expose all three parameters: `output_path`, `include_content_in_response`, and `overwrite`.

### STDIO Output Rule

Over the STDIO transport, stdout is reserved for MCP JSON-RPC messages, so server code must never print to stdout. Write diagnostics to stderr (`print(..., file=sys.stderr)`). `server.py` redirects stray stdout writes to stderr while running over STDIO, and `suppress_output.py` silences noisy library output. Logging is disabled by default. Setting `CRAWL4AI_MCP_LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, or `ERROR` sends log records to stderr. `server.py` reads it at import, before `.env` is loaded, so it must be in the process environment. `FASTMCP_LOG_LEVEL` does not produce log output.

## Development & Distribution

1. Development: edit logic in `crawl4ai_mcp/core/` and tool registrations in `crawl4ai_mcp/server_tools/`.
2. UVX distribution: installed from the GitHub repository (`uvx --from git+https://github.com/walksoda/crawl-mcp crawl-mcp`).
3. Docker distribution: container image built from `Dockerfile`.

### Development Workflow

```bash
# 1. Create a branch
git switch -c feat/your-feature

# 2. Develop and test
venv/bin/pytest tests/unit/
venv/bin/pytest tests/mcp/ -m "not slow"

# 3. Test with Docker (optional)
docker-compose up --build

# 4. Commit (stage files by name; avoid `git add .` so .env or venv/ are never included)
git add crawl4ai_mcp/core/your_module.py tests/unit/test_your_module.py
git commit -m "feat: add new functionality"
```

### Release

```bash
# 1. Update the version in pyproject.toml (the single source of the version)
# 2. Commit and tag
git add pyproject.toml
git commit -m "release: bump version to X.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# 3. Push the branch and the tag together
git push --atomic origin main vX.Y.Z
```

## Development Environment Setup

### Local Development Setup

Use a virtual environment named `venv` in the repository root. The MCP test client in `tests/conftest.py` starts the server with `./venv/bin/python`.

```bash
git clone https://github.com/walksoda/crawl-mcp.git
cd crawl-mcp
python -m venv venv            # or: uv venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -e ".[test]"       # with uv: uv pip install --python venv/bin/python -e ".[test]"
python -m playwright install chromium
```

Verify the environment:

```bash
which python                                          # should point to ./venv/bin/python
python -c "import crawl4ai_mcp; print(crawl4ai_mcp.__file__)"   # should point into this checkout
```

## Testing

```bash
venv/bin/pytest tests/unit/                          # Unit tests (fast, no MCP connection)
venv/bin/pytest tests/mcp/ -m "not slow"             # MCP tool tests (stdio server, real network)
venv/bin/pytest tests/unit/test_file_persistence.py::<test_name>   # Single test

./scripts/run_tests.sh quick                         # Unit tests
./scripts/run_tests.sh mcp                           # MCP tool tests, excluding slow
./scripts/run_tests.sh full cov                      # Everything, with coverage
./scripts/run_tests.sh youtube|crawl|search|file|utility   # One tool group
```

- `pytest.ini`: `asyncio_mode = auto`, default timeout 120 seconds, markers `slow`, `mcp`, `youtube`, `crawl`, `search`, `file`, `utility`.
- CI (Python 3.11 and 3.12) runs `tests/unit/`, then `tests/mcp/ -m "not slow"`, then `-m slow` (failures allowed).

Manual server checks:

```bash
python -m crawl4ai_mcp.server --help
python -m crawl4ai_mcp.server                                               # STDIO
python -m crawl4ai_mcp.server --transport streamable-http --host 127.0.0.1 --port 8000   # HTTP at /mcp
./scripts/run_http_server.sh --log-level DEBUG                              # same HTTP server, with logging
```

`--transport` accepts `stdio`, `streamable-http`, `http`, and `sse`. An invalid argument exits with status 2 and prints the error to stderr.

See the [HTTP Integration Guide](HTTP_INTEGRATION.md) for calling the HTTP server with curl or an MCP client.

### Documentation Check

`scripts/check_docs.py` compares the docs with the code. It checks that every tool parameter (and every key of the `request` dict tools) appears in `docs/API_REFERENCE.md` and `docs/ja/API_REFERENCE.md`, that dependency pins match `pyproject.toml`, that repository paths, links and anchors exist, that environment variables used in examples are read by the server, and that the prose has no emoji or hype words. `tests/unit/test_docs_sync.py` runs it in CI.

```bash
venv/bin/python scripts/check_docs.py              # docs against code
venv/bin/python scripts/check_docs.py --release    # also: CHANGELOG covers code changes since the latest tag
git config core.hooksPath .githooks                # run the --release check before every push
```

When you change a tool's parameters or behavior, update both API references, both READMEs and the `[Unreleased]` section of `CHANGELOG.md` in the same change.

## Configuration for Development

### Environment Variables

`config.py` loads `.env` from the project root (see `.env.example`).

```bash
# LLM providers (either per-provider keys or a full MCP_LLM_CONFIG JSON)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://resource.openai.azure.com"

# Web search (at least one configured engine)
export SEARCH_ENGINE=auto                   # auto (default) | google_custom | brave | exa | searxng
export GOOGLE_SEARCH_API_KEY="..."
export GOOGLE_SEARCH_ENGINE_ID="..."
export BRAVE_SEARCH_API_KEY="..."
export EXA_API_KEY="..."
export SEARXNG_URL="http://localhost:8080"  # self-hosted SearXNG, no key
```

### Local MCP Client Configuration

```json
{
  "mcpServers": {
    "crawl4ai-dev": {
      "command": "/path/to/crawl-mcp/venv/bin/python",
      "args": ["-m", "crawl4ai_mcp.server"],
      "cwd": "/path/to/crawl-mcp"
    }
  }
}
```

### Code Style and Standards

- Follow PEP 8
- Use type hints where appropriate
- Document functions with docstrings
- Return structured error dicts from tools instead of raising
- Use async/await for I/O operations

Write commit messages in English, keep them short, and do not use emoji. Use the format `type: brief description`, for example:
- `feat: add YouTube batch processing`
- `fix: resolve memory leak in large file processing`
- `docs: update API reference`

## Build and Packaging

### Docker

```bash
# Build the image
docker build -t crawl4ai-mcp:latest .

# STDIO mode (default)
docker-compose up --build
docker run -i crawl4ai-mcp:latest

# HTTP mode
docker-compose --profile http up --build crawl4ai-mcp-http
docker run -p 8000:8000 crawl4ai-mcp:latest \
  python -m crawl4ai_mcp.server --transport http --host 0.0.0.0 --port 8000
```

The Docker build installs dependencies from `requirements.txt`.

### Packaging and Dependencies

Excerpt from `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "crawl-mcp"
requires-python = ">=3.11"
dependencies = [
    "crawl4ai>=0.9.4,<0.10",
    "fastmcp>=2.14.2,<3",
    "playwright==1.55.0",
    # ...
]

[project.optional-dependencies]
test = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0", "pytest-timeout>=2.3.1", "pytest-cov>=5.0.0"]

[project.scripts]
crawl-mcp = "crawl4ai_mcp.server:main"
```

`pyproject.toml` is the source of truth for dependencies and the version. `requirements.txt` mirrors the dependencies for the Docker build and must be kept in sync when `pyproject.toml` changes.

## Debugging and Troubleshooting

Import errors:

```bash
source venv/bin/activate
pip install -e ".[test]"
python -c "import crawl4ai_mcp; print(crawl4ai_mcp.__file__)"
```

Browser issues:

```bash
python -m playwright install chromium
python -m playwright install --with-deps chromium   # Linux: also installs system libraries (sudo)
```

Server startup issues:

```bash
lsof -i :8000               # Check port availability
kill <PID>
```

The server writes diagnostics to stderr. Python logging is off unless `CRAWL4AI_MCP_LOG_LEVEL` is set. To investigate, run the server in a terminal with `CRAWL4AI_MCP_LOG_LEVEL=DEBUG` and watch stderr, or add temporary `print(..., file=sys.stderr)` calls. Never print to stdout.

CPU profiling:

```bash
python -m cProfile -o profile.prof -m crawl4ai_mcp.server
python -m pstats profile.prof
```

## Contributing Guidelines

1. Fork and clone the repository.
   ```bash
   git clone https://github.com/your-username/crawl-mcp.git
   cd crawl-mcp
   git remote add upstream https://github.com/walksoda/crawl-mcp.git
   ```

2. Create a feature branch.
   ```bash
   git switch -c feat/your-feature-name
   ```

3. Change `crawl4ai_mcp/core/` (logic) and `crawl4ai_mcp/server_tools/` (tool parameters and registration), and add tests under `tests/unit/`.

4. Run the tests.
   ```bash
   venv/bin/pytest tests/unit/
   venv/bin/pytest tests/mcp/ -m "not slow"
   docker build -t crawl4ai-mcp:test .     # optional
   ```

5. Commit, push, and open a pull request.
   ```bash
   git add <changed files>
   git commit -m "feat: add new feature description"
   git push origin feat/your-feature-name
   ```

### Pull Request Checklist

- [ ] Unit tests and MCP tool tests pass
- [ ] `scripts/check_docs.py` passes, and behavior changes are described in the docs and `CHANGELOG.md`
- [ ] `requirements.txt` updated if dependencies changed
- [ ] Commit messages follow the standards above
- [ ] No breaking changes without discussion

## Related Resources

- [API Reference](API_REFERENCE.md)
- [Configuration Examples](CONFIGURATION_EXAMPLES.md)
- [Usage Patterns](ADVANCED_USAGE.md)
- [Installation Guide](INSTALLATION.md)
- [HTTP Integration](HTTP_INTEGRATION.md)
