# Usage Patterns

Patterns and workflows that combine the server's tools.

Each example shows the `arguments` object for one tool call. Every key is a real parameter of the named tool. The [API Reference](API_REFERENCE.md) lists all parameters.

## Web crawling

### Crawling a site

`deep_crawl_site` follows links from a start page:

```json
{
  "url": "https://docs.example.com",
  "max_depth": 2,
  "max_pages": 10,
  "crawl_strategy": "best_first",
  "url_pattern": "*docs*",
  "score_threshold": 0.3
}
```

`max_depth` accepts 1 or 2, and `max_pages` accepts 1 to 10. Values outside these ranges return an error before any page is fetched. `crawl_strategy` is `bfs`, `dfs`, or `best_first`. `url_pattern` takes one pattern; the server passes it as a single filter and does not split comma-separated lists.

The response has the pages merged into one `markdown` field, and `crawled_pages` lists each page's `url`, `title`, `success`, and `error`. With `output_path`, each successful page is saved as its own `.md` file next to an `index.json`, and the response keeps only metadata unless `include_content_in_response` is `true`.

### JavaScript-heavy pages

For single-page applications, call `crawl_url` with `wait_for_js` and, if the page has a marker element, `wait_for_selector`:

```json
{
  "url": "https://spa-application.com",
  "wait_for_js": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded"
}
```

### HTML pages and document URLs

`crawl_url` renders remote `.html` and `.htm` URLs in the browser, like any other web page. Document URLs (PDF, DOCX, XLSX, PPTX, ZIP and similar) are downloaded and converted as files, and they skip the browser fallback chain. You can also pass them to `process_file` directly.

### Sites with bot protection

Start with `crawl_url` and the undetected browser:

```json
{
  "url": "https://protected-site.com",
  "use_undetected_browser": true,
  "wait_for_js": true,
  "timeout": 90
}
```

If that fails, use `crawl_url_with_fallback` with the same URL:

```json
{
  "url": "https://protected-site.com",
  "wait_for_js": true,
  "timeout": 90
}
```

`crawl_url_with_fallback` tries up to seven stages in order: a static fast path, a normal headless browser, Chromium stealth mode, simulated user behavior, a mobile user agent, the page's AMP or RSS version, and static HTML/JSON extraction. It caches the stage that worked for each domain and starts there next time. The response names the stage in `extracted_data.fallback_strategy_used` and `fallback_stage`. A stage whose page returns HTTP 4xx/5xx counts as failed, and the AMP stage also rejects pages that look like error pages.

## LLM-based extraction

### Extracting specific information

`intelligent_extract` takes a goal and optional instructions:

```json
{
  "url": "https://research-paper.com",
  "extraction_goal": "Extract methodology, results, and conclusions from academic paper",
  "content_filter": "llm",
  "custom_instructions": "Focus on quantitative results, statistical significance, and practical implications. Ignore references and author information.",
  "use_llm": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4.1"
}
```

`content_filter` is `bm25`, `pruning`, or `llm`. To narrow the page with BM25 first and then run the LLM, use `bm25` with a `filter_query`:

```json
{
  "url": "https://docs.example.com/reference",
  "extraction_goal": "Extract actionable technical information",
  "content_filter": "bm25",
  "filter_query": "technical documentation API reference",
  "use_llm": true
}
```

### LLM providers

Set API keys for the providers you use in the MCP client configuration:

```json
{
  "mcpServers": {
    "crawl4ai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/walksoda/crawl-mcp", "crawl-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "AZURE_OPENAI_API_KEY": "your-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com"
      }
    }
  }
}
```

The default provider is `openai` with model `gpt-4.1`. If the requested provider has no API key, the server tries `openai`, `aoai`, `anthropic`, and `ollama` in that order. [Configuration Examples](CONFIGURATION_EXAMPLES.md#environment-variables) describes `MCP_LLM_CONFIG`.

## Large content

The server does not summarize large pages unless you ask it to. Without `auto_summarize`, a response over about 25,000 tokens is truncated and carries a `recommendations` field. Two options return the full text.

Page through the content with `content_limit` and `content_offset`, then repeat with `content_offset` set to 20000, 40000, and so on:

```json
{
  "url": "https://example.com/very-long-article",
  "content_limit": 20000,
  "content_offset": 0
}
```

Or write the full text to a file with `output_path`. The response then contains only metadata and the file path:

```json
{
  "url": "https://example.com/very-long-article",
  "output_path": "/absolute/path/article.md"
}
```

The file always gets the unsliced content. Set `include_content_in_response` to keep the content in the response as well, and `overwrite` to replace an existing file.

### Summarization (opt-in)

With `auto_summarize: true`, `crawl_url` summarizes pages longer than about 15,000 tokens using the default LLM provider. It skips summarization when `content_limit` or `content_offset` is set.

```json
{
  "url": "https://example.com/very-long-article",
  "auto_summarize": true
}
```

`process_file` and `extract_youtube_transcript` also let you set the threshold, length, and provider:

```json
{
  "url": "https://example.com/report.pdf",
  "auto_summarize": true,
  "max_content_tokens": 15000,
  "summary_length": "long",
  "llm_provider": "anthropic"
}
```

| `summary_length` | Target | Detail level |
|------------------|--------|--------------|
| `short` | 2-3 paragraphs, about 400 tokens | Key points and main findings |
| `medium` (default) | 4-6 paragraphs, about 1000 tokens | Overview with important details |
| `long` | 8-12 paragraphs, about 2000 tokens | Analysis with examples and context |

### Chunking long documents

`enhanced_process_large_content` splits a page into chunks, keeps the most relevant ones, and can summarize them:

```json
{
  "url": "https://example.com/long-report",
  "chunking_strategy": "topic",
  "filtering_strategy": "bm25",
  "filter_query": "results methodology conclusions",
  "max_chunk_tokens": 2000,
  "chunk_overlap": 200,
  "extract_top_chunks": 10,
  "similarity_threshold": 0.5,
  "summarize_chunks": true,
  "merge_strategy": "hierarchical",
  "final_summary_length": "medium"
}
```

`chunking_strategy` is `topic`, `sentence`, `overlap`, or `regex`. `filtering_strategy` is `bm25`, `pruning`, or `llm`. `merge_strategy` is `hierarchical` or `linear`.

## Batch operations

### Limits per call

| Tool | Limit per call |
|------|----------------|
| `batch_crawl` | 3 URLs, processed one after another |
| `multi_url_crawl` | 5 URL configurations, processed one after another (`max_concurrent` has no effect) |
| `batch_extract_youtube_transcripts` | 3 URLs; optional `max_concurrent` (default 3, capped at 5) |
| `batch_search_google` | 3 queries; optional `max_concurrent` (default 3, capped at 5) |
| `search_and_crawl` | `crawl_top_results` up to 3 |
| `deep_crawl_site` | `max_depth` 1-2, `max_pages` 1-10 |

Split larger jobs into several calls.

### Crawling several URLs

`batch_crawl` takes up to three URLs and flat options:

```json
{
  "urls": [
    "https://site1.com/page1",
    "https://site1.com/page2",
    "https://site2.com/api-docs"
  ],
  "generate_markdown": true,
  "extract_media": false,
  "wait_for_js": false,
  "base_timeout": 60,
  "output_path": "/absolute/path/batch-run"
}
```

Here `output_path` is a directory. Each successful URL becomes a Markdown file, and `index.json` lists every result.

`multi_url_crawl` applies different options per URL:

```json
{
  "url_configurations": {
    "https://spa.example.com/dashboard": {"wait_for_js": true, "timeout": 60},
    "https://docs.example.com/guide": {"css_selector": "main"},
    "https://blog.example.com/post": {"extract_media": true}
  },
  "base_timeout": 30
}
```

The keys must be literal URLs. Each value can set `generate_markdown`, `extract_media`, `wait_for_js`, `timeout`, `use_undetected_browser`, and `css_selector`.

### YouTube transcripts

```json
{
  "request": {
    "urls": [
      "https://youtube.com/watch?v=VIDEO1",
      "https://youtube.com/watch?v=VIDEO2",
      "https://youtube.com/watch?v=VIDEO3"
    ],
    "languages": ["en", "ja"],
    "include_timestamps": true,
    "translate_to": "en",
    "timezone": "America/New_York"
  }
}
```

`timezone` takes an IANA name and defaults to `UTC`. The server uses it to convert each video's publish time and returns the result under `metadata.published_at`. `extract_youtube_transcript` and `get_youtube_video_info` accept the same parameter.

### Search and crawl

The search tools take a single `request` object. `search_and_crawl` searches and then crawls the top results:

```json
{
  "request": {
    "search_query": "enterprise API documentation best practices",
    "crawl_top_results": 3,
    "generate_markdown": true,
    "max_content_per_page": 5000
  }
}
```

`batch_search_google` runs up to three queries:

```json
{
  "request": {
    "queries": [
      "machine learning trends 2024",
      "AI adoption enterprise survey",
      "deep learning frameworks comparison"
    ],
    "num_results_per_query": 10,
    "search_genre": "pdf",
    "recent_days": 365
  }
}
```

`search_genre` accepts `pdf`, `documents`, `presentations`, `spreadsheets`, `japanese`, and `english`, in any letter case. `search_google`, `batch_search_google`, and `search_and_crawl` reject other values with an `invalid_search_genre` error and a `valid_genres` list, before any search runs. `get_search_genres` returns each genre with the operator it adds to the query.

## Workflows

### Research across several sources

1. Find pages with `search_google`.
2. Collect them with `search_and_crawl` (up to 3 results) or `batch_crawl` (up to 3 URLs), using `output_path` to keep the full text on disk.
3. Pull out the details you need with `extract_structured_data` or `intelligent_extract`.
4. Compare the saved Markdown files. For long documents, use `enhanced_process_large_content`.

### Mapping a documentation site

First collect the pages with `deep_crawl_site`:

```json
{
  "url": "https://api-docs.example.com",
  "max_depth": 2,
  "max_pages": 10,
  "crawl_strategy": "bfs",
  "url_pattern": "*docs*",
  "extract_media": true,
  "output_path": "/absolute/path/api-docs"
}
```

Then extract details from a key page with `intelligent_extract`:

```json
{
  "url": "https://api-docs.example.com/reference",
  "extraction_goal": "List API endpoints with their parameters and examples",
  "content_filter": "bm25",
  "filter_query": "API endpoint parameter example tutorial"
}
```

### Extracting contact information

`extract_entities` uses regular expressions by default:

```json
{
  "url": "https://company-directory.com",
  "entity_types": ["emails", "phones", "urls"],
  "deduplicate": true,
  "include_context": true,
  "custom_patterns": {
    "linkedin": "linkedin\\.com/in/[\\w-]+",
    "github": "github\\.com/[\\w-]+"
  }
}
```

The built-in types are `emails`, `phones`, `urls`, `dates`, `ips`, `prices`, `credit_cards`, and `coordinates`. Singular names such as `email` or `phone` also work, and letter case does not matter. Each `custom_patterns` entry defines a new type; add its name to `entity_types` to extract it. `custom_patterns` is used only in regex mode. Other types, such as `people` or `social_media`, need `"use_llm": true`. Without it, they are listed in `unsupported_entity_types` and the response carries a warning.

## Timeouts

| Tool | Parameter | Default |
|------|-----------|---------|
| `crawl_url`, `crawl_url_with_fallback` | `timeout` (seconds) | 60 |
| `extract_structured_data` | `timeout` (seconds) | 30 |
| `batch_crawl`, `multi_url_crawl` | `base_timeout` per URL | 30 |
| `deep_crawl_site` | `base_timeout` per page | 60 |
| `extract_youtube_transcript` | `fallback_timeout` for the page-crawl fallback | 60 |

## Errors and diagnostics

Every tool returns `success` and, on failure, an `error` message. Many failures also include `error_code` or `suggestions`.

When a crawl fails, try these steps in order:

1. `crawl_url` with default settings.
2. `crawl_url` with `use_undetected_browser: true`, or `crawl_url_with_fallback`.
3. Simpler settings: no `wait_for_js` and a shorter `timeout`.
4. For PDF and Office URLs, `process_file`.

Over STDIO, stdout carries only MCP messages and the server writes diagnostics to stderr. Python logging is off by default. Set `CRAWL4AI_MCP_LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, or `ERROR` to write log records to stderr. The server reads it at startup, before it loads `.env`, so set it in the process environment, for example in the `env` block of the MCP client configuration. `FASTMCP_LOG_LEVEL` has no effect. Look at the MCP client's server log (stderr) when you investigate a problem.

The server has no health-check endpoint. To check an HTTP server, call a cheap tool such as `get_supported_file_formats`; see the [HTTP Integration Guide](HTTP_INTEGRATION.md).

## Python client with retries

This example uses the `fastmcp` client, which is a dependency of this project, against a server started with `--transport streamable-http`:

```python
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"


async def crawl_with_retries(url: str) -> Optional[Dict[str, Any]]:
    """Try several crawl_url settings, then fall back to crawl_url_with_fallback."""
    strategies = [
        {"wait_for_js": True, "timeout": 60},
        {"wait_for_js": False, "timeout": 30},
        {"use_undetected_browser": True, "timeout": 60},
    ]

    async with Client(SERVER_URL) as client:
        for attempt, strategy in enumerate(strategies):
            result = await client.call_tool("crawl_url", {"url": url, **strategy})
            data = json.loads(result.content[0].text)
            if data.get("success"):
                return data
            logging.warning("Attempt %d failed: %s", attempt + 1, data.get("error"))
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

        result = await client.call_tool("crawl_url_with_fallback", {"url": url})
        data = json.loads(result.content[0].text)
        return data if data.get("success") else None


print(asyncio.run(crawl_with_retries("https://example.com")))
```

The [HTTP Integration Guide](HTTP_INTEGRATION.md) shows the same calls with curl.

## Tips

1. Start with plain `crawl_url` and add options only when a page needs them.
2. For long pages, save the text with `output_path` or page through it with `content_limit` and `content_offset`.
3. Keep batches to three URLs or queries per call.
4. Use `crawl_url_with_fallback` for sites with bot protection.
5. Send PDF and Office files to `process_file`.
6. Turn on `auto_summarize` only when you need it. It needs an LLM API key and uses tokens.
7. Respect the rate limits and quotas of external services.

## Related documentation

- [Configuration Examples](CONFIGURATION_EXAMPLES.md)
- [API Reference](API_REFERENCE.md)
- [HTTP Integration Guide](HTTP_INTEGRATION.md)
- [Development Guide](DEVELOPMENT.md)
- [Installation Guide](INSTALLATION.md)
