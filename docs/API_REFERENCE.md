# API Reference

Reference for the MCP tools provided by the Crawl4AI MCP Server.

## Tool Selection Guide

### Choose the Right Tool for Your Task

| Use case | Tool | What it does |
|-------------|---------------------|------------------|
| Single webpage | `crawl_url` | Basic crawling, JS support |
| Multiple pages (up to 10) | `deep_crawl_site` | Site mapping, link following |
| Web search | `search_web` | Google Custom Search, Brave, Exa or SearXNG |
| Search + Crawling | `search_and_crawl` | Web search + auto-crawl |
| Difficult sites | `crawl_url_with_fallback` | Multiple retry strategies |
| Structured data | `extract_structured_data` | CSS/LLM schemas |
| Intelligent extraction | `intelligent_extract` | Goal-based, filter-aware extraction |
| Entity extraction | `extract_entities` | Emails, phones, URLs, dates, IPs, prices |
| File processing | `process_file` | PDF, Office, ZIP conversion |
| Large content | `enhanced_process_large_content` | Chunking and filtering strategies |
| YouTube transcript | `extract_youtube_transcript` | Subtitle extraction |
| YouTube comments | `extract_youtube_comments` | Comment extraction with sorting |
| YouTube transcripts (batch) | `batch_extract_youtube_transcripts` | Up to 3 videos at once |
| Video metadata | `get_youtube_video_info` | Title, description, publish time, summary |
| Batch crawling | `batch_crawl` | Up to 3 URLs, processed sequentially |
| Multi-URL with config | `multi_url_crawl` | Up to 5 URLs, per-URL config |
| Batch search | `batch_search_web` | Up to 3 queries concurrently |
| Search genres | `get_search_genres` | List available genre filters |
| Supported file formats | `get_supported_file_formats` | List supported formats |

### Performance Guidelines

- Deep crawling is limited to 10 pages per call.
- Batch tools have per-call limits: 3 URLs, 5 URL configurations, 3 queries, or 3 videos.
- Set the total timeout to about `pages x base_timeout`.
- Files larger than 100MB are rejected.
- If a call fails, retry it manually.

### Best Practices

For JavaScript-heavy sites, set `wait_for_js: true`, raise the timeout to 30-60 seconds, and use `wait_for_selector` to wait for a specific element.

For LLM-based extraction, use `extract_structured_data` in LLM mode. If no LLM is configured, use CSS extraction instead. Adjust the extraction schema to the data you need.

## Result Persistence (output_path)

The information-gathering tools listed at the end of this section accept three optional parameters that write the full result to disk and return only metadata. An agent can then fetch large pages, long transcripts, or whole batches and read the file when it needs the content, instead of holding the raw payload in its context window.

Parameters:
- `output_path` (Optional[str]): Absolute path to write the result to. `None` (default) or `""` skips persistence. Single-file tools expect a file path (the extension is added when omitted); batch tools expect a directory path and write one file per item plus an `index.json`.
- `include_content_in_response` (bool): When `true`, the content is also returned inline (still subject to any `content_limit`/`content_offset`/`max_content_per_page` slicing). When `false` (default), only metadata and the file path are returned.
- `overwrite` (bool): Overwrite existing output files (default: `false`). For single-file tools this guards the file at `output_path`; for batch tools it guards the per-item files and `index.json` inside the directory (a regular file sitting at a batch `output_path` is always rejected).

For request-dict tools (`search_web`, `batch_search_web`, `search_and_crawl`, `batch_extract_youtube_transcripts`, and the `search_google`/`batch_search_google` aliases) these three are keys of the `request` dict instead of top-level parameters.

Behavior:
- Writes are atomic (temp file + `os.replace`); parent directories are created automatically.
- The full, unsliced payload is written before any slicing or truncation, so the file is complete even when the response is sliced.
- Each markdown file starts with a YAML frontmatter block (`url`, `title`, `fetched_at`, `source_tool`).
- Batch (directory) tools skip per-item persistence for items that report `success=false`; those items still appear in `index.json` with a null output path (`file: null` for `deep_crawl_site`/`search_and_crawl`/`batch_extract_youtube_transcripts`, `output_file: null` for `batch_crawl`/`multi_url_crawl`).

Tools that support `output_path`:
- Single-file (`output_path` is a file): `crawl_url`, `crawl_url_with_fallback`, `extract_structured_data`, `intelligent_extract`, `extract_entities`, `process_file`, `enhanced_process_large_content`, `extract_youtube_transcript`, `extract_youtube_comments`, `get_youtube_video_info`, `search_web`, `batch_search_web` (and the `search_google`/`batch_search_google` aliases)
- Directory/batch (`output_path` is a directory): `deep_crawl_site`, `search_and_crawl`, `batch_crawl`, `multi_url_crawl`, `batch_extract_youtube_transcripts`

## Web Crawling Tools

### `crawl_url`

Web crawling with JavaScript support, screenshot capture, and optional content summarization.

Parameters:
- `url` (str): URL to crawl (required)
- `css_selector` (Optional[str]): CSS selector for targeted extraction (default: None)
- `extract_media` (bool): Extract images and videos (default: False)
- `take_screenshot` (bool): Take a screenshot of the page (default: False)
- `generate_markdown` (bool): Generate markdown from page content (default: True)
- `include_cleaned_html` (bool): Include cleaned HTML in response (default: False)
- `wait_for_selector` (Optional[str]): Wait for a specific element to load (default: None)
- `timeout` (int): Timeout in seconds (default: 60)
- `wait_for_js` (bool): Wait for JavaScript to finish executing (default: False)
- `auto_summarize` (bool): Summarize large content with an LLM; only runs when explicitly `true` (default: False)
- `use_undetected_browser` (bool): Use undetected browser to bypass bot detection (default: False)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.md` file)

Response behavior:
- By default, returns markdown content only to reduce token usage
- Set `include_cleaned_html=True` to also receive cleaned HTML content
- Token limit: 25000 tokens (automatically truncated with recommendations if exceeded)
- Content is summarized only when `auto_summarize=true` (summarization is skipped while `content_limit`/`content_offset` paging is active). Without it, large pages are returned as-is and the response is truncated at about 25000 tokens; use `output_path` to save the full, unsummarized content to disk
- Remote `.html`/`.htm` URLs are rendered in the browser like any other web page. Document URLs (PDF, Office, ZIP, text, arXiv `/pdf/` links, etc.) are converted as files without a browser fallback
- If the first crawl fails, it is retried automatically with the undetected browser (except for document URLs)
- `take_screenshot`: with `output_path`, the PNG is saved next to the main file (`<stem>_screenshot.png`) and the response gets `screenshot_path`; without `output_path`, the screenshot is dropped from the response with a warning

### `deep_crawl_site`

Crawl a site recursively from a starting URL, following links up to the given depth.

Parameters:
- `url` (str): Starting URL (required)
- `max_depth` (int): Link depth, 1-2 (default: 2)
- `max_pages` (int): Maximum number of pages to crawl, 1-10 (default: 5)
- `crawl_strategy` (str): Crawling strategy, 'bfs', 'dfs' or 'best_first' (default: 'bfs')
- `include_external` (bool): Follow external links (default: False)
- `url_pattern` (Optional[str]): URL filter pattern, e.g. '*docs*', '*blog*' (default: None)
- `score_threshold` (float): Minimum relevance score 0.0-1.0 (default: 0.0)
- `extract_media` (bool): Extract images and videos from pages (default: False)
- `base_timeout` (int): Timeout per page in seconds (default: 60)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a directory of per-URL `.md` files + `index.json`)

Response behavior:
- `max_depth` or `max_pages` outside its range returns an error before any page is fetched. The `error_code` is `invalid_depth` or `depth_too_large` for `max_depth`, and `invalid_max_pages` or `max_pages_too_large` for `max_pages`
- `markdown` and `content` hold all crawled pages merged into one document
- `crawled_pages` lists each page with `url`, `title`, `success`, and `error`
- With `output_path`, each successful page is saved as its own `.md` file and listed in `index.json`. Unless `include_content_in_response=true`, the merged `markdown`/`content` and the page bodies are left out of the response

### `crawl_url_with_fallback`

Crawl with fallback strategies for anti-bot sites. Use content_offset/content_limit for pagination.

Parameters:
- `url` (str): URL to crawl (required)
- `css_selector` (Optional[str]): CSS selector (default: None)
- `extract_media` (bool): Extract media (default: False)
- `take_screenshot` (bool): Take screenshot (default: False)
- `generate_markdown` (bool): Generate markdown (default: True)
- `wait_for_selector` (Optional[str]): Element to wait for (default: None)
- `timeout` (int): Timeout in seconds (default: 60)
- `wait_for_js` (bool): Wait for JavaScript (default: False)
- `auto_summarize` (bool): Summarize content with an LLM; only runs when explicitly `true` (default: False)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.md` file)

Response behavior:
- Token limit: 25000 tokens (automatically truncated with recommendations if exceeded), same as `crawl_url`
- Content is summarized only when `auto_summarize=true`. Without it, large pages are returned as-is and the response is truncated; use `output_path` to save the full, unsummarized content to disk
- Remote `.html`/`.htm` URLs are rendered in the browser. Document URLs (PDF, Office, ZIP, text, etc.) are converted as files once, without the browser fallback stages
- `take_screenshot`: the screenshot is persisted only when `output_path` is set (saved as `<stem>_screenshot.png`, referenced by `screenshot_path`)
- A fallback stage that gets an HTTP 4xx/5xx status moves on to the next stage. The AMP stage also skips pages that look like error pages, so a missing `<url>/amp` page is not returned as the content

## Data Extraction Tools

### `extract_structured_data`

Extract structured data using CSS selectors or an LLM schema.

Parameters:
- `url` (str): Target URL (required)
- `extraction_type` (str): Extraction method, 'css', 'llm' or 'table' (default: 'css')
- `css_selectors` (Optional[Dict]): CSS selector mapping for targeted extraction (default: None)
- `extraction_schema` (Optional[Dict]): Schema definition for LLM extraction (default: None)
- `generate_markdown` (bool): Generate markdown from page content (default: False)
- `wait_for_js` (bool): Wait for JavaScript before extraction (default: False)
- `timeout` (int): Timeout in seconds (default: 30)
- `use_llm_table_extraction` (bool): Use LLM for table extraction (default: False)
- `table_chunking_strategy` (str): Table chunking method, 'intelligent', 'fixed' or 'semantic' (default: 'intelligent')
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file including `table_data`)

### `intelligent_extract`

Goal-based content extraction with configurable filtering and optional LLM enrichment.

Parameters:
- `url` (str): Target URL (required)
- `extraction_goal` (str): Natural language description of what to extract (required)
- `content_filter` (str): Filter strategy, 'bm25', 'pruning' or 'llm' (default: 'bm25')
- `filter_query` (Optional[str]): Keywords for filter-based strategies (default: None)
- `chunk_content` (bool): Split content into chunks for large documents (default: False)
- `use_llm` (bool): Use LLM for semantic extraction (default: True)
- `llm_provider` (Optional[str]): LLM provider, auto-detected if not specified (default: None)
- `llm_model` (Optional[str]): Specific LLM model, auto-detected if not specified (default: None)
- `custom_instructions` (Optional[str]): Additional instructions for the extraction process (default: None)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file)

### `extract_entities`

Extract specific entity types from a URL using pattern matching or LLM.

Parameters:
- `url` (str): Target URL (required)
- `entity_types` (List[str]): Entity types to extract (required). Regex-supported values: 'emails', 'phones', 'urls', 'dates', 'ips', 'prices', 'credit_cards', 'coordinates'. Other types (e.g. 'people', 'organizations', 'social_media') are extracted only with `use_llm=true`
- `custom_patterns` (Optional[Dict]): Custom regex patterns as `{type_name: regex}`; add `type_name` to `entity_types` to extract it. Used in regex mode only (default: None)
- `include_context` (bool): Include surrounding context for each entity (default: True)
- `deduplicate` (bool): Remove duplicate entities from results (default: True)
- `use_llm` (bool): Use LLM for named entity recognition (default: False)
- `llm_provider` (Optional[str]): LLM provider, auto-detected if not specified (default: None)
- `llm_model` (Optional[str]): Specific LLM model, auto-detected if not specified (default: None)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file)

Response behavior:
- Type names are case-insensitive, and singular forms are mapped to the plural names: 'email', 'phone', 'url', 'date', 'ip', 'price', 'credit_card', 'coordinate', 'person' (to 'people'), 'organization', 'location', 'product', 'event'. `entity_types_requested` shows the normalized names
- In regex mode, requested types without a pattern are listed in `unsupported_entity_types`, with a message in `warnings`

## File Processing Tools

### `process_file`

Convert files to Markdown using Microsoft MarkItDown.

Parameters:
- `url` (str): File URL or local path (PDF, Office, ZIP, etc.). Supports http/https URLs, `file://` URIs and absolute paths (required)
- `max_size_mb` (int): Maximum file size in MB (default: 100)
- `extract_all_from_zip` (bool): Extract all files from ZIP archives (default: True)
- `include_metadata` (bool): Include file metadata in response (default: True)
- `auto_summarize` (bool): Summarize large content with an LLM; only runs when explicitly `true` (default: False)
- `max_content_tokens` (int): Maximum tokens before summarization when `auto_summarize=true` (default: 15000)
- `summary_length` (str): Summary length, 'short', 'medium' or 'long' (default: 'medium')
- `llm_provider` (Optional[str]): LLM provider for summarization, auto-detected if not specified (default: None)
- `llm_model` (Optional[str]): Specific LLM model for summarization, auto-detected if not specified (default: None)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.md` file)

Supported formats:
- PDF: .pdf
- Microsoft Office: .docx, .pptx, .xlsx, .xls
- Archives: .zip
- Web/text: .html, .htm, .txt, .md, .csv, .rtf
- eBooks: .epub

`process_file` converts `.html`/`.htm` as files, while `crawl_url` and `crawl_url_with_fallback` render remote `.html`/`.htm` URLs in the browser.

### `get_supported_file_formats`

Returns the list of supported file formats and their capabilities. No parameters required.

### `enhanced_process_large_content`

Split large content into chunks, filter them, and optionally summarize them.

Parameters:
- `url` (str): URL to process (required)
- `chunking_strategy` (str): How to split content, 'topic', 'sentence', 'overlap' or 'regex' (default: 'sentence')
- `filtering_strategy` (str): How to filter chunks, 'bm25', 'pruning' or 'llm' (default: 'bm25')
- `filter_query` (Optional[str]): Keywords for BM25 filtering (default: None)
- `max_chunk_tokens` (int): Maximum tokens per chunk (default: 2000)
- `chunk_overlap` (int): Overlap tokens between consecutive chunks (default: 200)
- `extract_top_chunks` (int): Number of top-ranked chunks to return (default: 5)
- `similarity_threshold` (float): Minimum similarity score 0-1 for chunk inclusion (default: 0.5)
- `summarize_chunks` (bool): Summarize individual chunks before merging (default: False)
- `merge_strategy` (str): How to merge processed chunks, 'hierarchical' or 'linear' (default: 'linear')
- `final_summary_length` (str): Length of the final merged summary, 'short', 'medium' or 'long' (default: 'short')
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file with chunks and summaries)

## YouTube Processing Tools

### `extract_youtube_transcript`

Extract transcripts from YouTube videos with language preferences and optional translation.

Parameters:
- `url` (str): YouTube video URL (required)
- `languages` (Optional[List[str] | str]): Preferred languages in order of preference (default: ["ja", "en"])
- `translate_to` (Optional[str]): Target language for translation (default: None)
- `include_timestamps` (bool): Include timestamps in transcript (default: False)
- `preserve_formatting` (bool): Preserve original formatting (default: True)
- `include_metadata` (bool): Include video metadata (default: True)
- `auto_summarize` (bool): Summarize large transcripts with an LLM; only runs when explicitly `true` (default: False)
- `max_content_tokens` (int): Maximum tokens before summarization when `auto_summarize=true` (default: 15000)
- `summary_length` (str): Summary length, 'short', 'medium' or 'long' (default: 'medium')
- `llm_provider` (Optional[str]): LLM provider for summarization, auto-detected if not specified (default: None)
- `llm_model` (Optional[str]): Specific LLM model for summarization, auto-detected if not specified (default: None)
- `enable_crawl_fallback` (bool): Fall back to page crawl if transcript API fails (default: True)
- `fallback_timeout` (int): Timeout in seconds for crawl fallback (default: 60)
- `enrich_metadata` (bool): Enrich metadata (upload_date, view_count) via page crawl (default: True)
- `timezone` (str): IANA timezone name (e.g. 'Asia/Tokyo', 'America/New_York') used to convert the publish time into `metadata.published_at`; requires `enrich_metadata=True` (default: 'UTC')
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.md` file)

`metadata.published_at` holds the publish time converted to `timezone`; `metadata.published_at.original` always keeps the offset reported by YouTube.

### `extract_youtube_comments`

Extract and sort comments from a YouTube video.

Parameters:
- `url` (str): YouTube video URL (required)
- `sort_by` (str): Comment sort order, 'popular' or 'recent' (default: 'popular')
- `max_comments` (int): Maximum number of comments to retrieve, 1-1000 (default: 300)
- `comment_offset` (int): Number of comments to skip, for pagination (default: 0)
- `include_replies` (bool): Include replies to top-level comments (default: True)
- `content_offset` (int): Start position for content retrieval, 0-indexed (default: 0)
- `content_limit` (int): Maximum characters to return, 0 means unlimited (default: 0)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file with the full comment list, written before the token-limit reduction)

### `batch_extract_youtube_transcripts`

Extract transcripts from multiple YouTube videos in a single request (up to 3 videos).

Parameters:
- `request` (Dict): Request object with the following keys:
  - `urls` (List[str]): List of YouTube video URLs (max 3, required)
  - `languages` (List[str]): Preferred languages in order of preference (default: ["ja", "en"])
  - `include_timestamps` (bool): Include timestamps in transcripts (default: False)
  - `timezone` (str): IANA timezone name used to convert each video's publish time into `metadata.published_at`; `metadata.published_at.original` keeps YouTube's offset (default: 'UTC')
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a directory of per-video `.md` files + `index.json`)
  - Also accepted: `translate_to` (Optional[str], default: None), `preserve_formatting` (bool, default: True), `include_metadata` (bool, default: True), `max_concurrent` (int, default: 3, max 5)

### `get_youtube_video_info`

Get metadata and transcript information for a YouTube video.

Parameters:
- `video_url` (str): YouTube video URL (required)
- `summarize_transcript` (bool): Generate a summary of the transcript (default: False)
- `max_tokens` (int): Token limit for summarization (default: 25000)
- `llm_provider` (Optional[str]): LLM provider for summarization, auto-detected if not specified (default: None)
- `llm_model` (Optional[str]): Specific LLM model for summarization, auto-detected if not specified (default: None)
- `summary_length` (str): Summary length, 'short', 'medium' or 'long' (default: 'medium')
- `include_timestamps` (bool): Include timestamps in transcript output (default: True)
- `timezone` (str): IANA timezone name (e.g. 'Asia/Tokyo', 'America/New_York') used to convert the publish time returned under `published_at` (default: 'UTC')
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.md` file with video info + transcript)

Returns:
- Available transcript languages
- Manual/auto-generated distinction
- Translatable language information
- Publish time as `published_at` (converted to `timezone`; `published_at.original` keeps YouTube's offset)

## Search Tools

The search tools call a search engine API. Four engines are supported:

| Engine | `engine` value | API key | Notes |
|---|---|---|---|
| Google Custom Search JSON API | `google_custom` | `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` | Closed to new customers; Google ends the API on 2027-01-01 |
| Brave Search API | `brave` | `BRAVE_SEARCH_API_KEY` | Accepts query operators (`site:`, `filetype:`, `lang:`). At most 20 results per request; larger `num_results` values send one request per 20 results. Requests go out one at a time; an HTTP 429 whose `X-RateLimit-Reset` is 2 seconds or less is retried after that wait (at most twice) |
| Exa | `exa` | `EXA_API_KEY` | Reads the query as natural language. `site:` and `-site:` terms are sent as domain filters; other operators stay in the query text. Results include `published_date` and `author` when Exa returns them |
| SearXNG (self-hosted metasearch) | `searxng` | none; set `SEARXNG_URL` (an `http://` or `https://` URL) | Sends the query, with any operators, to the instance's engines (`SEARXNG_ENGINES`, default: the instance's own). Reads up to 5 result pages to reach `num_results` and drops duplicate URLs. Results include `source_engines`, `score` and `published_date` when SearXNG returns them. The instance must allow the `json` format (`search.formats` in `settings.yml`); otherwise it answers HTTP 403 and the error has a `suggestion` |

Engine selection:
- The `engine` request key picks the engine for one call. Without it, the `SEARCH_ENGINE` environment variable decides (default `auto`).
- `auto` tries the configured engines in the order set by `SEARCH_ENGINE_ORDER` (default `google_custom,brave,exa,searxng`). It moves to the next engine when a request fails (rate limit, rejected key, no credits, network error, 5xx). Zero results is a valid answer and does not move on. The response lists each try in `fallback_info.attempts`.
- A named engine that is not configured returns `error_code: search_engine_not_configured` before any network request. An unknown `engine` value returns `error_code: invalid_search_engine` with the `valid_engines` list.
- Every response has an `engine` key with the engine that answered.

How the common keys map to each engine:

| Key | `google_custom` | `brave` | `exa` | `searxng` |
|---|---|---|---|---|
| `language` | `hl` | `search_lang` | not used | `language`, combined with a two-letter `region` (for example `en-US`). A language that already has a region part (`zh-tw`) or a region such as `all` sends the language alone |
| `region` | `gl` | `country` (`uk` becomes `GB`; codes Brave does not list become `ALL`) | `userLocation` | part of `language` (`uk` becomes `GB`) |
| `recent_days` | `dateRestrict` | `freshness` (see below) | `startPublishedDate`, exactly N days before now | `time_range` (see below) |
| `safe_search` | `safe=active` | `safesearch=moderate` | `moderation` | `safesearch=1` |

For `brave`, `recent_days` of 1, 7, 31 and 365 use Brave's rolling windows (`pd`, `pw`, `pm`, `py`: the past 24 hours, 7 days, 31 days and 365 days). Other values become a date range in UTC from N days ago through today. Whole days are counted, so results can be up to one day older than N days.

For `searxng`, `recent_days` becomes the smallest SearXNG `time_range` that covers it: 1 is `day`, 2-7 `week`, 8-31 `month`, 32-365 `year`. Values over 365 apply no date filter. The range used is returned as `time_range`. When some of the engines behind SearXNG fail, the response lists them in `unresponsive_engines` (for example `"google: timeout"`). Zero results are `no_results` even when some engines are listed there. Only when `SEARXNG_ENGINES` is set and every engine in it is unresponsive is the result a failure that `auto` moves past.

### `search_web`

Run a web search and return the results with titles, snippets, URLs and metadata.

Parameters:
- `request` (Dict): Request object with the following keys:
  - `query` (str): Search query string (required)
  - `num_results` (int): Number of results to return, 1-100 (default: 10)
  - `search_genre` (Optional[str]): Genre filter, one of `pdf`, `documents`, `presentations`, `spreadsheets`, `japanese`, `english` (default: None)
  - `engine` (Optional[str]): `auto`, `google_custom`, `brave`, `exa` or `searxng` (default: the `SEARCH_ENGINE` setting)
  - `result_type` (str or List[str]): `web`, `discussions`, `faq`, `videos`, `news`, or several of them as a list or a comma-separated string (default: "web"). See below
  - `language` (str): Search language (default: "en")
  - `region` (str): Search region (default: "us")
  - `recent_days` (Optional[int]): Limit results to the past N days (default: None)
  - `content_limit` (int): Maximum characters to return per result, 0 means unlimited (default: 0)
  - `content_offset` (int): Start position for content retrieval (default: 0)
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file with the full results, written before slicing)
  - Also accepted: `safe_search` (bool, default: True)

Features:
- Each result has `rank`, `url`, `domain`, `title`, `snippet` and a URL `type` (for example `academic`, `documentation`, `video`)
- 6 search genres: `pdf`, `documents`, `presentations`, `spreadsheets` (file type filters), `japanese`, `english` (language filters). `google_custom`, `brave` and `searxng` add the genre's operator to the query (SearXNG passes it to its engines). `exa` has no file type filter: naming `exa` with a file type genre returns `error_code: unsupported_search_genre`, and `auto` skips `exa` for those genres. With `exa`, `japanese` sets the location to JP and `english` has no effect
- `search_genre` matching is case-insensitive. Any other value returns `success: false` with `error_code: invalid_search_genre` and the `valid_genres` list, before any network request. `batch_search_web` and `search_and_crawl` check `search_genre` and `engine` the same way
- When an engine returns zero results, the response suggests shorter queries in `alternative_queries`
- Safe search is on by default
- `result_type` picks Brave result sections besides web results: `discussions` (forum posts such as Reddit threads and GitHub issues, with `forum_name`, `num_answers`, `score`, `question` and `top_comment`), `faq` (question and answer excerpts from pages, with `question` and the answer as `snippet`), `videos` (with `duration`, `creator`, `publisher`) and `news` (with `source` and `age`). `web`, `discussions`, `faq` and `videos` come from one Brave web search request. `news` comes from a separate request to Brave's News Search endpoint (up to 50 articles per request), because the news section of web search is empty for most queries that name a specific event or a concept. News Search returns dated articles in general, so blog posts and documentation pages can appear too; for Japanese news pass `region: jp`. `results` holds the web results when `web` is requested, otherwise the first type listed; the other types go to `extra_results` keyed by type (an empty list when Brave returned none). Forum post and answer text is cut at 1,000 characters. If the news request fails while `news` is an extra type, the other results are still returned and the error is in `extra_errors`. An empty list in `extra_results` means Brave returned nothing for that type; a failed request always shows up in `extra_errors`
- Only `brave` returns types other than `web`: `auto` uses Brave alone, and naming `google_custom`, `exa` or `searxng` returns `error_code: unsupported_result_type`. Brave returns `discussions`, `faq` and `videos` for some queries only, `num_results` can only shorten them, and paging past 20 results adds web results only. When none of the requested sections has results, the response is `success: false` with a suggestion to reword or use `web`. An unknown type returns `error_code: invalid_result_type` before any network request

### `search_google`

Deprecated alias of `search_web`. Its `request` dict takes the same keys (`query`, `num_results`, `search_genre`, `engine`, `result_type`, `language`, `region`, `recent_days`, `content_limit`, `content_offset`, `output_path`, `include_content_in_response`, `overwrite`) and returns the same results. The engine is chosen as for `search_web`, so it is not limited to Google. Kept so that existing clients keep working.

### `search_and_crawl`

Run a web search and crawl the top results.

Parameters:
- `request` (Dict): Request object with the following keys:
  - `search_query` (str): Search query (required). It is sent to the engine as written
  - `crawl_top_results` (int): Number of top results to crawl, max 3 (default: 2)
  - `search_genre` (Optional[str]): Genre filter, same values as `search_web` (default: None)
  - `engine` (Optional[str]): Search engine, same values as `search_web` (default: the `SEARCH_ENGINE` setting)
  - `result_type` (str or List[str]): Result types, as in `search_web` (default: "web"). The pages in `results` are crawled (the forum threads with `discussions` alone); `extra_results` is returned as is
  - `recent_days` (Optional[int]): Limit search results to the past N days (default: None)
  - `generate_markdown` (bool): Generate markdown content from crawled pages (default: True)
  - `max_content_per_page` (int): Maximum content characters per crawled page in the response (default: 5000)
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a directory of per-page `.md` files + `index.json`, written before `max_content_per_page` truncation)

Returns:
- `engine`, `result_type`, search metadata, `search_results` and `extra_results` (`search_fallback_info` when `auto` moved past an engine)
- `crawled_pages`: one entry per crawled page with `search_rank`, `url`, `title`, `success`, `markdown` (or `content` when `generate_markdown` is false), `content_length`, `error` for failed pages, and `search_result` (the search hit). A page that fails or comes back empty is crawled again with the fallback crawler and marked `fallback_used`
- Success rates and processing statistics
- A combined summary of the search and crawl results

### `batch_search_web`

Run up to 3 web searches concurrently.

Parameters:
- `request` (Dict): Request object with the following keys:
  - `queries` (List[str]): List of search query strings (max 3, required)
  - `num_results_per_query` (int): Number of results per query, 1-100 (default: 10)
  - `search_genre` (Optional[str]): Genre filter applied to all queries, same values as `search_web` (default: None)
  - `engine` (Optional[str]): Search engine for all queries, same values as `search_web` (default: the `SEARCH_ENGINE` setting)
  - `result_type` (str or List[str]): Result types for all queries, as in `search_web` (default: "web"). Each entry in `results` carries its own `extra_results`
  - `recent_days` (Optional[int]): Limit results to the past N days (default: None)
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file with the full result set)
  - Also accepted: `language` (str, default: "en"), `region` (str, default: "us"), `max_concurrent` (int, default: 3, max 5)

Each entry in `results` has its own `engine` (and `fallback_info` when `auto` moved past an engine), so in `auto` mode different queries can be answered by different engines.

### `batch_search_google`

Deprecated alias of `batch_search_web`. Its `request` dict takes the same keys (`queries`, `num_results_per_query`, `search_genre`, `engine`, `result_type`, `recent_days`, `output_path`, `include_content_in_response`, `overwrite`) and returns the same results.

### `get_search_genres`

Returns the supported `search_genre` values and the search engine configuration. No parameters required.

Response keys:
- `genres`: genre name to description
- `query_operators`: genre name to the operator added to the query
- `configuration`: `default_engine`, `auto_order`, `configured_engines` (engines with an API key or, for `searxng`, `SEARXNG_URL`), `engines`, `result_types`, `genre_support` (how each engine applies genres), and the supported languages and regions
- `total_genres`, `usage_example`, and `notes`

## Batch Operations

### `batch_crawl`

Crawl multiple URLs in a single request (up to 3 URLs), processed sequentially with fallback.

Parameters:
- `urls` (List[str]): List of URLs to crawl (max 3, required)
- `base_timeout` (int): Timeout in seconds per URL (default: 30)
- `generate_markdown` (bool): Generate markdown from page content (default: True)
- `extract_media` (bool): Extract images and videos from pages (default: False)
- `wait_for_js` (bool): Wait for JavaScript to finish executing (default: False)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a directory of per-URL `.md` files + `index.json`; the list return shape is kept and each success item gains `output_file`)

### `multi_url_crawl`

Crawl multiple URLs with per-URL configuration options (up to 5 URLs).

Parameters:
- `url_configurations` (Dict): Per-URL configuration map, max 5 URLs, e.g. `{"https://site1.com": {"wait_for_js": true}}` (required)
- `pattern_matching` (str): URL pattern matching mode, 'wildcard' or 'regex' (default: 'wildcard')
- `default_config` (Optional[Dict]): Default configuration applied to all URLs (default: None)
- `base_timeout` (int): Timeout in seconds per URL (default: 30)
- `max_concurrent` (int): Has no effect; URLs are processed sequentially. Kept for compatibility (default: 3)
- `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a directory of per-URL `.md` files + `index.json`; the list return shape is kept and each success item gains `output_file`)

## Tool Categories

### By Complexity
- Simple: `crawl_url`, `process_file`, `get_youtube_video_info`, `get_search_genres`, `get_supported_file_formats`
- Moderate: `deep_crawl_site`, `search_web`, `extract_youtube_transcript`, `extract_youtube_comments`, `batch_crawl`
- Multi-step or LLM-backed: `extract_structured_data`, `intelligent_extract`, `extract_entities`, `search_and_crawl`, `batch_search_web`, `multi_url_crawl`, `batch_extract_youtube_transcripts`, `enhanced_process_large_content`

### By Use Case
- Content discovery: `search_web`, `search_and_crawl`, `batch_search_web`, `get_search_genres`
- Data extraction: `crawl_url`, `extract_structured_data`, `intelligent_extract`, `extract_entities`
- Media processing: `extract_youtube_transcript`, `extract_youtube_comments`, `batch_extract_youtube_transcripts`, `process_file`, `get_supported_file_formats`
- Site analysis: `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`, `multi_url_crawl`
- Large content: `enhanced_process_large_content`, `get_youtube_video_info`

## Integration Examples

For configuration examples, see [Configuration Examples](CONFIGURATION_EXAMPLES.md).

For HTTP API access, see the [HTTP Integration Guide](HTTP_INTEGRATION.md).

For more usage patterns, see the [Usage Patterns](ADVANCED_USAGE.md).
