# API Reference

Reference for the MCP tools provided by the Crawl4AI MCP Server.

## Tool Selection Guide

### Choose the Right Tool for Your Task

| Use case | Tool | What it does |
|-------------|---------------------|------------------|
| Single webpage | `crawl_url` | Basic crawling, JS support |
| Multiple pages (up to 10) | `deep_crawl_site` | Site mapping, link following |
| Search + Crawling | `search_and_crawl` | Google search + auto-crawl |
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
| Batch search | `batch_search_google` | Up to 3 queries concurrently |
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

For request-dict tools (`search_google`, `batch_search_google`, `search_and_crawl`, `batch_extract_youtube_transcripts`) these three are keys of the `request` dict instead of top-level parameters.

Behavior:
- Writes are atomic (temp file + `os.replace`); parent directories are created automatically.
- The full, unsliced payload is written before any slicing or truncation, so the file is complete even when the response is sliced.
- Each markdown file starts with a YAML frontmatter block (`url`, `title`, `fetched_at`, `source_tool`).
- Batch (directory) tools skip per-item persistence for items that report `success=false`; those items still appear in `index.json` with a null output path (`file: null` for `deep_crawl_site`/`search_and_crawl`/`batch_extract_youtube_transcripts`, `output_file: null` for `batch_crawl`/`multi_url_crawl`).

Tools that support `output_path`:
- Single-file (`output_path` is a file): `crawl_url`, `crawl_url_with_fallback`, `extract_structured_data`, `intelligent_extract`, `extract_entities`, `process_file`, `enhanced_process_large_content`, `extract_youtube_transcript`, `extract_youtube_comments`, `get_youtube_video_info`, `search_google`, `batch_search_google`
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

## Google Search Tools

### `search_google`

Perform Google search with genre filtering and metadata extraction.

Parameters:
- `request` (Dict): Request object with the following keys:
  - `query` (str): Search query string (required)
  - `num_results` (int): Number of results to return, 1-100 (default: 10)
  - `search_genre` (Optional[str]): Genre filter, one of `pdf`, `documents`, `presentations`, `spreadsheets`, `japanese`, `english` (default: None)
  - `language` (str): Search language (default: "en")
  - `region` (str): Search region (default: "us")
  - `recent_days` (Optional[int]): Limit results to the past N days (default: None)
  - `content_limit` (int): Maximum characters to return per result, 0 means unlimited (default: 0)
  - `content_offset` (int): Start position for content retrieval (default: 0)
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file with the full results, written before slicing)
  - Also accepted: `safe_search` (bool, default: True)

Features:
- Automatic title and snippet extraction from search results
- 6 search genres that add Google operators to the query: `pdf`, `documents`, `presentations`, `spreadsheets` (file type filters), `japanese`, `english` (language filters)
- `search_genre` matching is case-insensitive. Any other value returns `success: false` with `error_code: invalid_search_genre` and the `valid_genres` list, before any network request. `batch_search_google` and `search_and_crawl` check `search_genre` the same way
- URL classification and domain analysis
- Safe search enforced by default

### `search_and_crawl`

Perform Google search and automatically crawl top results.

Parameters:
- `request` (Dict): Request object with the following keys:
  - `search_query` (str): Google search query (required)
  - `crawl_top_results` (int): Number of top results to crawl, max 3 (default: 2)
  - `search_genre` (Optional[str]): Genre filter, same values as `search_google` (default: None)
  - `recent_days` (Optional[int]): Limit search results to the past N days (default: None)
  - `generate_markdown` (bool): Generate markdown content from crawled pages (default: True)
  - `max_content_per_page` (int): Maximum content characters per crawled page in the response (default: 5000)
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a directory of per-page `.md` files + `index.json`, written before `max_content_per_page` truncation)

Returns:
- Search metadata and crawled content
- Success rates and processing statistics
- A combined summary of the search and crawl results

### `batch_search_google`

Perform multiple Google searches concurrently (up to 3 queries).

Parameters:
- `request` (Dict): Request object with the following keys:
  - `queries` (List[str]): List of search query strings (max 3, required)
  - `num_results_per_query` (int): Number of results per query, 1-100 (default: 10)
  - `search_genre` (Optional[str]): Genre filter applied to all queries, same values as `search_google` (default: None)
  - `recent_days` (Optional[int]): Limit results to the past N days (default: None)
  - `output_path`, `include_content_in_response`, `overwrite`: see [Result Persistence](#result-persistence-output_path) (saves a single `.json` file with the full result set)
  - Also accepted: `language` (str, default: "en"), `region` (str, default: "us"), `max_concurrent` (int, default: 3, max 5)

### `get_search_genres`

Returns the supported `search_genre` values. No parameters required.

Response keys:
- `genres`: genre name to description
- `query_operators`: genre name to the operator added to the query
- `total_genres`, `usage_example`, `configuration` (search mode, supported languages and regions), and `notes`

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
- Moderate: `deep_crawl_site`, `search_google`, `extract_youtube_transcript`, `extract_youtube_comments`, `batch_crawl`
- Multi-step or LLM-backed: `extract_structured_data`, `intelligent_extract`, `extract_entities`, `search_and_crawl`, `batch_search_google`, `multi_url_crawl`, `batch_extract_youtube_transcripts`, `enhanced_process_large_content`

### By Use Case
- Content discovery: `search_google`, `search_and_crawl`, `batch_search_google`, `get_search_genres`
- Data extraction: `crawl_url`, `extract_structured_data`, `intelligent_extract`, `extract_entities`
- Media processing: `extract_youtube_transcript`, `extract_youtube_comments`, `batch_extract_youtube_transcripts`, `process_file`, `get_supported_file_formats`
- Site analysis: `deep_crawl_site`, `crawl_url_with_fallback`, `batch_crawl`, `multi_url_crawl`
- Large content: `enhanced_process_large_content`, `get_youtube_video_info`

## Integration Examples

For configuration examples, see [Configuration Examples](CONFIGURATION_EXAMPLES.md).

For HTTP API access, see the [HTTP Integration Guide](HTTP_INTEGRATION.md).

For more usage patterns, see the [Usage Patterns](ADVANCED_USAGE.md).
