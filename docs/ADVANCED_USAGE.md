# Advanced Usage Guide

Advanced patterns, techniques, and workflows for power users of the Crawl4AI MCP Server.

## 🚁 Advanced Web Crawling Techniques

### Deep Site Exploration

**Multi-depth crawling with strategic filtering:**
```json
{
  "url": "https://docs.example.com",
  "max_depth": 3,
  "crawl_strategy": "best_first",
  "url_pattern": "*docs*",
  "score_threshold": 0.7,
  "content_filter": "bm25",
  "filter_query": "API documentation tutorial guide"
}
```

**JavaScript-heavy sites optimization:**
```json
{
  "url": "https://spa-application.com",
  "wait_for_js": true,
  "simulate_user": true,
  "timeout": 60,
  "wait_for_selector": ".content-loaded",
  "execute_js": "window.scrollTo(0, document.body.scrollHeight); await new Promise(r => setTimeout(r, 2000));",
  "headers": {
    "User-Agent": "Mozilla/5.0 (compatible; CrawlBot/1.0)"
  }
}
```

### Fallback Strategies for Difficult Sites

**Multi-strategy crawling with automatic fallbacks:**
```json
{
  "url": "https://protected-site.com",
  "cookies": {
    "session_id": "your_session_cookie",
    "auth_token": "bearer_token"
  },
  "headers": {
    "Authorization": "Bearer your-api-key",
    "Accept": "text/html,application/xhtml+xml"
  },
  "user_agent": "CustomBot/1.0",
  "simulate_user": true,
  "timeout": 90
}
```

**Using crawl_url_with_fallback for maximum reliability:**
- Automatically tries multiple strategies
- Handles anti-bot protection
- Provides detailed failure analysis
- Returns partial results on timeout

## 🧠 AI-Powered Content Processing

### Intelligent Content Extraction

**Semantic understanding with custom instructions:**
```json
{
  "url": "https://research-paper.com",
  "extraction_goal": "Extract methodology, results, and conclusions from academic paper",
  "content_filter": "llm",
  "custom_instructions": "Focus on quantitative results, statistical significance, and practical implications. Ignore references and author information.",
  "use_llm": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4"
}
```

**Multi-provider LLM configuration:**
```json
{
  "mcpServers": {
    "crawl4ai-multi-llm": {
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

### Auto-Summarization for Large Content

**Configuration for different content types:**
```json
{
  "auto_summarize": true,
  "max_content_tokens": 50000,
  "summary_length": "long",
  "llm_provider": "anthropic",
  "chunk_content": true,
  "chunk_strategy": "topic"
}
```

**Summary length optimization:**
- `"short"`: 1-2 paragraphs (500-1000 tokens)
- `"medium"`: 3-5 paragraphs (1500-3000 tokens)
- `"long"`: Comprehensive analysis (5000-10000 tokens)

## 🔄 Batch Processing Operations

### Large-Scale Crawling Workflows

**Batch crawling with URL management:**
```json
{
  "urls": [
    "https://site1.com/page1",
    "https://site1.com/page2",
    "https://site2.com/api-docs",
    "https://site3.com/tutorials"
  ],
  "config": {
    "generate_markdown": true,
    "extract_media": false,
    "timeout": 45,
    "auto_summarize": true
  },
  "base_timeout": 60
}
```

**YouTube transcript batch processing:**
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
    "max_concurrent": 3,
    "translate_to": "en"
  }
}
```

### Search-and-Crawl Workflows

**Competitive analysis pipeline:**
```json
{
  "search_query": "enterprise API documentation best practices 2024",
  "num_search_results": 10,
  "crawl_top_results": 5,
  "search_genre": "technical",
  "extract_media": false,
  "generate_markdown": true
}
```

**Content research workflow:**
```json
{
  "request": {
    "queries": [
      "machine learning trends 2024",
      "AI adoption enterprise survey",
      "deep learning frameworks comparison"
    ],
    "num_results_per_query": 15,
    "search_genre": "academic",
    "max_concurrent": 2
  }
}
```

## 🎯 Complex Workflow Patterns

### Multi-Stage Content Analysis

**1. Discovery Phase:**
```bash
# Search for relevant content
search_google: "topic keywords research papers"
```

**2. Collection Phase:**
```bash
# Crawl top results with fallback protection
search_and_crawl: top 5 results → generate comprehensive analysis
```

**3. Analysis Phase:**
```bash
# Extract structured insights
extract_structured_data: "methodology, findings, implications"
```

**4. Synthesis Phase:**
```bash
# Batch process related content
batch_crawl: related URLs → auto-summarize → comparative analysis
```

### Site Documentation Mapping

**Complete documentation extraction:**
```json
{
  "url": "https://api-docs.example.com",
  "max_depth": 4,
  "crawl_strategy": "bfs",
  "url_pattern": "*docs*,*api*,*guide*",
  "extract_media": true,
  "content_filter": "bm25",
  "filter_query": "API endpoint parameter example tutorial",
  "auto_summarize": false,
  "generate_markdown": true
}
```

### Entity Extraction Workflows

**Contact information mining:**
```json
{
  "url": "https://company-directory.com",
  "entity_types": ["emails", "phones", "social_media"],
  "deduplicate": true,
  "include_context": true,
  "custom_patterns": {
    "linkedin": "linkedin\\.com/in/[\\w-]+",
    "github": "github\\.com/[\\w-]+",
    "departments": "department:\\s*([\\w\\s]+)"
  }
}
```

## 🔧 Performance Optimization

### Memory and Processing Management

**Large document handling:**
```json
{
  "chunk_content": true,
  "chunk_size": 8000,
  "chunk_strategy": "topic",
  "overlap_rate": 0.1,
  "max_content_tokens": 100000,
  "auto_summarize": true
}
```

**Timeout optimization strategies:**
```json
{
  "timeout": 120,
  "base_timeout": 45,
  "wait_for_js": true,
  "simulate_user": false,
  "cache_mode": "enabled"
}
```

### Concurrent Processing Control

**Batch operation limits:**
- `max_concurrent`: 3 (default, stable)
- `max_concurrent`: 5 (maximum, may impact stability)
- `max_concurrent`: 1 (sequential, most reliable)

**Rate limiting considerations:**
- YouTube API: 3 concurrent requests
- Google Search: 2 concurrent queries
- General crawling: 5 concurrent URLs

## 🛡️ Error Handling and Resilience

### Robust Configuration Patterns

**Production-ready setup with error recovery:**
```json
{
  "mcpServers": {
    "crawl4ai-production": {
      "transport": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/walksoda/crawl-mcp",
        "crawl-mcp"
      ],
      "env": {
        "CRAWL4AI_LANG": "en",
        "FASTMCP_LOG_LEVEL": "ERROR",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Fallback strategy implementation:**
1. **Primary attempt**: Standard crawl_url
2. **Fallback 1**: crawl_url_with_fallback
3. **Fallback 2**: Simplified parameters (no JS, basic extraction)
4. **Final fallback**: Manual URL validation and basic HTTP fetch

### Monitoring and Diagnostics

**Health check configuration:**
```json
{
  "url": "http://127.0.0.1:8000/health",
  "timeout": 5000,
  "expected_status": 200,
  "check_interval": 30000
}
```

**Debug configuration for troubleshooting:**
```json
{
  "env": {
    "FASTMCP_LOG_LEVEL": "DEBUG",
    "DEBUG": "1",
    "PYTHONUNBUFFERED": "1",
    "PLAYWRIGHT_DEBUG": "1"
  }
}
```

## 🎨 Advanced Content Filtering

### Multi-Stage Filtering Pipeline

**BM25 + LLM combination:**
```json
{
  "content_filter": "bm25",
  "filter_query": "technical documentation API reference",
  "use_llm": true,
  "custom_instructions": "After BM25 filtering, apply semantic analysis to extract only actionable technical information"
}
```

**Hierarchical content processing:**
```json
{
  "filtering_strategy": "llm",
  "extract_top_chunks": 15,
  "similarity_threshold": 0.8,
  "chunking_strategy": "topic",
  "merge_strategy": "hierarchical"
}
```

## 📊 Analytics and Reporting

### Advanced Metrics Collection

**Performance tracking:**
- Processing time per URL
- Success/failure rates
- Token usage by operation
- Cache hit ratios
- Memory usage patterns

**Content quality metrics:**
- Extraction completeness
- Relevance scoring
- Duplicate detection rates
- Format conversion success

### Custom Reporting Workflows

**Comprehensive site analysis:**
```bash
1. deep_crawl_site → structural mapping
2. extract_structured_data → content categorization
3. batch_crawl → multi-page extraction
4. multi_url_crawl → comparative analysis
5. generate_report → structured output
```

## 🔗 Integration Patterns

### API Integration Examples

**Python integration with error handling:**
```python
import asyncio
import logging
from typing import Optional, Dict, Any

async def robust_crawl_workflow(url: str, max_retries: int = 3) -> Optional[Dict[Any, Any]]:
    """Advanced crawling with automatic fallbacks and retry logic."""
    
    strategies = [
        {"wait_for_js": True, "timeout": 60},
        {"wait_for_js": False, "timeout": 30},
        {"simulate_user": False, "timeout": 15}
    ]
    
    for attempt, strategy in enumerate(strategies):
        try:
            result = await crawl_url(url, **strategy)
            if result.get("success"):
                return result
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < len(strategies) - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
    # Final fallback using crawl_url_with_fallback
    try:
        return await crawl_url_with_fallback(url)
    except Exception as e:
        logging.error(f"All crawling strategies failed for {url}: {e}")
        return None
```

**Node.js batch processing:**
```javascript
async function processUrlBatch(urls, options = {}) {
  const batchSize = options.batchSize || 5;
  const results = [];
  
  for (let i = 0; i < urls.length; i += batchSize) {
    const batch = urls.slice(i, i + batchSize);
    
    try {
      const batchResults = await Promise.allSettled(
        batch.map(url => crawlUrl(url, options))
      );
      
      results.push(...batchResults);
      
      // Rate limiting delay
      if (i + batchSize < urls.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error) {
      console.error(`Batch processing error:`, error);
    }
  }
  
  return results;
}
```

## 🚨 Advanced Troubleshooting

### Complex Issue Resolution

**Memory management for large operations:**
```json
{
  "chunk_content": true,
  "max_content_tokens": 25000,
  "auto_summarize": true,
  "cache_mode": "bypass",
  "timeout": 300
}
```

**Network and connectivity issues:**
```json
{
  "user_agent": "Mozilla/5.0 (compatible; Bot/1.0)",
  "headers": {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
  },
  "cookies": {},
  "simulate_user": true,
  "timeout": 90
}
```

### Performance Profiling

**Identifying bottlenecks:**
1. Enable DEBUG logging
2. Monitor token usage patterns
3. Track processing times by operation
4. Analyze memory usage during batch operations
5. Profile network request patterns

## 📚 Related Documentation

- **Configuration Examples**: [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **HTTP Integration**: [HTTP_INTEGRATION.md](HTTP_INTEGRATION.md)
- **Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)

## 💡 Pro Tips

1. **Start simple, scale complexity** - Begin with basic crawling, add features incrementally
2. **Monitor resource usage** - Track memory and processing time for optimization
3. **Use appropriate tools** - Match tool complexity to task requirements
4. **Implement fallbacks** - Always have backup strategies for critical workflows
5. **Cache strategically** - Balance performance with content freshness
6. **Test thoroughly** - Validate complex workflows in development environment
7. **Document patterns** - Record successful configurations for reuse
8. **Monitor API limits** - Respect rate limits and quotas for external services