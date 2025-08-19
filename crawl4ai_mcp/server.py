"""
Crawl4AI MCP Server - Unofficial Implementation

An unofficial Model Context Protocol server that provides advanced web crawling
and content extraction capabilities by wrapping the crawl4ai library.

🔗 Original Library: https://github.com/unclecode/crawl4ai by unclecode
🔗 This MCP Wrapper: https://github.com/walksoda/crawl-mcp by walksoda

⚠️  This is NOT an official crawl4ai project.
    This is a third-party implementation for MCP integration.

Author: walksoda
License: MIT (see LICENSE file)
Status: Unofficial third-party wrapper
"""

import os
import sys
import warnings

# Set environment variables before any imports
os.environ["FASTMCP_QUIET"] = "1"
os.environ["FASTMCP_NO_BANNER"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

# Prevent bash job control errors completely
os.environ["TERM"] = "dumb"
os.environ["SHELL"] = "/bin/sh"

# Redirect all output streams to devnull immediately
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, 2)  # stderr
os.close(devnull_fd)

# Suppress all warnings as early as possible
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Additional stderr suppression
sys.stderr = open(os.devnull, 'w')
sys.stdout = sys.stdout  # Keep stdout for MCP communication

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Annotated
from pydantic import BaseModel, Field

# Ensure logging is completely disabled
logging.disable(logging.CRITICAL)
logging.getLogger().disabled = True

# Additional Pydantic warning suppression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*class-based.*config.*deprecated.*")
warnings.filterwarnings("ignore", module="pydantic.*")
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler
from crawl4ai import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
    BM25ContentFilter,
    PruningContentFilter,
    LLMContentFilter,
    CacheMode,
)
from crawl4ai.chunking_strategy import (
    TopicSegmentationChunking,
    OverlappingWindowChunking,
    RegexChunking
)

# Custom sentence chunking implementation to replace problematic NlpSentenceChunking
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import re

# Import prompts from the dedicated module
from .prompts import (
    crawl_website_prompt,
    analyze_crawl_results_prompt,
    batch_crawl_setup_prompt,
    process_file_prompt
)

# Import models from the dedicated modules
from .models import (
    CrawlRequest,
    CrawlResponse,
    LargeContentRequest,
    LargeContentResponse,
    FileProcessRequest,
    FileProcessResponse,
    YouTubeTranscriptRequest,
    YouTubeTranscriptResponse,
    YouTubeBatchRequest,
    YouTubeBatchResponse,
    GoogleSearchRequest,
    GoogleSearchResponse,
    GoogleBatchSearchRequest,
    GoogleBatchSearchResponse,
    StructuredExtractionRequest
)

class CustomSentenceChunking:
    """Custom sentence-based chunking implementation to replace NlpSentenceChunking"""
    
    def __init__(self, max_sentences_per_chunk: int = 5):
        self.max_sentences_per_chunk = max_sentences_per_chunk
        try:
            # Ensure NLTK data is available
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
    
    def chunk(self, text: str) -> list:
        """Split text into sentence-based chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks


class CosineSimilarityFilter:
    """
    Advanced semantic filtering using OpenAI embeddings and cosine similarity.
    Selects chunks with highest semantic relevance to the query.
    """
    
    def __init__(self, query: str, similarity_threshold: float = 0.7, max_chunks: int = 10):
        self.query = query
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        self.client = openai.OpenAI()
    
    def get_embedding(self, text: str) -> list:
        """Get OpenAI embedding for text"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text.replace("\n", " ")[:8000]  # Limit input length
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 1536  # Default embedding size
    
    def filter_chunks(self, chunks: list) -> list:
        """Filter chunks based on cosine similarity to query"""
        if not chunks or not self.query:
            return chunks
        
        # Get query embedding
        query_embedding = self.get_embedding(self.query)
        if not any(query_embedding):
            return chunks  # Fallback if embedding fails
        
        # Calculate similarities for each chunk
        chunk_similarities = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            chunk_embedding = self.get_embedding(chunk_text)
            
            if any(chunk_embedding):
                similarity = cosine_similarity(
                    [query_embedding], [chunk_embedding]
                )[0][0]
                chunk_similarities.append((i, similarity))
        
        # Sort by similarity and filter
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select chunks above threshold and within max_chunks limit
        selected_indices = []
        for idx, similarity in chunk_similarities:
            if similarity >= self.similarity_threshold and len(selected_indices) < self.max_chunks:
                selected_indices.append((idx, similarity))
        
        # Return filtered chunks with similarity scores
        filtered_chunks = []
        for idx, similarity in selected_indices:
            chunk = chunks[idx]
            if isinstance(chunk, dict):
                chunk["relevance_score"] = similarity
            else:
                chunk = {"content": str(chunk), "relevance_score": similarity}
            filtered_chunks.append(chunk)
        
        return filtered_chunks


class AdaptiveContentAnalyzer:
    """
    Analyzes content characteristics to determine optimal processing strategies.
    """
    
    def __init__(self):
        self.content_stats = {}
    
    def analyze_content(self, content: str, url: str = "") -> dict:
        """Analyze content to determine optimal strategies"""
        stats = {
            "length": len(content),
            "word_count": len(content.split()),
            "paragraph_count": content.count('\n\n') + 1,
            "sentence_count": len(sent_tokenize(content)) if content else 0,
            "has_html_structure": bool('<h' in content.lower() or '<div' in content.lower()),
            "has_numbered_sections": bool(re.search(r'\n\d+\.', content)),
            "has_bullet_points": bool('•' in content or '*' in content),
            "url_type": self._classify_url(url),
            "language_complexity": self._estimate_complexity(content)
        }
        
        self.content_stats = stats
        return stats
    
    def _classify_url(self, url: str) -> str:
        """Classify URL type for strategy selection"""
        url_lower = url.lower()
        if 'wikipedia' in url_lower:
            return 'encyclopedia'
        elif 'docs.' in url_lower or '/docs/' in url_lower:
            return 'documentation'
        elif '.pdf' in url_lower:
            return 'document'
        elif 'blog' in url_lower or 'medium.com' in url_lower:
            return 'article'
        elif 'news' in url_lower:
            return 'news'
        else:
            return 'general'
    
    def _estimate_complexity(self, content: str) -> str:
        """Estimate content complexity based on structure"""
        if len(content) > 50000:
            return 'high'
        elif len(content) > 10000:
            return 'medium' 
        else:
            return 'low'
    
    def recommend_chunking_strategy(self) -> str:
        """Recommend optimal chunking strategy based on content analysis"""
        stats = self.content_stats
        
        # Decision logic based on content characteristics
        if stats.get('has_html_structure') and stats.get('url_type') == 'documentation':
            return 'regex'  # Best for structured docs
        elif stats.get('url_type') == 'encyclopedia' or stats.get('language_complexity') == 'high':
            return 'topic'  # Best for complex content
        elif stats.get('sentence_count', 0) > 100:
            return 'sentence'  # Good for narrative content
        elif stats.get('length', 0) > 20000:
            return 'overlap'  # Good for very long content
        else:
            return 'topic'  # Default fallback
    
    def recommend_filtering_strategy(self, has_query: bool = False) -> str:
        """Recommend optimal filtering strategy based on content analysis"""
        stats = self.content_stats
        
        # Decision logic for filtering
        if has_query and stats.get('language_complexity') in ['medium', 'high']:
            return 'cosine'  # Best for semantic understanding
        elif has_query and stats.get('url_type') in ['documentation', 'article']:
            return 'bm25'  # Good for keyword-based filtering
        elif stats.get('length', 0) > 30000:
            return 'pruning'  # Good for very large content
        elif has_query:
            return 'bm25'  # Default with query
        else:
            return 'pruning'  # Default without query


class AdaptiveChunking:
    """
    Adaptive chunking that automatically selects the best strategy based on content.
    """
    
    def __init__(self):
        self.analyzer = AdaptiveContentAnalyzer()
        self.strategies = {
            'topic': lambda: TopicSegmentationChunking(num_keywords=5),
            'sentence': lambda: CustomSentenceChunking(max_sentences_per_chunk=5),
            'overlap': lambda max_tokens, overlap: OverlappingWindowChunking(
                window_size=max_tokens, overlap=overlap
            ),
            'regex': lambda: RegexChunking(
                patterns=[r'\n\n', r'\n#{1,6}\s', r'\n\d+\.', r'\n[A-Z][^.]*:']
            )
        }
    
    def get_optimal_strategy(self, content: str, url: str = "", 
                           max_chunk_tokens: int = 8000, chunk_overlap: int = 500):
        """Get optimal chunking strategy for the content"""
        self.analyzer.analyze_content(content, url)
        strategy_name = self.analyzer.recommend_chunking_strategy()
        
        if strategy_name == 'overlap':
            return self.strategies[strategy_name](max_chunk_tokens, chunk_overlap), strategy_name
        else:
            return self.strategies[strategy_name](), strategy_name


class AdaptiveFiltering:
    """
    Adaptive filtering that automatically selects the best strategy based on content.
    """
    
    def __init__(self):
        self.analyzer = AdaptiveContentAnalyzer()
    
    def get_optimal_filter(self, content: str, filter_query: str = "", url: str = ""):
        """Get optimal filtering strategy for the content"""
        self.analyzer.analyze_content(content, url)
        strategy_name = self.analyzer.recommend_filtering_strategy(bool(filter_query))
        
        if strategy_name == 'cosine' and filter_query:
            return CosineSimilarityFilter(
                query=filter_query,
                similarity_threshold=0.7,
                max_chunks=10
            ), strategy_name
        elif strategy_name == 'bm25' and filter_query:
            return BM25ContentFilter(
                user_query=filter_query,
                bm25_threshold=1.0,
                language='english'
            ), strategy_name
        elif strategy_name == 'pruning':
            return PruningContentFilter(threshold=0.5), strategy_name
        else:
            # Fallback to BM25 if query exists, otherwise pruning
            if filter_query:
                return BM25ContentFilter(
                    user_query=filter_query,
                    bm25_threshold=1.0,
                    language='english'
                ), 'bm25'
            else:
                return PruningContentFilter(threshold=0.5), 'pruning'
from .strategies import (
    CustomCssExtractionStrategy,
    XPathExtractionStrategy,
    create_extraction_strategy,
)
from . import TOOL_SELECTION_GUIDE, WORKFLOW_GUIDE, COMPLEXITY_GUIDE
from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from .suppress_output import suppress_stdout_stderr
from .file_processor import FileProcessor
from .youtube_processor import YouTubeProcessor
from .google_search_processor import GoogleSearchProcessor


# Model definitions have been moved to the models/ directory


# Suppress crawl4ai verbose output completely
logging.getLogger("crawl4ai").setLevel(logging.CRITICAL)
logging.getLogger("playwright").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Disable all logging to console for crawl4ai related modules
for logger_name in ["crawl4ai", "playwright", "asyncio", "urllib3"]:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False

# Initialize FastMCP server with minimal output
mcp = FastMCP("Crawl4AI MCP Server")

# Disable FastMCP debug logging
logging.getLogger("fastmcp").setLevel(logging.CRITICAL)
logging.getLogger("mcp").setLevel(logging.CRITICAL)

# Initialize FileProcessor for MarkItDown integration
file_processor = FileProcessor()

# Initialize YouTubeProcessor for transcript extraction (youtube-transcript-api v1.1.0+)
youtube_processor = YouTubeProcessor()

# Initialize GoogleSearchProcessor for search functionality
google_search_processor = GoogleSearchProcessor()


async def summarize_web_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    target_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Summarize web page content using LLM with enhanced metadata preservation
    
    Args:
        content: The web page content to summarize (markdown or text)
        title: Page title for context
        url: Source URL for reference
        summary_length: "short", "medium", or "long" summary
        llm_provider: LLM provider to use
        llm_model: Specific model to use
        target_tokens: Target token count for summary (if specified)
        
    Returns:
        Dictionary with summary and metadata
    """
    try:
        # Import config here to avoid circular imports
        try:
            from .config import get_llm_config
        except ImportError:
            from config import get_llm_config
        
        # Define summary lengths with enhanced token targets
        length_configs = {
            "short": {
                "target_length": "2-3 paragraphs",
                "detail_level": "key points and main conclusions only",
                "target_tokens": target_tokens or 400
            },
            "medium": {
                "target_length": "4-6 paragraphs", 
                "detail_level": "main topics with important details and examples",
                "target_tokens": target_tokens or 1000
            },
            "long": {
                "target_length": "8-12 paragraphs",
                "detail_level": "comprehensive overview with subtopics, examples, and analysis",
                "target_tokens": target_tokens or 2000
            }
        }
        
        config = length_configs.get(summary_length, length_configs["medium"])
        
        # Extract domain for context
        domain = ""
        if url:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
            except:
                domain = url
        
        # Prepare enhanced instruction for LLM
        page_context = f"""
Page Information:
- Title: {title}
- URL: {url}
- Domain: {domain}
"""
        
        instruction = f"""
        Summarize this web page content in {config['target_length']}.
        Focus on {config['detail_level']}.
        Target length: approximately {config['target_tokens']} tokens.
        
        {page_context}
        
        Structure your summary with:
        1. Brief overview including page title and source context
        2. Key sections or categories discussed
        3. Important information, data, or insights
        4. Relevant examples, quotes, or specific details
        5. Conclusions or actionable information
        
        Make the summary informative and well-structured, preserving important technical details and maintaining the original context.
        IMPORTANT: Preserve the page title, URL, and domain in your response for reference.
        """
        
        # Get LLM configuration
        llm_config = get_llm_config(llm_provider, llm_model)
        
        # Create the prompt for summarization
        prompt = f"""
        {instruction}
        
        Please provide a JSON response with the following structure:
        {{
            "summary": "The comprehensive summary of the content (approximately {config['target_tokens']} tokens)",
            "page_title": "{title}",
            "page_url": "{url}",
            "domain": "{domain}",
            "key_topics": ["List", "of", "main", "topics", "covered"],
            "content_type": "Type/category of the webpage (e.g., 'Documentation', 'Article', 'Blog Post')",
            "main_insights": ["Key", "insights", "or", "takeaways"],
            "technical_details": ["Important", "technical", "information", "if", "any"],
            "summary_token_count": "Estimated token count of summary"
        }}
        
        Web content to summarize:
        {content}
        """
        
        # Use the LLM config to make direct API call
        if hasattr(llm_config, 'provider'):
            provider_info = llm_config.provider.split('/')
            provider = provider_info[0] if provider_info else 'openai'
            model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'
            
            if provider == 'openai':
                import openai
                
                # Get API key from config
                api_key = llm_config.api_token or os.environ.get('OPENAI_API_KEY')
                
                if not api_key:
                    raise ValueError("OpenAI API key not found")
                
                client = openai.AsyncOpenAI(api_key=api_key)
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries of web page content while preserving important metadata."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=min(5000, config['target_tokens'] * 2)  # Allow up to 2x target for flexibility
                )
                
                extracted_content = response.choices[0].message.content
            else:
                raise ValueError(f"Provider {provider} not supported in direct mode")
        else:
            raise ValueError("Invalid LLM config format")
        
        if extracted_content:
            try:
                import json
                # Clean up the extracted content if it's wrapped in markdown
                content_to_parse = extracted_content
                if content_to_parse.startswith('```json'):
                    content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
                
                summary_data = json.loads(content_to_parse) if isinstance(content_to_parse, str) else content_to_parse
                
                # Ensure metadata is preserved
                return {
                    "success": True,
                    "summary": summary_data.get("summary", "Summary generation failed"),
                    "page_title": summary_data.get("page_title", title),
                    "page_url": summary_data.get("page_url", url),
                    "domain": summary_data.get("domain", domain),
                    "key_topics": summary_data.get("key_topics", []),
                    "content_type": summary_data.get("content_type", "Unknown"),
                    "main_insights": summary_data.get("main_insights", []),
                    "technical_details": summary_data.get("technical_details", []),
                    "summary_length": summary_length,
                    "target_tokens": config['target_tokens'],
                    "estimated_summary_tokens": len(summary_data.get("summary", "")) // 4,  # Rough estimate
                    "original_length": len(content),
                    "compressed_ratio": len(summary_data.get("summary", "")) / len(content) if content else 0,
                    "llm_provider": llm_config.get("provider") if isinstance(llm_config, dict) else "unknown",
                    "llm_model": llm_config.get("model") if isinstance(llm_config, dict) else "unknown",
                    "source_url": url,
                    "source_title": title
                }
            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback: treat as plain text summary
                return {
                    "success": True,
                    "summary": str(extracted_content),
                    "page_title": title,
                    "page_url": url,
                    "domain": domain,
                    "key_topics": [],
                    "content_type": "Unknown",
                    "main_insights": [],
                    "technical_details": [],
                    "summary_length": summary_length,
                    "target_tokens": config['target_tokens'],
                    "estimated_summary_tokens": len(str(extracted_content)) // 4,
                    "original_length": len(content),
                    "compressed_ratio": len(str(extracted_content)) / len(content) if content else 0,
                    "llm_provider": llm_config.get("provider") if isinstance(llm_config, dict) else "unknown",
                    "llm_model": llm_config.get("model") if isinstance(llm_config, dict) else "unknown",
                    "fallback_mode": True,
                    "source_url": url,
                    "source_title": title
                }
        else:
            return {
                "success": False,
                "error": "LLM extraction returned empty result",
                "summary_length": summary_length
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Summarization failed: {str(e)}",
            "summary_length": summary_length
        }


async def _internal_crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract content using various methods, with optional deep crawling.
    
    Args:
        request: CrawlRequest containing URL and extraction parameters
        
    Returns:
        CrawlResponse with crawled content and metadata
    """
    try:
        # Check if URL is a YouTube video
        if youtube_processor.is_youtube_url(request.url):
            # ⚠️ WARNING: YouTube transcript extraction is currently experiencing issues
            # due to API specification changes. Attempting extraction but may fail.
            try:
                youtube_result = await youtube_processor.process_youtube_url(
                    url=request.url,
                    languages=["ja", "en"],  # Default language preferences
                    include_timestamps=True,
                    preserve_formatting=True,
                    include_metadata=True
                )
                
                if youtube_result['success']:
                    transcript_data = youtube_result['transcript']
                    return CrawlResponse(
                        success=True,
                        url=request.url,
                        title=f"YouTube Video Transcript: {youtube_result['video_id']}",
                        content=transcript_data.get('full_text'),
                        markdown=transcript_data.get('clean_text'),
                        extracted_data={
                            "video_id": youtube_result['video_id'],
                            "processing_method": "youtube_transcript_api",
                            "language_info": youtube_result.get('language_info'),
                            "transcript_stats": {
                                "word_count": transcript_data.get('word_count'),
                                "segment_count": transcript_data.get('segment_count'),
                                "duration": transcript_data.get('duration_formatted')
                            },
                            "metadata": youtube_result.get('metadata')
                        }
                    )
                else:
                    # If YouTube transcript extraction fails, provide helpful error message
                    error_msg = youtube_result.get('error', 'Unknown error')
                    suggestion = youtube_result.get('suggestion', '')
                    
                    full_error = f"YouTube transcript extraction failed: {error_msg}"
                    if suggestion:
                        full_error += f"\n\nSuggestion: {suggestion}"
                    
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=full_error
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"YouTube processing error: {str(e)}"
                )
        
        # Check if URL points to a file that should be processed with MarkItDown
        if file_processor.is_supported_file(request.url):
            # Redirect to file processing for supported file formats
            try:
                file_result = await file_processor.process_file_from_url(
                    request.url,
                    max_size_mb=100  # Default size limit
                )
                
                if file_result['success']:
                    return CrawlResponse(
                        success=True,
                        url=request.url,
                        title=file_result.get('title'),
                        content=file_result.get('content'),
                        markdown=file_result.get('content'),  # MarkItDown already provides markdown
                        extracted_data={
                            "file_type": file_result.get('file_type'),
                            "size_bytes": file_result.get('size_bytes'),
                            "is_archive": file_result.get('is_archive', False),
                            "metadata": file_result.get('metadata'),
                            "archive_contents": file_result.get('archive_contents'),
                            "processing_method": "markitdown"
                        }
                    )
                else:
                    return CrawlResponse(
                        success=False,
                        url=request.url,
                        error=f"File processing failed: {file_result.get('error')}"
                    )
            except Exception as e:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error=f"File processing error: {str(e)}"
                )
        
        # Setup deep crawling strategy if max_depth is specified
        deep_crawl_strategy = None
        if request.max_depth is not None and request.max_depth > 0:
            # Create filter chain
            filters = []
            if request.url_pattern:
                filters.append(URLPatternFilter(patterns=[request.url_pattern]))
            if not request.include_external:
                # Extract domain from URL for domain filtering
                from urllib.parse import urlparse
                domain = urlparse(request.url).netloc
                filters.append(DomainFilter(allowed_domains=[domain]))
            
            filter_chain = FilterChain(filters) if filters else None
            
            # Select crawling strategy
            if request.crawl_strategy == "dfs":
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            elif request.crawl_strategy == "best_first":
                deep_crawl_strategy = BestFirstCrawlingStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )
            else:  # Default to BFS
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=request.max_depth,
                    max_pages=request.max_pages,
                    include_external=request.include_external,
                    filter_chain=filter_chain,
                    score_threshold=request.score_threshold
                )

        config = CrawlerRunConfig(
            css_selector=request.css_selector,
            screenshot=request.take_screenshot,
            wait_for=request.wait_for_selector,
            page_timeout=request.timeout * 1000,
            exclude_all_images=not request.extract_media,
            verbose=False,  # Disable verbose output
            log_console=False,  # Disable console logging
            deep_crawl_strategy=deep_crawl_strategy,
        )
        
        # Setup advanced content filtering
        content_filter_strategy = None
        if request.content_filter == "bm25" and request.filter_query:
            content_filter_strategy = BM25ContentFilter(user_query=request.filter_query)
        elif request.content_filter == "pruning":
            content_filter_strategy = PruningContentFilter(threshold=0.5)
        elif request.content_filter == "llm" and request.filter_query:
            content_filter_strategy = LLMContentFilter(
                instructions=f"Filter content related to: {request.filter_query}"
            )

        # Setup cache mode
        cache_mode = CacheMode.ENABLED
        if not request.enable_caching or request.cache_mode == "disabled":
            cache_mode = CacheMode.DISABLED
        elif request.cache_mode == "bypass":
            cache_mode = CacheMode.BYPASS

        # Configure chunking if requested
        chunking_strategy = None
        if request.chunk_content:
            from crawl4ai.chunking_strategy import SlidingWindowChunking
            step_size = int(request.chunk_size * (1 - request.overlap_rate))
            chunking_strategy = SlidingWindowChunking(
                window_size=request.chunk_size,
                step=step_size
            )

        # Create config parameters
        config_params = {
            "css_selector": request.css_selector,
            "screenshot": request.take_screenshot,
            "wait_for": request.wait_for_selector,
            "page_timeout": request.timeout * 1000,
            "exclude_all_images": not request.extract_media,
            "verbose": False,
            "log_console": False,
            "deep_crawl_strategy": deep_crawl_strategy,
            "cache_mode": cache_mode
        }
        
        if chunking_strategy:
            config_params["chunking_strategy"] = chunking_strategy
        
        # Add content filter if supported by current crawl4ai version
        try:
            config = CrawlerRunConfig(**config_params, content_filter=content_filter_strategy)
        except TypeError:
            # Fallback for older versions without content_filter support
            config = CrawlerRunConfig(**config_params)

        # Setup browser configuration with lightweight WebKit preference
        browser_config = {
            "headless": True,
            "verbose": False,
            "browser_type": "webkit"  # Use lightweight WebKit by default
        }
        
        if request.user_agent:
            browser_config["user_agent"] = request.user_agent
        
        if request.headers:
            browser_config["headers"] = request.headers

        # Suppress output to avoid JSON parsing errors
        with suppress_stdout_stderr():
            # Try WebKit first, fallback to Chromium if needed
            result = None
            browsers_to_try = ["webkit", "chromium"]
            
            for browser_type in browsers_to_try:
                try:
                    current_browser_config = browser_config.copy()
                    current_browser_config["browser_type"] = browser_type
                    
                    async with AsyncWebCrawler(**current_browser_config) as crawler:
                        # Handle authentication
                        if request.cookies:
                            # Set cookies if provided
                            await crawler.set_cookies(request.cookies)
                        
                        # Execute custom JavaScript if provided
                        if request.execute_js:
                            config.js_code = request.execute_js
                        
                        # Run crawler with config
                        arun_params = {"url": request.url, "config": config}
                        
                        result = await crawler.arun(**arun_params)
                        break  # Success, no need to try other browsers
                        
                except Exception as browser_error:
                    # If this is the last browser to try, raise the error
                    if browser_type == browsers_to_try[-1]:
                        raise browser_error
                    # Otherwise, try the next browser
                    continue
        
        # Handle different result types (single result vs list from deep crawling)
        if isinstance(result, list):
            # Deep crawling returns a list of results
            if not result:
                return CrawlResponse(
                    success=False,
                    url=request.url,
                    error="No results returned from deep crawling"
                )
            
            # Process multiple results from deep crawling
            all_content = []
            all_markdown = []
            all_media = []
            crawled_urls = []
            
            for page_result in result:
                if hasattr(page_result, 'success') and page_result.success:
                    crawled_urls.append(page_result.url if hasattr(page_result, 'url') else 'unknown')
                    if hasattr(page_result, 'cleaned_html') and page_result.cleaned_html:
                        all_content.append(f"=== {page_result.url} ===\n{page_result.cleaned_html}")
                    if hasattr(page_result, 'markdown') and page_result.markdown:
                        all_markdown.append(f"=== {page_result.url} ===\n{page_result.markdown}")
                    if hasattr(page_result, 'media') and page_result.media and request.extract_media:
                        all_media.extend(page_result.media)
            
            return CrawlResponse(
                success=True,
                url=request.url,
                title=f"Deep crawl of {len(crawled_urls)} pages",
                content="\n\n".join(all_content) if all_content else "No content extracted",
                markdown="\n\n".join(all_markdown) if all_markdown else "No markdown content",
                media=all_media if request.extract_media else None,
                extracted_data={
                    "crawled_pages": len(crawled_urls),
                    "crawled_urls": crawled_urls,
                    "processing_method": "deep_crawling"
                }
            )
        
        elif hasattr(result, 'success') and result.success:
            # For deep crawling, result might contain multiple pages
            if deep_crawl_strategy and hasattr(result, 'crawled_pages'):
                # Combine content from all crawled pages
                all_content = []
                all_markdown = []
                all_media = []
                
                for page in result.crawled_pages:
                    if page.cleaned_html:
                        all_content.append(f"=== {page.url} ===\n{page.cleaned_html}")
                    if page.markdown:
                        all_markdown.append(f"=== {page.url} ===\n{page.markdown}")
                    if page.media and request.extract_media:
                        all_media.extend(page.media)
                
                # Prepare content for potential summarization
                combined_content = "\n\n".join(all_content) if all_content else result.cleaned_html
                combined_markdown = "\n\n".join(all_markdown) if all_markdown else result.markdown
                title_to_use = result.metadata.get("title", "")
                extracted_data = {"crawled_pages": len(result.crawled_pages)} if hasattr(result, 'crawled_pages') else {}
                
                # Apply auto-summarization if enabled and content exceeds token limit
                if request.auto_summarize and combined_content:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(combined_content) // 4
                    
                    # Only summarize if content exceeds the specified token limit
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = combined_markdown or combined_content
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model,
                                target_tokens=request.max_content_tokens
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                combined_content = summary_result["summary"]
                                combined_markdown = summary_result["summary"]
                                
                                extracted_data.update({
                                    "summarization_applied": True,
                                    "original_content_length": len("\n\n".join(all_content) if all_content else result.cleaned_html),
                                    "original_tokens_estimate": estimated_tokens,
                                    "summary_length": request.summary_length,
                                    "target_tokens": summary_result.get("target_tokens", request.max_content_tokens),
                                    "estimated_summary_tokens": summary_result.get("estimated_summary_tokens", 0),
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    # Preserve page metadata from summary
                                    "page_title_preserved": summary_result.get("page_title", ""),
                                    "page_url_preserved": summary_result.get("page_url", ""),
                                    "domain_preserved": summary_result.get("domain", ""),
                                    "auto_summarization_trigger": f"Content exceeded {request.max_content_tokens} tokens"
                                })
                            else:
                                # Summarization failed, add error info but keep original content
                                extracted_data.update({
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                })
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data.update({
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            })
                    else:
                        # Content is below threshold - preserve original content and add info
                        extracted_data.update({
                            "auto_summarize_requested": True,
                            "original_content_preserved": True,
                            "content_below_threshold": True,
                            "tokens_estimate": estimated_tokens,
                            "max_tokens_threshold": request.max_content_tokens,
                            "reason": f"Content ({estimated_tokens} tokens) is below threshold ({request.max_content_tokens} tokens)"
                        })
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=combined_content,
                    markdown=combined_markdown,
                    media=all_media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            else:
                # Check if auto-summarization should be applied
                content_to_use = result.cleaned_html
                markdown_to_use = result.markdown
                extracted_data = None
                title_to_use = result.metadata.get("title", "")
                
                # Apply auto-summarization if enabled and content exceeds token limit
                if request.auto_summarize and content_to_use:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(content_to_use) // 4
                    
                    # Only summarize if content exceeds the specified token limit
                    if estimated_tokens > request.max_content_tokens:
                        try:
                            # Use markdown content for summarization if available, otherwise use cleaned HTML
                            content_for_summary = markdown_to_use or content_to_use
                            
                            summary_result = await summarize_web_content(
                                content=content_for_summary,
                                title=title_to_use,
                                url=request.url,
                                summary_length=request.summary_length,
                                llm_provider=request.llm_provider,
                                llm_model=request.llm_model,
                                target_tokens=request.max_content_tokens
                            )
                            
                            if summary_result.get("success"):
                                # Replace content with summary and preserve original in extracted_data
                                content_to_use = summary_result["summary"]
                                markdown_to_use = summary_result["summary"]  # Use same summary for both
                                
                                extracted_data = {
                                    "summarization_applied": True,
                                    "original_content_length": len(result.cleaned_html),
                                    "original_tokens_estimate": estimated_tokens,
                                    "summary_length": request.summary_length,
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    "auto_summarization_trigger": f"Content exceeded {request.max_content_tokens} tokens"
                                }
                            else:
                                # Summarization failed, add error info but keep original content
                                extracted_data = {
                                    "summarization_attempted": True,
                                    "summarization_error": summary_result.get("error", "Unknown error"),
                                    "original_content_preserved": True
                                }
                        except Exception as e:
                            # Summarization failed, add error info but keep original content
                            extracted_data = {
                                "summarization_attempted": True,
                                "summarization_error": f"Exception during summarization: {str(e)}",
                                "original_content_preserved": True
                            }
                    else:
                        # Content is below threshold - preserve original content and add info
                        extracted_data = {
                            "auto_summarize_requested": True,
                            "original_content_preserved": True,
                            "content_below_threshold": True,
                            "tokens_estimate": estimated_tokens,
                            "max_tokens_threshold": request.max_content_tokens,
                            "reason": f"Content ({estimated_tokens} tokens) is below threshold ({request.max_content_tokens} tokens)"
                        }
                
                response = CrawlResponse(
                    success=True,
                    url=request.url,
                    title=title_to_use,
                    content=content_to_use,
                    markdown=markdown_to_use,
                    media=result.media if request.extract_media else None,
                    screenshot=result.screenshot if request.take_screenshot else None,
                    extracted_data=extracted_data
                )
            return response
        else:
            # Handle case where result doesn't have success attribute or failed
            error_msg = "Failed to crawl URL"
            if hasattr(result, 'error_message'):
                error_msg = f"Failed to crawl URL: {result.error_message}"
            elif hasattr(result, 'error'):
                error_msg = f"Failed to crawl URL: {result.error}"
            else:
                error_msg = f"Failed to crawl URL: Unknown error (result type: {type(result)})"
            
            return CrawlResponse(
                success=False,
                url=request.url,
                error=error_msg
            )
                
    except Exception as e:
        error_message = f"Crawling error: {str(e)}"
        
        # Enhanced error handling for browser and UVX issues
        if "playwright" in str(e).lower() or "browser" in str(e).lower() or "executable doesn't exist" in str(e).lower():
            import os
            import sys
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            
            if is_uvx_env:
                error_message += "\n\n🔧 UVX Environment Browser Setup Required:\n" \
                    f"UVX environments require manual browser installation due to isolation:\n\n" \
                    f"Method 1 - Lightweight Headless Installation (Recommended):\n" \
                    f"  uvx --with playwright playwright install --with-deps --only-shell\n" \
                    f"  uvx --with playwright playwright install --with-deps webkit\n\n" \
                    f"Method 2 - System-wide Lightweight Installation:\n" \
                    f"  pip install playwright\n" \
                    f"  playwright install --only-shell webkit\n\n" \
                    f"Method 3 - Restart Claude Desktop and retry\n\n" \
                    f"💡 Verification & Notes:\n" \
                    f"  - Run get_system_diagnostics() to check installation\n" \
                    f"  - Headless shell (~100MB) + WebKit (~180MB) = most efficient\n" \
                    f"  - Standard Chromium (~281MB) if lightweight options fail\n" \
                    f"  - If problems persist, switch to STDIO local setup"
            else:
                error_message += "\n\n🔧 Browser Setup Required:\n" \
                    f"1. Install Lightweight Playwright browsers (Recommended):\n" \
                    f"   playwright install --only-shell  # Headless Chromium (~100MB)\n" \
                    f"   playwright install webkit        # WebKit (~180MB)\n\n" \
                    f"2. Alternative - Install with system dependencies:\n" \
                    f"   playwright install --with-deps --only-shell webkit\n\n" \
                    f"3. For Linux system dependencies (if needed):\n" \
                    f"   sudo apt-get install libnss3 libnspr4 libasound2 libxss1\n\n" \
                    f"4. Verification:\n" \
                    f"   - Run get_system_diagnostics() to confirm installation\n" \
                    f"   - Check browser files in platform cache directory\n" \
                    f"   - Total space: ~280MB (vs ~561MB for full browsers)"
        
        return CrawlResponse(
            success=False,
            url=request.url,
            error=error_message,
            extracted_data={
                'error_type': 'browser_setup_required',
                'uvx_environment': 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable),
                'diagnostic_tool': 'get_system_diagnostics',
                'installation_commands': [
                    'playwright install webkit',
                    'playwright install chromium'
                ]
            }
        )




# Import all tools from the tools module
from .tools import (
    # YouTube tools (Phase1: Import with private names for @mcp.tool decorator)
    extract_youtube_transcript as _extract_youtube_transcript,
    batch_extract_youtube_transcripts as _batch_extract_youtube_transcripts,
    get_youtube_video_info as _get_youtube_video_info,
    get_youtube_api_setup_guide as _get_youtube_api_setup_guide,
    
    # File processing tools (Phase2: Import with private names)
    process_file as _process_file,
    get_supported_file_formats as _get_supported_file_formats,
    enhanced_process_large_content as _enhanced_process_large_content,
    
    # Web crawling tools (Phase2: Import with private names)
    crawl_url as _crawl_url,
    deep_crawl_site as _deep_crawl_site,
    crawl_url_with_fallback as _crawl_url_with_fallback,
    intelligent_extract as _intelligent_extract,
    extract_entities as _extract_entities,
    extract_structured_data as _extract_structured_data,
    
    # Search tools (Phase2: Import with private names)
    search_google as _search_google,
    batch_search_google as _batch_search_google,
    search_and_crawl as _search_and_crawl,
    get_search_genres as _get_search_genres,
    
    # Utility tools (Phase2: Import with private names)
    get_llm_config_info as _get_llm_config_info,
    batch_crawl as _batch_crawl,
    get_tool_selection_guide as _get_tool_selection_guide
)

# get_system_diagnostics is defined in this file, not imported from tools

# Phase1: YouTube tools with @mcp.tool decorators
@mcp.tool
async def extract_youtube_transcript(
    url: Annotated[str, Field(description="YouTube video URL. Supports formats: https://www.youtube.com/watch?v=VIDEO_ID, https://youtu.be/VIDEO_ID")],
    languages: Annotated[Optional[Union[List[str], str]], Field(description="Array of language codes in preference order. Can be array like [\"ja\", \"en\"] or string like '[\"ja\", \"en\"]' (default: [\"ja\", \"en\"])")] = ["ja", "en"],
    translate_to: Annotated[Optional[str], Field(description="Target language code for translation. Examples: 'en' (English), 'ja' (Japanese), 'es' (Spanish), 'fr' (French), 'de' (German) (default: None)")] = None,
    include_timestamps: Annotated[bool, Field(description="Include timestamps in transcript (default: True)")] = True,
    preserve_formatting: Annotated[bool, Field(description="Preserve original formatting (default: True)")] = True,
    include_metadata: Annotated[bool, Field(description="Include video metadata (default: True)")] = True,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize long transcripts using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'azure', 'together', 'groq', auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model: 'gpt-4o', 'claude-3-sonnet', 'gemini-pro', or provider-specific model names, auto-detected if not specified (default: None)")] = None
) -> Dict[str, Any]:
    """
    Extract YouTube video transcripts with timestamps and optional AI summarization.
    
    Works with public videos that have captions. No authentication required.
    Auto-detects available languages and falls back appropriately.
    
    Note: Automatic transcription may contain errors.
    """
    # Handle string-encoded array for languages parameter
    if isinstance(languages, str):
        try:
            import json
            # Try to parse as JSON string
            languages = json.loads(languages)
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, try to extract language codes manually
            import re
            # Extract quoted strings from the parameter
            matches = re.findall(r'"([^"]*)"', languages)
            if matches:
                languages = matches
            else:
                # Fallback to default
                languages = ["ja", "en"]
    
    return await _extract_youtube_transcript(
        url, languages, translate_to, include_timestamps, preserve_formatting,
        include_metadata, auto_summarize, max_content_tokens, summary_length,
        llm_provider, llm_model
    )

@mcp.tool
async def batch_extract_youtube_transcripts(
    request: Annotated[Dict[str, Any], Field(description="YouTubeBatchRequest dictionary containing: urls (required list of YouTube URLs), languages (default: ['ja', 'en']), max_concurrent (default: 3, max: 5), include_timestamps, translate_to, preserve_formatting, include_metadata (all optional booleans)")]
) -> Dict[str, Any]:
    """
    Extract transcripts from multiple YouTube videos using youtube-transcript-api.
    
    Processes multiple YouTube URLs concurrently with controlled rate limiting.
    No authentication required for public videos with captions.
    
    Note: Automatic transcription may contain errors.
    """
    return await _batch_extract_youtube_transcripts(request)

@mcp.tool
async def get_youtube_video_info(
    video_url: Annotated[str, Field(description="YouTube video URL. Supports formats: https://www.youtube.com/watch?v=VIDEO_ID, https://youtu.be/VIDEO_ID")],
    summarize_transcript: Annotated[bool, Field(description="Summarize long transcripts using LLM (default: False)")] = False,
    max_tokens: Annotated[int, Field(description="Token limit before triggering summarization (default: 25000)")] = 25000,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'azure', 'together', 'groq' (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model: 'gpt-4o', 'claude-3-sonnet', 'gemini-pro', or provider-specific model names (default: auto-detected)")] = None,
    summary_length: Annotated[str, Field(description="Summary length - 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    include_timestamps: Annotated[bool, Field(description="Preserve key timestamps in summary (default: True)")] = True
) -> Dict[str, Any]:
    """
    Get YouTube video information with optional transcript summarization.
    
    Retrieves basic video information and transcript availability using youtube-transcript-api.
    No authentication required for public videos.
    
    Note: Automatic transcription may contain errors.
    """
    return await _get_youtube_video_info(
        video_url, summarize_transcript, max_tokens, llm_provider, llm_model,
        summary_length, include_timestamps
    )

@mcp.tool
async def get_youtube_api_setup_guide() -> Dict[str, Any]:
    """
    Get setup information for youtube-transcript-api integration.
    
    Provides information about current youtube-transcript-api setup.
    No authentication or API keys required for basic transcript extraction.
    """
    return await _get_youtube_api_setup_guide()

# Phase2: File processing tools with @mcp.tool decorators
@mcp.tool
async def process_file(
    url: Annotated[str, Field(description="URL of the file to process. Examples: https://example.com/document.pdf, https://site.com/report.docx, https://files.com/data.xlsx, https://docs.com/archive.zip")],
    max_size_mb: Annotated[int, Field(description="Maximum file size in MB (default: 100)")] = 100,
    extract_all_from_zip: Annotated[bool, Field(description="Whether to extract all files from ZIP archives (default: True)")] = True,
    include_metadata: Annotated[bool, Field(description="Whether to include file metadata (default: True)")] = True,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False,
    max_content_tokens: Annotated[int, Field(description="Maximum tokens before triggering auto-summarization (default: 15000)")] = 15000,
    summary_length: Annotated[str, Field(description="Summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium",
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'azure', 'together', 'groq', auto-detected if not specified (default: None)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific LLM model: 'gpt-4o', 'claude-3-sonnet', 'gemini-pro', or provider-specific model names, auto-detected if not specified (default: None)")] = None
) -> Dict[str, Any]:
    """
    Convert documents to markdown text with optional AI summarization.
    
    Supports: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), ZIP archives, ePub.
    Auto-detects file format and applies appropriate conversion method.
    """
    return await _process_file(
        url, max_size_mb, extract_all_from_zip, include_metadata, auto_summarize,
        max_content_tokens, summary_length, llm_provider, llm_model
    )

@mcp.tool
async def get_supported_file_formats() -> Dict[str, Any]:
    """
    Get list of supported file formats for file processing.
    
    Provides comprehensive information about supported file formats and their capabilities.
    
    Parameters: None
    
    Returns dictionary with supported file formats and descriptions.
    """
    return await _get_supported_file_formats()

@mcp.tool
async def enhanced_process_large_content(
    url: Annotated[str, Field(description="Target URL to process. Examples: https://long-article.com, https://research-paper.site.com/document, https://content.example.com/page")],
    chunking_strategy: Annotated[str, Field(description="Chunking method: 'topic' (semantic boundaries), 'sentence' (sentence-based), 'overlap' (sliding window), 'regex' (custom pattern) (default: 'topic')")] = "topic",
    filtering_strategy: Annotated[str, Field(description="Content filtering method: 'bm25' (keyword relevance), 'pruning' (structure-based), 'llm' (AI-powered) (default: 'bm25')")] = "bm25", 
    filter_query: Annotated[Optional[str], Field(description="Query for BM25 filtering, keywords related to desired content (default: None)")] = None,
    max_chunk_tokens: Annotated[int, Field(description="Maximum tokens per chunk (default: 8000)")] = 8000,
    chunk_overlap: Annotated[int, Field(description="Token overlap between chunks (default: 500)")] = 500,
    extract_top_chunks: Annotated[int, Field(description="Number of top relevant chunks to extract (default: 10)")] = 10,
    similarity_threshold: Annotated[float, Field(description="Minimum similarity threshold for relevant chunks (default: 0.7)")] = 0.7,
    summarize_chunks: Annotated[bool, Field(description="Whether to summarize individual chunks (default: True)")] = True,
    merge_strategy: Annotated[str, Field(description="Chunk summary merging approach: 'hierarchical' (tree-based progressive), 'linear' (sequential concatenation) (default: 'hierarchical')")] = "hierarchical",
    final_summary_length: Annotated[str, Field(description="Final summary length: 'short', 'medium', 'long' (default: 'medium')")] = "medium"
) -> Dict[str, Any]:
    """
    Enhanced processing for large content using advanced chunking and filtering.
    
    Uses BM25 filtering and intelligent chunking to reduce token usage while preserving semantic boundaries.
    Supports hierarchical summarization for progressive content refinement.
    """
    return await _enhanced_process_large_content(
        url, chunking_strategy, filtering_strategy, filter_query, max_chunk_tokens,
        chunk_overlap, extract_top_chunks, similarity_threshold, summarize_chunks,
        merge_strategy, final_summary_length
    )

# Phase2: Web crawling tools with @mcp.tool decorators
@mcp.tool
async def crawl_url(
    url: Annotated[str, Field(description="Target URL to crawl. Examples: https://example.com, https://news.site.com/article")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction. Examples: '.article-content', '#main-content', 'div.post', 'article p' (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction. Examples: '//div[@class=\"content\"]', '//article//p', '//h1[@id=\"title\"]' (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element to load. CSS selector or XPath. Examples: '.content-loaded', '#dynamic-content', '[data-loaded=\"true\"]' (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False
) -> Dict[str, Any]:
    """
    Extract content from web pages with JavaScript support. Auto-detects PDFs and documents.
    
    Core web crawling tool with comprehensive configuration options.
    Essential for SPAs: set wait_for_js=true for JavaScript-heavy sites.
    
    NOTE: If this tool fails with 503 errors, rate limiting, or anti-bot protection 
    (especially for sites like HackerNews, Reddit, social media), try using 
    crawl_url_with_fallback which implements multiple retry strategies with 
    realistic browser simulation and anti-bot evasion techniques.
    """
    return await _crawl_url(
        url, css_selector, xpath, extract_media, take_screenshot, generate_markdown,
        wait_for_selector, timeout, wait_for_js, auto_summarize
    )

@mcp.tool
async def deep_crawl_site(
    url: Annotated[str, Field(description="Starting URL for multi-page crawling. Examples: https://docs.example.com, https://site.com/documentation, https://wiki.company.com")],
    max_depth: Annotated[int, Field(description="Link levels to follow from start URL (default: 2)")] = 2,
    max_pages: Annotated[int, Field(description="Maximum pages to crawl (default: 5)")] = 5,
    crawl_strategy: Annotated[str, Field(description="Crawling approach: 'bfs' (breadth-first, balanced coverage), 'dfs' (depth-first, follow links deeply), 'best_first' (relevance-based prioritization) (default: 'bfs')")] = "bfs",
    include_external: Annotated[bool, Field(description="Follow external domain links (default: False)")] = False,
    url_pattern: Annotated[Optional[str], Field(description="Wildcard filter like '*docs*' or '*api*' (default: None)")] = None,
    score_threshold: Annotated[float, Field(description="Minimum relevance score 0.0-1.0 (default: 0.0)")] = 0.0,
    extract_media: Annotated[bool, Field(description="Include images/videos (default: False)")] = False,
    base_timeout: Annotated[int, Field(description="Timeout per page in seconds (default: 60)")] = 60
) -> Dict[str, Any]:
    """
    Crawl multiple related pages from a website (maximum 5 pages for stability).
    
    Multi-page crawling with configurable depth and filtering options.
    Perfect for documentation sites and content discovery.
    """
    return await _deep_crawl_site(
        url, max_depth, max_pages, crawl_strategy, include_external,
        url_pattern, score_threshold, extract_media, base_timeout
    )

@mcp.tool
async def crawl_url_with_fallback(
    url: Annotated[str, Field(description="Target URL to crawl. Examples: https://example.com, https://difficult-site.com")],
    css_selector: Annotated[Optional[str], Field(description="CSS selector for content extraction. Examples: '.article-content', '#main-content', 'div.post', 'article p' (default: None)")] = None,
    xpath: Annotated[Optional[str], Field(description="XPath selector for content extraction. Examples: '//div[@class=\"content\"]', '//article//p', '//h1[@id=\"title\"]' (default: None)")] = None,
    extract_media: Annotated[bool, Field(description="Whether to extract media files (default: False)")] = False,
    take_screenshot: Annotated[bool, Field(description="Whether to take a screenshot (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Whether to generate markdown (default: True)")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Wait for specific element to load. CSS selector or XPath. Examples: '.content-loaded', '#dynamic-content', '[data-loaded=\"true\"]' (default: None)")] = None,
    timeout: Annotated[int, Field(description="Request timeout in seconds (default: 60)")] = 60,
    wait_for_js: Annotated[bool, Field(description="Wait for JavaScript to complete (default: False)")] = False,
    auto_summarize: Annotated[bool, Field(description="Automatically summarize large content using LLM (default: False)")] = False
) -> Dict[str, Any]:
    """
    Enhanced crawling with multiple fallback strategies for difficult sites.
    
    Uses multiple fallback strategies when normal crawling fails. Same parameters as crawl_url 
    but with enhanced reliability for sites with aggressive anti-bot protection.
    """
    return await _crawl_url_with_fallback(
        url, css_selector, xpath, extract_media, take_screenshot, generate_markdown,
        wait_for_selector, timeout, wait_for_js, auto_summarize
    )

@mcp.tool
async def intelligent_extract(
    url: Annotated[str, Field(description="Target webpage URL. Examples: https://example.com/page, https://news.site.com/article, https://company.com/about")],
    extraction_goal: Annotated[str, Field(description="Specific data to extract, be precise. Examples: 'contact information and pricing', 'product details and specifications', 'article title and author', 'company address and phone number'")],
    content_filter: Annotated[str, Field(description="Pre-filter content method: 'bm25' (keyword relevance), 'pruning' (structure-based), 'llm' (AI-powered) (default: 'bm25')")] = "bm25",
    filter_query: Annotated[Optional[str], Field(description="Keywords for BM25 filtering to improve accuracy. Space-separated terms related to your extraction goal (default: None)")] = None,
    chunk_content: Annotated[bool, Field(description="Split large content for better processing (default: False)")] = False,
    use_llm: Annotated[bool, Field(description="Enable LLM processing (default: True)")] = True,
    custom_instructions: Annotated[Optional[str], Field(description="Additional guidance for LLM. Examples: 'Focus on technical details', 'Extract only recent information', 'Prioritize pricing and contact info' (default: None)")] = None,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'azure', 'together', 'groq' (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model: 'gpt-4o', 'claude-3-sonnet', 'gemini-pro', or provider-specific model names (default: auto-detected)")] = None
) -> Dict[str, Any]:
    """
    AI-powered extraction of specific data from web pages using LLM semantic understanding.
    
    Uses LLM to extract specific information based on your extraction goal.
    Pre-filtering improves accuracy and reduces processing time.
    """
    return await _intelligent_extract(
        url, extraction_goal, content_filter, filter_query, chunk_content,
        use_llm, custom_instructions, llm_provider, llm_model
    )

@mcp.tool
async def extract_entities(
    url: Annotated[str, Field(description="Target webpage URL. Examples: https://example.com/page, https://news.site.com/article, https://company.com/about")],
    entity_types: Annotated[List[str], Field(description="List of entity types to extract. Supported types: emails, phones, urls, dates, ips, social_media, prices, credit_cards, coordinates, names (with LLM)")],
    custom_patterns: Annotated[Optional[Dict[str, str]], Field(description="Custom regex patterns for specialized extraction. Dictionary with entity_name -> regex_pattern pairs (default: None)")] = None,
    include_context: Annotated[bool, Field(description="Include surrounding text context (default: True)")] = True,
    deduplicate: Annotated[bool, Field(description="Remove duplicate entities (default: True)")] = True,
    use_llm: Annotated[bool, Field(description="Use AI for named entity recognition (default: False)")] = False,
    llm_provider: Annotated[Optional[str], Field(description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'azure', 'together', 'groq' (default: auto-detected)")] = None,
    llm_model: Annotated[Optional[str], Field(description="Specific model name (default: auto-detected)")] = None
) -> Dict[str, Any]:
    """
    Extract specific entity types from web pages using regex patterns or LLM.
    
    Supports regex: emails, phones, urls, dates, ips, social_media, prices, credit_cards, coordinates
    Supports LLM: names (people/organizations/locations) when use_llm=True
    """
    return await _extract_entities(
        url, entity_types, custom_patterns, include_context, deduplicate,
        use_llm, llm_provider, llm_model
    )

@mcp.tool
async def extract_structured_data(
    request: Annotated[Dict[str, Any], Field(description="StructuredExtractionRequest dictionary containing: url (required), extraction_schema (required), extraction_type ('css' or 'llm'), css_selectors (for CSS extraction), instruction (for LLM extraction), llm_provider and llm_model (optional)")]
) -> Dict[str, Any]:
    """
    Extract structured data from a URL using CSS selectors or LLM-based extraction.
    
    Extract data matching a predefined schema using CSS selectors or LLM processing.
    Useful for consistent data extraction from similar page structures.
    """
    from crawl4ai_mcp.models import StructuredExtractionRequest
    return await _extract_structured_data(StructuredExtractionRequest(**request))

# Phase2: Search tools with @mcp.tool decorators
@mcp.tool
async def search_google(
    request: Annotated[Dict[str, Any], Field(description="GoogleSearchRequest dictionary containing: query (required), num_results (max 20, default: 10), search_genre (optional: 'pdf', 'documents', 'recent', 'japanese', 'english', etc.), country_code (default: 'US'), language_code (default: 'en')")],
    include_current_date: Annotated[bool, Field(description="Append current date to query for latest results (default: True)")] = True
) -> Dict[str, Any]:
    """
    Perform Google search with genre filtering and extract structured results with metadata.
    
    Returns web search results with titles, snippets, URLs, and metadata.
    Supports targeted search genres for better results.
    """
    return await _search_google(request, include_current_date)

@mcp.tool
async def batch_search_google(
    request: Annotated[Dict[str, Any], Field(description="GoogleBatchSearchRequest dictionary containing: queries (required list), max_concurrent (default: 3), num_results_per_query (max 20, default: 10), search_genre (optional: 'pdf', 'documents', 'recent', 'japanese', 'english'), country_code (default: 'US'), language_code (default: 'en')")],
    include_current_date: Annotated[bool, Field(description="Append current date to queries for latest results (default: True)")] = True
) -> Dict[str, Any]:
    """
    Perform multiple Google searches in batch with analysis.
    
    Process multiple search queries concurrently with controlled rate limiting.
    Includes batch processing statistics and result analysis.
    """
    return await _batch_search_google(request, include_current_date)

@mcp.tool
async def search_and_crawl(
    search_query: Annotated[str, Field(description="Search terms, be specific for better results. Examples: 'machine learning tutorials', 'Python web scraping libraries', 'climate change research 2024'")],
    num_search_results: Annotated[int, Field(description="Number of search results to retrieve (default: 5, max: 20)")] = 5,
    crawl_top_results: Annotated[int, Field(description="Number of top results to crawl (default: 3, max: 10)")] = 3,
    extract_media: Annotated[bool, Field(description="Include images/videos from crawled pages (default: False)")] = False,
    generate_markdown: Annotated[bool, Field(description="Convert crawled content to markdown (default: True)")] = True,
    search_genre: Annotated[Optional[str], Field(description="Content type filter for targeted results. Options: 'pdf', 'documents', 'presentations', 'spreadsheets', 'recent', 'japanese', 'english' (default: None)")] = None,
    base_timeout: Annotated[int, Field(description="Base timeout, auto-scales with crawl count (default: 30)")] = 30,
    include_current_date: Annotated[bool, Field(description="Add current date to query (default: True)")] = True
) -> Dict[str, Any]:
    """
    Perform Google search and automatically crawl top results for full content analysis.
    
    Combines search discovery with full content extraction. Auto-scales timeout based on crawl count.
    Returns both search results and full page content.
    """
    return await _search_and_crawl(
        search_query, num_search_results, crawl_top_results, extract_media,
        generate_markdown, search_genre, base_timeout, include_current_date
    )

@mcp.tool
async def get_search_genres() -> Dict[str, Any]:
    """
    Get list of available search genres for content filtering.
    
    Provides comprehensive information about available search genres and their capabilities.
    
    Parameters: None
    
    Returns dictionary with available genres and their descriptions.
    """
    return await _get_search_genres()

# Phase2: Utility tools with @mcp.tool decorators
@mcp.tool
async def get_llm_config_info() -> Dict[str, Any]:
    """
    Get information about the current LLM configuration.
    
    Provides details about available LLM providers, models, and API key status.
    
    Parameters: None
    
    Returns dictionary with LLM configuration details including available providers and models.
    """
    return await _get_llm_config_info()

@mcp.tool
async def batch_crawl(
    urls: Annotated[List[str], Field(description="List of URLs to crawl")],
    config: Annotated[Optional[Dict[str, Any]], Field(description="Optional configuration parameters (default: None)")] = None,
    base_timeout: Annotated[int, Field(description="Base timeout in seconds, adjusted based on URL count (default: 30)")] = 30
) -> List[Dict[str, Any]]:
    """
    Crawl multiple URLs in batch.
    
    Process multiple URLs concurrently for efficiency. Timeout auto-scales based on URL count.
    
    Parameters:
    - urls: List of URLs to crawl (required)
    - config: Optional configuration parameters (default: None)
    - base_timeout: Base timeout in seconds, adjusted based on URL count (default: 30)
    
    Example:
    {"urls": ["https://example.com/page1", "https://example.com/page2"], "config": {"generate_markdown": true}}
    
    Returns List of CrawlResponse objects for each URL.
    """
    from crawl4ai_mcp.models import CrawlResponse
    result = await _batch_crawl(urls, config, base_timeout)
    # Convert CrawlResponse objects to dictionaries for JSON serialization
    return [response.__dict__ if hasattr(response, '__dict__') else response for response in result]

@mcp.tool
async def get_tool_selection_guide() -> Dict[str, Any]:
    """
    Get comprehensive tool selection guide for AI agents.
    
    Provides complete mapping of use cases to appropriate tools, workflows, and complexity guides.
    Essential for tool selection, workflow planning, and understanding capabilities.
    
    Parameters: None
    
    Returns dictionary with tool selection guide, workflows, and complexity mapping.
    """
    return await _get_tool_selection_guide()

# All 21 tools are now using @mcp.tool decorators
# get_system_diagnostics will be registered after its definition

# to avoid conflict with batch_crawl from utilities module
@mcp.resource("uri://crawl4ai/config")
async def get_crawler_config() -> str:
    """
    Get the default crawler configuration options.
    
    Returns:
        JSON string with configuration options
    """
    config_options = {
        "browser_config": {
            "headless": True,
            "viewport_width": 1920,
            "viewport_height": 1080,
            "user_agent": "Mozilla/5.0 (compatible; Crawl4AI/1.0)",
        },
        "crawler_config": {
            "timeout": 30,
            "extract_media": False,
            "screenshot": False,
            "generate_markdown": True,
            "remove_overlay_elements": True,
        },
        "extraction_strategies": {
            "css": "Use CSS selectors for targeted extraction",
            "xpath": "Use XPath expressions for complex queries",
            "llm": "Use LLM-based extraction with custom schemas",
        }
    }
    
    return json.dumps(config_options, indent=2)


@mcp.resource("uri://crawl4ai/examples")
async def get_usage_examples() -> str:
    """
    Get usage examples for the Crawl4AI MCP server.
    
    Returns:
        JSON string with example requests
    """
    examples = {
        "basic_crawl": {
            "description": "Basic URL crawling",
            "request": {
                "url": "https://example.com",
                "generate_markdown": True,
                "take_screenshot": False
            }
        },
        "css_extraction": {
            "description": "Extract specific content using CSS selectors",
            "request": {
                "url": "https://news.ycombinator.com",
                "css_selector": ".storylink"
            }
        },
        "structured_extraction": {
            "description": "Extract structured data with schema",
            "request": {
                "url": "https://example-store.com/product/123",
                "schema": {
                    "name": "Product name",
                    "price": "Product price",
                    "description": "Product description"
                },
                "extraction_type": "css",
                "css_selectors": {
                    "name": "h1.product-title",
                    "price": ".price",
                    "description": ".description"
                }
            }
        }
    }
    
    return json.dumps(examples, indent=2)


@mcp.resource("uri://crawl4ai/file-processing")
async def get_file_processing_examples() -> str:
    """
    Get usage examples for file processing with MarkItDown.
    
    Returns:
        JSON string with file processing examples
    """
    examples = {
        "pdf_processing": {
            "description": "Process PDF document and extract text content",
            "request": {
                "url": "https://example.com/document.pdf",
                "max_size_mb": 50,
                "include_metadata": True
            }
        },
        "office_document": {
            "description": "Process Microsoft Word document",
            "request": {
                "url": "https://example.com/report.docx",
                "max_size_mb": 25,
                "include_metadata": True
            }
        },
        "excel_spreadsheet": {
            "description": "Process Excel spreadsheet",
            "request": {
                "url": "https://example.com/data.xlsx",
                "max_size_mb": 30,
                "include_metadata": True
            }
        },
        "powerpoint_presentation": {
            "description": "Process PowerPoint presentation",
            "request": {
                "url": "https://example.com/slides.pptx",
                "max_size_mb": 40,
                "include_metadata": True
            }
        },
        "zip_archive": {
            "description": "Process ZIP archive containing multiple files",
            "request": {
                "url": "https://example.com/documents.zip",
                "max_size_mb": 100,
                "extract_all_from_zip": True,
                "include_metadata": True
            }
        },
        "large_file_processing": {
            "description": "Process large file with custom size limit",
            "request": {
                "url": "https://example.com/large-document.pdf",
                "max_size_mb": 200,
                "include_metadata": False
            }
        }
    }
    
    return json.dumps(examples, indent=2)



# Register MCP prompts with decorators
@mcp.prompt
def analyze_crawl_results_prompt_wrapper(crawl_data: str):
    return analyze_crawl_results_prompt(crawl_data)

@mcp.prompt  
def batch_crawl_setup_prompt_wrapper(urls: str):
    return batch_crawl_setup_prompt(urls)

@mcp.prompt
def process_file_prompt_wrapper(file_url: str, file_type: str = "auto"):
    return process_file_prompt(file_url, file_type)


def setup_playwright_browsers():
    """
    Dynamic UVX-compatible Playwright browser setup with intelligent version management
    - Dynamically detect UVX environment requirements
    - Check for existing Playwright Chromium cache
    - Provide clear manual installation instructions
    - Support language-aware messaging (Japanese/English)
    """
    import os
    import platform
    from pathlib import Path
    import logging
    import glob
    import subprocess
    import re
    import sys

    logger = logging.getLogger("crawl4ai_mcp.server")

    # Language detection
    lang = os.environ.get('CRAWL4AI_LANG', os.environ.get('LANG', 'en'))
    is_japanese = lang.startswith('ja')
    
    def get_dynamic_minimum_version():
        """Calculate minimum Chromium version based on current Playwright version"""
        try:
            # Get current Playwright version
            from importlib.metadata import version
            playwright_version = version('playwright')
            logger.info(f"Current Playwright version: {playwright_version}")
            
            # Calculate minimum Chromium version based on Playwright version
            version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', playwright_version)
            if version_match:
                major, minor, patch = map(int, version_match.groups())
                if major == 1:
                    if minor >= 55:
                        return "140.0.0.0"  # Future versions
                    elif minor >= 54:
                        return "139.0.0.0"  # chromium-1181+ (Playwright 1.54+)
                    elif minor >= 52:
                        return "136.0.0.0"  # chromium-1169+ (Playwright 1.52+)
                    elif minor >= 50:
                        return "130.0.0.0"  # chromium-1100+ (Playwright 1.50+)
                    else:
                        return "120.0.0.0"
                else:
                    return "120.0.0.0"
            else:
                return "137.0.0.0"  # Fallback for unparseable versions
                
        except ImportError:
            logger.warning("Playwright not installed, using conservative minimum")
            return "137.0.0.0"
        except Exception as e:
            logger.warning(f"Error detecting Playwright version: {e}")
            return "137.0.0.0"
    
    def detect_uvx_requirements():
        """Try to detect UVX environment requirements"""
        try:
            # Check if running in UVX environment
            is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
            if is_uvx_env:
                logger.info("UVX environment detected")
                # In UVX environment, be more conservative with version requirements
                return "139.0.0.0"  # Ensure compatibility with latest UVX packages
            else:
                return None
        except Exception as e:
            logger.warning(f"Error detecting UVX environment: {e}")
            return None
    
    # Dynamic minimum version calculation
    uvx_requirement = detect_uvx_requirements()
    dynamic_minimum = get_dynamic_minimum_version()
    
    # Use the more stringent requirement
    if uvx_requirement:
        minimum_version = max(uvx_requirement, dynamic_minimum, key=lambda x: [int(i) for i in x.split('.')])
        logger.info(f"Using UVX-compatible minimum version: {minimum_version}")
    else:
        minimum_version = dynamic_minimum
        logger.info(f"Using dynamic minimum version: {minimum_version}")
    
    # Check for existing Playwright cache (platform-aware paths)
    if platform.system() == "Windows":
        # Windows uses AppData\Local\ms-playwright
        cache_pattern = str(Path.home() / "AppData" / "Local" / "ms-playwright" / "chromium-*")
    else:
        # Unix systems use .cache/ms-playwright
        cache_pattern = str(Path.home() / ".cache" / "ms-playwright" / "chromium-*")
    
    cache_dirs = glob.glob(cache_pattern)
    valid_cache = False
    current_version = None
    
    for cache_dir in cache_dirs:
        if platform.system() == "Windows":
            chrome_path = Path(cache_dir) / "chrome-win" / "chrome.exe"
        else:
            chrome_path = Path(cache_dir) / "chrome-linux" / "chrome"
        
        if chrome_path.exists():
            try:
                if platform.system() == "Windows":
                    # Windows: Extract version from directory name (chromium-1181 format)
                    # to avoid GUI window opening with --version
                    dir_name = Path(cache_dir).name
                    build_match = re.search(r'chromium-(\d+)', dir_name)
                    if build_match:
                        build_number = int(build_match.group(1))
                        # Map build numbers to Chrome versions (based on official Playwright mappings)
                        if build_number >= 1181:
                            current_version = "139.0.0.0"  # Latest builds
                        elif build_number >= 1178:
                            current_version = "138.0.0.0"  # chromium-1178 → 138.0.7204.15
                        elif build_number >= 1169:
                            current_version = "136.0.0.0"  # chromium-1169 → 136.0.7103.25
                        elif build_number >= 1100:
                            current_version = "130.0.0.0"  # Older builds
                        else:
                            current_version = "120.0.0.0"  # Very old builds
                else:
                    # Unix: Use --version command (works without GUI)
                    result = subprocess.run(
                        [str(chrome_path), "--version"], 
                        capture_output=True, text=True, timeout=10
                    )
                    version_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', result.stdout)
                    if version_match:
                        current_version = version_match.group(1)
                    else:
                        continue
                
                # Simple version comparison - split and compare numeric parts
                current_parts = list(map(int, current_version.split('.')))
                minimum_parts = list(map(int, minimum_version.split('.')))
                
                if current_parts >= minimum_parts:
                    valid_cache = True
                    break
            except Exception as e:
                logger.warning(f"Failed to check Chromium version: {e}")
    
    # Provide guidance based on cache status
    if valid_cache:
        if is_japanese:
            print(f"✅ 有効なPlaywrightキャッシュを検出: {current_version}")
            print("   UVXは既存のキャッシュを自動的に使用します。")
        else:
            print(f"✅ Valid Playwright cache found: {current_version}")
            print("   UVX will automatically use existing cache.")
        logger.info(f"Using existing Playwright cache: {current_version}")
        return True
    else:
        if is_japanese:
            if current_version:
                print(f"⚠️  古いPlaywrightキャッシュ: {current_version} < {minimum_version}")
            else:
                print("⚠️  Playwrightキャッシュがありません。")
            print("\n📋 手動でChromiumキャッシュをインストール:")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate") 
            print("   pip install playwright")
            print("   python -m playwright install chromium")
            print("\n🎯 インストール後のUVX実行:")
            print("   uvx --from crawl4ai-dxt-correct crawl4ai_mcp")
        else:
            if current_version:
                print(f"⚠️  Outdated Playwright cache: {current_version} < {minimum_version}")
            else:
                print("⚠️  No Playwright cache found.")
            print("\n📋 Manual Chromium cache installation:")
            print("   python3 -m venv venv")
            if platform.system() == "Windows":
                print("   venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
            print("   pip install playwright")
            print("   python -m playwright install chromium")
            print("\n🎯 UVX execution after installation:")
            print("   uvx --from crawl4ai-dxt-correct crawl4ai_mcp")
        
        logger.warning("Manual Playwright Chromium installation required")
        return False


# Installation status storage (module level) - removed to avoid syntax errors


@mcp.tool
async def get_system_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive system diagnostics for troubleshooting UVX and browser issues.
    
    Returns detailed information about the environment, browser installations,
    and provides specific recommendations for fixing issues.
    """
    import sys
    import os
    import time
    import subprocess
    from pathlib import Path
    
    try:
        # Environment detection
        is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
        is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        # Python environment info
        python_info = {
            'version': sys.version,
            'executable': str(sys.executable),
            'is_uvx_environment': is_uvx_env,
            'is_virtual_environment': is_venv,
            'platform': os.name,
            'python_path': sys.path[:3]  # First 3 entries for brevity
        }
        
        # Check Playwright installation with robust version detection
        playwright_available = False
        playwright_version = None
        try:
            import playwright
            playwright_available = True
            
            # Try multiple methods to get version
            try:
                # Method 1: Try importlib.metadata (Python 3.8+)
                from importlib.metadata import version
                playwright_version = version('playwright')
            except Exception:
                try:
                    # Method 2: Try pkg_resources (fallback)
                    import pkg_resources
                    playwright_version = pkg_resources.get_distribution('playwright').version
                except Exception:
                    try:
                        # Method 3: Try direct attribute access
                        playwright_version = playwright.__version__
                    except Exception:
                        # Method 4: Check if we can get it from the sync_api
                        try:
                            from playwright.sync_api import Playwright
                            playwright_version = "Available (version detection failed)"
                        except Exception:
                            playwright_version = "Unknown"
                            
        except ImportError:
            pass
        
        # Use Playwright's native browser detection (best practice)
        browser_status = {
            'chromium_installed': False,
            'browser_list': [],
            'installation_output': None
        }
        
        try:
            # Use playwright install --list to get accurate browser status
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "--list"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                browser_status['installation_output'] = result.stdout
                # Parse output to determine if chromium is installed
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'chromium' in line.lower():
                        browser_status['browser_list'].append(line.strip())
                        if 'installed' in line.lower() or '✓' in line:
                            browser_status['chromium_installed'] = True
                            
        except Exception as e:
            browser_status['installation_output'] = f"Error checking browsers: {str(e)}"
        
        # Installation status (simplified without global state)
        installation_status = {
            "note": "Installation status tracking simplified to avoid global variable issues",
            "recommendation": "Run browser installation commands manually if needed"
        }
        
        # Generate recommendations
        recommendations = []
        issues_found = []
        
        if not playwright_available:
            issues_found.append("Playwright library not available")
            recommendations.append("Install Playwright: pip install playwright>=1.40.0")
        
        if not browser_status['chromium_installed']:
            issues_found.append("Chromium browser not properly installed")
            recommendations.extend([
                "Install Chromium browser with dependencies:",
                "python -m playwright install --with-deps chromium",
                "This will install the correct revision managed by Playwright"
            ])
        
        if is_uvx_env and issues_found:
            recommendations.append("UVX environments may have browser persistence issues - consider STDIO local setup for development")
        
        # Test basic functionality
        can_create_crawler = False
        crawler_error = None
        try:
            # Test if we can import crawler (doesn't actually create instance)
            from crawl4ai import AsyncWebCrawler
            can_create_crawler = True
        except ImportError as e:
            crawler_error = f"Import error: {str(e)}"
        except Exception as e:
            crawler_error = f"Unexpected error: {str(e)}"
        
        return {
            "success": True,
            "timestamp": str(int(time.time())),
            "environment": python_info,
            "playwright": {
                "available": playwright_available,
                "version": playwright_version,
                "can_create_crawler": can_create_crawler,
                "crawler_error": crawler_error
            },
            "browsers": browser_status,
            "installation_history": installation_status,
            "issues_found": issues_found,
            "recommendations": recommendations,
            "ready_for_crawling": (
                playwright_available and 
                browser_status['chromium_installed'] and
                can_create_crawler
            )
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        return {
            "success": False,
            "error": f"Diagnostics failed: {str(e)}",
            "error_type": type(e).__name__,
            "error_details": error_details,
            "partial_info": {
                "python_executable": str(sys.executable),
                "platform": os.name,
                "working_directory": os.getcwd()
            },
            "recommendations": [
                "Check the error_details field for specific failure information",
                "Verify that crawl4ai dependencies are properly installed",
                "For UVX environments, consider switching to STDIO local setup",
                "Try running: pip install --upgrade playwright crawl4ai"
            ]
        }


# get_system_diagnostics now uses @mcp.tool decorator


def main():
    """Main entry point for the MCP server."""
    import sys
    
    # Setup Playwright browsers following best practices
    setup_playwright_browsers()
    
    # Ensure all output is suppressed
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Crawl4AI MCP Server")
        print("Usage: python -m crawl4ai_mcp.server [--transport TRANSPORT] [--host HOST] [--port PORT]")
        print("Transports: stdio (default), streamable-http, sse")
        return
    
    # Parse command line arguments
    transport = "stdio"
    host = "127.0.0.1"
    port = 8000
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1
    
    # Run the server
    if transport == "stdio":
        mcp.run()
    elif transport == "streamable-http" or transport == "http":
        mcp.run(transport="streamable-http", host=host, port=port)
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        print(f"Unknown transport: {transport}")
        print("Available transports: stdio, streamable-http, sse")
        sys.exit(1)


if __name__ == "__main__":
    main()