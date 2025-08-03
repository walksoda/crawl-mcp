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
    llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize web page content using LLM when content is too large for context.
    
    Args:
        content: The web page content to summarize (markdown or text)
        title: Page title for context
        url: Source URL for reference
        summary_length: "short", "medium", or "long" summary
        llm_provider: LLM provider to use
        llm_model: Specific model to use
        
    Returns:
        Dictionary with summary and metadata
    """
    try:
        # Import config here to avoid circular imports
        try:
            from .config import get_llm_config
        except ImportError:
            from config import get_llm_config
        
        # Define summary lengths
        length_configs = {
            "short": {
                "target_length": "2-3 paragraphs",
                "detail_level": "key points and main conclusions only"
            },
            "medium": {
                "target_length": "4-6 paragraphs", 
                "detail_level": "main topics with important details and examples"
            },
            "long": {
                "target_length": "8-12 paragraphs",
                "detail_level": "comprehensive overview with subtopics, examples, and analysis"
            }
        }
        
        config = length_configs.get(summary_length, length_configs["medium"])
        
        # Prepare instruction for LLM
        instruction = f"""
        Summarize this web page content in {config['target_length']}.
        Focus on {config['detail_level']}.
        
        Page Information:
        - Title: {title}
        - URL: {url}
        
        Structure your summary with:
        1. Brief overview of the page purpose and main topic
        2. Key sections or categories discussed
        3. Important information, data, or insights
        4. Relevant examples, quotes, or specific details
        5. Conclusions or actionable information
        
        Make the summary informative and well-structured, preserving important technical details and maintaining the original context.
        """
        
        # Get LLM configuration
        llm_config = get_llm_config(llm_provider, llm_model)
        
        # Create the prompt for summarization
        prompt = f"""
        {instruction}
        
        Please provide a JSON response with the following structure:
        {{
            "summary": "The comprehensive summary of the content",
            "key_topics": ["List", "of", "main", "topics", "covered"],
            "content_type": "Type/category of the webpage (e.g., 'Documentation', 'Article', 'Blog Post')",
            "main_insights": ["Key", "insights", "or", "takeaways"],
            "technical_details": ["Important", "technical", "information", "if", "any"]
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
                        {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries of web page content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2500  # Generous token limit for detailed summaries
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
                
                return {
                    "success": True,
                    "summary": summary_data.get("summary", "Summary generation failed"),
                    "key_topics": summary_data.get("key_topics", []),
                    "content_type": summary_data.get("content_type", "Unknown"),
                    "main_insights": summary_data.get("main_insights", []),
                    "technical_details": summary_data.get("technical_details", []),
                    "summary_length": summary_length,
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
                    "key_topics": [],
                    "content_type": "Unknown",
                    "main_insights": [],
                    "technical_details": [],
                    "summary_length": summary_length,
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
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and combined_content:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(combined_content) // 4
                    
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
                                llm_model=request.llm_model
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
                                    "compression_ratio": summary_result.get("compressed_ratio", 0),
                                    "key_topics": summary_result.get("key_topics", []),
                                    "content_type": summary_result.get("content_type", "Unknown"),
                                    "main_insights": summary_result.get("main_insights", []),
                                    "technical_details": summary_result.get("technical_details", []),
                                    "llm_provider": summary_result.get("llm_provider", "unknown"),
                                    "llm_model": summary_result.get("llm_model", "unknown"),
                                    "auto_summarization_trigger": f"Deep crawl content exceeded {request.max_content_tokens} tokens"
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
                
                # Apply auto-summarization if enabled and content is large
                if request.auto_summarize and content_to_use:
                    # Rough token estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(content_to_use) // 4
                    
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
                                llm_model=request.llm_model
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
                    f"1. Run system diagnostics: get_system_diagnostics()\n" \
                    f"2. Manual browser installation:\n" \
                    f"   - uvx --with playwright playwright install webkit\n" \
                    f"   - Or system-wide: playwright install webkit\n" \
                    f"3. Restart Claude Desktop after installation\n" \
                    f"4. If issues persist, consider STDIO local setup\n\n" \
                    f"💡 WebKit is lightweight (~180MB) vs Chromium (~281MB)"
            else:
                error_message += "\n\n🔧 Browser Setup Required:\n" \
                    f"1. Install Playwright browsers:\n" \
                    f"   playwright install webkit  # Lightweight option\n" \
                    f"   playwright install chromium  # Full compatibility\n" \
                    f"2. For system dependencies: sudo apt-get install libnss3 libnspr4 libasound2\n" \
                    f"3. Run diagnostics: get_system_diagnostics()"
        
        # Check if we have installation status information
        try:
            if _playwright_installation_status and _playwright_installation_status.get('needs_manual_setup'):
                error_message += f"\n\n📊 Installation Status: {_playwright_installation_status}"
        except (NameError, KeyError):
            # Ignore if global variable not accessible
            pass
        
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
    # YouTube tools
    extract_youtube_transcript,
    batch_extract_youtube_transcripts,
    get_youtube_video_info,
    get_youtube_api_setup_guide,
    
    # File processing tools
    process_file,
    get_supported_file_formats,
    enhanced_process_large_content,
    
    # Web crawling tools
    crawl_url,
    deep_crawl_site,
    crawl_url_with_fallback,
    intelligent_extract,
    extract_entities,
    extract_structured_data,
    
    # Search tools
    search_google,
    batch_search_google,
    search_and_crawl,
    get_search_genres,
    
    # Utility tools
    get_llm_config_info,
    batch_crawl,
    get_tool_selection_guide
)

# Apply MCP decorators to all tools

# YouTube tools
mcp.tool(extract_youtube_transcript)
mcp.tool(batch_extract_youtube_transcripts)
mcp.tool(get_youtube_video_info)
mcp.tool(get_youtube_api_setup_guide)

# File processing tools
mcp.tool(process_file)
mcp.tool(get_supported_file_formats)
mcp.tool(enhanced_process_large_content)

# Web crawling tools
mcp.tool(crawl_url)
mcp.tool(deep_crawl_site)
mcp.tool(crawl_url_with_fallback)
mcp.tool(intelligent_extract)
mcp.tool(extract_entities)
mcp.tool(extract_structured_data)

# Search tools
mcp.tool(search_google)
mcp.tool(batch_search_google)
mcp.tool(search_and_crawl)
mcp.tool(get_search_genres)

# Utility tools
mcp.tool(get_llm_config_info)
mcp.tool(batch_crawl)
mcp.tool(get_tool_selection_guide)
mcp.tool(get_system_diagnostics)

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


def setup_lightweight_playwright_browsers():
    """Setup lightweight Playwright browsers with enhanced error reporting for UVX environments."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    global _playwright_installation_status
    
    # Detect UVX environment
    is_uvx_env = 'UV_PROJECT_ENVIRONMENT' in os.environ or 'UVX' in str(sys.executable)
    
    try:
        # Check if Playwright browsers are installed
        possible_cache_dirs = [
            Path.home() / ".cache" / "ms-playwright",
            Path.home() / "Library" / "Caches" / "ms-playwright",  # macOS
            Path.home() / "AppData" / "Local" / "ms-playwright"    # Windows
        ]
        
        webkit_installed = False
        chromium_installed = False
        installation_attempted = False
        
        # Enhanced browser detection
        for cache_dir in possible_cache_dirs:
            if cache_dir.exists():
                # Check for WebKit with more robust detection
                webkit_dirs = list(cache_dir.glob("webkit-*"))
                for webkit_dir in webkit_dirs:
                    webkit_executables = list(webkit_dir.rglob("*webkit*")) + list(webkit_dir.rglob("*Playwright*"))
                    if webkit_executables:
                        webkit_installed = True
                        break
                
                # Check for Chromium with more robust detection  
                chromium_dirs = list(cache_dir.glob("chromium-*"))
                for chromium_dir in chromium_dirs:
                    chrome_executables = list(chromium_dir.rglob("chrome*")) + list(chromium_dir.rglob("*chromium*"))
                    if chrome_executables:
                        chromium_installed = True
                        break
                
                if webkit_installed or chromium_installed:
                    break
        
        # Install browsers if needed
        if not webkit_installed and not chromium_installed:
            installation_attempted = True
            
            # Try WebKit first (most lightweight ~180MB)
            try:
                result = subprocess.run([
                    sys.executable, "-m", "playwright", "install", "webkit"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # WebKit installation successful
                    webkit_installed = True
                else:
                    # WebKit failed, try Chromium headless shell
                    result2 = subprocess.run([
                        sys.executable, "-m", "playwright", "install", "--only-shell"
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result2.returncode == 0:
                        chromium_installed = True
                    else:
                        # Both installations failed - this will be handled in error reporting
                        pass
                        
            except subprocess.TimeoutExpired:
                # Installation timeout - browsers might still be installing in background
                pass
            except FileNotFoundError:
                # Playwright not available in the environment
                pass
            except Exception as e:
                # Other installation errors will be handled below
                pass
        
        # For UVX environments, add additional diagnostic information
        if is_uvx_env and installation_attempted and not (webkit_installed or chromium_installed):
            # Store error info for later diagnostic reporting
            _playwright_installation_status = {
                'uvx_environment': True,
                'installation_attempted': True,
                'webkit_installed': webkit_installed,
                'chromium_installed': chromium_installed,
                'cache_dirs_checked': [str(d) for d in possible_cache_dirs if d.exists()],
                'needs_manual_setup': True
            }
        
    except Exception as e:
        # Store diagnostic info even for unexpected errors
        _playwright_installation_status = {
            'uvx_environment': is_uvx_env,
            'setup_error': str(e),
            'needs_manual_setup': True
        }


# Global variable to store installation status
_playwright_installation_status = {}


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
        
        # Check Playwright installation
        playwright_available = False
        playwright_version = None
        try:
            import playwright
            playwright_available = True
            playwright_version = playwright.__version__
        except ImportError:
            pass
        
        # Browser detection with detailed info
        possible_cache_dirs = [
            Path.home() / ".cache" / "ms-playwright",
            Path.home() / "Library" / "Caches" / "ms-playwright",  # macOS
            Path.home() / "AppData" / "Local" / "ms-playwright"    # Windows
        ]
        
        browser_status = {
            'webkit_installed': False,
            'chromium_installed': False,
            'cache_directories': [],
            'browser_details': []
        }
        
        for cache_dir in possible_cache_dirs:
            if cache_dir.exists():
                browser_status['cache_directories'].append(str(cache_dir))
                
                # Check WebKit
                webkit_dirs = list(cache_dir.glob("webkit-*"))
                for webkit_dir in webkit_dirs:
                    webkit_executables = list(webkit_dir.rglob("*webkit*")) + list(webkit_dir.rglob("*Playwright*"))
                    if webkit_executables:
                        browser_status['webkit_installed'] = True
                        browser_status['browser_details'].append({
                            'type': 'webkit',
                            'path': str(webkit_dir),
                            'executables': [str(f) for f in webkit_executables[:3]]  # Limit for brevity
                        })
                
                # Check Chromium
                chromium_dirs = list(cache_dir.glob("chromium-*"))
                for chromium_dir in chromium_dirs:
                    chrome_executables = list(chromium_dir.rglob("chrome*")) + list(chromium_dir.rglob("*chromium*"))
                    if chrome_executables:
                        browser_status['chromium_installed'] = True
                        browser_status['browser_details'].append({
                            'type': 'chromium',
                            'path': str(chromium_dir),
                            'executables': [str(f) for f in chrome_executables[:3]]  # Limit for brevity
                        })
        
        # Installation status from setup function
        try:
            installation_status = _playwright_installation_status.copy() if _playwright_installation_status else {}
        except (NameError, KeyError):
            installation_status = {}
        
        # Generate recommendations
        recommendations = []
        issues_found = []
        
        if not playwright_available:
            issues_found.append("Playwright library not available")
            recommendations.append("Install Playwright: pip install playwright>=1.40.0")
        
        if not (browser_status['webkit_installed'] or browser_status['chromium_installed']):
            issues_found.append("No Playwright browsers found")
            if is_uvx_env:
                recommendations.extend([
                    "For UVX environments, try manual installation:",
                    "1. uvx --from git+https://github.com/walksoda/crawl-mcp --with playwright playwright install webkit",
                    "2. Or install system-wide: playwright install webkit",
                    "3. Restart Claude Desktop after installation"
                ])
            else:
                recommendations.extend([
                    "Install browsers manually:",
                    "playwright install webkit  # Lightweight option (~180MB)",
                    "playwright install chromium  # Full compatibility option (~281MB)"
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
                (browser_status['webkit_installed'] or browser_status['chromium_installed']) and
                can_create_crawler
            )
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Diagnostics failed: {str(e)}",
            "recommendations": [
                "Try running system diagnostic again",
                "Check if crawl4ai dependencies are properly installed",
                "For UVX environments, consider switching to STDIO local setup"
            ]
        }


def main():
    """Main entry point for the MCP server."""
    import sys
    
    # Setup lightweight Playwright browsers if needed
    setup_lightweight_playwright_browsers()
    
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