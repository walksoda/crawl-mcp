"""Unified LLM client for content summarization.

This module provides a unified interface for interacting with different LLM
providers (OpenAI, Anthropic, Ollama) for content summarization tasks.
"""

import json
import os
from typing import Dict, Any, Optional, Tuple

from ..constants import (
    SUMMARY_LENGTH_CONFIGS,
    SUPPORTED_LLM_PROVIDERS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    SUMMARIZATION_TEMPERATURE,
    LLM_API_TIMEOUT,
    MAX_CONTENT_FOR_LLM,
)


def get_llm_config_safe(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Tuple[str, str]:
    """
    Get LLM configuration safely with fallback to defaults.

    Args:
        provider: LLM provider name (openai, anthropic, ollama)
        model: Model name to use

    Returns:
        Tuple of (provider, model)
    """
    try:
        from ..config import get_llm_config
        llm_config = get_llm_config(provider, model)

        if hasattr(llm_config, 'provider'):
            provider_info = llm_config.provider.split('/')
            resolved_provider = provider_info[0] if provider_info else DEFAULT_LLM_PROVIDER
            resolved_model = provider_info[1] if len(provider_info) > 1 else DEFAULT_LLM_MODEL
            return resolved_provider, resolved_model

    except ImportError:
        pass

    # Fallback to defaults
    return provider or DEFAULT_LLM_PROVIDER, model or DEFAULT_LLM_MODEL


class LLMClient:
    """Unified LLM client for content summarization."""

    # Default URL for Ollama local server
    OLLAMA_DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = SUMMARIZATION_TEMPERATURE,
        timeout: int = LLM_API_TIMEOUT
    ):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider (openai, anthropic, ollama, aoai)
            model: Model name
            api_key: API key (optional, will use environment variable if not provided)
            base_url: Base URL for API (required for ollama, optional for Azure OpenAI)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url  # Don't set default here; handle per-provider
        self.temperature = temperature
        self.timeout = timeout

    async def summarize(
        self,
        content: str,
        title: str = "",
        url: str = "",
        summary_length: str = "medium",
        content_type: str = "document",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        target_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Summarize content using LLM with enhanced metadata preservation.

        Args:
            content: The content to summarize
            title: Title of the content
            url: Source URL
            summary_length: "short", "medium", or "long"
            content_type: Type of content (document, video, webpage)
            llm_provider: Override provider for this call
            llm_model: Override model for this call
            target_tokens: Target token count for summary
            metadata: Additional metadata to preserve

        Returns:
            Dictionary with summary and metadata
        """
        try:
            # Get provider and model
            provider = llm_provider or self.provider
            model = llm_model or self.model

            if not provider or not model:
                provider, model = get_llm_config_safe(provider, model)

            if provider not in SUPPORTED_LLM_PROVIDERS:
                return {
                    "success": False,
                    "error": f"Provider {provider} not supported. Supported: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
                }

            # Get summary configuration
            config = SUMMARY_LENGTH_CONFIGS.get(
                summary_length,
                SUMMARY_LENGTH_CONFIGS["medium"]
            )
            if target_tokens:
                config = {**config, "target_tokens": target_tokens}

            # Truncate content if too long
            if len(content) > MAX_CONTENT_FOR_LLM:
                content = content[:MAX_CONTENT_FOR_LLM] + "\n\n[Content truncated for summarization]"

            # Build prompt
            prompt = self._build_summarization_prompt(
                content=content,
                title=title,
                url=url,
                config=config,
                content_type=content_type,
                metadata=metadata
            )

            # Call appropriate provider
            if provider == "openai":
                extracted_content = await self._call_openai(prompt, model, config)
            elif provider == "anthropic":
                extracted_content = await self._call_anthropic(prompt, model, config)
            elif provider == "ollama":
                extracted_content = await self._call_ollama(prompt, model)
            elif provider == "aoai":
                extracted_content = await self._call_azure_openai(prompt, model, config)
            else:
                return {
                    "success": False,
                    "error": f"Provider {provider} not implemented"
                }

            # Parse and return response
            return self._parse_response(
                extracted_content=extracted_content,
                title=title,
                url=url,
                content=content,
                config=config,
                summary_length=summary_length,
                content_type=content_type,
                provider=provider,
                model=model,
                metadata=metadata
            )

        except Exception as e:
            return {
                "success": False,
                "error": f"Summarization failed: {str(e)}"
            }

    def _build_summarization_prompt(
        self,
        content: str,
        title: str,
        url: str,
        config: Dict[str, Any],
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the summarization prompt."""
        # Build metadata context
        metadata_lines = []
        if title:
            metadata_lines.append(f"- Title: {title}")
        if url:
            metadata_lines.append(f"- Source: {url}")
        if content_type:
            metadata_lines.append(f"- Content Type: {content_type}")

        if metadata:
            for key, value in metadata.items():
                if value and key not in ["title", "url", "content_type"]:
                    metadata_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        metadata_context = "\n".join(metadata_lines) if metadata_lines else "No metadata available"

        instruction = f"""
Summarize this {content_type} content in {config['target_length']}.
Focus on {config['detail_level']}.
Target length: approximately {config['target_tokens']} tokens.

Content Information:
{metadata_context}

Structure your summary with:
1. Brief overview including title and context
2. Main topics or sections covered
3. Key insights, findings, or conclusions
4. Important details or examples mentioned

Make the summary informative and well-structured, preserving important details.
IMPORTANT: Preserve the title and source information in your response.
"""

        prompt = f"""
{instruction}

Please provide a JSON response with the following structure:
{{
    "summary": "The summarized content (approximately {config['target_tokens']} tokens)",
    "key_topics": ["List", "of", "main", "topics"],
    "main_insights": ["Key", "findings", "or", "insights"],
    "content_type": "{content_type}"
}}

Content to summarize:
{content}
"""
        return prompt

    async def _call_openai(
        self,
        prompt: str,
        model: str,
        config: Dict[str, Any]
    ) -> str:
        """Call OpenAI API."""
        import openai

        api_key = self.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")

        client = openai.AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes content while preserving important metadata."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=min(4000, config['target_tokens'] * 2)
        )

        return response.choices[0].message.content

    async def _call_anthropic(
        self,
        prompt: str,
        model: str,
        config: Dict[str, Any]
    ) -> str:
        """Call Anthropic API."""
        import anthropic

        api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found")

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=model,
            max_tokens=min(4000, config['target_tokens'] * 2),
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    async def _call_ollama(
        self,
        prompt: str,
        model: str
    ) -> str:
        """Call Ollama API."""
        import aiohttp

        # Use provided base_url or default to local Ollama server
        base_url = self.base_url or self.OLLAMA_DEFAULT_URL

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature}
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '')
                else:
                    error_text = await response.text()
                    raise ValueError(
                        f"Ollama API request failed: {response.status} - {error_text}"
                    )

    async def _call_azure_openai(
        self,
        prompt: str,
        model: str,
        config: Dict[str, Any]
    ) -> str:
        """Call Azure OpenAI API."""
        import openai

        api_key = self.api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        # Use explicit base_url if provided, otherwise fall back to environment variable
        # Note: self.base_url is None by default, so environment variable will be used
        api_base = self.base_url if self.base_url else os.environ.get('AZURE_OPENAI_ENDPOINT')
        api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

        if not api_key:
            raise ValueError("Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable.")
        if not api_base:
            raise ValueError("Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable or provide base_url.")

        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes content while preserving important metadata."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=min(4000, config['target_tokens'] * 2)
        )

        return response.choices[0].message.content

    def _parse_response(
        self,
        extracted_content: str,
        title: str,
        url: str,
        content: str,
        config: Dict[str, Any],
        summary_length: str,
        content_type: str,
        provider: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Parse LLM response and build result dictionary."""
        if not extracted_content:
            return {
                "success": False,
                "error": "LLM returned empty result"
            }

        try:
            # Clean up the extracted content if it's wrapped in markdown
            content_to_parse = extracted_content
            if content_to_parse.startswith('```json'):
                content_to_parse = content_to_parse.replace('```json', '').replace('```', '').strip()
            elif content_to_parse.startswith('```'):
                content_to_parse = content_to_parse.replace('```', '').strip()

            summary_data = json.loads(content_to_parse)

            result = {
                "success": True,
                "summary": summary_data.get("summary", "Summary generation failed"),
                "key_topics": summary_data.get("key_topics", []),
                "main_insights": summary_data.get("main_insights", []),
                "content_type": summary_data.get("content_type", content_type),
                "summary_length": summary_length,
                "target_tokens": config['target_tokens'],
                "estimated_summary_tokens": len(summary_data.get("summary", "")) // 4,
                "original_length": len(content),
                "compression_ratio": (
                    len(summary_data.get("summary", "")) / len(content)
                    if content else 0
                ),
                "llm_provider": provider,
                "llm_model": model,
            }

            # Add title and url if provided
            if title:
                result["title"] = title
            if url:
                result["source_url"] = url

            # Add any additional metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in result and value:
                        result[key] = value

            return result

        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat as plain text summary
            return {
                "success": True,
                "summary": str(extracted_content),
                "key_topics": [],
                "main_insights": [],
                "content_type": content_type,
                "summary_length": summary_length,
                "target_tokens": config['target_tokens'],
                "estimated_summary_tokens": len(str(extracted_content)) // 4,
                "original_length": len(content),
                "compression_ratio": (
                    len(str(extracted_content)) / len(content)
                    if content else 0
                ),
                "llm_provider": provider,
                "llm_model": model,
                "title": title,
                "source_url": url,
                "fallback_mode": True,
            }


# Convenience function for direct use
async def summarize_content(
    content: str,
    title: str = "",
    url: str = "",
    summary_length: str = "medium",
    content_type: str = "document",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    target_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to summarize content using default LLM client.

    This function creates a temporary LLMClient instance and calls summarize.
    For repeated calls, consider creating an LLMClient instance directly.
    """
    client = LLMClient()
    return await client.summarize(
        content=content,
        title=title,
        url=url,
        summary_length=summary_length,
        content_type=content_type,
        llm_provider=llm_provider,
        llm_model=llm_model,
        target_tokens=target_tokens,
        metadata=metadata
    )
