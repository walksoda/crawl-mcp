"""
LLM Extraction Client for unified LLM API calls in extraction tasks.

This module provides a unified client for making LLM API calls across different
providers (OpenAI, Anthropic, Ollama) with support for JSON response parsing.
"""

import json
import os
from typing import Any, Dict, Optional

# Ollama default URL
OLLAMA_DEFAULT_URL = "http://localhost:11434"


class LLMExtractionClient:
    """Unified LLM client for extraction tasks.

    Provides a simple interface for making LLM API calls across different providers
    with built-in JSON response parsing and error handling.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the LLM extraction client.

        Args:
            provider: LLM provider (openai, anthropic, ollama)
            model: Model name
            api_key: API key (optional, will use environment variable if not provided)
            base_url: Base URL for API (required for ollama, optional for others)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def from_config(cls, llm_provider: Optional[str] = None, llm_model: Optional[str] = None) -> "LLMExtractionClient":
        """Create client from LLM configuration.

        Args:
            llm_provider: Override provider (optional)
            llm_model: Override model (optional)

        Returns:
            Configured LLMExtractionClient instance
        """
        try:
            from ..config import get_llm_config
        except ImportError:
            from config import get_llm_config

        llm_config = get_llm_config(llm_provider, llm_model)

        # Parse provider info from config
        provider_info = llm_config.provider.split('/')
        provider = provider_info[0] if provider_info else 'openai'
        model = provider_info[1] if len(provider_info) > 1 else 'gpt-4o'

        return cls(
            provider=provider,
            model=model,
            api_key=llm_config.api_token,
            base_url=llm_config.base_url
        )

    async def call_llm(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> str:
        """Make an LLM API call.

        Args:
            prompt: User prompt to send
            system_message: System message for context
            temperature: Sampling temperature (default 0.1 for extraction)
            max_tokens: Maximum tokens in response

        Returns:
            Raw LLM response text

        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        if self.provider == 'openai':
            return await self._call_openai(prompt, system_message, temperature, max_tokens)
        elif self.provider == 'anthropic':
            return await self._call_anthropic(prompt, system_message, temperature, max_tokens)
        elif self.provider == 'ollama':
            return await self._call_ollama(prompt, temperature)
        else:
            raise ValueError(f"LLM provider '{self.provider}' not supported. Supported: openai, anthropic, ollama")

    async def _call_openai(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call OpenAI API."""
        import openai

        api_key = self.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")

        client = openai.AsyncOpenAI(api_key=api_key, base_url=self.base_url)

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    async def _call_anthropic(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Anthropic API."""
        import anthropic

        api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found")

        client = anthropic.AsyncAnthropic(api_key=api_key)

        # Anthropic uses system parameter separately
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system_message:
            kwargs["system"] = system_message

        response = await client.messages.create(**kwargs)

        return response.content[0].text

    async def _call_ollama(
        self,
        prompt: str,
        temperature: float
    ) -> str:
        """Call Ollama API."""
        import aiohttp

        base_url = self.base_url or OLLAMA_DEFAULT_URL

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '')
                else:
                    error_text = await response.text()
                    raise ValueError(f"Ollama API request failed: {response.status} - {error_text}")

    @staticmethod
    def clean_json_response(content: str) -> str:
        """Remove markdown code block markers from JSON response.

        Args:
            content: Raw LLM response that may contain ```json markers

        Returns:
            Cleaned content ready for JSON parsing
        """
        if not content:
            return content

        content = content.strip()

        # Remove ```json or ``` markers
        if content.startswith('```json'):
            content = content[7:]  # Remove '```json'
        elif content.startswith('```'):
            content = content[3:]  # Remove '```'

        if content.endswith('```'):
            content = content[:-3]  # Remove trailing '```'

        return content.strip()

    @staticmethod
    def parse_json_response(content: str) -> Dict[str, Any]:
        """Parse JSON response with automatic markdown cleanup.

        Args:
            content: Raw LLM response (may contain markdown markers)

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        cleaned = LLMExtractionClient.clean_json_response(content)
        return json.loads(cleaned)

    async def extract_json(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Make an LLM call and parse JSON response.

        Convenience method that combines call_llm and parse_json_response.

        Args:
            prompt: User prompt to send
            system_message: System message for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON as dictionary

        Raises:
            ValueError: If LLM call fails
            json.JSONDecodeError: If JSON parsing fails
        """
        response = await self.call_llm(prompt, system_message, temperature, max_tokens)
        if not response:
            raise ValueError("LLM returned empty response")
        return self.parse_json_response(response)


# Convenience function for quick extraction
async def extract_with_llm(
    prompt: str,
    system_message: str = "",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    parse_json: bool = True
) -> Dict[str, Any] | str:
    """Quick LLM extraction with automatic configuration.

    Args:
        prompt: User prompt to send
        system_message: System message for context
        llm_provider: Override provider (optional)
        llm_model: Override model (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        parse_json: Whether to parse response as JSON

    Returns:
        Parsed JSON dict or raw string based on parse_json flag
    """
    client = LLMExtractionClient.from_config(llm_provider, llm_model)

    if parse_json:
        return await client.extract_json(prompt, system_message, temperature, max_tokens)
    else:
        return await client.call_llm(prompt, system_message, temperature, max_tokens)
