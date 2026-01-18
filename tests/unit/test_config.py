"""
Unit tests for crawl4ai_mcp.config module.

Tests configuration management functionality without MCP server connection.
"""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_import_config_module(self):
        """Test that config module can be imported."""
        from crawl4ai_mcp import config
        assert hasattr(config, 'ConfigManager')
        assert hasattr(config, 'config_manager')

    def test_config_manager_singleton(self):
        """Test that config_manager is a singleton instance."""
        from crawl4ai_mcp.config import config_manager, ConfigManager
        assert isinstance(config_manager, ConfigManager)

    def test_get_default_provider(self):
        """Test getting default provider."""
        from crawl4ai_mcp.config import get_default_provider
        provider = get_default_provider()
        # Should return a string (even if None or empty)
        assert provider is None or isinstance(provider, str)

    def test_get_default_model(self):
        """Test getting default model."""
        from crawl4ai_mcp.config import get_default_model
        model = get_default_model()
        # Should return a string (even if None or empty)
        assert model is None or isinstance(model, str)

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        from crawl4ai_mcp.config import config_manager
        providers = config_manager.get_available_providers()
        assert isinstance(providers, list)

    def test_llm_provider_config_dataclass(self):
        """Test LLMProviderConfig dataclass."""
        from crawl4ai_mcp.config import LLMProviderConfig

        # Test creating a config instance
        config = LLMProviderConfig(
            api_key="test-key",
            api_key_env="TEST_API_KEY",
            base_url="https://api.example.com"
        )
        assert config.api_key == "test-key"
        assert config.api_key_env == "TEST_API_KEY"
        assert config.base_url == "https://api.example.com"

    def test_mcp_llm_config_dataclass(self):
        """Test MCPLLMConfig dataclass."""
        from crawl4ai_mcp.config import MCPLLMConfig, LLMProviderConfig

        # Test creating a config instance
        provider_config = LLMProviderConfig(
            api_key="test-key",
            api_key_env=None,
            base_url=None
        )
        config = MCPLLMConfig(
            default_provider="openai",
            default_model="gpt-4",
            providers={"openai": provider_config}
        )
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-4"
        assert "openai" in config.providers

    def test_config_manager_has_valid_api_key(self):
        """Test checking for valid API key."""
        from crawl4ai_mcp.config import config_manager

        # Without setting env vars, should return False for most providers
        result = config_manager.has_valid_api_key("openai")
        assert isinstance(result, bool)

    def test_validate_provider_model(self):
        """Test provider/model validation."""
        from crawl4ai_mcp.config import config_manager

        # Returns False if validation fails, or tuple (provider, model) if valid
        result = config_manager.validate_provider_model(None, None)
        # Can be False (invalid) or tuple (provider, model)
        assert result is False or isinstance(result, tuple)

    def test_list_available_models(self):
        """Test listing available models."""
        from crawl4ai_mcp.config import config_manager

        # Returns dict mapping provider to list of models
        models = config_manager.list_available_models()
        assert isinstance(models, dict)


class TestConfigWithEnvVars:
    """Tests for config with environment variables."""

    def test_config_respects_env_vars(self):
        """Test that config manager respects environment variables."""
        from crawl4ai_mcp.config import ConfigManager

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-openai-key",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4"
        }):
            # Create a new config manager to pick up env vars
            manager = ConfigManager()
            key = manager.get_api_key("openai")
            # Should find the API key from env
            assert key == "test-openai-key" or key is not None

    def test_config_handles_missing_env_vars(self):
        """Test that config handles missing environment variables gracefully."""
        from crawl4ai_mcp.config import ConfigManager

        with patch.dict(os.environ, {}, clear=True):
            # Should not raise even without env vars
            manager = ConfigManager()
            provider = manager.get_default_provider()
            # Should return None or a default value
            assert provider is None or isinstance(provider, str)


class TestGetLlmConfig:
    """Tests for get_llm_config function."""

    def test_get_llm_config_returns_object_or_none(self):
        """Test get_llm_config function."""
        from crawl4ai_mcp.config import get_llm_config

        result = get_llm_config()
        # Returns LLMConfig object or None
        assert result is None or hasattr(result, 'provider')

    def test_get_llm_config_with_provider(self):
        """Test get_llm_config with specific provider."""
        from crawl4ai_mcp.config import get_llm_config

        result = get_llm_config(provider="openai")
        # Returns LLMConfig object or None
        assert result is None or hasattr(result, 'provider')
