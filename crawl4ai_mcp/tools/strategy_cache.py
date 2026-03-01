"""Strategy cache and fingerprint profiles for Crawl4AI MCP Server.

This module is a backward-compatibility facade.
All implementation has been moved to crawl4ai_mcp.infra.strategy_cache
and crawl4ai_mcp.infra.fingerprint.
"""

from ..infra.strategy_cache import (  # noqa: F401
    StrategyCache,
    get_strategy_cache,
)
from ..infra.fingerprint import (  # noqa: F401
    FingerprintProfile,
    get_fingerprint_config,
)
