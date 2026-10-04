"""Backward-compatible import path. Implementation: processors.web_search"""

from .web_search import *  # noqa: F401, F403
from .web_search import GoogleSearchProcessor, WebSearchProcessor  # noqa: F401
