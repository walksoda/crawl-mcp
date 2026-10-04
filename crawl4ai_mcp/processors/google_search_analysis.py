"""Backward-compatible import path. Implementation: processors.web_search_analysis"""

from .web_search_analysis import WebSearchAnalysisMixin  # noqa: F401

GoogleSearchAnalysisMixin = WebSearchAnalysisMixin
