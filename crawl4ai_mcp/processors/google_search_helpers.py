"""
Google Search Helper Classes
SearchRequest dataclass and RateLimiter for search API management.
"""

import os
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SearchRequest:
    """Represents a search request with rate limiting info"""
    timestamp: datetime
    method: str  # 'googlesearch' or 'custom_search'
    status: str  # 'success', 'error', '429'


class RateLimiter:
    """Rate limiting manager for search APIs"""

    def __init__(self):
        self.requests = defaultdict(list)  # method -> list of SearchRequest
        self.rpm_limits = {
            'googlesearch': int(os.getenv('GOOGLESEARCH_PYTHON_RPM', '60')),
            'custom_search': int(os.getenv('CUSTOM_SEARCH_API_RPM', '100'))
        }

    def can_make_request(self, method: str) -> bool:
        """Check if a request can be made without exceeding rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Clean old requests
        self.requests[method] = [
            req for req in self.requests[method]
            if req.timestamp > cutoff
        ]

        return len(self.requests[method]) < self.rpm_limits[method]

    def record_request(self, method: str, status: str):
        """Record a completed request"""
        self.requests[method].append(SearchRequest(
            timestamp=datetime.now(),
            method=method,
            status=status
        ))

    def get_wait_time(self, method: str) -> float:
        """Get suggested wait time in seconds before next request"""
        if self.can_make_request(method):
            return 0.0

        # Find oldest request in current window
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        oldest_request = min(
            (req for req in self.requests[method] if req.timestamp > cutoff),
            key=lambda r: r.timestamp,
            default=None
        )

        if oldest_request:
            wait_until = oldest_request.timestamp + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            return max(0.0, wait_seconds + 1.0)  # Add 1 second buffer

        return 0.0
