"""
Client-side requests-per-minute limiter for search engines.

Only google_custom has a client-side limit (CUSTOM_SEARCH_API_RPM, default
100). Brave and Exa enforce their limits server-side with HTTP 429.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional


@dataclass
class SearchRequest:
    """One recorded search request"""
    timestamp: datetime
    method: str  # engine name
    status: str  # 'success', 'error', '429'


class RateLimiter:
    """Rate limiting manager for search APIs"""

    def __init__(self):
        self.requests = defaultdict(list)  # engine -> list of SearchRequest
        self.rpm_limits: Dict[str, int] = {
            'google_custom': int(os.getenv('CUSTOM_SEARCH_API_RPM', '100')),
        }

    def _recent(self, method: str):
        cutoff = datetime.now() - timedelta(minutes=1)
        self.requests[method] = [r for r in self.requests[method] if r.timestamp > cutoff]
        return self.requests[method]

    def limit_for(self, method: str) -> Optional[int]:
        return self.rpm_limits.get(method)

    def can_make_request(self, method: str) -> bool:
        """Check if a request can be made without exceeding rate limit"""
        limit = self.limit_for(method)
        return limit is None or len(self._recent(method)) < limit

    def record_request(self, method: str, status: str):
        """Record a completed request (history older than one minute is dropped)"""
        self._recent(method).append(SearchRequest(datetime.now(), method, status))

    def get_wait_time(self, method: str) -> float:
        """Seconds to wait before the next request is allowed"""
        if self.can_make_request(method):
            return 0.0
        oldest = min(self._recent(method), key=lambda r: r.timestamp)
        wait = (oldest.timestamp + timedelta(minutes=1) - datetime.now()).total_seconds()
        return max(0.0, wait + 1.0)
