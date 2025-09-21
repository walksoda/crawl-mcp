"""
Time period parsing utilities for Crawl4AI MCP Server.
"""

from typing import Optional, Any, Union
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re


def parse_time_period(period: Optional[Any]) -> Optional[Union[int, str]]:
    """
    Parse time period string or integer to days or date string.

    Supports:
    - Integer: direct number of days (e.g., 7, 30)
    - String formats:
      - "7d" or "7days": 7 days
      - "2w" or "2weeks": 14 days
      - "1mo" or "1month": 1 month (using proper month calculation)
      - "3mo" or "3months": 3 months (using proper month calculation)
      - "1y" or "1year": 1 year (using proper year calculation)

    Returns:
        Number of days as integer for day/week periods,
        or date string for month/year periods,
        or None if period is None/invalid
    """
    if period is None:
        return None

    # If already an integer, return it
    if isinstance(period, int):
        return period

    # If not a string, try to convert
    if not isinstance(period, str):
        try:
            return int(period)
        except (ValueError, TypeError):
            return None

    # Parse string formats
    period = period.strip().lower()

    # Try direct integer conversion first
    try:
        return int(period)
    except ValueError:
        pass

    # Parse unit-based formats
    match = re.match(r'^(\d+)\s*([a-z]+)$', period)
    if not match:
        return None

    number = int(match.group(1))
    unit = match.group(2)

    # Handle days and weeks with simple multiplication
    if unit in ['d', 'day', 'days']:
        return number * 1
    elif unit in ['w', 'week', 'weeks']:
        return number * 7

    # Handle months and years with proper date calculation
    elif unit in ['mo', 'month', 'months']:
        # Calculate the exact date N months ago
        target_date = datetime.now() - relativedelta(months=number)
        return target_date.strftime("%Y-%m-%d")
    elif unit in ['y', 'year', 'years']:
        # Calculate the exact date N years ago
        target_date = datetime.now() - relativedelta(years=number)
        return target_date.strftime("%Y-%m-%d")

    return None