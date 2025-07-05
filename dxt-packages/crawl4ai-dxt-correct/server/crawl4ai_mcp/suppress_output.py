"""
Output suppression utilities for crawl4ai MCP server.
"""

import sys
import os
from contextlib import contextmanager
from io import StringIO


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output."""
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Redirect stdout and stderr to devnull
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@contextmanager
def capture_output():
    """Context manager to capture and return stdout/stderr output."""
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create StringIO objects to capture output
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    
    try:
        # Redirect stdout and stderr to our StringIO objects
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr
        yield captured_stdout, captured_stderr
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr