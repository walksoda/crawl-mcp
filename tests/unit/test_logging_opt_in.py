"""CRAWL4AI_MCP_LOG_LEVEL turns on stderr logging; stdout stays clean."""

import os
import subprocess
import sys
import textwrap


SCRIPT = textwrap.dedent("""
    import logging, sys
    from crawl4ai_mcp.server import _isolate_stdout_for_stdio

    _isolate_stdout_for_stdio()

    sys.stdout.buffer.write(b'{"jsonrpc":"2.0","id":1}\\n')
    sys.stdout.buffer.flush()

    logging.getLogger("crawl4ai_mcp.core.test").debug("debug-marker-123")
    logging.getLogger("crawl4ai_mcp.core.test").error("error-marker-456")
""")


def _run(log_level=None):
    env = dict(os.environ)
    env.pop("CRAWL4AI_MCP_LOG_LEVEL", None)
    if log_level is not None:
        env["CRAWL4AI_MCP_LOG_LEVEL"] = log_level
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout == b'{"jsonrpc":"2.0","id":1}\n'
    return proc.stderr.decode(errors="replace")


def test_log_level_env_logs_to_stderr_only():
    stderr = _run("debug")
    assert "debug-marker-123" in stderr
    assert "error-marker-456" in stderr
    assert "DEBUG crawl4ai_mcp.core.test: debug-marker-123" in stderr


def test_level_filters_lower_records():
    stderr = _run("ERROR")
    assert "debug-marker-123" not in stderr
    assert "error-marker-456" in stderr


def test_logging_disabled_by_default():
    stderr = _run()
    assert "debug-marker-123" not in stderr
    assert "error-marker-456" not in stderr


def test_invalid_level_keeps_logging_disabled():
    stderr = _run("verbose")
    assert "ignoring CRAWL4AI_MCP_LOG_LEVEL" in stderr
    assert "error-marker-456" not in stderr
