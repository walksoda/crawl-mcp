"""stdout must carry only MCP messages when serving over stdio."""

import subprocess
import sys
import textwrap


SCRIPT = textwrap.dedent("""
    import os, subprocess, sys
    from crawl4ai_mcp.server import _isolate_stdout_for_stdio

    _isolate_stdout_for_stdio()

    # What the MCP stdio transport does: wrap sys.stdout.buffer for messages
    sys.stdout.buffer.write(b'{"jsonrpc":"2.0","id":1}\\n')
    sys.stdout.buffer.flush()

    # Stray output from tool code, C-level writes and child processes
    print("Stage 1/7: stray print")
    sys.stdout.flush()
    os.write(1, b"raw fd write\\n")
    subprocess.run(["echo", "child process"], check=True)
""")


def test_only_protocol_bytes_reach_stdout():
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")

    assert proc.stdout == b'{"jsonrpc":"2.0","id":1}\n'
    stderr = proc.stderr.decode(errors="replace")
    for stray in ("Stage 1/7: stray print", "raw fd write", "child process"):
        assert stray in stderr
