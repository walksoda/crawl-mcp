"""
Crawl4AI MCP Server - FastMCP 2.0 Version

Uses FastMCP 2.0.0 which doesn't have banner output issues.
Clean STDIO transport compatible for perfect MCP communication.
"""

import argparse
import os
import sys
import warnings

# Set environment variables before any imports
os.environ["FASTMCP_QUIET"] = "true"
os.environ["FASTMCP_NO_BANNER"] = "true"
os.environ["FASTMCP_SILENT"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TERM"] = "dumb"
os.environ["SHELL"] = "/bin/sh"

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import logging

# Logging is off by default: the stdio transport reserves stdout for MCP
# messages and most clients surface stderr noise as errors. Set
# CRAWL4AI_MCP_LOG_LEVEL (DEBUG, INFO, WARNING or ERROR) to log to stderr.
LOG_LEVEL_ENV = "CRAWL4AI_MCP_LOG_LEVEL"
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _configure_logging() -> None:
    """Disable logging unless CRAWL4AI_MCP_LOG_LEVEL asks for it.

    When enabled, the root logger writes to stderr only, never stdout.
    """
    raw = os.environ.get(LOG_LEVEL_ENV, "").strip()
    level = raw.upper()
    if level not in _LOG_LEVELS:
        if raw:
            sys.stderr.write(
                f"crawl4ai-mcp: ignoring {LOG_LEVEL_ENV}={raw!r} "
                f"(expected one of {', '.join(_LOG_LEVELS)}); logging disabled\n"
            )
        logging.disable(logging.CRITICAL)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


_configure_logging()

# Import FastMCP 2.0 - no banner output!
from fastmcp import FastMCP

from .server_tools import register_all_tools

# Create MCP server with clean initialization
mcp = FastMCP("Crawl4AI")

# Tool module loading state (owned by server.py)
_tools_imported = False
_tool_modules = (None, None, None, None, None)


def _load_tool_modules() -> bool:
    """Load tool modules only when needed.

    Returns:
        True if modules were loaded successfully, False otherwise.
    """
    global _tools_imported, _tool_modules
    if _tools_imported:
        return True

    try:
        from .tools import web_crawling as _wc
        from .tools import search as _s
        from .tools import youtube as _yt
        from .tools import file_processing as _fp
        from .tools import utilities as _ut

        _tool_modules = (_wc, _s, _yt, _fp, _ut)
        _tools_imported = True
        return True
    except ImportError:
        try:
            from crawl4ai_mcp.tools import web_crawling as _wc
            from crawl4ai_mcp.tools import search as _s
            from crawl4ai_mcp.tools import youtube as _yt
            from crawl4ai_mcp.tools import file_processing as _fp
            from crawl4ai_mcp.tools import utilities as _ut

            _tool_modules = (_wc, _s, _yt, _fp, _ut)
            _tools_imported = True
            return True
        except ImportError:
            _tools_imported = False
            return False


def is_tools_imported() -> bool:
    """Check if tool modules are imported."""
    return _tools_imported


def get_tool_modules():
    """Get the loaded tool modules.

    Returns:
        Tuple of (web_crawling, search, youtube, file_processing, utilities).
    """
    if not _tools_imported:
        _load_tool_modules()
    return _tool_modules


def _get_modules():
    """Get tool modules, loading them if needed.

    Returns the tuple (web_crawling, search, youtube, file_processing, utilities)
    or None if modules couldn't be loaded.
    """
    if not _tools_imported:
        _load_tool_modules()
    if not _tools_imported:
        return None
    return _tool_modules


# Register all MCP tools
register_all_tools(mcp, _get_modules)


class _StdoutRedirectedToStderr:
    """sys.stdout replacement for the stdio transport.

    Text writes (print and friends) go to stderr, while ``.buffer`` is the
    real stdout pipe, which the MCP stdio transport wraps for its messages.
    """

    def __init__(self, protocol_buffer, text_sink):
        self.buffer = protocol_buffer
        self._sink = text_sink

    def write(self, text):
        return self._sink.write(text)

    def flush(self):
        self._sink.flush()

    def __getattr__(self, name):
        return getattr(self._sink, name)


def _isolate_stdout_for_stdio() -> None:
    """Reserve stdout for MCP messages when serving over stdio.

    Any other output on stdout (print calls in tool code, libraries, C
    extensions or child processes writing to fd 1) would be read by the
    client as malformed JSON-RPC. Keep a private duplicate of the stdout
    pipe for the protocol and point fd 1 and sys.stdout text writes at stderr.
    """
    sys.stdout.flush()
    protocol_buffer = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = _StdoutRedirectedToStderr(protocol_buffer, sys.stderr)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crawl4ai_mcp.server",
        description="Crawl4AI MCP Server",
        epilog=(
            f"Set {LOG_LEVEL_ENV}=DEBUG|INFO|WARNING|ERROR to log to stderr "
            "(logging is disabled by default)."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "http", "sse"),
        default="stdio",
        help="transport to serve (default: stdio; http is an alias of streamable-http)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host for HTTP transports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port for HTTP transports (default: 8000)",
    )
    return parser


def main(argv=None):
    """Parse command line arguments and run the MCP server.

    argparse writes --help to stdout and exits before any transport starts;
    argument errors go to stderr with exit status 2.
    """
    args = _build_arg_parser().parse_args(argv)
    transport = "streamable-http" if args.transport == "http" else args.transport

    try:
        if transport == "stdio":
            _isolate_stdout_for_stdio()
            mcp.run()
        else:
            mcp.run(transport=transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
