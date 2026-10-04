"""crawl4ai's legacy migration must not rewrite the shared ~/.crawl4ai cache.

The migration ignores CRAWL4_AI_BASE_DIRECTORY and always opens
~/.crawl4ai/crawl4ai.db, so it is triggered here with a separate base
directory that has no version.txt (as on a fresh install).
"""

import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest


SENTINEL = "<html>sentinel</html>"

SCRIPT = textwrap.dedent("""
    import asyncio, sys
    if sys.argv[1] == "guarded":
        import crawl4ai_mcp
    from crawl4ai.async_database import async_db_manager
    asyncio.run(async_db_manager.initialize())
""")


def _make_home_db(home):
    db_dir = home / ".crawl4ai"
    db_dir.mkdir(parents=True)
    with sqlite3.connect(db_dir / "crawl4ai.db") as db:
        db.execute(
            "CREATE TABLE crawled_data (url TEXT PRIMARY KEY, html TEXT,"
            " cleaned_html TEXT, markdown TEXT, extracted_content TEXT,"
            " screenshot TEXT)"
        )
        db.execute(
            "INSERT INTO crawled_data VALUES (?, ?, '', '', '', '')",
            ("https://example.invalid/", SENTINEL),
        )
    return db_dir


def _home_html(db_dir):
    with sqlite3.connect(db_dir / "crawl4ai.db") as db:
        return db.execute("SELECT html FROM crawled_data").fetchone()[0]


def _initialize(tmp_path, mode):
    home = tmp_path / mode / "home"
    base = tmp_path / mode / "base"
    db_dir = _make_home_db(home)
    env = dict(os.environ, HOME=str(home), CRAWL4_AI_BASE_DIRECTORY=str(base))
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, mode],
        capture_output=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return db_dir, base, proc


def test_migration_does_not_touch_home_cache(tmp_path):
    db_dir, _, _ = _initialize(tmp_path, "unguarded")
    if _home_html(db_dir) == SENTINEL:
        pytest.skip("crawl4ai no longer migrates ~/.crawl4ai on version change")

    db_dir, base, proc = _initialize(tmp_path, "guarded")

    assert _home_html(db_dir) == SENTINEL
    assert sorted(p.name for p in db_dir.iterdir()) == ["crawl4ai.db"]
    assert b"skipped crawl4ai legacy cache migration" in proc.stderr
    # The rest of initialization still runs against the configured directory
    assert (base / ".crawl4ai" / "crawl4ai.db").is_file()
    assert (base / ".crawl4ai" / "version.txt").is_file()
