"""Disable crawl4ai's legacy cache migration.

On a version change crawl4ai runs ``crawl4ai.migrations.run_migration()``,
which converts inline cache columns to file storage. It is not idempotent:
on a current database it hashes the stored hashes again, so cached results
come back as 16-character hash strings. It also ignores
``CRAWL4_AI_BASE_DIRECTORY`` and always opens ``~/.crawl4ai/crawl4ai.db``,
so a process configured with a separate cache directory still rewrites the
default one.

The migration is only needed for databases from very old crawl4ai releases,
so it is replaced with a no-op. The schema update (adding new columns) and
the ``version.txt`` update are separate steps and still run.

The patch is applied when ``crawl4ai.migrations`` is imported, so installing
the guard does not import crawl4ai itself.
"""

import importlib.abc
import importlib.machinery
import sys

_TARGET = "crawl4ai.migrations"


async def _skip_migration(db_path=None):
    print("crawl4ai_mcp: skipped crawl4ai legacy cache migration", file=sys.stderr)


def _patch(module) -> None:
    module.run_migration = _skip_migration


class _MigrationGuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Disable the migration now if loaded, otherwise when it is imported."""
    module = sys.modules.get(_TARGET)
    if module is not None:
        _patch(module)
    if not any(isinstance(f, _MigrationGuardFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MigrationGuardFinder())
