"""The docs must match the code (see scripts/check_docs.py).

CHANGELOG coverage is checked only before a push (.githooks/pre-push),
since unreleased work normally has no entry yet.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"
spec = importlib.util.spec_from_file_location("check_docs", SCRIPT)
check_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_docs)


def test_every_tool_parameter_is_documented():
    schemas = asyncio.run(check_docs._tool_schemas())
    assert check_docs.check_schema(schemas) == []


@pytest.mark.parametrize("check", ["check_versions", "check_refs", "check_env", "check_style"])
def test_docs(check):
    assert getattr(check_docs, check)() == []
