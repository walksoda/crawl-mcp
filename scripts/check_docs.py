#!/usr/bin/env python3
"""Check that the documentation still matches the code.

Checks (all run by default):
  schema    every MCP tool and parameter (request-dict keys included) is
            documented in docs/API_REFERENCE.md and docs/ja/API_REFERENCE.md
  versions  dependency pins quoted in the docs match pyproject.toml
  refs      repository paths and relative links in the docs exist, and
            same-file anchors resolve
  env       environment variables set in doc examples are read somewhere
  style     no emoji or hype words in the prose (code is ignored)

With --release, also check that CHANGELOG.md describes the code changes
made since the latest tag. Used by .githooks/pre-push.

Exit status is 1 when any problem is found.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_REFERENCES = [ROOT / "docs" / "API_REFERENCE.md", ROOT / "docs" / "ja" / "API_REFERENCE.md"]
CODE_PATHS = ("crawl4ai_mcp/", "pyproject.toml", "requirements.txt", "Dockerfile")

# Variables read by Python, Playwright, crawl4ai or the shell rather than by
# this project's code
EXTERNAL_ENV_VARS = {
    "PATH", "HOME", "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
    "PLAYWRIGHT_BROWSERS_PATH", "CRAWL4_AI_BASE_DIRECTORY",
}


def doc_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "README_ja.md", ROOT / "CHANGELOG.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.is_file()]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# --- schema -----------------------------------------------------------------

async def _tool_schemas() -> dict[str, dict]:
    sys.path.insert(0, str(ROOT))
    from crawl4ai_mcp.server import mcp

    tools = await mcp.get_tools()
    return {name: tool.parameters for name, tool in tools.items()}


def request_keys(description: str) -> list[str]:
    """Keys listed in a request-dict description ("Dict with: a, b (...), c.")."""
    text = re.sub(r"\([^()]*\)", "", description)
    keys = []
    for marker in ("Dict with:", "persistence keys:"):
        if marker not in text:
            continue
        segment = text.split(marker, 1)[1].split(".", 1)[0]
        for item in segment.split(","):
            item = item.strip()
            if re.fullmatch(r"[a-z][a-z0-9_]*", item):
                keys.append(item)
    return keys


def doc_sections(path: Path) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^### `(\w+)`", line)
        if heading:
            current = heading.group(1)
            sections[current] = ""
        elif line.startswith(("## ", "### ")):
            current = None
        elif current:
            sections[current] += line + "\n"
    return sections


def check_schema(schemas: dict[str, dict]) -> list[str]:
    problems = []
    for doc in API_REFERENCES:
        sections = doc_sections(doc)
        for name in sorted(set(sections) - set(schemas)):
            problems.append(f"{rel(doc)}: documents `{name}`, which is not a tool")
        for name, schema in sorted(schemas.items()):
            if name not in sections:
                problems.append(f"{rel(doc)}: tool `{name}` has no section")
                continue
            params = list(schema.get("properties", {}))
            if params == ["request"]:
                params += request_keys(schema["properties"]["request"].get("description", ""))
            for param in params:
                if f"`{param}`" not in sections[name]:
                    problems.append(f"{rel(doc)}: `{name}` does not document `{param}`")
    return problems


# --- versions ---------------------------------------------------------------

def check_versions() -> list[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = {}
    for dep in pyproject["project"]["dependencies"]:
        match = re.match(r"([A-Za-z0-9_.-]+)\s*(.*)", dep)
        pins[match.group(1).lower()] = match.group(2).replace(" ", "")
    problems = []
    for doc in doc_files():
        if doc.name == "CHANGELOG.md":
            continue  # records past pins on purpose
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for package, spec in re.findall(r"\b(crawl4ai|playwright)((?:[<>=!]=?[\w.*]+,?)+)", line):
                expected = pins.get(package)
                if expected and spec.rstrip(",") != expected:
                    problems.append(
                        f"{rel(doc)}:{lineno}: {package}{spec} does not match pyproject.toml ({package}{expected})"
                    )
    return problems


# --- refs -------------------------------------------------------------------

def slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    result = set()
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        heading = re.match(r"^#{1,6} (.+)", line)
        if heading and not in_code:
            base = slug(heading.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            result.add(base if count == 0 else f"{base}-{count}")
    return result


def check_refs() -> list[str]:
    problems = []
    path_pattern = re.compile(
        r"(?<![\w/.~-])((?:scripts|examples|docs|tests|configs)/[\w./-]*\w)"
    )
    for doc in doc_files():
        if doc.name == "CHANGELOG.md":
            continue  # may name files that no longer exist
        own_anchors = anchors(doc)
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for target in path_pattern.findall(line):
                if "your_" in target or "X.Y.Z" in target:
                    continue  # placeholder in an example
                if not (ROOT / target).exists():
                    problems.append(f"{rel(doc)}:{lineno}: `{target}` does not exist")
            for link, anchor in re.findall(r"\]\(([^)#\s]*)(#[^)\s]*)?\)", line):
                if re.match(r"[a-z]+:", link):
                    continue  # external URL
                target = (doc.parent / link) if link else doc
                if link and not target.exists():
                    problems.append(f"{rel(doc)}:{lineno}: link `{link}` does not exist")
                elif anchor and target.suffix == ".md":
                    known = own_anchors if not link else anchors(target)
                    if anchor[1:] not in known:
                        problems.append(f"{rel(doc)}:{lineno}: anchor `{link}{anchor}` does not exist")
    return problems


# --- env --------------------------------------------------------------------

def check_env() -> list[str]:
    known = set(EXTERNAL_ENV_VARS)
    sources = list((ROOT / "crawl4ai_mcp").rglob("*.py"))
    sources += [ROOT / name for name in (".env.example", "Dockerfile", "docker-compose.yml")]
    for source in sources:
        if source.is_file():
            known |= set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", source.read_text(encoding="utf-8")))
    set_patterns = [
        r'"([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)"\s*:',          # JSON "env" blocks
        r"\bexport\s+([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)=",    # shell export
        r"^\s*([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)=(?![\"']?\$\()",  # .env lines, not NAME=$(...)
        r"(?:-e|--env)\s+([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)=",  # docker run
    ]
    problems = []
    for doc in doc_files():
        if doc.name == "CHANGELOG.md":
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in set_patterns:
                for name in re.findall(pattern, line):
                    if name not in known:
                        problems.append(f"{rel(doc)}:{lineno}: {name} is not read by the server")
    return problems


# --- style ------------------------------------------------------------------

EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
HYPE_WORDS = re.compile(
    r"\b(comprehensive|powerful|seamless(?:ly)?|robust|cutting-edge|state-of-the-art|"
    r"production[- ]ready|leverag(?:e|es|ing))\b|包括的|強力な|画期的|シームレス",
    re.I,
)


def prose_lines(path: Path):
    """(lineno, text) for lines outside fenced code, with inline code removed."""
    in_code = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            yield lineno, re.sub(r"`[^`]*`", "", line)


def check_style() -> list[str]:
    problems = []
    for doc in doc_files():
        for lineno, line in prose_lines(doc):
            if EMOJI.search(line):
                problems.append(f"{rel(doc)}:{lineno}: emoji in prose")
            for word in {m.group(0) for m in HYPE_WORDS.finditer(line)}:
                problems.append(f"{rel(doc)}:{lineno}: hype word \"{word}\"")
    return problems


# --- changelog --------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def check_changelog() -> list[str]:
    tag = git("describe", "--tags", "--abbrev=0")
    changed = [p for p in git("diff", "--name-only", f"{tag}..HEAD").splitlines() if p.startswith(CODE_PATHS)]
    if not changed:
        return []
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[([^\]]+)\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    allowed = {"Unreleased"} | ({version} if f"v{version}" != tag else set())
    if not match or match.group(1) not in allowed:
        return [f"CHANGELOG.md: code changed since {tag} ({len(changed)} files) but the top section is not "
                f"[Unreleased] or [{version}]"]
    if not re.search(r"^- ", match.group(2), re.M):
        return [f"CHANGELOG.md: [{match.group(1)}] has no entries for the code changes since {tag}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", action="store_true", help="also check CHANGELOG.md against the latest tag")
    args = parser.parse_args()

    checks = {
        "schema": lambda: check_schema(asyncio.run(_tool_schemas())),
        "versions": check_versions,
        "refs": check_refs,
        "env": check_env,
        "style": check_style,
    }
    if args.release:
        checks["changelog"] = check_changelog

    failed = False
    for name, check in checks.items():
        problems = check()
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        print(f"[{name}] {status}")
        for problem in problems:
            print(f"  {problem}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
