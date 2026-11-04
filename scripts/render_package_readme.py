"""Derive the package-index README from the GitHub README, without Mermaid."""
from pathlib import Path
import argparse
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/geniusgeek/autorsi/blob/main/"
SOURCE_NOTICE = (
    "AutoRSI is a Python framework for controlled, iterative model improvement. "
    "**Version 0.1.0a1 is an MIT-licensed alpha, [available on PyPI]"
    "(https://pypi.org/project/autorsi/0.1.0a1/):** APIs may change. "
    "The source repository and linked source files require repository access while the repository is private."
)
PACKAGE_NOTICE = """AutoRSI is a Python framework for controlled, iterative model improvement. **Version 0.1.0a1 is an MIT-licensed alpha:** APIs may change.

The source repository and linked source files require repository access while the repository is private.
"""
FLOW = """```text
Objective + protected metrics -> Propose bounded edit -> Isolated candidate
    -> Evaluate -> Pass: promote at inference safe point
                -> Fail: keep current serving weights
    -> Repeat only while the configured steps and budget remain
    -> Return accepted checkpoint (if any) and journal
```"""


def render(source: str) -> str:
    result, count = re.subn(r"```mermaid\n.*?\n```", FLOW, source, flags=re.DOTALL)
    if count != 1:
        raise ValueError("expected exactly one Mermaid workflow in README.md")
    if result.count(SOURCE_NOTICE) != 1 or result.count("## Install the alpha") != 1:
        raise ValueError("README release notice changed; review the package notice substitution")
    result = result.replace(SOURCE_NOTICE, PACKAGE_NOTICE)
    return re.sub(r"\]\((?![a-z]+:|#)([^)]+)\)", lambda match: "](" + BASE + match[1] + ")", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated README is stale")
    args = parser.parse_args()
    expected = render((ROOT / "README.md").read_text(encoding="utf-8"))
    destination = ROOT / "PACKAGE_README.md"
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != expected:
            raise SystemExit("PACKAGE_README.md is stale; run python scripts/render_package_readme.py")
    else:
        destination.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
