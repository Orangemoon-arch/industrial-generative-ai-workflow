"""Fail when README references a missing local image."""

from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"


def main() -> None:
    content = README.read_text(encoding="utf-8")
    references = re.findall(r"!\[[^\]]*\]\((?!https?://)([^)]+)\)", content)
    missing = [reference for reference in references if not (REPOSITORY_ROOT / reference).is_file()]
    if missing:
        raise SystemExit("Missing README assets: " + ", ".join(missing))
    print(f"README asset check passed: {len(references)} local images")


if __name__ == "__main__":
    main()
