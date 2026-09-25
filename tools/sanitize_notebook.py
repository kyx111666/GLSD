#!/usr/bin/env python3
"""Remove execution state from a notebook before it is published."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def sanitize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook.pop("signature", None)
    metadata = notebook.setdefault("metadata", {})
    for key in ("colab", "execution", "widgets", "accelerator", "gpuClass"):
        metadata.pop(key, None)
    for cell in notebook.get("cells", []):
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
        cell.pop("metadata", None)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: sanitize_notebook.py NOTEBOOK [NOTEBOOK ...]", file=sys.stderr)
        return 2
    for argument in sys.argv[1:]:
        sanitize(Path(argument))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
