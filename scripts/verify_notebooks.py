"""Execute example notebooks and check their committed outputs.

Pass --update to regenerate those outputs intentionally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="refresh committed outputs")
    args = parser.parse_args()
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        parser.error(f"No example notebooks in {NOTEBOOKS}")

    for path in paths:
        saved = nbformat.read(path, as_version=4)
        nbformat.validate(saved)
        executed = NotebookClient(
            nbformat.read(path, as_version=4), kernel_name="python3", timeout=90
        ).execute()

        if args.update:
            for cell in executed.cells:
                if cell.cell_type == "code":
                    cell.metadata.pop("execution", None)
            nbformat.write(executed, path)
            print(f"Updated {path.name}")
            continue

        for index, (before, after) in enumerate(zip(saved.cells, executed.cells, strict=True)):
            if before.cell_type != "code":
                continue
            if before.execution_count is None:
                raise AssertionError(f"{path.name} cell {index}: missing saved execution count")
            if before.outputs != after.outputs:
                raise AssertionError(
                    f"{path.name} cell {index}: recorded outputs differ; "
                    "run python scripts/verify_notebooks.py --update"
                )

    if not args.update:
        print(f"Verified {len(paths)} executed example notebooks; all stored outputs match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
