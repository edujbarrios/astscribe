from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from astscribe import NotebookAnalyzer, __version__, explain


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astscribe",
        description="Explain Python ML code and inspect Jupyter notebooks without executing them.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Python source file, Jupyter notebook, or '-' to read Python source from stdin",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--style",
        choices=("scientific", "educational", "concise"),
        default="scientific",
        help="rendering style for Python source (default: scientific)",
    )
    parser.add_argument(
        "--report",
        choices=("methodology", "pipeline", "techniques", "dependencies", "diagnostics", "impact"),
        default="methodology",
        help="notebook report to render (default: methodology)",
    )
    parser.add_argument("--cell", type=int, help="notebook cell for an impact report")
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="include source evidence in methodology reports",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if a notebook code cell is not valid Python instead of skipping it",
    )
    return parser


def _render_notebook(args: argparse.Namespace) -> str:
    notebook = NotebookAnalyzer.from_ipynb(
        args.path,
        skip_invalid_python=not args.strict,
    )
    if args.report == "methodology":
        return notebook.render_methodology(include_evidence=args.evidence)
    if args.report == "pipeline":
        return notebook.render_pipeline()
    if args.report == "techniques":
        return notebook.render_techniques()
    if args.report == "dependencies":
        return notebook.render_dependency_graph()
    if args.report == "diagnostics":
        return notebook.render_diagnostics()
    if args.cell is None:
        raise ValueError("--cell is required when --report impact is selected")
    return notebook.render_impact(args.cell)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        if args.path.suffix.lower() == ".ipynb":
            output = _render_notebook(args)
        else:
            source = sys.stdin.read() if str(args.path) == "-" else args.path.read_text(encoding="utf-8")
            output = str(explain(source, style=args.style))
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))

    sys.stdout.write(output)
    if output and not output.endswith("\n"):
        sys.stdout.write("\n")
    return 0
