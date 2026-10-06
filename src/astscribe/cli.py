from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from astscribe import NotebookAnalyzer, __version__, analyze, explain


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
        choices=(
            "methodology",
            "pipeline",
            "techniques",
            "dependencies",
            "diagnostics",
            "impact",
            "ranking",
        ),
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
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--dot",
        action="store_true",
        help="render --report dependencies as Graphviz DOT",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="emit structured JSON instead of rendered text",
    )
    return parser


def _notebook_payload(notebook: NotebookAnalyzer, args: argparse.Namespace) -> Any:
    if args.report == "methodology":
        return notebook.methodology().to_dict()
    if args.report == "pipeline":
        return notebook.pipeline().to_dict()
    if args.report == "techniques":
        return {"items": [finding.to_dict() for finding in notebook.techniques()]}
    if args.report == "dependencies":
        return notebook.dependency_graph().to_dict()
    if args.report == "diagnostics":
        return notebook.diagnostics().to_dict()
    if args.report == "ranking":
        return {"items": [summary.to_dict() for summary in notebook.impact_ranking()]}
    if args.cell is None:
        raise ValueError("--cell is required when --report impact is selected")
    return notebook.impact(args.cell).to_dict()


def _render_notebook(args: argparse.Namespace) -> str:
    notebook = NotebookAnalyzer.from_ipynb(
        args.path,
        skip_invalid_python=not args.strict,
    )
    if args.dot and args.report != "dependencies":
        raise ValueError("--dot is only supported with --report dependencies")
    if args.json:
        return json.dumps(_notebook_payload(notebook, args), indent=2, sort_keys=True)
    if args.report == "methodology":
        return notebook.render_methodology(include_evidence=args.evidence)
    if args.report == "pipeline":
        return notebook.render_pipeline()
    if args.report == "techniques":
        return notebook.render_techniques()
    if args.report == "dependencies":
        return notebook.dependency_dot() if args.dot else notebook.render_dependency_graph()
    if args.report == "diagnostics":
        return notebook.render_diagnostics()
    if args.report == "ranking":
        return notebook.render_impact_ranking()
    if args.cell is None:
        raise ValueError("--cell is required when --report impact is selected")
    return notebook.render_impact(args.cell)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        is_notebook = args.path.suffix.lower() == ".ipynb"
        if args.dot and not is_notebook:
            raise ValueError("--dot is only supported with notebook --report dependencies")
        if is_notebook:
            output = _render_notebook(args)
        else:
            source = sys.stdin.read() if str(args.path) == "-" else args.path.read_text(encoding="utf-8")
            if args.json:
                output = json.dumps(analyze(source).to_dict(), indent=2, sort_keys=True)
            else:
                output = str(explain(source, style=args.style))
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))

    sys.stdout.write(output)
    if output and not output.endswith("\n"):
        sys.stdout.write("\n")
    return 0
