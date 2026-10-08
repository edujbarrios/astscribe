"""Keep all README Python examples and their displayed outputs in sync."""

from __future__ import annotations

import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
EXAMPLE = re.compile(
    r"```python\n(?P<source>.*?)\n```\n\n\*\*Output\*\*\n\n```text\n(?P<output>.*?)\n```",
    re.DOTALL,
)


def _examples() -> list[tuple[str, str]]:
    return [(match["source"], match["output"]) for match in EXAMPLE.finditer(README.read_text())]


def test_every_python_readme_example_has_recorded_output() -> None:
    text = README.read_text(encoding="utf-8")
    assert text.count("```python\n") == len(_examples()) == 5


@pytest.mark.parametrize(("source", "expected"), _examples())
def test_readme_examples_execute_as_documented(source: str, expected: str) -> None:
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(source, "README.md", "exec"), {"__name__": "__main__"})
    assert output.getvalue().rstrip("\n") == expected


def test_readme_stdin_cli_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "astscribe", "-", "--style", "concise"],
        input="import torch\nmodel.eval()\nwith torch.no_grad():\n    outputs = model(inputs)\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == (
        "Performs PyTorch inference using evaluation-oriented execution semantics.\n"
    )
    assert result.stdout.strip() in README.read_text(encoding="utf-8")
