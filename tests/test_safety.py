from pathlib import Path

from astscribe import analyze


def test_analyzed_code_is_never_executed(tmp_path: Path) -> None:
    victim = tmp_path / "important.file"
    victim.write_text("keep me", encoding="utf-8")
    code = f'''\nimport os\nos.remove({str(victim)!r})\nraise RuntimeError("THIS MUST NEVER RUN")\n'''
    analyze(code)
    assert victim.read_text(encoding="utf-8") == "keep me"
