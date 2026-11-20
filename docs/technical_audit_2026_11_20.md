# ASTScribe technical audit — 2026-11-20

Reviewed the current 0.14.1 source, import inventory, CLI, test matrix, packaging and release workflow.

## Addressed in 0.14.2

- **Correctness:** Import inventory counted imports from statically unreachable literal branches even though semantic analysis and dependency tracking already avoid these branches. Filter only provably unreachable paths and retain source locations and `inside_scope` semantics for surviving imports.
- **Interoperability:** CLI Python source decoding used fixed UTF-8, rejecting legal PEP 263 encoded source files and UTF-8 BOM. Use `tokenize.open` for Python files.
- **Release safety:** The release workflow built a wheel and checked package metadata but did not independently run regression tests and static quality checks before tagging/publishing. Gate the tag step on a Python 3.10–3.14 test matrix and Ruff/mypy jobs.

## Existing safeguards verified by source inspection

- No required third-party runtime dependencies; source inspection does not execute notebook cells.
- Unsupported notebook syntax is explicitly skipped or rejected in strict mode.
- PyPI Trusted Publishing via OIDC; no long-lived upload secret configured in the workflow.
- A separate test workflow checks Python 3.10–3.14, package builds, linting, type checking and saved notebook outputs.

## Follow-up opportunities (not implemented)

1. **Fuzz and property testing:** Add Hypothesis-based AST and malformed-notebook test generation, including nested scopes and syntax across Python versions.
2. **Performance:** Benchmark notebook parse times and memory for long notebooks; set actionable performance budgets before micro-optimizing.
3. **Cross-version static semantics:** Test how Python 3.10–3.14 grammar changes affect source evidence and import provenance.
4. **Supply-chain hardening:** Pin third-party GitHub Actions by commit digest and automate dependency review, retaining trusted publishing.
5. **Integration cases:** Expand regressions using real-world (syntheticized, nonexecuted) PyTorch, Transformers and multimodal notebook snippets.

This audit is source-level and CI-level, not a penetration test or a claim
that all Python notebooks can be analyzed exactly.
