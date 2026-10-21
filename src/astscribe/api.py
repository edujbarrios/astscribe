from __future__ import annotations

import ast

from astscribe.parser import (
    ImportTable,
    ParsedSource,
    SymbolTable,
    build_symbol_table,
    parse_source,
    update_import_table,
)
from astscribe.parser.reachability import prune_statically_unreachable
from astscribe.patterns import detect_inference, detect_training_step
from astscribe.semantics import DEFAULT_REGISTRY, SemanticRegistry
from astscribe.sir import AnalysisResult


def _analyze_parsed(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
    registry: SemanticRegistry | None = None,
) -> tuple[AnalysisResult, ImportTable, SymbolTable]:
    """Analyze statements using the bindings visible at each source position.

    Incoming imports and symbols are the state *before* this source. We also
    return the final state so notebook analysis can carry it to later cells.
    """
    snapshots: list[tuple[ParsedSource, ImportTable, SymbolTable]] = []
    current_imports = imports.copy()
    current_symbols = symbols.copy()

    # Use the same proven-unreachable branch policy for semantic rules and
    # source-order symbol tracking; preserve original AST source locations.
    effective_tree = prune_statically_unreachable(parsed.tree)
    for statement in effective_tree.body:
        single = ParsedSource(
            source=parsed.source,
            tree=ast.Module(body=[statement], type_ignores=[]),
            lines=parsed.lines,
            cell=parsed.cell,
        )
        snapshots.append((single, current_imports.copy(), current_symbols.copy()))

        # Assignment right-hand sides use bindings from before the assignment,
        # not imports or constants established by later statements.
        next_symbols = build_symbol_table(
            single.tree, current_imports, base=current_symbols,
            source=parsed.source, cell=parsed.cell,
        )
        current_imports = update_import_table(single.tree, base=current_imports)
        current_symbols = next_symbols

    result = AnalysisResult(source=parsed.source)
    selected_registry = registry if registry is not None else DEFAULT_REGISTRY
    for framework, analyzers in selected_registry.analyzers().items():
        framework_detected = False
        for analyzer in analyzers:
            for statement, statement_imports, statement_symbols in snapshots:
                semantic = analyzer(statement, statement_imports, statement_symbols)
                if semantic.operations or semantic.claims:
                    framework_detected = True
                    result.operations.extend(semantic.operations)
                    result.claims.extend(semantic.claims)
        if framework_detected:
            result.frameworks.append(framework)

    # A context manager is recognized by the legacy whole-tree visitor
    # before sibling call expressions. Keep that evidence ordering stable in
    # scientific/README output after splitting the source into statements.
    context_rules = {"pytorch.no_grad", "pytorch.inference_mode", "pytorch.autocast"}
    result.claims.sort(key=lambda claim: 0 if claim.rule in context_rules else 1)

    # A binding configured in a prior notebook cell may be referenced several
    # times in this cell, but its contextual evidence must be reported once.
    unique_claims = []
    seen_claims: set[tuple[str, str, int | None, int | None, str]] = set()
    for claim in result.claims:
        key = (
            claim.rule, claim.evidence.source, claim.evidence.cell,
            claim.evidence.line_start, claim.text,
        )
        if key not in seen_claims:
            seen_claims.add(key)
            unique_claims.append(claim)
    result.claims = unique_claims

    result.training_step = detect_training_step(result.operations)
    result.inference = detect_inference(result.operations)
    return result, current_imports, current_symbols


def analyze(source: str, *, registry: SemanticRegistry | None = None) -> AnalysisResult:
    parsed = parse_source(source)
    result, _, _ = _analyze_parsed(parsed, ImportTable(), SymbolTable(), registry)
    return result


def explain(
    source: str,
    style: str = "scientific",
    *,
    return_result: bool = False,
    registry: SemanticRegistry | None = None,
) -> str | AnalysisResult:
    result = analyze(source, registry=registry)
    return result if return_result else result.render(style)
