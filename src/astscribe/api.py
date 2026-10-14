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

    for statement in parsed.tree.body:
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
