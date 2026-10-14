from __future__ import annotations

from astscribe.parser import (
    ImportTable,
    ParsedSource,
    SymbolTable,
    build_import_table,
    build_symbol_table,
    parse_source,
)
from astscribe.patterns import detect_inference, detect_training_step
from astscribe.semantics import DEFAULT_REGISTRY, SemanticRegistry
from astscribe.sir import AnalysisResult


def _analyze_parsed(
    parsed: ParsedSource,
    imports: ImportTable,
    symbols: SymbolTable,
    registry: SemanticRegistry | None = None,
) -> AnalysisResult:
    result = AnalysisResult(source=parsed.source)
    selected_registry = registry if registry is not None else DEFAULT_REGISTRY
    for framework, analyzers in selected_registry.analyzers().items():
        framework_detected = False
        for analyzer in analyzers:
            semantic = analyzer(parsed, imports, symbols)
            if semantic.operations or semantic.claims:
                framework_detected = True
                result.operations.extend(semantic.operations)
                result.claims.extend(semantic.claims)
        if framework_detected:
            result.frameworks.append(framework)

    result.training_step = detect_training_step(result.operations)
    result.inference = detect_inference(result.operations)
    return result


def analyze(source: str, *, registry: SemanticRegistry | None = None) -> AnalysisResult:
    parsed = parse_source(source)
    imports = build_import_table(parsed.tree)
    symbols = build_symbol_table(parsed.tree, imports, source=source)
    return _analyze_parsed(parsed, imports, symbols, registry)


def explain(
    source: str,
    style: str = "scientific",
    *,
    return_result: bool = False,
    registry: SemanticRegistry | None = None,
) -> str | AnalysisResult:
    result = analyze(source, registry=registry)
    return result if return_result else result.render(style)
