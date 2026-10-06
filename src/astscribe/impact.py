from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from astscribe.dependency import CellDependencyEdge, NotebookDependencyGraph


@dataclass(frozen=True)
class ImpactHop:
    producer_cell: int
    consumer_cell: int
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class ImpactPath:
    target_cell: int
    hops: tuple[ImpactHop, ...]

    @property
    def distance(self) -> int:
        return len(self.hops)

    @property
    def cells(self) -> tuple[int, ...]:
        if not self.hops:
            return (self.target_cell,)
        return (self.hops[0].producer_cell,) + tuple(
            hop.consumer_cell for hop in self.hops
        )


@dataclass(frozen=True)
class NotebookImpactReport:
    source_cell: int
    direct_dependents: tuple[int, ...]
    affected_cells: tuple[int, ...]
    required_ancestors: tuple[int, ...]
    paths: tuple[ImpactPath, ...]

    @property
    def blast_radius(self) -> int:
        return len(self.affected_cells)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["blast_radius"] = self.blast_radius
        data["paths"] = [
            {
                "target_cell": path.target_cell,
                "distance": path.distance,
                "cells": path.cells,
                "hops": [asdict(hop) for hop in path.hops],
            }
            for path in self.paths
        ]
        return data

    def render(self) -> str:
        lines = [f"# Impact analysis for Cell {self.source_cell}", ""]
        lines.append(f"Blast radius: {self.blast_radius} downstream cell(s).")

        if self.direct_dependents:
            direct = ", ".join(str(cell) for cell in self.direct_dependents)
            lines.append(f"Direct dependents: {direct}.")
        else:
            lines.append("Direct dependents: none.")

        if self.required_ancestors:
            required = ", ".join(str(cell) for cell in self.required_ancestors)
            lines.append(f"Static prerequisites: {required}.")
        else:
            lines.append("Static prerequisites: none.")

        if self.paths:
            lines.extend(["", "Propagation paths:"])
            for path in self.paths:
                cells = " -> ".join(str(cell) for cell in path.cells)
                symbols = " | ".join(
                    ", ".join(hop.symbols) for hop in path.hops
                )
                lines.append(
                    f"- Cell {path.target_cell} (distance {path.distance}): "
                    f"{cells} [{symbols}]"
                )

        return "\n".join(lines)


@dataclass(frozen=True)
class CellImpactSummary:
    cell: int
    direct_dependents: int
    affected_cells: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _cell_ids(graph: NotebookDependencyGraph) -> set[int]:
    return {node.cell for node in graph.nodes}


def _validate_cell(graph: NotebookDependencyGraph, cell: int) -> None:
    if cell not in _cell_ids(graph):
        raise ValueError(f"Cell {cell} is not present in the analyzed dependency graph.")


def _outgoing_edges(graph: NotebookDependencyGraph) -> dict[int, tuple[CellDependencyEdge, ...]]:
    grouped: dict[int, list[CellDependencyEdge]] = {}
    for edge in graph.edges:
        grouped.setdefault(edge.producer_cell, []).append(edge)
    return {
        cell: tuple(sorted(edges, key=lambda edge: edge.consumer_cell))
        for cell, edges in grouped.items()
    }


def _incoming_edges(graph: NotebookDependencyGraph) -> dict[int, tuple[CellDependencyEdge, ...]]:
    grouped: dict[int, list[CellDependencyEdge]] = {}
    for edge in graph.edges:
        grouped.setdefault(edge.consumer_cell, []).append(edge)
    return {
        cell: tuple(sorted(edges, key=lambda edge: edge.producer_cell))
        for cell, edges in grouped.items()
    }


def _transitive_cells(
    start: int,
    adjacency: dict[int, tuple[CellDependencyEdge, ...]],
    *,
    downstream: bool,
) -> tuple[int, ...]:
    visited: set[int] = set()
    queue: deque[int] = deque([start])

    while queue:
        current = queue.popleft()
        for edge in adjacency.get(current, ()):
            neighbor = edge.consumer_cell if downstream else edge.producer_cell
            if neighbor in visited or neighbor == start:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return tuple(sorted(visited))


def _shortest_downstream_paths(
    graph: NotebookDependencyGraph,
    source_cell: int,
) -> tuple[ImpactPath, ...]:
    outgoing = _outgoing_edges(graph)
    queue: deque[tuple[int, tuple[ImpactHop, ...]]] = deque([(source_cell, ())])
    paths: dict[int, tuple[ImpactHop, ...]] = {}

    while queue:
        current, current_hops = queue.popleft()
        for edge in outgoing.get(current, ()):
            target = edge.consumer_cell
            if target in paths or target == source_cell:
                continue
            hops = current_hops + (
                ImpactHop(
                    producer_cell=edge.producer_cell,
                    consumer_cell=edge.consumer_cell,
                    symbols=edge.symbols,
                ),
            )
            paths[target] = hops
            queue.append((target, hops))

    return tuple(
        ImpactPath(target_cell=cell, hops=hops)
        for cell, hops in sorted(paths.items(), key=lambda item: (len(item[1]), item[0]))
    )


def build_impact_report(
    graph: NotebookDependencyGraph,
    cell: int,
) -> NotebookImpactReport:
    _validate_cell(graph, cell)
    outgoing = _outgoing_edges(graph)
    incoming = _incoming_edges(graph)

    direct = tuple(edge.consumer_cell for edge in outgoing.get(cell, ()))
    affected = _transitive_cells(cell, outgoing, downstream=True)
    required = _transitive_cells(cell, incoming, downstream=False)
    paths = _shortest_downstream_paths(graph, cell)

    return NotebookImpactReport(
        source_cell=cell,
        direct_dependents=direct,
        affected_cells=affected,
        required_ancestors=required,
        paths=paths,
    )


def rank_cells_by_impact(
    graph: NotebookDependencyGraph,
) -> tuple[CellImpactSummary, ...]:
    summaries = [
        CellImpactSummary(
            cell=node.cell,
            direct_dependents=len(graph.children(node.cell)),
            affected_cells=build_impact_report(graph, node.cell).blast_radius,
        )
        for node in graph.nodes
    ]
    return tuple(
        sorted(
            summaries,
            key=lambda item: (-item.affected_cells, -item.direct_dependents, item.cell),
        )
    )


def render_impact_ranking(summaries: tuple[CellImpactSummary, ...]) -> str:
    if not summaries:
        return "No analyzable notebook cells are available."

    lines = ["# Notebook impact ranking", ""]
    for summary in summaries:
        lines.append(
            f"- Cell {summary.cell}: {summary.affected_cells} affected cell(s), "
            f"{summary.direct_dependents} direct dependent(s)."
        )
    return "\n".join(lines)
