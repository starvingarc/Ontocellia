from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContributionNode:
    id: str
    kind: str
    label: str
    tick: int | None = None
    cell_id: int | None = None
    fate: str | None = None
    gene_id: str | None = None
    record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContributionEdge:
    source_id: str
    target_id: str
    relation: str
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContributionScore:
    node_id: str
    positive: float = 0.0
    negative: float = 0.0
    total: float = 0.0
    confidence: float = 0.0
    evidence_count: int = 0

    def add(self, *, positive: float = 0.0, negative: float = 0.0) -> None:
        self.positive = round(self.positive + max(0.0, positive), 6)
        self.negative = round(self.negative + max(0.0, negative), 6)
        self.total = round(self.positive - self.negative, 6)
        self.evidence_count += 1
        self.confidence = round(min(1.0, self.evidence_count / 5.0), 6)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContributionReport:
    nodes: list[ContributionNode]
    edges: list[ContributionEdge]
    scores: list[ContributionScore]
    top_cells: list[dict[str, Any]]
    top_genes: list[dict[str, Any]]
    top_matrix_records: list[dict[str, Any]]
    negative_paths: list[dict[str, Any]]
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": {
                "nodes": [node.as_dict() for node in self.nodes],
                "edges": [edge.as_dict() for edge in self.edges],
            },
            "scores": [score.as_dict() for score in self.scores],
            "top_cells": list(self.top_cells),
            "top_genes": list(self.top_genes),
            "top_matrix_records": list(self.top_matrix_records),
            "negative_paths": list(self.negative_paths),
            "summary": dict(self.summary),
        }

    def write(self, output: str | Path) -> dict[str, str]:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "graph": output_dir / "contribution_graph.json",
            "report": output_dir / "contribution_report.md",
            "cell_contributions": output_dir / "cell_contributions.csv",
            "gene_contributions": output_dir / "gene_contributions.csv",
            "matrix_contributions": output_dir / "matrix_contributions.csv",
            "summary": output_dir / "contribution_summary.json",
        }
        _write_json(paths["graph"], self.as_dict()["graph"])
        _write_json(paths["summary"], self.summary)
        paths["report"].write_text(_markdown_report(self), encoding="utf-8")
        _write_csv(paths["cell_contributions"], ["cell_id", "node_id", "total", "positive", "negative", "evidence_count"], self.top_cells)
        _write_csv(paths["gene_contributions"], ["gene_id", "node_id", "total", "positive", "negative", "evidence_count"], self.top_genes)
        _write_csv(paths["matrix_contributions"], ["record_id", "node_id", "total", "positive", "negative", "evidence_count"], self.top_matrix_records)
        return {key: str(path) for key, path in paths.items()}


class ContributionAttributionRuntime:
    def analyze(
        self,
        *,
        tissue: Any | None = None,
        trace: list[dict[str, Any]] | None = None,
        summary: dict[str, Any] | None = None,
        actions: list[Any] | None = None,
        tool_results: list[Any] | None = None,
        execution_results: list[Any] | None = None,
        validation_results: list[Any] | None = None,
    ) -> ContributionReport:
        builder = _ContributionBuilder()
        events = [dict(event) for event in (trace if trace is not None else getattr(getattr(tissue, "trace", None), "events", []) or [])]
        action_items = [_to_dict(action) for action in actions or []]
        tool_items = [_to_dict(result) for result in tool_results or []]
        execution_items = [_to_dict(result) for result in execution_results or []]
        validation_items = [_to_dict(result) for result in validation_results or []]
        if tissue is not None:
            builder.add_tissue_cells(tissue)
            builder.add_matrix_records(getattr(getattr(tissue, "environment", None), "matrix", None))
        builder.add_summary(summary or {})
        for action in action_items:
            builder.add_action(action)
        for event in events:
            builder.add_trace_event(event)
        for result in tool_items:
            builder.add_tool_result(result)
        for result in execution_items:
            builder.add_execution_result(result)
        for result in validation_items:
            builder.add_validation_result(result)
        return builder.report()

    def analyze_artifacts(
        self,
        *,
        trace_path: str | Path,
        summary_path: str | Path | None = None,
        actions_path: str | Path | None = None,
        tool_results_path: str | Path | None = None,
        execution_results_path: str | Path | None = None,
        validation_results_path: str | Path | None = None,
    ) -> ContributionReport:
        return self.analyze(
            trace=_read_json_list(trace_path),
            summary=_read_json_dict(summary_path),
            actions=_read_json_list(actions_path),
            tool_results=_read_json_list(tool_results_path),
            execution_results=_read_json_list(execution_results_path),
            validation_results=_read_json_list(validation_results_path),
        )


class _ContributionBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, ContributionNode] = {}
        self.edges: dict[tuple[str, str, str], ContributionEdge] = {}
        self.scores: dict[str, ContributionScore] = {}
        self.handoff_sources: dict[str, int] = {}
        self.result_ids: set[str] = set()

    def add_tissue_cells(self, tissue: Any) -> None:
        for cell in getattr(tissue, "cells", {}).values():
            node_id = _cell_node_id(getattr(cell, "id", None))
            self.node(node_id, "cell", f"cell {getattr(cell, 'id', '')}", cell_id=getattr(cell, "id", None), fate=getattr(cell, "fate", None))
            for gene_id in getattr(cell, "expressed_gene_ids", []) or []:
                self.node(_gene_node_id(gene_id), "gene", str(gene_id), gene_id=str(gene_id))
                self.edge(_gene_node_id(gene_id), node_id, "expressed_by", "cell expressed gene")
                self.score(_gene_node_id(gene_id), positive=0.1)

    def add_matrix_records(self, matrix: Any | None) -> None:
        for record in getattr(matrix, "records", []) or []:
            data = _to_dict(record)
            record_id = str(data.get("id", ""))
            if not record_id:
                continue
            node_id = _matrix_node_id(record_id)
            self.node(
                node_id,
                "matrix_record",
                record_id,
                tick=_int_or_none(data.get("created_tick")),
                cell_id=_int_or_none(data.get("source_cell_id")),
                fate=data.get("fate"),
                record_id=record_id,
                metadata={key: data.get(key) for key in ("kind", "tags", "validation_status", "status", "references") if key in data},
            )
            status = str(data.get("validation_status") or "unverified")
            if status == "validated":
                self.score(node_id, positive=0.25)
            elif status in {"failed", "contradicted"}:
                self.score(node_id, negative=0.25)
            for reference in data.get("references", []) or []:
                self.edge(_reference_node_id(str(reference)), node_id, "referenced", "matrix record reference")

    def add_summary(self, summary: dict[str, Any]) -> None:
        for action in summary.get("actions", []) or []:
            if isinstance(action, dict):
                self.add_action(action)

    def add_action(self, action: dict[str, Any]) -> str:
        action_id = _action_node_id(action)
        cell_id = _int_or_none(action.get("cell_id"))
        self.node(action_id, "action", _action_label(action), cell_id=cell_id, fate=action.get("fate"), metadata={"intent_type": action.get("intent_type") or action.get("gene_id")})
        if cell_id is not None:
            self.node(_cell_node_id(cell_id), "cell", f"cell {cell_id}", cell_id=cell_id, fate=action.get("fate"))
            self.edge(_cell_node_id(cell_id), action_id, "emitted", "cell emitted action")
            self.score(_cell_node_id(cell_id), positive=0.15)
        for gene_id in action.get("expressed_gene_ids", []) or ([] if "gene_id" not in action else [action["gene_id"]]):
            gene_node = _gene_node_id(str(gene_id))
            self.node(gene_node, "gene", str(gene_id), gene_id=str(gene_id))
            self.edge(gene_node, action_id, "expressed_by", "gene shaped action")
            self.score(gene_node, positive=0.15)
        for record_id in _context_record_ids(action):
            matrix_node = _matrix_node_id(record_id)
            self.node(matrix_node, "matrix_record", record_id, record_id=record_id)
            self.edge(matrix_node, action_id, "referenced", "action used matrix context")
            self.score(matrix_node, positive=0.2)
        return action_id

    def add_trace_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "matrix_deposit":
            self.add_matrix_records(_MatrixProxy([event]))
        elif event_type == "llm_effector":
            action = event.get("intent")
            if isinstance(action, dict):
                action_id = self.add_action(action)
                cell_id = _int_or_none(event.get("cell_id"))
                if cell_id is not None:
                    self.edge(_cell_node_id(cell_id), action_id, "emitted", "llm effector emitted intent")
                for record_id in [str(item) for item in event.get("context_record_ids", []) or []]:
                    self.node(_matrix_node_id(record_id), "matrix_record", record_id, record_id=record_id)
                    self.edge(_matrix_node_id(record_id), action_id, "referenced", "prompt context reference")
                    self.score(_matrix_node_id(record_id), positive=0.25)
        elif event_type == "message_emitted":
            message_id = str(event.get("id") or event.get("message_id") or "")
            if message_id:
                node_id = _message_node_id(message_id)
                sender = _int_or_none(event.get("sender_cell_id"))
                self.node(node_id, "message", message_id, tick=_int_or_none(event.get("tick")), cell_id=sender, metadata={"kind": event.get("kind")})
                if sender is not None:
                    self.edge(_cell_node_id(sender), node_id, "emitted", "cell emitted message")
        elif event_type == "handoff_requested":
            request_id = str(event.get("request_id") or "")
            source = _int_or_none(event.get("source_cell_id"))
            if request_id:
                node_id = _handoff_node_id(request_id)
                self.node(node_id, "handoff", request_id, tick=_int_or_none(event.get("tick")), cell_id=source, fate=event.get("target_fate"))
                self.handoff_sources[request_id] = source or 0
                message_id = str(event.get("message_id") or "")
                if message_id:
                    self.edge(_message_node_id(message_id), node_id, "handoff_requested", "message requested handoff")
                if source is not None:
                    self.edge(_cell_node_id(source), node_id, "emitted", "cell requested handoff")
        elif event_type == "handoff_completed":
            request_id = str(event.get("request_id") or "")
            recipient = _int_or_none(event.get("recipient_cell_id"))
            accepted = bool(event.get("accepted", True))
            if request_id:
                handoff_node = _handoff_node_id(request_id)
                self.node(handoff_node, "handoff", request_id, tick=_int_or_none(event.get("tick")))
                if recipient is not None:
                    recipient_node = _cell_node_id(recipient)
                    self.node(recipient_node, "cell", f"cell {recipient}", cell_id=recipient)
                    self.edge(handoff_node, recipient_node, "handoff_completed", "handoff completed")
                    self.score(recipient_node, positive=0.35 if accepted else 0.0, negative=0.0 if accepted else 0.35)
                source = self.handoff_sources.get(request_id)
                if source:
                    self.score(_cell_node_id(source), positive=0.45 if accepted else 0.0, negative=0.0 if accepted else 0.45)
                self.score(handoff_node, positive=0.35 if accepted else 0.0, negative=0.0 if accepted else 0.35)
        elif event_type in {"tool_completed", "tool_skipped", "execution_completed", "execution_skipped", "validation_hook_completed", "validation_hook_skipped"}:
            if "invocation" in event:
                self.add_tool_result(event)
            elif "request" in event:
                self.add_execution_result(event)
            elif "result" in event:
                self.add_validation_result(event["result"])

    def add_tool_result(self, result: dict[str, Any]) -> None:
        invocation = result.get("invocation") if isinstance(result.get("invocation"), dict) else result.get("request", {})
        invocation = invocation if isinstance(invocation, dict) else {}
        result_id = str(invocation.get("id") or result.get("id") or f"tool-{len(self.nodes)}")
        if result_id in self.result_ids:
            return
        self.result_ids.add(result_id)
        node_id = _tool_node_id(result_id)
        cell_id = _int_or_none(invocation.get("cell_id"))
        passed = bool(result.get("passed", False))
        status = str(result.get("status") or "")
        self.node(node_id, "tool_result", result_id, cell_id=cell_id, metadata={"status": status, "interface": invocation.get("interface")})
        if cell_id is not None:
            self.edge(_cell_node_id(cell_id), node_id, "emitted", "cell requested tool")
        for record_id in [str(item) for item in invocation.get("context_record_ids", []) or []]:
            self.node(_matrix_node_id(record_id), "matrix_record", record_id, record_id=record_id)
            self.edge(_matrix_node_id(record_id), node_id, "referenced", "tool used context")
            self.score(_matrix_node_id(record_id), positive=0.15)
        positive = 0.55 if passed and status not in {"skipped", "dry_run"} else 0.0
        negative = 0.55 if (not passed or status in {"skipped", "dry_run", "failed"}) else 0.0
        self.score(node_id, positive=positive, negative=negative)
        if cell_id is not None:
            self.score(_cell_node_id(cell_id), positive=positive * 0.5, negative=negative * 0.5)

    def add_execution_result(self, result: dict[str, Any]) -> None:
        request = result.get("request") if isinstance(result.get("request"), dict) else {}
        invocation = {
            "id": request.get("id") or result.get("id"),
            "cell_id": request.get("cell_id"),
            "intent_type": request.get("intent_type"),
            "interface": request.get("interface"),
            "context_record_ids": [],
        }
        self.add_tool_result({"invocation": invocation, **result})

    def add_validation_result(self, result: dict[str, Any]) -> None:
        name = str(result.get("name") or "validation")
        node_id = f"validation:{name}:{len([key for key in self.nodes if key.startswith('validation:')])}"
        passed = bool(result.get("passed", False))
        self.node(node_id, "validation_result", name, metadata={"target": result.get("target"), "evidence": result.get("evidence")})
        self.score(node_id, positive=0.6 if passed else 0.0, negative=0.6 if not passed else 0.0)
        action_nodes = [node.id for node in self.nodes.values() if node.kind == "action"]
        for action_node in action_nodes:
            self.edge(action_node, node_id, "validated", "validation applied to action")
            self.score(action_node, positive=0.25 if passed else 0.0, negative=0.25 if not passed else 0.0)

    def node(self, node_id: str, kind: str, label: str, **kwargs: Any) -> ContributionNode:
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if node.fate is None and kwargs.get("fate") is not None:
                node.fate = str(kwargs["fate"])
            if node.cell_id is None and kwargs.get("cell_id") is not None:
                node.cell_id = _int_or_none(kwargs["cell_id"])
            node.metadata.update({key: value for key, value in dict(kwargs.get("metadata", {})).items() if value is not None})
            return node
        node = ContributionNode(node_id, kind, label, **kwargs)
        self.nodes[node_id] = node
        return node

    def edge(self, source_id: str, target_id: str, relation: str, evidence: str = "") -> None:
        if not source_id or not target_id:
            return
        self.edges.setdefault((source_id, target_id, relation), ContributionEdge(source_id, target_id, relation, evidence))

    def score(self, node_id: str, *, positive: float = 0.0, negative: float = 0.0) -> None:
        self.scores.setdefault(node_id, ContributionScore(node_id)).add(positive=positive, negative=negative)

    def report(self) -> ContributionReport:
        for node_id in self.nodes:
            self.scores.setdefault(node_id, ContributionScore(node_id))
        ordered_nodes = sorted(self.nodes.values(), key=lambda node: (node.kind, node.id))
        ordered_edges = sorted(self.edges.values(), key=lambda edge: (edge.source_id, edge.target_id, edge.relation))
        ordered_scores = sorted(self.scores.values(), key=lambda score: (-score.total, score.node_id))
        top_cells = self._top("cell", "cell_id")
        top_genes = self._top("gene", "gene_id")
        top_matrix = self._top("matrix_record", "record_id")
        negative_paths = [
            {**score.as_dict(), "label": self.nodes.get(score.node_id, ContributionNode(score.node_id, "unknown", score.node_id)).label}
            for score in sorted(self.scores.values(), key=lambda item: (-item.negative, item.node_id))
            if score.negative > 0
        ][:10]
        summary = {
            "nodes": len(ordered_nodes),
            "edges": len(ordered_edges),
            "scored_nodes": len(ordered_scores),
            "top_cell_id": top_cells[0]["cell_id"] if top_cells else None,
            "top_gene_id": top_genes[0]["gene_id"] if top_genes else None,
            "top_matrix_record_id": top_matrix[0]["record_id"] if top_matrix else None,
            "negative_paths": len(negative_paths),
        }
        return ContributionReport(ordered_nodes, ordered_edges, ordered_scores, top_cells, top_genes, top_matrix, negative_paths, summary)

    def _top(self, kind: str, field_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node in self.nodes.values():
            if node.kind != kind:
                continue
            score = self.scores.get(node.id, ContributionScore(node.id))
            value = getattr(node, field_name)
            if value is None:
                continue
            rows.append({field_name: value, **score.as_dict()})
        rows.sort(key=lambda row: (-float(row["total"]), str(row[field_name])))
        return rows[:10]


@dataclass(slots=True)
class _MatrixProxy:
    records: list[dict[str, Any]]


def _action_node_id(action: dict[str, Any]) -> str:
    if action.get("id"):
        return f"action:{action['id']}"
    cell = action.get("cell_id", "unknown")
    intent = action.get("intent_type") or action.get("gene_id") or "action"
    target = str(action.get("target") or action.get("interface_id") or "target").replace(" ", "-")
    return f"action:cell-{cell}:{intent}:{target}"


def _action_label(action: dict[str, Any]) -> str:
    return str(action.get("intent_type") or action.get("gene_id") or "action")


def _context_record_ids(action: dict[str, Any]) -> list[str]:
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    return [str(item) for item in payload.get("context_record_ids", []) or []]


def _cell_node_id(cell_id: Any) -> str:
    return f"cell:{cell_id}"


def _gene_node_id(gene_id: str) -> str:
    return f"gene:{gene_id}"


def _matrix_node_id(record_id: str) -> str:
    return f"matrix:{record_id}"


def _message_node_id(message_id: str) -> str:
    return f"message:{message_id}"


def _handoff_node_id(request_id: str) -> str:
    return f"handoff:{request_id}"


def _tool_node_id(result_id: str) -> str:
    return f"tool:{result_id}"


def _reference_node_id(reference: str) -> str:
    if reference.startswith(("matrix:", "cell:", "gene:", "action:", "tool:", "validation:", "handoff:", "message:")):
        return reference
    if reference.startswith(("tool-", "execution-")):
        return _tool_node_id(reference)
    return f"reference:{reference}"


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    return dict(asdict(value))


def _read_json_list(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


def _read_json_dict(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _markdown_report(report: ContributionReport) -> str:
    lines = [
        "# Contribution Attribution Report",
        "",
        f"- Nodes: {report.summary['nodes']}",
        f"- Edges: {report.summary['edges']}",
        f"- Top cell: {report.summary['top_cell_id']}",
        f"- Top gene: {report.summary['top_gene_id']}",
        f"- Top matrix record: {report.summary['top_matrix_record_id']}",
        "",
        "## Top Cells",
        "",
        "| Cell | Total | Positive | Negative |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report.top_cells:
        lines.append(f"| {row['cell_id']} | {row['total']:.3f} | {row['positive']:.3f} | {row['negative']:.3f} |")
    lines.extend(["", "## Negative Paths", ""])
    if not report.negative_paths:
        lines.append("No negative contribution paths were detected.")
    else:
        for row in report.negative_paths:
            lines.append(f"- `{row['node_id']}` negative={row['negative']:.3f}: {row['label']}")
    return "\n".join(lines) + "\n"


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
