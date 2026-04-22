from __future__ import annotations

from dataclasses import dataclass, field

from ontocellia.config import GeneAsset, GeneKind


@dataclass(slots=True)
class GeneRegistry:
    genes: list[GeneAsset] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)

    def add(self, gene: GeneAsset) -> None:
        self.genes.append(gene)
        self.events.append({"event": "add", "gene": gene.name, "kind": gene.kind.value})

    def evaluate(self, context: dict[str, float]) -> dict[str, float]:
        strategy = 0.0
        warning = 0.0
        active_genes: list[str] = []
        for gene in self.genes:
            if gene.matches(context):
                active_genes.append(gene.name)
                if gene.kind is GeneKind.STRATEGY:
                    strategy += gene.magnitude
                elif gene.kind is GeneKind.WARNING:
                    warning += gene.magnitude
        return {"strategy": strategy, "warning": warning, "active_names": active_genes}
