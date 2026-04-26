from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


FATE_BY_GENE_CATEGORY = {
    "exploration": "explorer",
    "planning": "planner",
    "implementation": "builder",
    "regeneration": "repair",
    "repair": "repair",
    "verification": "reviewer",
    "review": "reviewer",
    "memory": "memory",
    "quiescence": "quiescent",
}


@dataclass(slots=True)
class Gene:
    """Lowest-level endogenous control unit in an agent genome."""

    id: str
    category: str
    morphogen_affinity: list[str]
    encoded_response: list[str]
    expression_window: list[str] = field(default_factory=list)
    inhibitors: list[str] = field(default_factory=list)
    suppression_cues: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    validation_hooks: list[str] = field(default_factory=list)
    heritability: dict[str, Any] = field(default_factory=dict)

    @property
    def fate_bias(self) -> str:
        return FATE_BY_GENE_CATEGORY.get(self.category, self.category)

    def expression_score(self, morphogens: Any) -> float:
        if isinstance(morphogens, dict):
            signal = lambda name: float(morphogens.get(name, 0.0))
        else:
            signal = morphogens.signal
        promoter_score = sum(signal(name) for name in self.morphogen_affinity)
        inhibitor_score = sum(signal(name) for name in self.inhibitors)
        return max(0.0, promoter_score - inhibitor_score)


@dataclass(slots=True)
class RegulatoryElement:
    id: str
    kind: str
    target_gene_id: str
    signals: list[str] = field(default_factory=list)
    strength: float = 0.0

    def modifier(self, morphogens: dict[str, float]) -> float:
        signal_score = sum(float(morphogens.get(signal, 0.0)) for signal in self.signals) if self.signals else 1.0
        magnitude = self.strength * signal_score
        if self.kind in {"promoter", "enhancer"}:
            return magnitude
        if self.kind in {"inhibitor", "silencer"}:
            return -magnitude
        raise ValueError(f"Unknown regulatory element kind: {self.kind}")


@dataclass(slots=True)
class EpigeneticMarks:
    fate_locks: dict[str, float] = field(default_factory=dict)
    gene_locks: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExpressionContext:
    morphogens: dict[str, float]
    cell_stage: str = "stem"
    current_fate: str = "stem"
    energy: float = 1.0
    stress: float = 0.0
    competence: dict[str, float] = field(default_factory=dict)
    lineage_history: list[dict[str, Any]] = field(default_factory=list)
    epigenetic_marks: EpigeneticMarks = field(default_factory=EpigeneticMarks)
    organ_feedback: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExpressedGeneProgram:
    gene: Gene
    score: float
    fate_bias: str
    encoded_response: list[str]
    validation_hooks: list[str]
    constraints: dict[str, Any]


@dataclass(slots=True)
class LineageMutation:
    source_gene_id: str
    changed_fields: dict[str, Any]
    objective: str
    validation_result: dict[str, Any]
    lineage_id: str


@dataclass(slots=True)
class AgentGenome:
    """Heritable program shared by a tissue lineage."""

    genes: list[Gene]
    metadata: dict[str, Any] = field(default_factory=dict)
    regulatory_elements: list[RegulatoryElement] = field(default_factory=list)
    epigenetic_defaults: EpigeneticMarks = field(default_factory=EpigeneticMarks)
    mutation_history: list[LineageMutation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_regulatory_targets()

    def gene_by_id(self, gene_id: str) -> Gene:
        for gene in self.genes:
            if gene.id == gene_id:
                return gene
        raise KeyError(f"unknown gene: {gene_id}")

    def expressed_for_fate(self, fate: str, morphogens: Any, limit: int = 2) -> list[Gene]:
        context = ExpressionContext(morphogens=_morphogen_dict(morphogens), current_fate=fate)
        return [program.gene for program in self.express(context, limit=limit) if program.fate_bias == fate]

    def dominant_fate_for(self, morphogens: Any) -> str:
        programs = self.express(ExpressionContext(morphogens=_morphogen_dict(morphogens)), limit=1)
        if not programs:
            return "quiescent"
        return programs[0].fate_bias

    def express(self, context: ExpressionContext, limit: int = 3) -> list[ExpressedGeneProgram]:
        merged_marks = _merge_epigenetic_marks(self.epigenetic_defaults, context.epigenetic_marks)
        programs: list[ExpressedGeneProgram] = []
        for gene in self.genes:
            score = self._score_gene(gene, context, merged_marks)
            if score <= 0.0:
                continue
            programs.append(
                ExpressedGeneProgram(
                    gene=gene,
                    score=score,
                    fate_bias=gene.fate_bias,
                    encoded_response=list(gene.encoded_response),
                    validation_hooks=list(gene.validation_hooks),
                    constraints=dict(gene.constraints),
                )
            )
        programs.sort(key=lambda program: (-program.score, program.gene.id))
        return programs[:limit]

    def mutate(self, mutation: LineageMutation) -> "AgentGenome":
        genes = [self._mutated_gene(gene, mutation) for gene in self.genes]
        return AgentGenome(
            genes=genes,
            metadata=dict(self.metadata),
            regulatory_elements=[replace(element) for element in self.regulatory_elements],
            epigenetic_defaults=EpigeneticMarks(
                fate_locks=dict(self.epigenetic_defaults.fate_locks),
                gene_locks=dict(self.epigenetic_defaults.gene_locks),
            ),
            mutation_history=[*self.mutation_history, mutation],
        )

    def _score_gene(self, gene: Gene, context: ExpressionContext, marks: EpigeneticMarks) -> float:
        score = gene.expression_score(context.morphogens)
        for element in self.regulatory_elements:
            if element.target_gene_id == gene.id:
                score += element.modifier(context.morphogens)
        if context.current_fate != gene.fate_bias:
            score -= float(marks.fate_locks.get(context.current_fate, 0.0))
        score -= float(marks.gene_locks.get(gene.id, 0.0))
        score *= float(context.competence.get(gene.fate_bias, context.competence.get(gene.id, 1.0)))
        score += float(context.organ_feedback.get(gene.fate_bias, context.organ_feedback.get(gene.id, 0.0)))
        cost = float(gene.constraints.get("cost", 0.0))
        if cost:
            score -= cost * max(0.0, 0.5 - context.energy)
            score -= cost * max(0.0, context.stress - 0.5)
        return max(0.0, score)

    def _mutated_gene(self, gene: Gene, mutation: LineageMutation) -> Gene:
        if gene.id != mutation.source_gene_id:
            return replace(gene)
        fields = {**mutation.changed_fields}
        return replace(gene, **fields)

    def _validate_regulatory_targets(self) -> None:
        gene_ids = {gene.id for gene in self.genes}
        for element in self.regulatory_elements:
            if element.target_gene_id not in gene_ids:
                raise ValueError(f"regulatory element {element.id} has unknown target gene: {element.target_gene_id}")


def _morphogen_dict(morphogens: Any) -> dict[str, float]:
    if isinstance(morphogens, dict):
        return {str(name): float(value) for name, value in morphogens.items()}
    signals = getattr(morphogens, "signals", None)
    if isinstance(signals, dict):
        return {str(name): float(value) for name, value in signals.items()}
    return {}


def _merge_epigenetic_marks(defaults: EpigeneticMarks, local: EpigeneticMarks) -> EpigeneticMarks:
    return EpigeneticMarks(
        fate_locks={**defaults.fate_locks, **local.fate_locks},
        gene_locks={**defaults.gene_locks, **local.gene_locks},
    )
