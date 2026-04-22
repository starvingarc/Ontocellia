from __future__ import annotations

import numpy as np

from ontocellia.architecture import GenomeInput, GenomeProgram, LocalContext, NeighborhoodState
from ontocellia.architecture.models import CellTransition
from ontocellia.compiler import CompiledGenome
from ontocellia.config import OntocelliaConfig

KernelOutput = CellTransition


class GenomeKernel(GenomeProgram):
    def __init__(self, config: OntocelliaConfig, compiled_genome: CompiledGenome | None = None):
        super().__init__(config=config, compiled_genome=compiled_genome)

    def step(
        self,
        cell,
        local_fields: dict[str, float],
        gradients: dict[str, np.ndarray],
        neighbor_summary: dict[str, np.ndarray | float],
        gene_context: dict[str, float],
    ) -> KernelOutput:
        field_names = self.compiled_genome.field_order if self.compiled_genome is not None else list(local_fields)
        background_names = {"ECM", "mechanical_stress", "crowding", "resource_availability"}
        local_context = LocalContext(
            diffusive_fields=dict(local_fields),
            gradient_fields={name: np.asarray(gradients.get(name, np.zeros(2, dtype=float)), dtype=float) for name in field_names},
            neighbor_messages={
                "contact_inhibition": float(neighbor_summary.get("contact_inhibition", 0.0)),
                "neighbor_quiescence": float(neighbor_summary.get("neighbor_quiescence", 0.0)),
                "community_signal": float(neighbor_summary.get("community_signal", 0.0)),
            },
            mechanical_resource_context={name: float(local_fields.get(name, 0.0)) for name in local_fields if name in background_names},
            local_risk=float(local_fields.get("damage", 0.0) + local_fields.get("crowding", 0.0) * 0.5),
            global_signals={},
        )
        genome_input = GenomeInput(
            self_state=cell,
            neighborhood_state=NeighborhoodState.from_summary(neighbor_summary),
            local_context=local_context,
            history_state=cell.summarize_history(),
        )
        return super().step(genome_input, gene_context)


__all__ = ["GenomeKernel", "KernelOutput"]
