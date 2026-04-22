from __future__ import annotations

from dataclasses import dataclass

from ontocellia.architecture.models import OrganFeedback


@dataclass(slots=True)
class OrganSelectionField:
    strength: float = 0.18

    def evaluate(self, runtime) -> dict[str, float]:
        latest = runtime.metrics.history[-1] if runtime.metrics.history else {}
        coverage = float(latest.get("coverage", 0.0))
        diversity = float(latest.get("development_diversity", latest.get("heterogeneity", 0.0)))
        avg_energy = float(latest.get("avg_energy", 0.0))
        avg_stress = float(latest.get("avg_stress", 0.0))
        repair = float(latest.get("repair_activation", latest.get("repair_fraction", 0.0)))
        graph_density = float(latest.get("graph_density", 0.0))
        task_completion = min(1.0, 0.45 * coverage + 0.3 * diversity + 0.25 * graph_density)
        structural_stability = max(0.0, 1.0 - avg_stress)
        energy_cost = max(0.0, 1.0 - avg_energy)
        redundancy = graph_density
        recovery_capacity = repair
        return {
            "task_completion": task_completion,
            "structural_stability": structural_stability,
            "energy_cost": energy_cost,
            "redundancy": redundancy,
            "recovery_capacity": recovery_capacity,
            "diversity": diversity,
        }

    def feedback(self, runtime) -> OrganFeedback:
        latest = runtime.metrics.history[-1] if runtime.metrics.history else {}
        targets = runtime.built_environment.evaluation if getattr(runtime, "built_environment", None) is not None else {}
        coverage = float(latest.get("coverage", 0.0))
        diversity = float(latest.get("development_diversity", latest.get("heterogeneity", 0.0)))
        risk = float(latest.get("risk_exposure", 0.0))
        min_coverage = float(targets.get("min_coverage", 0.25))
        min_diversity = float(targets.get("min_diversity", 0.45))
        max_risk = float(targets.get("max_risk_exposure", 0.45))

        task_bias = self.strength * max(0.0, min_coverage - coverage)
        resource_bias = self.strength * 0.8 * max(0.0, min_coverage - coverage)
        reward_bias = self.strength * max(0.0, min_diversity - diversity)
        damage_tolerance = self.strength * max(0.0, max_risk - risk)
        selection_pressure = self.strength * (max(0.0, min_coverage - coverage) + max(0.0, min_diversity - diversity))
        target_regions = []
        for zone in getattr(runtime, "goals", []):
            if "center" in zone and "radius" in zone:
                center = zone["center"]
                radius = float(zone["radius"])
                target_regions.append((float(center[0]), float(center[1]), radius))
        return OrganFeedback(
            task_pressure_bias=task_bias,
            resource_pressure_bias=resource_bias,
            damage_tolerance_bias=damage_tolerance,
            reward_field_bias=reward_bias,
            selection_pressure=selection_pressure,
            target_regions=target_regions,
        )
