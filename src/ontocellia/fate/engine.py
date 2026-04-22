from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ontocellia.config import FATE_NAMES, OntocelliaConfig


@dataclass(slots=True)
class FateEngine:
    config: OntocelliaConfig

    def update(self, cell, fate_logits: np.ndarray) -> None:
        logits = np.asarray(fate_logits, dtype=float)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        base_commitment = self.config.commitment_decay * cell.commitment + (1.0 - self.config.commitment_decay) * probs
        if self.config.enable_epigenetic_lock:
            dominant_index = int(cell.commitment.argmax())
            lock_vec = np.zeros_like(base_commitment)
            lock_vec[dominant_index] = self.config.lock_strength * max(cell.epigenetic_lock, 0.1)
            base_commitment += lock_vec
            base_commitment /= base_commitment.sum()
            cell.epigenetic_lock = float(np.clip(cell.epigenetic_lock + 0.04 * base_commitment.max(), 0.05, 1.0))
        else:
            cell.epigenetic_lock = max(0.05, cell.epigenetic_lock * 0.98)

        cell.fate_dist = probs
        cell.commitment = np.clip(base_commitment, 0.0, 1.0)
        cell.commitment /= cell.commitment.sum()
        candidate = FATE_NAMES[int(cell.commitment.argmax())]
        if cell.commitment.max() >= self.config.commitment_threshold or probs.max() >= 0.6:
            cell.previous_fate = cell.current_fate
            cell.current_fate = candidate
