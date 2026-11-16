"""One player's behavioral strategy keyed by information state.

The key packs (own beads, ordered transcript of announced totals) into one
integer. Column a of a row announces own beads + a + 1. The table is the
hand-off format between a solver and engine_certificate.
"""

from dataclasses import dataclass

import numpy as np

from src.solver.public_tree import HANDS, TOTALS, Layer


def info_key(own_beads: int, totals: tuple[int, ...]) -> int:
    code = 0
    for g in totals:
        code = code * TOTALS + (g - 2)
    return (code * 8 + len(totals)) * HANDS + own_beads - 1


@dataclass(frozen=True, slots=True)
class StrategyTable:
    choose: np.ndarray  # (10,) bead choice probabilities
    keys: np.ndarray  # (n,) info_key of each guess information set
    probs: np.ndarray  # (n, 10)

    def save(self, path: str) -> None:
        np.savez(path, choose=self.choose, keys=self.keys, probs=self.probs)

    @staticmethod
    def load(path: str) -> "StrategyTable":
        data = np.load(path)
        return StrategyTable(data["choose"], data["keys"], data["probs"])

    def rows(self) -> dict[int, int]:
        return dict(zip(self.keys.tolist(), range(len(self.keys))))


def transcript_codes(layers: list[Layer]) -> list[np.ndarray]:
    """Base-19 code of each public node's transcript, layer by layer."""
    codes = [np.zeros(1, dtype=np.int64)]
    for layer in layers[1:]:
        codes.append(codes[-1][layer.parent] * TOTALS + (layer.total.astype(np.int64) - 2))
    return codes


def export_table(layers: list[Layer], behavior, choose, p: int) -> StrategyTable:
    """Collect player p's behavior from the public-tree arrays."""
    codes = transcript_codes(layers)
    keys, probs = [], []
    for d, layer in enumerate(layers):
        if layer.actor != p:
            continue
        keys.append((codes[d][layer.slot_node[p]] * 8 + d) * HANDS + layer.slot_hand[p] - 1)
        probs.append(behavior[d])
    return StrategyTable(np.asarray(choose[p], dtype=float), np.concatenate(keys), np.concatenate(probs))
