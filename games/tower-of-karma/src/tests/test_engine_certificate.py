import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from src.solver.array_best_response import saddle_gap
from src.solver.best_response import certify as rational_certify
from src.solver.build_tree import build_tree
from src.solver.cfr_plus import CFRPlus
from src.solver.engine_certificate import certify
from src.solver.primitives import Phase
from src.solver.public_tree import build_public_tree
from src.solver.sequences import build_sequence_index
from src.solver.strategy_table import StrategyTable, export_table, info_key

sys.setrecursionlimit(10000)


def rational_behavior(index, p: int, table: StrategyTable):
    rows = table.rows()
    out = {}
    for info in index.players[p].information_sets:
        state = info.information_state
        actions = [index.players[p].sequences[s].action for s in info.action_sequence_ids]
        if state.phase in (Phase.CHOOSE_P0, Phase.CHOOSE_P1):
            weights = [Fraction(float(table.choose[a.count - 1])) for a in actions]
        else:
            probs = table.probs[rows[info_key(state.own_beads, tuple(m.total for m in state.public_history))]]
            weights = [Fraction(float(probs[a.total - state.own_beads - 1])) for a in actions]
        total = sum(weights)
        out[state] = {a: w / total for a, w in zip(actions, weights)}
    return out


class EngineCertificateTests(unittest.TestCase):
    def test_matches_rational_and_array_certificates(self) -> None:
        for horizon, iterations in ((1, 50), (2, 40)):
            layers = build_public_tree(horizon)
            cfr = CFRPlus(layers)
            for _ in range(iterations):
                cfr.iterate()
            behavior, choose = cfr.average()
            with tempfile.TemporaryDirectory() as tmp:
                paths = [str(Path(tmp) / f"p{p}.npz") for p in (0, 1)]
                for p in (0, 1):
                    export_table(layers, behavior, choose, p).save(paths[p])
                lower, upper = certify(horizon, *paths)
                tables = [StrategyTable.load(path) for path in paths]
            array_lower, array_upper = saddle_gap(layers, behavior, choose)
            tree = build_tree(horizon)
            index = build_sequence_index(tree)
            exact = rational_certify(tree, rational_behavior(index, 0, tables[0]), rational_behavior(index, 1, tables[1]))
            self.assertGreater(upper - lower, 1e-4)
            for got, want in zip((lower, upper), exact):
                self.assertAlmostEqual(got, float(want), places=12)
            self.assertAlmostEqual(lower, array_lower, places=12)
            self.assertAlmostEqual(upper, array_upper, places=12)

    def test_parallel_matches_serial(self) -> None:
        layers = build_public_tree(2)
        cfr = CFRPlus(layers)
        for _ in range(20):
            cfr.iterate()
        behavior, choose = cfr.average()
        with tempfile.TemporaryDirectory() as tmp:
            paths = [str(Path(tmp) / f"p{p}.npz") for p in (0, 1)]
            for p in (0, 1):
                export_table(layers, behavior, choose, p).save(paths[p])
            serial = certify(2, *paths)
            parallel = certify(2, *paths, workers=2)
        for a, b in zip(serial, parallel):
            self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
