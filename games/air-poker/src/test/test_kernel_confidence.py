"""K4 gate: the confidence sequences cover the exact kernel on 19-card decks over 1,000 seeds."""

import unittest

import numpy as np

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.kernel.confidence import boxes, interval, intervals, log_mixture, tighten
from src.kernel.exact import exact_kernel_counts
from src.kernel.sampler import DIGIT, ROW_OFFSET, sample_round_classes, sample_sequence
from src.kernel.table import HIDDEN, KernelCounts, Sequences, class_code
from src.sensitivity.sampler import NOJ, TIE, W0, W0C, W1, W1C, HandTables
from src.test.test_kernel_dp import seeded_decks

TABLES = HandTables()
PAIRS = [(a, b) for a in TEAM_0_METAL_CARDS for b in TEAM_1_METAL_CARDS]
REACHABLE = (W0, W0C, W1, W1C, NOJ)


class IntervalTest(unittest.TestCase):
    def test_empty_sample_gives_the_whole_interval(self):
        self.assertEqual(interval(0, 0, 0.01), (0.0, 1.0))

    def test_zero_successes_match_the_closed_form(self):
        for n in (1, 10, 1000):
            lo, hi = interval(0, n, 0.01)
            self.assertEqual(lo, 0.0)
            self.assertAlmostEqual(hi, 1 - np.exp((np.log(0.01) + log_mixture(0, n)) / n), places=12)
            self.assertLess(hi, (np.log(100) + 0.5 * np.log(n) + 1) / n)

    def test_interval_contains_the_frequency_and_shrinks_and_is_symmetric(self):
        widths = []
        for n in (10, 100, 1000, 10000):
            k = n // 3
            lo, hi = interval(k, n, 1e-11)
            self.assertLess(lo, k / n)
            self.assertGreater(hi, k / n)
            widths.append(hi - lo)
            lo_r, hi_r = interval(n - k, n, 1e-11)
            self.assertAlmostEqual(lo_r, 1 - hi, places=12)
            self.assertAlmostEqual(hi_r, 1 - lo, places=12)
        self.assertEqual(widths, sorted(widths, reverse=True))
        # Normal scale at n = 1e4, p = 1/3: 2 * sqrt(2 log(1/alpha) p (1 - p) / n) = 0.067.
        self.assertLess(widths[-1], 0.08)

    def test_vectorized_intervals_match_the_scalar(self):
        k = np.array([[0, 3], [7, 7]])
        n = np.array([[10], [7]])
        lo, hi = intervals(k, n, 0.05)
        for i, j in np.ndindex(2, 2):
            self.assertEqual((lo[i, j], hi[i, j]), interval(k[i, j], n[i, 0], 0.05))


class CoverageGateTest(unittest.TestCase):
    def test_1000_seeds_on_19_card_decks_stay_covered_at_every_sample_size(self):
        alpha = 0.05
        draws = 2000
        rng = np.random.default_rng(20261029)
        violations = 0
        final_widths = []
        for seed, deck in enumerate(seeded_decks(19, 1000, 19, 19)):
            a, b = PAIRS[int(rng.integers(len(PAIRS)))]
            exact = exact_kernel_counts(TABLES, deck, a, b)
            truth = exact / exact.sum()
            classes = sample_round_classes(deck, a, b, 1000 + seed, 0, draws)
            cumulative = np.cumsum(np.eye(6, dtype=np.int64)[classes], axis=0)
            n = np.arange(1, draws + 1)[:, None]
            lo, hi = intervals(cumulative, n, alpha / 5)
            inside = (lo <= truth) & (truth <= hi)
            if not inside[:, list(REACHABLE)].all():
                violations += 1
            final_widths.append(float((hi[-1] - lo[-1])[list(REACHABLE)].mean()))
        self.assertLessEqual(violations, alpha * 1000, violations)
        self.assertLess(np.mean(final_widths), 0.1)


class BoxTest(unittest.TestCase):
    def test_boxes_zero_the_tie_and_tightening_keeps_a_point_in_the_box(self):
        counts = np.array([[[900, 50, 40, 10, 0, 0], [0, 0, 0, 0, 0, 1000]]])
        lo, hi = boxes(counts, 1e-3)
        self.assertEqual(lo[..., TIE].tolist(), [[0.0, 0.0]])
        self.assertEqual(hi[..., TIE].tolist(), [[0.0, 0.0]])
        p = counts / counts.sum(-1, keepdims=True)
        self.assertTrue(np.all(lo <= p + 1e-15))
        self.assertTrue(np.all(hi >= p - 1e-15))
        self.assertTrue(np.all(hi[0, 1, :5] < 0.03))
        wide_lo = np.array([[0.0, 0.0, 0.5]])
        wide_hi = np.array([[0.2, 0.2, 1.0]])
        t_lo, t_hi = tighten(wide_lo, wide_hi)
        np.testing.assert_allclose(t_lo, [[0.0, 0.0, 0.6]])
        np.testing.assert_allclose(t_hi, [[0.2, 0.2, 1.0]])


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequences = Sequences()

    def test_sequence_and_prefix_counts(self):
        self.assertEqual(len(self.sequences), 14_400)
        self.assertEqual([len(index) for index in self.sequences.prefix_index], [25, 400, 3600, 14_400, 14_400])
        self.assertEqual(KernelCounts(self.sequences).entries, 21_904_800)

    def test_sequences_sharing_a_number_prefix_add_into_one_entry_and_hidden_pools(self):
        table = KernelCounts(self.sequences)
        shared = [s for s in range(len(self.sequences))
                  if self.sequences.targets[s, 0].tolist() == [25, 8] and self.sequences.targets[s, 1].tolist() == [26, 15]]
        self.assertEqual(len(shared), 36)
        outputs = {}
        for s in shared[:3]:
            outputs[s], errors = sample_sequence(self.sequences.targets[s], 9, s, 20_000)
            self.assertEqual(errors, 0)
            table.add(s, outputs[s])
        prefix_1 = self.sequences.prefix_of_sequence[1, shared[0]]
        expected_round_2 = sum(c[ROW_OFFSET[1]:ROW_OFFSET[1] + 5] for c in outputs.values())
        np.testing.assert_array_equal(table.revealed[1][prefix_1], expected_round_2)
        self.assertEqual(int(table.revealed[1][prefix_1].sum()), 60_000)
        pooled = table.pooled(2)
        prefix_2 = self.sequences.prefix_of_sequence[2, shared[0]]
        for first in REACHABLE:
            code = class_code((first, None))
            expected = sum(table.revealed[2][prefix_2][DIGIT[first] * 5 + DIGIT[second]] for second in REACHABLE)
            np.testing.assert_array_equal(pooled[prefix_2][code], expected)
        np.testing.assert_array_equal(pooled[prefix_2][class_code((None, None))],
                                      table.revealed[2][prefix_2].sum(axis=0))
        same_three_rounds = sum(1 for s in shared[:3] if self.sequences.prefix_of_sequence[2, s] == prefix_2)
        self.assertEqual(same_three_rounds, 2)  # the third sequence plays (36, 47) in round 3
        self.assertEqual(int(pooled[prefix_2][class_code((None, None))].sum()), 20_000 * same_three_rounds)
        lo, hi = table.boxes(2, 1e-3)
        self.assertEqual(lo.shape, (3600, 36, 6))
        self.assertTrue(np.all(lo <= hi))
        p = pooled[prefix_2] / np.maximum(pooled[prefix_2].sum(-1, keepdims=True), 1)
        self.assertTrue(np.all(lo[prefix_2] <= p + 1e-15))
        self.assertTrue(np.all(hi[prefix_2] >= p - 1e-15))

    def test_class_code_uses_base_six_with_hidden_five(self):
        self.assertEqual(HIDDEN, 5)
        self.assertEqual(class_code(()), 0)
        self.assertEqual(class_code((W0C, None, NOJ)), 1 * 36 + 5 * 6 + 4)


if __name__ == "__main__":
    unittest.main()
