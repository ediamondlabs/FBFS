"""CPU-only checks of the paper's algebra, moments, and payload formulas.

The gate polynomial and noise calibration are extracted from the implementation
without importing its GPU dependencies. These are not model-training tests.
"""

import ast
from fractions import Fraction as F
import itertools
import math
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / "federated/common.py").read_text())


def extract(name, prefix=False):
    node = next(n for n in SOURCE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if prefix:
        stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "in_sum" for t in n.targets))
        node.body = node.body[:stop] + [ast.Return(value=ast.Name(id="sigma", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"math": math, "torch": ScalarTorch}
    exec(compile(module, "federated/common.py", "exec"), scope)
    return scope[name]


class ScalarTorch:
    zeros_like = staticmethod(lambda a: 0)
    ones_like = staticmethod(lambda a: 1)
    stack = staticmethod(lambda values, dim: values)


class Weights(list):
    def __getitem__(self, key):
        return super().__getitem__(key[-1] if isinstance(key, tuple) else key)


def relevance(rows, labels, cls):
    positive = [row for row, y in zip(rows, labels) if y == cls]
    negative = [row for row, y in zip(rows, labels) if y != cls]
    if not positive or not negative:
        raise ValueError("Pooled class and complement coverage required")
    return [F(sum(column), len(positive)) - F(sum(other), len(negative))
            for column, other in zip(zip(*positive), zip(*negative))]


def correlation(rows, g, h):
    n = len(rows)
    sg, sh = sum(r[g] for r in rows), sum(r[h] for r in rows)
    q = sum(r[g] * r[h] for r in rows)
    denominator = math.sqrt(sg * (n - sg) * sh * (n - sh))
    return (n * q - sg * sh) / denominator if denominator else 0


class GateAlgebraTests(unittest.TestCase):
    def test_all_truth_tables(self):
        gates = extract("bin_op_all")
        weighted = extract("bin_op_weighted")
        for a, b in itertools.product((0, 1), repeat=2):
            for operation in range(16):
                expected = (operation >> (3 - 2 * a - b)) & 1
                self.assertEqual(gates(a, b)[operation], expected)
                one_hot = Weights(int(i == operation) for i in range(16))
                self.assertEqual(weighted(a, b, one_hot), expected)

    def test_multilinear_identity_with_exact_rationals(self):
        gates, weighted = extract("bin_op_all"), extract("bin_op_weighted")
        rng = random.Random(42)
        for _ in range(200):
            a, b = (F(rng.randrange(101), 100) for _ in range(2))
            counts = [rng.randrange(10) for _ in range(16)]
            weights = Weights(F(count, sum(counts)) for count in counts)
            expected = sum(w * value for w, value in zip(weights, gates(a, b)))
            self.assertEqual(weighted(a, b, weights), expected)

    def test_light_truth_entries_define_all_boolean_operations(self):
        for operation in range(16):
            t00, t01, t10, t11 = [(operation >> i) & 1 for i in (3, 2, 1, 0)]
            for a, b in itertools.product((0, 1), repeat=2):
                value = t00*(1-a)*(1-b) + t01*(1-a)*b + t10*a*(1-b) + t11*a*b
                self.assertEqual(value, (operation >> (3 - 2*a - b)) & 1)


class MomentTests(unittest.TestCase):
    def setUp(self):
        self.rows = [[0, 1, a, 1-a, b, a] for a, b in itertools.product((0, 1), repeat=2)] * 3
        self.labels = [i % 3 for i in range(len(self.rows))]

    def test_partitioned_scores_equal_pooled_scores(self):
        # Class-specific shards include clients with no examples of a class.
        shards = [[i for i, y in enumerate(self.labels) if y == c] for c in range(3)]
        for cls in range(3):
            sums = [[0] * 6 for _ in range(3)]
            counts = [0] * 3
            for indices in shards:
                for i in indices:
                    y = self.labels[i]
                    counts[y] += 1
                    sums[y] = [v + x for v, x in zip(sums[y], self.rows[i])]
            reconstructed = [F(sums[cls][g], counts[cls])
                             - F(sum(sums[c][g] for c in range(3) if c != cls),
                                 len(self.rows) - counts[cls]) for g in range(6)]
            self.assertEqual(reconstructed, relevance(self.rows, self.labels, cls))
        # Coincident functions are still two source-indexed candidates.
        self.assertEqual(len(reconstructed), 6)
        self.assertEqual(reconstructed[2], reconstructed[5])

    def test_missing_class_or_complement_is_undefined(self):
        for cls in (0, 1):
            with self.assertRaises(ValueError):
                relevance(self.rows, [0] * len(self.rows), cls)

    def test_pair_moments_reconstruct_pearson_and_joint_counts(self):
        n = len(self.rows)
        shards = [self.rows[:1], self.rows[1:5], self.rows[5:]]
        for g, h in itertools.product(range(6), repeat=2):
            sg = sum(sum(r[g] for r in shard) for shard in shards)
            sh = sum(sum(r[h] for r in shard) for shard in shards)
            q = sum(sum(r[g]*r[h] for r in shard) for shard in shards)
            joints = (q, sg-q, sh-q, n-sg-sh+q)
            direct = tuple(sum((r[g], r[h]) == pair for r in self.rows)
                           for pair in ((1, 1), (1, 0), (0, 1), (0, 0)))
            self.assertEqual(joints, direct)
            centered = [(F(r[g])-F(sg,n), F(r[h])-F(sh,n)) for r in self.rows]
            denominator = math.sqrt(float(sum(a*a for a, b in centered)
                                          * sum(b*b for a, b in centered)))
            pearson = float(sum(a*b for a, b in centered))/denominator if denominator else 0
            self.assertAlmostEqual(correlation(self.rows, g, h), pearson)
        self.assertEqual(correlation(self.rows, 2, 3), -1)
        self.assertEqual(correlation(self.rows, 2, 4), 0)
        self.assertEqual(correlation(self.rows, 0, 1), 0)


class BoundsAndBudgetTests(unittest.TestCase):
    def test_convex_merge_preserves_common_strict_argmax(self):
        rng = random.Random(42)
        for _ in range(100):
            weights = [rng.random() for _ in range(5)]
            weights = [w/sum(weights) for w in weights]
            logits = [[rng.random() for _ in range(16)] for _ in range(5)]
            for row in logits:
                row[6] = 2
            merged = [sum(w*row[i] for w, row in zip(weights, logits)) for i in range(16)]
            self.assertEqual(max(range(16), key=merged.__getitem__), 6)

    def test_suppression_sufficient_condition(self):
        owner = [-2, 4, 1]
        nonowners = [[5, 2, 0], [6, 3, 1], [4, 1, 1]]
        m, upper_owner_margin = 3, 6
        weight = F(1, 4)
        self.assertGreater((1-weight)*m, weight*upper_owner_margin)
        merged = [weight*owner[i] + (1-weight)*sum(row[i] for row in nonowners)/3
                  for i in range(3)]
        self.assertEqual(max(range(3), key=merged.__getitem__), 0)

    def test_communication_and_round_parity(self):
        for p, g, h, basic, diverse, rounds in (
            (5347328, 51200, 83552, 27.451248168945312, 42.71766662597656, (3, 5)),
            (39739392, 1638400, 2483712, 251.9922637939453, 267.25868225097656, (4, 4)),
        ):
            b = 5*p + 4*10*g + 8*10
            d = b + 4*2000**2 + 4*2000
            self.assertAlmostEqual(b/2**20, basic)
            self.assertAlmostEqual(d/2**20, diverse)
            self.assertEqual((math.ceil(b/(2*p)), math.ceil(d/(2*p))), rounds)
            self.assertEqual(8*5*h, 40*h)


class PrivacyCalibrationTests(unittest.TestCase):
    def test_actual_sigma_prefix_satisfies_analytic_gaussian_bound(self):
        calibrate = extract("bfs_federated_dp", prefix=True)
        delta = 1e-5
        for epsilon in (.5, 1, 4, 8):
            sigma = calibrate([object()], None, None, None, epsilon=epsilon, delta=delta)
            mu = 1/sigma
            cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
            achieved = cdf(mu/2-epsilon/mu) - math.exp(epsilon)*cdf(-mu/2-epsilon/mu)
            self.assertLessEqual(achieved, delta*(1+1e-10))

    def test_joint_contribution_clipping(self):
        for active in (0, 1, 100, 7500):
            norm = math.sqrt(2*active+2)
            clipped_norm = norm*min(1, 1/norm)
            self.assertLessEqual(clipped_norm, 1+1e-15)


if __name__ == "__main__":
    unittest.main()
