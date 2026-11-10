import unittest

import pandas as pd

from ccr.metrics import by_corruption, summarize, wilson_interval


class WilsonTests(unittest.TestCase):
    def test_known_value(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

    def test_edges_stay_in_unit_interval(self):
        low, high = wilson_interval(0, 40)
        self.assertEqual(low, 0.0)
        self.assertTrue(0 < high < 0.15)
        low, high = wilson_interval(40, 40)
        self.assertEqual(high, 1.0)
        self.assertTrue(0.85 < low < 1)

    def test_narrows_with_more_samples(self):
        a = wilson_interval(80, 100)
        b = wilson_interval(800, 1000)
        self.assertLess(b[1] - b[0], a[1] - a[0])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


def make_frame():
    rows = []
    # (seed, corruption, severity, accuracy)
    for seed, scale in ((0, 1.0), (1, 0.8)):
        rows += [
            (seed, "clean", 0, 0.9 * scale),
            (seed, "noise", 1, 0.8 * scale), (seed, "noise", 2, 0.6 * scale),
            (seed, "blur", 1, 0.7 * scale), (seed, "blur", 2, 0.5 * scale),
        ]
    return pd.DataFrame(
        [dict(feature="hog", model="logreg", regime="clean", seed=s, corruption=c, severity=v, accuracy=a)
         for s, c, v, a in rows]
    )


class SummaryTests(unittest.TestCase):
    def test_summarize_values(self):
        s = summarize(make_frame()).iloc[0]
        self.assertAlmostEqual(s["clean_acc"], (0.9 + 0.72) / 2)
        self.assertAlmostEqual(s["corrupt_acc"], (0.65 + 0.52) / 2)
        self.assertAlmostEqual(s["acc_drop"], s["clean_acc"] - s["corrupt_acc"])
        self.assertAlmostEqual(s["retention"], 0.65 / 0.9 / 2 + 0.52 / 0.72 / 2)
        self.assertEqual(s["n_seeds"], 2)
        self.assertAlmostEqual(s["corrupt_acc_std"], abs(0.65 - 0.52) / 2 ** 0.5)

    def test_by_corruption_averages_severities_and_seeds(self):
        out = by_corruption(make_frame()).set_index("corruption")["accuracy"]
        self.assertAlmostEqual(out["noise"], (0.7 + 0.56) / 2)
        self.assertAlmostEqual(out["blur"], (0.6 + 0.48) / 2)
        self.assertNotIn("clean", out.index)


if __name__ == "__main__":
    unittest.main()
