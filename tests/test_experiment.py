import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from ccr.cli import main
from ccr.config import parse_config
from ccr import experiment
from ccr.experiment import run

RAW = {
    "dataset": "digits",
    "n_train": 200,
    "n_test": 100,
    "seeds": [0],
    "features": ["color_hist", "hog"],
    "models": ["logreg"],
    "corruptions": ["gaussian_noise", "occlusion"],
    "severities": [1, 3],
    "regimes": {"clean": {}, "aug": {"corruptions": "all", "severities": [1, 2], "p": 0.5}},
}


class ExperimentTests(unittest.TestCase):
    def test_row_count_and_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = run(parse_config(RAW), tmp)
            # seeds * features * models * regimes * (clean + corruptions * severities)
            self.assertEqual(len(df), 1 * 2 * 1 * 2 * (1 + 2 * 2))
            self.assertTrue(df["accuracy"].between(0, 1).all())
            self.assertTrue((df["ci_low"] <= df["accuracy"]).all())
            self.assertTrue((df["accuracy"] <= df["ci_high"]).all())
            self.assertTrue((df["n_test"] == 100).all())
            clean = df[(df["corruption"] == "clean") & (df["feature"] == "hog")]
            self.assertTrue((clean["accuracy"] > 0.5).all())
            self.assertTrue((Path(tmp) / "results.csv").exists())
            self.assertTrue((Path(tmp) / "config.yaml").exists())

    def test_clean_rows_do_not_depend_on_condition_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = run(parse_config(RAW), tmp)
            clean = df[df["corruption"] == "clean"]
            self.assertTrue((clean["severity"] == 0).all())
            self.assertEqual(len(clean), 4)

    def test_same_config_same_results(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            pd.testing.assert_frame_equal(run(parse_config(RAW), a), run(parse_config(RAW), b))

    def test_seeds_change_the_subsets(self):
        raw = dict(RAW, seeds=[0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            df = run(parse_config(raw), tmp)
        a = df[df["seed"] == 0]["accuracy"].to_numpy()
        b = df[df["seed"] == 1]["accuracy"].to_numpy()
        self.assertFalse((a == b).all())

    def test_noise_hurts_hog_more_at_higher_severity(self):
        raw = dict(RAW, features=["hog"], corruptions=["gaussian_noise"], severities=[1, 5],
                   n_train=400, n_test=300, regimes={"clean": {}})
        with tempfile.TemporaryDirectory() as tmp:
            df = run(parse_config(raw), tmp)
        acc = df.set_index("severity")["accuracy"]
        self.assertGreater(acc[0], acc[5])
        self.assertGreaterEqual(acc[1], acc[5])


class SeedStreamTests(unittest.TestCase):
    def test_streams_are_distinct(self):
        ids = [experiment._TRAIN_SUBSET, experiment._TEST_SUBSET, experiment._MODEL_INIT]
        self.assertEqual(len(set(ids)), 3)
        self.assertNotEqual(experiment._TRAIN_ENTROPY, experiment._TEST_ENTROPY)
        derived = {experiment._derive(0, i) for i in ids}
        self.assertEqual(len(derived), 3)


class CliTests(unittest.TestCase):
    def test_run_then_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(RAW))
            out = Path(tmp) / "out"
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(main(["run", "--config", str(cfg_path), "--out", str(out)]), 0)
            self.assertIn("retention", buf.getvalue())
            for name in ("results.csv", "summary.csv", "by_corruption.csv", "config.yaml"):
                self.assertTrue((out / name).exists(), name)
            for name in ("regimes_logreg.png", "curves_logreg_clean.png", "curves_logreg_aug.png"):
                self.assertGreater((out / "figures" / name).stat().st_size, 1000, name)

            other = Path(tmp) / "again"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["report", str(out / "results.csv"), "--out", str(other)])
            pd.testing.assert_frame_equal(pd.read_csv(out / "summary.csv"), pd.read_csv(other / "summary.csv"))


if __name__ == "__main__":
    unittest.main()
