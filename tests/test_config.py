import unittest

from ccr.config import parse_config


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = parse_config({"dataset": "digits"})
        self.assertEqual(cfg.severities, [1, 2, 3, 4, 5])
        self.assertEqual(len(cfg.corruptions), 6)
        self.assertEqual(list(cfg.regimes), ["clean"])
        self.assertEqual(cfg.regimes["clean"].corruptions, [])

    def test_regime_exclusion(self):
        cfg = parse_config({
            "dataset": "digits",
            "regimes": {"aug": {"corruptions": "all", "exclude": ["jpeg"], "severities": [3, 1], "p": 1}},
        })
        r = cfg.regimes["aug"]
        self.assertNotIn("jpeg", r.corruptions)
        self.assertEqual(len(r.corruptions), 5)
        self.assertEqual(r.severities, [1, 3])
        self.assertEqual(r.p, 1.0)

    def test_bad_configs(self):
        bad = [
            {},
            {"dataset": "imagenet"},
            {"dataset": "digits", "typo_key": 1},
            {"dataset": "digits", "n_train": 0},
            {"dataset": "digits", "n_test": 2.5},
            {"dataset": "digits", "seeds": []},
            {"dataset": "digits", "seeds": [0, 0]},
            {"dataset": "digits", "seeds": [-1]},
            {"dataset": "digits", "features": ["hog", "sift"]},
            {"dataset": "digits", "features": ["hog", "hog"]},
            {"dataset": "digits", "models": ["resnet"]},
            {"dataset": "digits", "corruptions": ["fog"]},
            {"dataset": "digits", "corruptions": "everything"},
            {"dataset": "digits", "severities": [0, 1]},
            {"dataset": "digits", "severities": []},
            {"dataset": "digits", "regimes": {"a": {"p": 0}}},
            {"dataset": "digits", "regimes": {"a": {"p": 1.5}}},
            {"dataset": "digits", "regimes": {"a": {"bogus": 1}}},
            {"dataset": "digits", "regimes": {"a": {"corruptions": ["jpeg"], "exclude": ["jpeg"]}}},
            {"dataset": "digits", "regimes": {"a": {"severities": [7]}}},
            {"dataset": "digits", "regimes": {}},
        ]
        for raw in bad:
            with self.assertRaises(ValueError, msg=str(raw)):
                parse_config(raw)


if __name__ == "__main__":
    unittest.main()
