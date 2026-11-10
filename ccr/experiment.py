import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ccr.corruptions import augment_batch, corrupt_batch
from ccr.data import load_dataset, stratified_subset
from ccr.features import extract
from ccr.metrics import wilson_interval
from ccr.models import make_model

log = logging.getLogger(__name__)

# stream ids, so subsets, model init, and corruption noise never share a generator
_TRAIN_SUBSET, _TEST_SUBSET, _MODEL_INIT = 0, 1, 2
_TRAIN_ENTROPY, _TEST_ENTROPY = 1, 2


def _derive(seed, stream):
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def run(cfg, out_dir, download=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    x_train_all, y_train_all, x_test_all, y_test_all = load_dataset(cfg.dataset, cfg.data_root, download)
    log.info("loaded %s: %d train, %d test", cfg.dataset, len(x_train_all), len(x_test_all))

    conditions = [("clean", 0)] + [(c, s) for c in cfg.corruptions for s in cfg.severities]
    rows = []
    for seed in cfg.seeds:
        tr = stratified_subset(y_train_all, cfg.n_train, _derive(seed, _TRAIN_SUBSET))
        te = stratified_subset(y_test_all, cfg.n_test, _derive(seed, _TEST_SUBSET))
        x_train, y_train = x_train_all[tr], y_train_all[tr]
        x_test, y_test = x_test_all[te], y_test_all[te]

        fitted = {}
        for regime_name, regime in cfg.regimes.items():
            if regime.corruptions:
                x_fit = augment_batch(x_train, regime.corruptions, regime.severities, regime.p,
                                      (seed, _TRAIN_ENTROPY))
            else:
                x_fit = x_train
            feats = extract(x_fit, cfg.features)
            for spec in cfg.features:
                for model_name in cfg.models:
                    est = make_model(model_name, _derive(seed, _MODEL_INIT))
                    est.fit(feats[spec], y_train)
                    fitted[(spec, model_name, regime_name)] = est
            log.info("seed %d: fitted regime %s", seed, regime_name)

        for corruption, severity in conditions:
            if corruption == "clean":
                x_eval = x_test
            else:
                x_eval = corrupt_batch(x_test, corruption, severity, (seed, _TEST_ENTROPY))
            feats = extract(x_eval, cfg.features)
            for (spec, model_name, regime_name), est in fitted.items():
                correct = int((est.predict(feats[spec]) == y_test).sum())
                low, high = wilson_interval(correct, len(y_test))
                rows.append({
                    "dataset": cfg.dataset, "seed": seed, "feature": spec, "model": model_name,
                    "regime": regime_name, "corruption": corruption, "severity": severity,
                    "n_test": len(y_test), "accuracy": correct / len(y_test),
                    "ci_low": low, "ci_high": high,
                })
        log.info("seed %d: evaluated %d conditions", seed, len(conditions))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    return df
