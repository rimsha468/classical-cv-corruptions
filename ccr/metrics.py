from math import sqrt

import pandas as pd


def wilson_interval(correct, total, z=1.96):
    if total <= 0:
        raise ValueError("total must be positive")
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(df):
    """One row per (feature, model, regime), averaged over seeds.

    corrupt_acc is the mean accuracy over every corruption and severity in the
    run, retention is corrupt_acc / clean_acc.
    """
    keys = ["feature", "model", "regime", "seed"]
    clean = df[df["corruption"] == "clean"].groupby(keys)["accuracy"].mean().rename("clean_acc")
    corrupt = df[df["corruption"] != "clean"].groupby(keys)["accuracy"].mean().rename("corrupt_acc")
    per_seed = pd.concat([clean, corrupt], axis=1).reset_index()
    per_seed["acc_drop"] = per_seed["clean_acc"] - per_seed["corrupt_acc"]
    per_seed["retention"] = per_seed["corrupt_acc"] / per_seed["clean_acc"]
    out = per_seed.groupby(["feature", "model", "regime"], as_index=False).agg(
        clean_acc=("clean_acc", "mean"),
        corrupt_acc=("corrupt_acc", "mean"),
        corrupt_acc_std=("corrupt_acc", "std"),
        acc_drop=("acc_drop", "mean"),
        retention=("retention", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return out


def by_corruption(df):
    """Mean accuracy over severities, per (feature, model, regime, corruption)."""
    d = df[df["corruption"] != "clean"]
    per_seed = d.groupby(["feature", "model", "regime", "corruption", "seed"], as_index=False)["accuracy"].mean()
    return per_seed.groupby(["feature", "model", "regime", "corruption"], as_index=False)["accuracy"].mean()
