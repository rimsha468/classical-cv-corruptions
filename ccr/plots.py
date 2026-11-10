import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ccr.metrics import summarize  # noqa: E402


def severity_curves(df, model, regime, path):
    sub = df[(df["model"] == model) & (df["regime"] == regime)]
    stats = sub.groupby(["feature", "corruption", "severity"])["accuracy"].agg(["mean", "std"]).reset_index()
    clean = stats[stats["corruption"] == "clean"].set_index("feature")
    corruptions = sorted(set(stats["corruption"]) - {"clean"})
    features = sorted(set(stats["feature"]))

    cols = min(3, len(corruptions))
    rows = math.ceil(len(corruptions) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharey=True, squeeze=False)
    for ax, corruption in zip(axes.ravel(), corruptions):
        for feat in features:
            part = stats[(stats["feature"] == feat) & (stats["corruption"] == corruption)].sort_values("severity")
            x = np.r_[0, part["severity"].to_numpy()]
            y = np.r_[clean.loc[feat, "mean"], part["mean"].to_numpy()]
            ax.plot(x, y, marker="o", ms=3, label=feat)
            spread = np.r_[0.0, part["std"].fillna(0).to_numpy()]
            if spread.any():
                ax.fill_between(x, y - spread, y + spread, alpha=0.15)
        ax.set_title(corruption)
        ax.set_xlabel("severity (0 = clean)")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(corruptions):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(f"{model}, trained with regime '{regime}'")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def regime_bars(df, model, path):
    s = summarize(df[df["model"] == model])
    features = sorted(s["feature"].unique())
    regimes = list(dict.fromkeys(s["regime"]))
    width = 0.8 / len(regimes)
    fig, ax = plt.subplots(figsize=(1.6 * len(features) + 3, 4))
    for j, regime in enumerate(regimes):
        part = s[s["regime"] == regime].set_index("feature").reindex(features)
        x = np.arange(len(features)) + j * width
        ax.bar(x, part["corrupt_acc"], width, yerr=part["corrupt_acc_std"].fillna(0), label=regime, capsize=2)
    ax.set_xticks(np.arange(len(features)) + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_ylabel("mean accuracy over all corruptions")
    ax.set_title(f"{model}: effect of training regime")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
