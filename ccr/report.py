from pathlib import Path

from ccr.metrics import by_corruption, summarize
from ccr.plots import regime_bars, severity_curves


def make_report(df, out_dir):
    out = Path(out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    by_corruption(df).to_csv(out / "by_corruption.csv", index=False)

    for model in sorted(df["model"].unique()):
        regime_bars(df, model, figures / f"regimes_{model}.png")
        for regime in df["regime"].unique():
            severity_curves(df, model, regime, figures / f"curves_{model}_{regime}.png")
    return summary
