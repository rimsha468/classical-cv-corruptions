import argparse
import logging
from pathlib import Path

import pandas as pd

from ccr.config import load_config
from ccr.experiment import run
from ccr.report import make_report


def _print_summary(summary):
    print(summary.round(4).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ccr", description="Corruption robustness of classical CV features")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment and write results and figures")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--no-download", action="store_true", help="fail instead of downloading the dataset")

    p_rep = sub.add_parser("report", help="rebuild tables and figures from a results.csv")
    p_rep.add_argument("results")
    p_rep.add_argument("--out", help="output directory, defaults to the folder of results.csv")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "run":
        cfg = load_config(args.config)
        df = run(cfg, args.out, download=not args.no_download)
        _print_summary(make_report(df, args.out))
    else:
        results = Path(args.results)
        df = pd.read_csv(results)
        _print_summary(make_report(df, args.out or results.parent))
    return 0
