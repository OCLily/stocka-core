"""Command line: `stocka demo [...]` and `stocka --version`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stocka import __version__
from stocka.config import ConfigError, load_config
from stocka.demo import run_demo
from stocka.providers import LocalFilesProvider, SyntheticProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stocka",
        description="Data-quality scores and risk measures for equity research. Not investment advice.",
    )
    parser.add_argument("--version", action="version", version=f"stocka-core {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run the engine on synthetic or local data and print a Markdown report")
    demo.add_argument("--seed", type=int, help="seed for the synthetic data (default from config: 1)")
    demo.add_argument(
        "--tickers", type=int,
        help="number of synthetic tickers, 1 to 99 (default from config: 20; ignored with --data-dir)",
    )
    demo.add_argument("--config", type=Path, help="TOML config file, see config/example.toml")
    demo.add_argument("--data-dir", type=Path, help="read prices.csv and fundamentals/*.json instead of synthetic data")
    demo.add_argument("--out", type=Path, help="write the report to this file instead of stdout")
    return parser


def _demo(args: argparse.Namespace) -> int:
    config = load_config(args.config).with_overrides(seed=args.seed, tickers=args.tickers)
    if args.data_dir is not None:
        provider = LocalFilesProvider(args.data_dir)
    else:
        provider = SyntheticProvider(config.demo.seed, config.demo.tickers)
    report = run_demo(provider, config)
    if args.out is None:
        sys.stdout.write(report)
    else:
        args.out.write_text(report, encoding="utf-8")
        print(f"report written to {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _demo(args)
    except (ConfigError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"stocka: error: {exc}", file=sys.stderr)
        return 2
