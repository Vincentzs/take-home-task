"""Command-line entry point. Subcommands cover the full pipeline and stage debugging."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from pipeline.config import load_settings
from pipeline.runner import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="P&ID-to-graph + SOP cross-reference pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to config.toml; defaults from pipeline.config.Settings apply otherwise",
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/p&id/diagram.pdf"),
        help="path to P&ID PDF",
    )
    p_run.add_argument(
        "--sop",
        type=Path,
        default=Path("data/sop/sop.docx"),
        help="path to SOP DOCX",
    )
    p_run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (overrides Settings.output_dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(config_path=args.config)
        if args.command == "run":
            if args.output is not None:
                settings = replace(settings, output_dir=args.output)
            if not args.pdf.exists():
                print(f"PDF not found: {args.pdf}", file=sys.stderr)
                return 3
            if not args.sop.exists():
                print(f"SOP not found: {args.sop}", file=sys.stderr)
                return 3
            run_pipeline(args.pdf, args.sop, settings)
            return 0
        return 2
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"unexpected error: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
