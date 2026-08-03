"""Command-line entry point: ``omdg``."""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="omdg",
                                 description="Generate data-catalog descriptions to CSV.")
    ap.add_argument("source", choices=["datahub", "databricks", "snowflake"],
                    help="metadata source to pull from")
    ap.add_argument("--keyword", default="", help="fully-qualified-name filter")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", default=None, help="LLM provider (overrides env)")
    ap.add_argument("--model", default=None, help="LLM model id (overrides env)")
    ap.add_argument("--code-path", default=None)
    ap.add_argument("--doc-path", default=None)
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.provider:  cfg.llm_provider = args.provider
    if args.model:     cfg.llm_model = args.model
    if args.code_path: cfg.code_path = args.code_path
    if args.doc_path:  cfg.doc_path = args.doc_path
    if args.out:       cfg.output_csv = args.out

    results = run_pipeline(args.source, cfg, keyword=args.keyword, limit=args.limit)
    tables = sum(1 for r in results if r.object_type == "table")
    cols = sum(1 for r in results if r.object_type == "column")
    print(f"Wrote {len(results)} rows ({tables} tables, {cols} columns) -> {cfg.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
