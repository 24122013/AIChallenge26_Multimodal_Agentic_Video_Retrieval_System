"""CLI for exporting one live KIS, grounded-QA, or TRAKE query to CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from backend.app.services.submission.csv_export import export_query_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an AIC KIS/QA/TRAKE query as ranked CSV.")
    parser.add_argument("--task", required=True, choices=("kis", "qa", "trake"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=100, help="Number of rows (1-100)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path under data/submissions/ (default: data/submissions/<task>_result.csv)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exported = export_query_csv(args.query, args.task, args.top_k)
    root = Path("data/submissions").resolve()
    output = (args.output or (root / exported.filename)).resolve()
    if output != root and root not in output.parents:
        raise SystemExit("--output must stay under data/submissions/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(exported.content, encoding="utf-8", newline="")
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
