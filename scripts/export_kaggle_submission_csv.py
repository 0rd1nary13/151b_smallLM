#!/usr/bin/env python3
"""Convert JSONL model outputs into a Kaggle-ready CSV submission.

The CSE 151B Spring 2026 Kaggle competition expects a CSV or Parquet file with
**943 data rows plus a header** (see the competition upload dialog). This script
**always writes exactly 943 data rows**: if the JSONL has more than 943 lines,
only the **first 943** are converted (see warning below). If it has fewer than
943, the script exits with an error.

Download ``sample_submission.csv`` from the Data tab and pass ``--id-column`` /
``--response-column`` if the template uses different names than ``id`` and
``response``.

Each input line must be a JSON object containing at least the id and response
fields (e.g. outputs from the starter notebook with ``SAVE_EVAL=False``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Private leaderboard row count (Kaggle upload dialog).
KAGGLE_SUBMISSION_ROWS: int = 943


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from ``path``."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def main() -> None:
    """Parse CLI arguments and write the submission CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL file with one object per line (must include id + response).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        metavar="NAME",
        help="CSV column name for the question id (default: id).",
    )
    parser.add_argument(
        "--response-column",
        default="response",
        metavar="NAME",
        help="CSV column name for the model response (default: response).",
    )
    args = parser.parse_args()

    records = _read_jsonl_rows(args.input)
    n = len(records)
    if n < KAGGLE_SUBMISSION_ROWS:
        raise SystemExit(
            f"Too few rows: JSONL has {n}, need at least {KAGGLE_SUBMISSION_ROWS} for a Kaggle submission."
        )
    if n > KAGGLE_SUBMISSION_ROWS:
        print(
            f"WARNING: JSONL has {n} rows; only the first {KAGGLE_SUBMISSION_ROWS} are written. "
            "For a valid submission, the input should be predictions on the **private** test set "
            "(943 questions), not a truncated public run.",
            file=sys.stderr,
        )
        records = records[:KAGGLE_SUBMISSION_ROWS]

    for i, row in enumerate(records):
        if "id" not in row:
            raise SystemExit(f"Record {i} missing 'id' field: keys={list(row)!r}")
        if "response" not in row:
            raise SystemExit(f"Record {i} missing 'response' field: keys={list(row)!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[args.id_column, args.response_column],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    args.id_column: row["id"],
                    args.response_column: row["response"],
                }
            )

    print(f"Wrote {len(records)} data rows + header to {args.output.resolve()}")


if __name__ == "__main__":
    main()
