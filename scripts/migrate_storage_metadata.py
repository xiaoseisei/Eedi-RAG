from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.storage_migration import apply_metadata_migration, plan_metadata_migration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan/apply Chroma metadata migration from DuckDB")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--diff", type=Path, help="write the dry-run diff JSON to this path")
    parser.add_argument("--apply", action="store_true", help="apply to an isolated target copy")
    parser.add_argument("--target-chroma", type=Path, help="new directory required with --apply")
    args = parser.parse_args(argv)

    diff = plan_metadata_migration(args.db, args.chroma)
    if args.diff:
        args.diff.parent.mkdir(parents=True, exist_ok=True)
        args.diff.write_text(json.dumps(diff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diff, ensure_ascii=False))
    if not args.apply:
        return 0
    if args.target_chroma is None:
        parser.error("--target-chroma is required with --apply")
    result = apply_metadata_migration(args.db, args.chroma, args.target_chroma)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
