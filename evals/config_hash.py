from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.contracts import canonical_system_config_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute the canonical L1 system_config hash")
    parser.add_argument("--input", required=True, type=Path, help="JSON object containing no credentials")
    args = parser.parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        parser.error("config JSON must be an object")
    print(canonical_system_config_hash(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
