# Business Graph v1 split evaluation dataset

"
        "This directory is a lossless projection of the approved source Gold.
"
        "The source Gold and its human approval remain authoritative.

"
        "- `queries.jsonl` is the only SUT input view.
"
        "- `expected_facts.jsonl`, `expected_graph_paths.jsonl`, and
"
        "  `expected_evidence.jsonl` are evaluator-only views.
"
        "- `split_manifest.json` records source hash, case IDs, and the fact that
"
        "  no dev/holdout split was invented.

"
        "Never regenerate expected data from runtime output or mark this derived
"
        "directory approved independently of the source Gold review.
