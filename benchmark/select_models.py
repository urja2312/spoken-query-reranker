"""Which fine-tuned checkpoints need evaluating, and which are aliases of another.

Protocol A (early stopping) and protocol B (fixed 3 epochs) can resolve to identical weights:
best_epoch == 0 means the early-stopped model IS the base re-ranker; best_epoch == epochs means
the early-stopped and final models are the same file. Evaluating an alias would only re-score
identical weights, so each is mapped to its source and the analysis copies the rows.
Usage: select_models.py [seed ...]. Prints the unique model tags for run_bench.py; writes results/training/aliases.json.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
logs = HERE / "results" / "training"
seeds = sys.argv[1:] or ["0"]
models = ["ce-l6"]
aliases = json.loads((logs / "aliases.json").read_text()) if (logs / "aliases.json").exists() else {}
variants = os.environ.get("VARIANTS", "clean noisy mixed").split()
for seed in seeds:
  for variant in variants:
    log = json.loads((logs / f"ft-{variant}-s{seed}.json").read_text())
    best, last = f"ft-{variant}-s{seed}", f"ft-{variant}-s{seed}-last"
    if log["best_epoch"] == 0:
        aliases[best] = "ce-l6"
    else:
        models.append(best)
    if log["best_epoch"] == int(log["args"]["epochs"]):
        aliases[last] = aliases.get(best, best)
    else:
        models.append(last)
(logs / "aliases.json").write_text(json.dumps(aliases, indent=2))
print(" ".join(models))
print(f"aliases: {aliases}", file=sys.stderr)
