"""Merge downloaded Kaggle run outputs (out/ folders) into this checkout, then re-run analyze.py.

Each run contributes its per-query result rows (union of configurations), training logs and
checkpoint aliases; results, latency and figures come from whichever run produced them.
Usage: python -I merge_kaggle.py <out_dir> [<out_dir> ...]
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for out in map(Path, sys.argv[1:]):
    for path in sorted((out / "cache").glob("*/*.json")):
        dest = HERE / "cache" / path.parent.name / path.name
        incoming = json.loads(path.read_text())
        if path.name.startswith("results_") and dest.exists():
            merged = json.loads(dest.read_text())
            merged["rows"].update(incoming.get("rows", {}))
            merged["abstain"].update(incoming.get("abstain", {}))
            incoming = merged
        dest.write_text(json.dumps(incoming, ensure_ascii=False))
    training = out / "results" / "training"
    if training.is_dir():
        (HERE / "results" / "training").mkdir(parents=True, exist_ok=True)
        aliases_path = HERE / "results" / "training" / "aliases.json"
        aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
        for log in training.glob("*.json"):
            if log.name == "aliases.json":
                aliases.update(json.loads(log.read_text()))
            else:
                shutil.copy(log, HERE / "results" / "training" / log.name)
        aliases_path.write_text(json.dumps(aliases, indent=2))
    if (out / "results" / "latency.json").exists():
        shutil.copy(out / "results" / "latency.json", HERE / "results" / "latency.json")
    (HERE / "logs").mkdir(exist_ok=True)
    for log in (out / "logs").glob("*.log") if (out / "logs").is_dir() else []:
        shutil.copy(log, HERE / "logs" / f"kaggle-{out.parent.name}-{log.name}")
    print(f"merged {out}")
