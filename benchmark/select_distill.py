"""Pick the stronger-distillation configuration on validation only (Part 3, C6).

Every candidate's fixed-budget (protocol B) checkpoint is scored on one shared validation set:
the 120 BEIR validation queries (typed + Whisper transcript) and the 200 HeySQuAD validation
questions (typed + real-speech transcript), BM25 top-25, mean nDCG@10 over the four parts.
No test data is touched. Prints the winner's name; writes results/training/distill_selection.json.
Usage: python -I select_distill.py ft-d1-s0 ft-d2t1-s0 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, metrics  # noqa: E402
from bench.first_stage import BM25  # noqa: E402
from bench.rerankers import CrossEncoderScorer  # noqa: E402

DEPTH = 25


def main() -> None:
    names = sys.argv[1:]
    sets = {"beir": (data.CACHE / "trainset.json", "noisy"), "heysquad": (data.CACHE / "trainset_heysquad.json", "real")}
    corpora, bm25, val = {}, {}, {}
    for key, (path, noisy) in sets.items():
        exs = [e for e in data.load_json(path)["examples"] if e["role"] == "val"]
        for e in exs:
            if e["dataset"] not in corpora:
                corpora[e["dataset"]] = data.load(e["dataset"])
                bm25[e["dataset"]] = BM25(corpora[e["dataset"]])
        for form, field in (("clean", "clean"), ("noisy", noisy)):
            val[f"{key}/{form}"] = [(e["dataset"], e[field], [d for d, _ in bm25[e["dataset"]].search(e[field], DEPTH)],
                                     e["positives"]) for e in exs]
    report = {}
    for name in names:
        scorer = CrossEncoderScorer(f"{name}-last")
        parts = {}
        for part, runs in val.items():
            scores = []
            for ds_name, query, run, rel in runs:
                if not run:
                    scores.append(0.0); continue
                s = scorer(query, [corpora[ds_name].doc_text(d) for d in run])
                scores.append(metrics.evaluate([run[i] for i in np.argsort(-s)], rel)["ndcg@10"])
            parts[part] = float(np.mean(scores))
        report[name] = {**parts, "mean": float(np.mean(list(parts.values())))}
        print(name, {k: round(v, 4) for k, v in report[name].items()}, file=sys.stderr, flush=True)
        del scorer
    best = max(report, key=lambda n: report[n]["mean"])
    data.save_json(HERE / "results" / "training" / "distill_selection.json", {"candidates": report, "selected": best})
    print(best)


if __name__ == "__main__":
    main()
