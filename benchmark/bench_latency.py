"""Per-query latency of each pipeline stage, as a voice turn would experience it.

Batch size = the re-rank depth (one forward pass per turn), warm model, 40 sampled queries per
dataset. Run on a quiet machine: contention inflates every number here.
Usage: python -I bench_latency.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data  # noqa: E402
from bench.first_stage import BM25, Dense  # noqa: E402
from bench.rerankers import MODELS  # noqa: E402

SAMPLE = 40
DEPTHS = (10, 25, 50, 100)


def summarise(ms: list[float]) -> dict[str, float]:
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "mean": float(np.mean(ms))}


def main() -> None:
    from sentence_transformers import CrossEncoder
    out = data.load_json(HERE / "results" / "latency.json", {})
    import platform
    import torch
    out["_hardware"] = {"cpu": f"{platform.processor() or platform.machine()} ({__import__('os').cpu_count()} threads)",
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else data.best_device()}
    for name in data.DATASETS:
        ds = data.load(name)
        qids = random.Random(7).sample(sorted(ds.queries), SAMPLE)
        bm25, dense = BM25(ds), Dense(ds, device="cpu")
        entry = out.setdefault(name, {})
        entry["bm25"] = summarise([_time(lambda q=ds.queries[q]: bm25.search(q, 100)) for q in qids])
        entry["dense_query_cpu"] = summarise([_time(lambda q=q: dense.search_many({q: ds.queries[q]}, 100))
                                              for q in qids])
        heads = {q: [ds.doc_text(d) for d, _ in bm25.search(ds.queries[q], 100)] for q in qids}
        for device in ("cpu", data.best_device()):
            for tag, (checkpoint, max_length) in MODELS.items():
                model = CrossEncoder(checkpoint, max_length=max_length, device=device)
                model.predict([("warm", "up")] * 8, show_progress_bar=False)
                for depth in DEPTHS:
                    ms = []
                    for q in qids:
                        pairs = [(ds.queries[q], t) for t in heads[q][:depth]]
                        if pairs:
                            ms.append(_time(lambda p=pairs: model.predict(p, batch_size=len(p),
                                                                           show_progress_bar=False)))
                    entry.setdefault(device, {}).setdefault(tag, {})[str(depth)] = summarise(ms)
                    print(f"[{name}] {device} {tag} @{depth}: p50 {entry[device][tag][str(depth)]['p50']:.0f} ms",
                          flush=True)
                del model
        data.save_json(HERE / "results" / "latency.json", out)


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    main()
