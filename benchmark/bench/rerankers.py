"""Cross-encoder scoring with a resumable on-disk cache, and evaluation through the production seam."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .data import Dataset, best_device, cache_dir, load_json, save_json
from .first_stage import REPO

sys.path.insert(0, str(REPO / "src"))
import oncall_reranker  # noqa: E402  (production seam under test)

# tag -> (checkpoint, max input tokens). `ce-l6` is the production configuration; the other two
# are ablations of model depth and of the input budget.
MODELS = {
    "ce-l6": ("cross-encoder/ms-marco-MiniLM-L-6-v2", oncall_reranker.MAX_LENGTH),    # 22.7M
    "ce-l12": ("cross-encoder/ms-marco-MiniLM-L-12-v2", oncall_reranker.MAX_LENGTH),  # 33.4M
    "ce-l6-512": ("cross-encoder/ms-marco-MiniLM-L-6-v2", 512),
    # Part 3: modern rerankers, same 256-token input budget so only the model differs.
    "bge-base": ("BAAI/bge-reranker-base", oncall_reranker.MAX_LENGTH),               # 278M
    "bge-m3": ("BAAI/bge-reranker-v2-m3", oncall_reranker.MAX_LENGTH),                # 568M
}


def resolve(tag: str) -> tuple[str, int]:
    """A named checkpoint, or a locally fine-tuned one saved by finetune.py (models/ft-*)."""
    if tag in MODELS:
        return MODELS[tag]
    local = Path(__file__).resolve().parents[1] / "models" / tag
    if tag.startswith("ft-") and local.is_dir():
        return str(local), oncall_reranker.MAX_LENGTH
    raise KeyError(f"unknown reranker {tag!r}")


class CrossEncoderScorer:
    def __init__(self, tag: str, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.tag = tag
        checkpoint, max_length = resolve(tag)
        self.model = CrossEncoder(checkpoint, max_length=max_length, device=device or best_device())
        # BENCH_FP16=1 halves inference cost on a GPU. Used only for the SD-QA runs, so every model
        # in one table shares one precision; the Part 1 BEIR tables stay fp32.
        if os.environ.get("BENCH_FP16") == "1" and str(self.model.device).startswith("cuda"):
            self.model.model.half()

    def __call__(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=64,
                                             show_progress_bar=False, convert_to_numpy=True),
                          dtype=np.float32)


def score_pool(ds: Dataset, scorer: CrossEncoderScorer, cond: str, queries: dict[str, str],
               pools: dict[str, list[str]], tag_suffix: str = "") -> dict[str, dict[str, float]]:
    """Score every (query, doc) in each query's pool once; resumable across runs."""
    path = cache_dir(ds.name) / f"ce_{scorer.tag}_{cond}{tag_suffix}.json"
    cache: dict[str, dict[str, float]] = load_json(path, {})
    todo = [qid for qid in queries if any(d not in cache.get(qid, {}) for d in pools.get(qid, []))]
    for n, qid in enumerate(todo, start=1):
        have = cache.setdefault(qid, {})
        need = [d for d in pools[qid] if d not in have]
        for doc_id, score in zip(need, scorer(queries[qid], [ds.doc_text(d) for d in need])):
            have[doc_id] = float(score)
        if n % 50 == 0 or n == len(todo):
            save_json(path, cache)
            print(f"    [{ds.name}/{cond}/{scorer.tag}{tag_suffix}] scored {n}/{len(todo)} queries", flush=True)
    return cache


def seam_rerank(query: str, run: list[tuple[str, float]], scores: dict[str, float], *,
                depth: int, alpha: float = 0.0, tag: str = "model") -> list[str]:
    """Re-rank the top `depth` of a first-stage run through `oncall_reranker.rerank` itself.

    Documents below the depth keep first-stage order beneath the re-ranked head — the standard
    re-ranking protocol, and exactly what a deployment with that pool size would return.
    """
    head = run[:depth]
    if not head:                       # first stage found nothing: there is nothing to re-rank
        return [doc_id for doc_id, _ in run]
    ids = [doc_id for doc_id, _ in head]
    oncall_reranker.set_scorer(lambda q, passages: [scores[d] for d in ids], name=tag)
    try:
        ranked, mode = oncall_reranker.rerank(
            query, [{"doc_id": d, "body": "", "score": s} for d, s in head],
            limit=len(head), alpha=alpha)
    finally:
        oncall_reranker.set_scorer(None)
    if mode != tag:
        raise RuntimeError(f"seam did not re-rank: mode={mode!r}")
    return [hit["doc_id"] for hit in ranked] + [doc_id for doc_id, _ in run[depth:]]
