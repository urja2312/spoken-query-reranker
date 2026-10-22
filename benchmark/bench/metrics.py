"""Ranking metrics, per query, matching trec_eval / BEIR conventions (linear graded gain)."""

from __future__ import annotations

import math

CUTOFFS = (1, 3, 10, 100)


def evaluate(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    """All metrics for one query. `ranked` is doc ids best-first; `rel` maps doc id -> grade>0."""
    out: dict[str, float] = {}
    dcg = sum(rel.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(ideal))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    rr = 0.0
    for i, doc in enumerate(ranked[:10]):
        if doc in rel:
            rr = 1.0 / (i + 1)
            break
    out["mrr@10"] = rr
    # Success@1: did the single passage a voice agent would read aloud answer the question?
    out["hit@1"] = 1.0 if ranked and ranked[0] in rel else 0.0
    for k in CUTOFFS:
        out[f"recall@{k}"] = len(set(ranked[:k]) & set(rel)) / len(rel)
    return out


def mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0
