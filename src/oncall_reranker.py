"""Optional neural re-ranking over the BM25 candidate pool.

WHY THIS EXISTS
---------------
`oncall_kb` ranks passages with BM25, which matches **words**. A caller who says "the printing
is light and streaky" shares no words with a manual section headed "Faded or Missing Print", so
BM25 scores it near zero and the agent honestly reports a miss — while the answer sits in the
corpus. The stopword lists in `oncall_kb` are a patch on the same wound: hand-listing the words
that mislead a lexical matcher does not generalise to the next paraphrase.

A re-ranker is the second opinion. BM25 stays as the fast recall stage — it scans the whole
corpus and shortlists `CANDIDATE_POOL` passages. A model then scores each (query, passage) pair
on whether the passage actually *answers* the question, and the pool is re-ordered before the
top `limit` are returned.

THE RULES THIS MODULE FOLLOWS
-----------------------------
1. **Off by default.** Without `ONCALL_RERANKER` set, nothing here runs and retrieval behaves
   exactly as it did before — same DB query, same row count, same order.
2. **No new dependency on the brain.** This module is stdlib-only until a backend is activated;
   the heavy import happens lazily, inside the backend loader.
3. **Degrades honestly, like everything else here.** A missing model, a failed import, or a
   scorer that raises means BM25 order is returned unchanged and the reported mode says so.
   A re-ranker that cannot run is never allowed to turn a good lexical hit into a miss.
4. **BM25 scores are preserved.** Re-ranking adds `rerank_score` and `bm25_rank` rather than
   overwriting `score`, so rank movement is measurable — which is the experiment.

TRAINING A MODEL FOR THIS SEAM
------------------------------
The scorer contract is deliberately tiny:

    scorer(query: str, passages: list[str]) -> list[float]     # higher is more relevant

Anything satisfying it plugs in via `set_scorer`, so a training notebook can evaluate a
candidate model against the live retrieval path without touching `oncall_kb`.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Callable, Sequence

# How many BM25 hits to re-rank. Large enough that the right passage is almost always somewhere
# in the pool (BM25 is good at recall, bad at precision), small enough that a cross-encoder
# scoring the pool stays inside a voice turn's latency budget.
CANDIDATE_POOL = 25
# Cross-encoder input budget in tokens. 512 is the model maximum, but on a CPU the cost grows
# with sequence length and the answer to "does this passage answer that question" is almost
# always decided by its opening; 256 halves the per-pair cost on a voice turn.
MAX_LENGTH = 256

Scorer = Callable[[str, list[str]], Sequence[float]]

_scorer: Scorer | None = None
_backend: str = ""
_load_error: str = ""
_loaded = False
# The demo server is a ThreadingHTTPServer: two callers arriving together must not both load
# a model into memory. The lock guards resolution only, never scoring.
_lock = threading.Lock()


def _env_float(name: str) -> float | None:
    raw = str(os.environ.get(name, "")).strip()
    try:
        value = float(raw) if raw else None
    except ValueError:
        return None
    return value if value is None or math.isfinite(value) else None


def pool_size() -> int:
    """`ONCALL_RERANKER_POOL`, else `CANDIDATE_POOL`. Re-read per call so it is tunable live."""
    raw = str(os.environ.get("ONCALL_RERANKER_POOL", "")).strip()
    return max(1, int(raw)) if raw.isdigit() else CANDIDATE_POOL


# --------------------------------------------------------------------------- configuration


def _configured_backend() -> str:
    """Which backend the environment asks for. Empty means the seam stays closed."""
    name = str(os.environ.get("ONCALL_RERANKER", "")).strip().lower()
    return "" if name in {"", "0", "off", "false", "no"} else name


def set_scorer(scorer: Scorer | None, *, name: str = "injected") -> None:
    """Install a scoring function directly, bypassing the environment.

    This is the seam a training script and the tests use: point it at a freshly trained model,
    run the real retrieval path, measure. Passing None closes the seam again.
    """
    global _scorer, _backend, _loaded, _load_error
    _scorer, _backend, _loaded, _load_error = scorer, (name if scorer else ""), True, ""


def _load() -> None:
    """Resolve the configured backend once. A failure disables re-ranking, never raises."""
    if _loaded:
        return
    with _lock:
        if not _loaded:
            _resolve()


def _resolve() -> None:
    global _scorer, _backend, _loaded, _load_error
    _loaded = True
    backend = _configured_backend()
    if not backend:
        return
    if backend == "cross_encoder":
        try:
            # Imported lazily and only when asked for: the brain must stay stdlib-only.
            from sentence_transformers import CrossEncoder            # type: ignore

            model_name = os.environ.get(
                "ONCALL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            raw_length = str(os.environ.get("ONCALL_RERANKER_MAXLEN", "")).strip()
            max_length = int(raw_length) if raw_length.isdigit() else MAX_LENGTH
            model = CrossEncoder(model_name, max_length=max_length)
            _scorer = lambda query, passages: model.predict(          # noqa: E731
                [(query, passage) for passage in passages])
            _backend = f"cross_encoder:{model_name}"
        except Exception as exc:                                      # noqa: BLE001
            _load_error = f"{type(exc).__name__}: {exc}"
    else:
        _load_error = f"unknown backend {backend!r}"


def active() -> bool:
    """True when a scorer is loaded and re-ranking will actually happen."""
    _load()
    return _scorer is not None


def candidate_pool(limit: int) -> int:
    """How many rows retrieval should fetch. Unchanged when the seam is closed."""
    return max(int(limit), pool_size()) if active() else int(limit)


def status() -> dict[str, Any]:
    """Report configuration for the desk, without pretending an absent model is present."""
    _load()
    return {"active": _scorer is not None, "backend": _backend,
            "requested": _configured_backend(), "candidate_pool": pool_size(),
            "alpha": _env_float("ONCALL_RERANKER_ALPHA") or 0.0,
            "min_score": _env_float("ONCALL_RERANKER_MIN_SCORE"), "error": _load_error}


def warmup() -> dict[str, Any]:
    """Load the model and run one throwaway pair at startup, not inside a caller's first turn.

    The first forward pass pays for weight loading and kernel compilation; on a live call that
    cost lands as dead air on whichever caller happens to be first.
    """
    if active():
        try:
            _scorer("warm up", ["warm up"])                          # type: ignore[misc]
        except Exception as exc:                                      # noqa: BLE001
            return {**status(), "warmup_error": f"{type(exc).__name__}: {exc}"}
    return status()


def reset() -> None:
    """Forget the resolved backend so the next call re-reads the environment (tests use this)."""
    global _scorer, _backend, _loaded, _load_error
    _scorer, _backend, _loaded, _load_error = None, "", False, ""


# --------------------------------------------------------------------------- the seam


def _minmax(values: Sequence[float]) -> list[float]:
    low, high = min(values), max(values)
    return [0.0] * len(values) if high == low else [(v - low) / (high - low) for v in values]


def rerank(query: str, candidates: list[dict], *, limit: int, alpha: float | None = None,
           min_score: float | None = None) -> tuple[list[dict], str]:
    """Re-order first-stage candidates by learned relevance. Returns (results, mode).

    `mode` is the honest record of what happened, and is reported to the caller:
      * `bm25`              — the seam is closed; this is plain first-stage order.
      * `<backend>`         — the pool was scored and re-ordered by that model.
      * `<backend>:abstained` — the best passage scored below `min_score`; nothing is returned,
                              so the agent says "not in the manuals" instead of reading a
                              confidently-ranked wrong passage.
      * `bm25_fallback:…`   — a model was configured but failed; first-stage order was kept.

    `alpha` blends the two stages after per-query min-max normalisation:
    `alpha * first_stage + (1 - alpha) * model`. 0 trusts the model alone, 1 reproduces the
    first stage. `min_score` is in the model's own output units (logits for the ms-marco
    cross-encoders). Both default to the environment (`ONCALL_RERANKER_ALPHA`,
    `ONCALL_RERANKER_MIN_SCORE`), and to "model only, never abstain" when unset.

    A scorer that raises, returns the wrong number of scores, or returns something non-numeric
    is treated as absent. Retrieval is on the critical path of a live call: a broken re-ranker
    must cost accuracy, never the answer. The caller's dicts are never mutated.
    """
    ordered = [{**hit, "bm25_rank": position} for position, hit in enumerate(candidates, start=1)]

    if not ordered or not active():
        return ordered[:limit], "bm25"

    try:
        scores = _scorer(str(query), [str(hit.get("body", "")) for hit in ordered])  # type: ignore[misc]
        values = [float(score) for score in scores]
        if len(values) != len(ordered):
            raise ValueError(
                f"scorer returned {len(values)} scores for {len(ordered)} passages")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("scorer returned a non-finite score")
    except Exception as exc:                                          # noqa: BLE001
        return ordered[:limit], f"bm25_fallback:{type(exc).__name__}"

    weight = _env_float("ONCALL_RERANKER_ALPHA") if alpha is None else float(alpha)
    weight = min(1.0, max(0.0, weight or 0.0))
    if weight > 0.0:
        first = _minmax([float(hit.get("score", 0.0) or 0.0) for hit in ordered])
        model = _minmax(values)
        final = [weight * f + (1.0 - weight) * m for f, m in zip(first, model)]
    else:
        final = values

    for hit, raw, fused in zip(ordered, values, final):
        hit["rerank_score"] = round(raw, 4)          # display only — never the sort key
        if weight > 0.0:
            hit["fused_score"] = round(fused, 4)
    # Sort on the raw values. Stable, so first-stage order breaks exact ties, and an untrained
    # model that scores everything equally degrades to today's behaviour, not to shuffling.
    order = sorted(range(len(ordered)), key=lambda index: -final[index])

    threshold = _env_float("ONCALL_RERANKER_MIN_SCORE") if min_score is None else float(min_score)
    if threshold is not None and values[order[0]] < threshold:
        return [], f"{_backend}:abstained"
    return [ordered[index] for index in order[:limit]], _backend
