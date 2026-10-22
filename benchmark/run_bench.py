"""Run the retrieval/reranking grid for one or more datasets and query conditions.

Every stage is cached under cache/<dataset>/, so the grid is resumable and cheap to re-run.
Usage: python -I run_bench.py --datasets scifact nfcorpus fiqa --conds text
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, metrics  # noqa: E402
from bench.first_stage import BM25, DEPTH, Dense, rrf  # noqa: E402
from bench.rerankers import CrossEncoderScorer, score_pool, seam_rerank  # noqa: E402

RERANK_DEPTHS = (10, 25, 50, 100)
HYBRID_DEPTHS = (25, 100)
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
ALPHA_DEPTHS = (25, 100)
# Out-of-domain probes for abstention: queries whose answers cannot be in the target corpus.
OOD_SOURCE = {"scifact": "fiqa", "nfcorpus": "fiqa", "fiqa": "scifact"}
OOD_QUERIES = 300
ABSTAIN_DEPTH = 25


def query_set(ds: data.Dataset, cond: str) -> dict[str, str] | None:
    if cond == "text":
        return dict(ds.queries)
    transcripts = data.load_json(data.cache_dir(ds.name) / f"transcripts_{cond}.json", {})
    if not all(qid in transcripts for qid in ds.queries):
        return None
    return {qid: transcripts[qid] for qid in ds.queries}


def first_stage(ds, cond, queries, bm25, dense):
    path = data.cache_dir(ds.name) / f"runs_{cond}.json"
    runs = data.load_json(path)
    if runs is None:
        t = time.time()
        bm = {qid: bm25.search(q, DEPTH) for qid, q in queries.items()}
        dn = dense.search_many(queries, DEPTH)
        runs = {"bm25": bm, "dense": dn, "hybrid": {qid: rrf(bm[qid], dn[qid]) for qid in queries}}
        data.save_json(path, runs)
        print(f"  first stage ({cond}): {time.time() - t:.1f}s", flush=True)
    return {name: {qid: [tuple(x) for x in run] for qid, run in by_q.items()} for name, by_q in runs.items()}


def evaluate_grid(ds, queries, runs, scores_by_model, *, full_pool=True, max_depth=100):
    rows: dict[str, dict[str, dict]] = {}
    extras: dict[str, dict] = {}

    def add(name, qid, ranked):
        rows.setdefault(name, {})[qid] = metrics.evaluate(ranked, ds.qrels[qid])

    for qid, query in queries.items():
        bm, dn, hy = runs["bm25"][qid], runs["dense"][qid], runs["hybrid"][qid]
        add("BM25", qid, [d for d, _ in bm])
        add("Dense (bge-small)", qid, [d for d, _ in dn])
        add("Hybrid RRF", qid, [d for d, _ in hy])
        for tag, scores in scores_by_model.items():
            sc = scores.get(qid, {})
            for depth in (d for d in RERANK_DEPTHS if d <= max_depth):
                add(f"BM25 > {tag} @{depth}", qid, seam_rerank(query, bm, sc, depth=depth, tag=tag))
            for depth in (d for d in HYBRID_DEPTHS if full_pool and d <= max_depth):
                add(f"Hybrid > {tag} @{depth}", qid, seam_rerank(query, hy, sc, depth=depth, tag=tag))
            for depth in (d for d in ALPHA_DEPTHS if d <= max_depth):
                for alpha in ALPHAS:
                    add(f"BM25 > {tag} @{depth} | alpha={alpha}", qid,
                        seam_rerank(query, bm, sc, depth=depth, alpha=alpha, tag=tag))
            head = bm[:ABSTAIN_DEPTH]
            if head and max_depth >= ABSTAIN_DEPTH:
                ranked = seam_rerank(query, bm, sc, depth=ABSTAIN_DEPTH, tag=tag)
                extras.setdefault(tag, {})[qid] = {
                    "ce_top": max(sc[d] for d, _ in head), "bm25_top": head[0][1],
                    "ce_hit1": float(ranked[0] in ds.qrels[qid]),
                    "bm25_hit1": float(head[0][0] in ds.qrels[qid]),
                    "answerable": float(any(d in ds.qrels[qid] for d, _ in head)),
                }
    return rows, extras


def ood_probe(ds, bm25, scorer, tag):
    """Top first-stage and model scores for queries the corpus cannot answer."""
    source = data.load(OOD_SOURCE[ds.name])
    qids = sorted(source.queries)[:OOD_QUERIES]
    queries = {f"ood-{q}": source.queries[q] for q in qids}
    runs = {qid: bm25.search(q, ABSTAIN_DEPTH) for qid, q in queries.items()}
    pools = {qid: [d for d, _ in run] for qid, run in runs.items()}
    scores = score_pool(ds, scorer, "ood", queries, pools)
    return {qid: {"ce_top": max((scores[qid][d] for d in pools[qid]), default=float("-inf")),
                  "bm25_top": runs[qid][0][1] if runs[qid] else 0.0}
            for qid in queries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(data.DATASETS))
    parser.add_argument("--conds", nargs="+", default=["text"])
    parser.add_argument("--models", nargs="+", default=["ce-l6", "ce-l12"])
    parser.add_argument("--ood", action="store_true", help="also run the abstention OOD probe")
    parser.add_argument("--pool", choices=("full", "bm25", "both"), default="full",
                        help="full = BM25 top-100 + hybrid top-100; bm25 = BM25 top-N; "
                             "both = BM25 top-N + hybrid top-N")
    parser.add_argument("--pool-depth", type=int, default=100)
    args = parser.parse_args()

    scorers = {tag: CrossEncoderScorer(tag) for tag in args.models}
    for name in args.datasets:
        ds = data.load(name)
        print(f"== {name}: {len(ds.corpus)} docs, {len(ds.queries)} test queries", flush=True)
        bm25, dense = BM25(ds), Dense(ds)
        for cond in args.conds:
            queries = query_set(ds, cond)
            if queries is None:
                print(f"  skip {cond}: transcripts not ready", flush=True)
                continue
            runs = first_stage(ds, cond, queries, bm25, dense)
            if args.pool == "full":
                pools = {qid: list(dict.fromkeys([d for d, _ in runs["bm25"][qid]] +
                                                 [d for d, _ in runs["hybrid"][qid]]))
                         for qid in queries}
            elif args.pool == "both":
                pools = {qid: list(dict.fromkeys([d for d, _ in runs["bm25"][qid][:args.pool_depth]] +
                                                 [d for d, _ in runs["hybrid"][qid][:args.pool_depth]]))
                         for qid in queries}
            else:
                pools = {qid: [d for d, _ in runs["bm25"][qid][:args.pool_depth]] for qid in queries}
            scores = {tag: score_pool(ds, s, cond, queries, pools) for tag, s in scorers.items()}
            t = time.time()
            rows, extras = evaluate_grid(ds, queries, runs, scores, full_pool=args.pool != "bm25",
                                         max_depth=args.pool_depth)
            # Merge: each model run adds its configurations without discarding earlier ones.
            path = data.cache_dir(name) / f"results_{cond}.json"
            merged = data.load_json(path, {"rows": {}, "abstain": {}})
            merged["rows"].update(rows)
            merged["abstain"].update(extras)
            data.save_json(path, merged)
            print(f"  evaluated {len(rows)} configs on {cond} in {time.time() - t:.1f}s; "
                  f"BM25 nDCG@10={metrics.mean(list(rows['BM25'].values()), 'ndcg@10'):.4f}", flush=True)
        if args.ood:
            for tag, scorer in scorers.items():
                data.save_json(data.cache_dir(name) / f"ood_{tag}.json", ood_probe(ds, bm25, scorer, tag))
                print(f"  OOD probe done for {tag}", flush=True)


if __name__ == "__main__":
    main()
