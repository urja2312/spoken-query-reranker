"""Part 2 tables: retrieval on SD-QA real speech (11 English dialects, two Google ASR models each).

Reads cache/sdqa/results_*.json and transcripts produced by pipeline_part2.sh (MODE A and B), writes
results/sdqa_results.md. Pure analysis — no model is run.
Usage: python -I analyze_sdqa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech, stats  # noqa: E402

OUT = HERE / "results" / "sdqa_results.md"
DIALECT = {"usa": "US", "gbr": "UK", "aus": "Australia", "irl": "Ireland", "nzl": "New Zealand",
           "zaf": "South Africa", "nga": "Nigeria", "kenya": "Kenya", "phl": "Philippines",
           "ind_n": "India-North", "ind_s": "India-South"}
FIRST = [("BM25", "BM25"), ("Dense (bge-small)", "Dense"), ("Hybrid RRF", "Hybrid"), ("BM25 > ce-l6 @25", "BM25→CE")]
VARIANTS = [("zero", "Zero-shot"), ("clean", "FT-clean"), ("noisy", "FT-noisy"), ("mixed", "FT-mixed"),
            ("distill", "FT-distill")]


def parse(cond: str) -> tuple[str, str]:
    """asr_ind_s_en_in -> ('ind_s', 'en-IN')."""
    body = cond[len("asr_"):]
    dialect, model = body.rsplit("_en_", 1)
    return dialect, f"en-{model.upper()}"


def per_query(rows, key, metric, qids):
    return np.array([rows[key][q][metric] for q in qids], dtype=float)


def variant_scores(rows, v, metric, qids):
    """Zero-shot, or the protocol-B (fixed 3 epochs) model averaged over the seeds present."""
    if v == "zero":
        return per_query(rows, "BM25 > ce-l6 @25", metric, qids) if "BM25 > ce-l6 @25" in rows else None
    runs = [per_query(rows, f"BM25 > ft-{v}-s{s}-last @25", metric, qids)
            for s in (0, 1, 2) if f"BM25 > ft-{v}-s{s}-last @25" in rows]
    return np.mean(runs, axis=0) if runs else None


def main() -> None:
    cache = data.cache_dir("sdqa")
    conds = data.load_json(cache / "conditions.json", [])
    ds = data.load("sdqa")
    res = {c: data.load_json(cache / f"results_{c}.json") for c in ["text"] + conds}
    res = {c: r for c, r in res.items() if r}
    if "text" not in res:
        raise SystemExit("no SD-QA results yet")
    qids = sorted(res["text"]["rows"]["BM25"])
    lines = ["# Part 2 — SD-QA real speech results\n",
             f"{len(qids)} answerable TyDi-QA English questions; corpus of {len(ds.corpus):,} Wikipedia passages "
             "from 929 articles. Speech: real speakers, 11 English dialects; transcripts from Google ASR with a "
             "localised model and the en-US model (released by SD-QA). Re-ranking: BM25 top-25. Fine-tuned "
             "models: protocol B (fixed 3 epochs), per-query scores averaged over seeds 0–2. Scoring in fp16.\n"]

    # ------------------------------------------------------------- table 1: degradation per condition
    lines += ["\n## 1. Degradation by dialect and ASR model (nDCG@10)\n",
              "| Dialect | ASR model | WER | " + " | ".join(n for _, n in FIRST) + " | BM25→CE vs typed |",
              "|---|---|---|" + "---|" * len(FIRST) + "---|"]
    text_rows = res["text"]["rows"]
    base = {k: per_query(text_rows, k, "ndcg@10", qids).mean() for k, _ in FIRST}
    lines.append("| *Typed (gold)* | — | 0.000 | " + " | ".join(f"{base[k]:.4f}" for k, _ in FIRST) + " | — |")
    agg: dict[str, list[float]] = {}
    for c in conds:
        if c not in res:
            continue
        dialect, model = parse(c)
        tr = data.load_json(cache / f"transcripts_{c}.json", {})
        wer = np.mean([speech.wer(ds.queries[q], tr.get(q, "")) for q in qids])
        rows = res[c]["rows"]
        vals = {k: per_query(rows, k, "ndcg@10", qids).mean() for k, _ in FIRST}
        drop = (vals["BM25 > ce-l6 @25"] - base["BM25 > ce-l6 @25"]) / base["BM25 > ce-l6 @25"] * 100
        lines.append(f"| {DIALECT.get(dialect, dialect)} | {model} | {wer:.3f} | "
                     + " | ".join(f"{vals[k]:.4f}" for k, _ in FIRST) + f" | {drop:+.1f}% |")
        agg.setdefault("all", []).append(drop)
        if dialect.startswith("ind"):
            agg.setdefault("indian", []).append(drop)
    if agg:
        lines.append(f"\nMean BM25→CE change vs typed: all conditions {np.mean(agg['all']):+.1f}%, "
                     f"Indian-English conditions {np.mean(agg.get('indian', [np.nan])):+.1f}%.\n")

    # ------------------------------------------------------------- table 2: rerankers on real speech
    groups = {"Typed (gold)": ["text"],
              "All 21 real-speech conditions": [c for c in conds if c in res],
              "Indian English (4 conditions)": [c for c in conds if c in res and parse(c)[0].startswith("ind")],
              "Indian English, Indian ASR model": [c for c in conds if c in res and parse(c)[0].startswith("ind") and parse(c)[1] == "en-IN"],
              "US English": [c for c in conds if c in res and parse(c)[0] == "usa"]}
    lines += ["\n## 2. Fine-tuned rerankers on real speech (nDCG@10; per-query scores averaged over the conditions in each group)\n",
              "Significance: paired randomization test on the group-averaged per-query nDCG@10, Holm-corrected "
              "within each group. † p<0.05 vs zero-shot; ‡ p<0.05 vs FT-clean.\n",
              "Last column: uncorrected p for FT-distill vs zero-shot.\n",
              "| Group | " + " | ".join(n for _, n in VARIANTS) + " | distill vs zero-shot, raw p |",
              "|---|" + "---|" * (len(VARIANTS) + 1)]
    for gname, gconds in groups.items():
        if not gconds:
            continue
        per_var = {}
        for v, _ in VARIANTS:
            mats = [variant_scores(res[c]["rows"], v, "ndcg@10", qids) for c in gconds]
            if all(m is not None for m in mats):
                per_var[v] = np.mean(mats, axis=0)
        tests = {f"{v}|zero": stats.randomization_test(per_var[v], per_var["zero"]) for v in per_var if v != "zero"}
        tests.update({f"{v}|clean": stats.randomization_test(per_var[v], per_var["clean"])
                      for v in ("noisy", "mixed", "distill") if v in per_var and "clean" in per_var})
        adj = stats.holm(tests) if tests else {}
        cells = []
        for v, _ in VARIANTS:
            if v not in per_var:
                cells.append("—"); continue
            mark = ("†" if adj.get(f"{v}|zero", 1) < 0.05 else "") + ("‡" if adj.get(f"{v}|clean", 1) < 0.05 else "")
            cells.append(f"{per_var[v].mean():.4f}{mark}")
        raw = tests.get("distill|zero", np.nan)
        lines.append(f"| {gname} | " + " | ".join(cells) + f" | {raw:.4f} |")

    # ------------------------------------------------------------- table 3: localised vs US ASR
    lines += ["\n## 3. Localised vs en-US recogniser, same speakers (BM25→CE zero-shot nDCG@10)\n",
              "| Dialect | Localised model | Localised | en-US | Δ (localised − US) |", "|---|---|---|---|---|"]
    by_dialect: dict[str, dict[str, str]] = {}
    for c in conds:
        if c in res:
            d, m = parse(c)
            by_dialect.setdefault(d, {})[m] = c
    for d, ms in by_dialect.items():
        loc = [m for m in ms if m != "en-US"]
        if not loc or "en-US" not in ms:
            continue
        a = per_query(res[ms[loc[0]]]["rows"], "BM25 > ce-l6 @25", "ndcg@10", qids)
        b = per_query(res[ms["en-US"]]["rows"], "BM25 > ce-l6 @25", "ndcg@10", qids)
        p = stats.randomization_test(a, b)
        lines.append(f"| {DIALECT.get(d, d)} | {loc[0]} | {a.mean():.4f} | {b.mean():.4f} | {a.mean() - b.mean():+.4f}"
                     f"{' *' if p < 0.05 else ''} |")
    lines.append("\n\\* p < 0.05 (paired randomization test, uncorrected).\n")

    # ------------------------------------------------------------- table 4: which first stage to re-rank
    pipes = [("BM25", "BM25"), ("Dense (bge-small)", "Dense"), ("Hybrid RRF", "Hybrid"),
             ("BM25 > ce-l6 @25", "BM25→CE"), ("Hybrid > ce-l6 @25", "Hybrid→CE"),
             ("Hybrid > ft-distill-s{s}-last @25", "Hybrid→FT-distill")]
    lines += ["\n## 4. Which first stage to re-rank (nDCG@10; Recall@100 of the first stage in brackets)\n",
              "p-values: paired randomization test, uncorrected (exploratory; not pre-registered).\n",
              "| Group | " + " | ".join(n for _, n in pipes) + " | Hybrid→CE vs Dense p | Hybrid→distill vs Hybrid→CE p |",
              "|---|" + "---|" * (len(pipes) + 2)]
    for gname, gconds in groups.items():
        if not gconds:
            continue
        vals = {}
        for key, name in pipes:
            keys = [key.format(s=s) for s in (0, 1, 2)] if "{s}" in key else [key]
            if not all(k in res[c]["rows"] for c in gconds for k in keys):
                continue
            vals[name] = np.mean([np.mean([per_query(res[c]["rows"], k, "ndcg@10", qids) for k in keys], axis=0)
                                  for c in gconds], axis=0)
        recall = {name: np.mean([per_query(res[c]["rows"], key, "recall@100", qids).mean() for c in gconds])
                  for key, name in pipes[:3]}
        cells = [f"{vals[n].mean():.4f}" + (f" ({recall[n]:.3f})" if n in recall else "") if n in vals else "—"
                 for _, n in pipes]
        p1 = stats.randomization_test(vals["Hybrid→CE"], vals["Dense"]) if {"Hybrid→CE", "Dense"} <= set(vals) else np.nan
        p2 = (stats.randomization_test(vals["Hybrid→FT-distill"], vals["Hybrid→CE"])
              if {"Hybrid→FT-distill", "Hybrid→CE"} <= set(vals) else np.nan)
        lines.append(f"| {gname} | " + " | ".join(cells) + f" | {p1:.4f} | {p2:.4f} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
