"""Part 3 tables for the pre-registered tests C1, C2, C3, C6 and the descriptive analyses.

Reads cache/<dataset>/results_*.json and transcripts merged from the Part 3 Kaggle jobs.
Writes results/part3_results.md. Pure analysis — no model is run.
Usage: python -I analyze_part3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech, stats  # noqa: E402

OUT = HERE / "results" / "part3_results.md"
SEEDS = (0, 1, 2)
DIALECT = {"usa": "US", "gbr": "UK", "aus": "Australia", "irl": "Ireland", "nzl": "New Zealand",
           "zaf": "South Africa", "nga": "Nigeria", "kenya": "Kenya", "phl": "Philippines",
           "ind_n": "India-North", "ind_s": "India-South"}
_res: dict = {}
# 43 SD-QA questions have no Kenyan recording in the SD-QA repository (their Whisper transcripts are
# empty in both models). They are dropped from every SD-QA condition here, so per-query averages over
# dialects compare the same questions. The rule does not depend on any model output.
MISSING = set(data.load_json(data.cache_dir("sdqa") / "missing_audio.json", []))


def res(ds: str, cond: str) -> dict | None:
    if (ds, cond) not in _res:
        r = data.load_json(data.cache_dir(ds) / f"results_{cond}.json")
        _res[(ds, cond)] = r["rows"] if r else None
    return _res[(ds, cond)]


def whisper_conds(tag: str = "wb") -> list[str]:
    return [c for c in data.load_json(data.cache_dir("sdqa") / "conditions_whisper.json", []) if c.startswith(f"asr_{tag}_")]


# Test groups: name -> list of (dataset, condition); per-query scores are averaged over conditions.
def test_groups() -> dict[str, list[tuple[str, str]]]:
    wb = whisper_conds("wb")
    return {"SD-QA-Whisper (11 dialects)": [("sdqa", c) for c in wb],
            "SD-QA-Whisper, Indian": [("sdqa", c) for c in wb if c.split("_", 2)[2].startswith("ind")],
            "HeySQuAD-dev": [("heysquad", "asr_wb")],
            "SLUE-SQA-5 test": [("slue", "asr_wb")]}


def scores(pairs, key, metric="ndcg@10"):
    """Per-query vector averaged over the (dataset, condition) pairs, or None if any is missing.
    `key` may contain {s}: then it is averaged over the seeds too (fine-tuned models)."""
    keys = [key.format(s=s) for s in SEEDS] if "{s}" in key else [key]
    mats = []
    for ds, cond in pairs:
        rows = res(ds, cond)
        if rows is None or any(k not in rows for k in keys):
            return None
        qids = sorted(q for q in rows[keys[0]] if not (ds == "sdqa" and q in MISSING))
        mats.append(np.mean([[rows[k][q][metric] for q in qids] for k in keys], axis=0))
    if not mats or len({m.shape for m in mats}) != 1:
        return None
    return np.mean(mats, axis=0)


def typed(pairs, key, metric="ndcg@10"):
    return scores(sorted({(ds, "text") for ds, _ in pairs}), key, metric)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    L = ["# Part 3 results\n", "Pre-registered in paper/PREREGISTRATION_PART3.md. nDCG@10; re-ranking depth 25; "
         "fine-tuned models: protocol B, per-query scores averaged over seeds 0–2; Part 3 scoring in fp16.\n"]
    groups = test_groups()

    # ---------------------------------------------------------------- C1
    L += ["\n## C1. First stage: Hybrid → ce-l6 vs BM25 → ce-l6 on new real-speech test sets (confirmatory)\n",
          "| Test set | Queries | BM25 | Dense | BM25 → ce-l6 | Hybrid → ce-l6 | Δ | p | p (Holm, 3 tests) |", "|---|---|---|---|---|---|---|---|---|"]
    c1 = {}
    for g in ("SD-QA-Whisper (11 dialects)", "HeySQuAD-dev", "SLUE-SQA-5 test"):
        a, b = scores(groups[g], "Hybrid > ce-l6 @25"), scores(groups[g], "BM25 > ce-l6 @25")
        if a is not None and b is not None:
            c1[g] = (a, b, stats.randomization_test(a, b))
    holm = stats.holm({g: v[2] for g, v in c1.items()}) if c1 else {}
    for g, (a, b, p) in c1.items():
        bm, dn = scores(groups[g], "BM25"), scores(groups[g], "Dense (bge-small)")
        L.append(f"| {g} | {len(a)} | {bm.mean():.4f} | {dn.mean():.4f} | {b.mean():.4f} | {a.mean():.4f} | "
                 f"{a.mean() - b.mean():+.4f} | {fmt_p(p)} | {fmt_p(holm[g])} |")

    # ---------------------------------------------------------------- zero-shot modern rerankers
    L += ["\n## Zero-shot rerankers on typed and spoken queries (descriptive)\n",
          "Relative change typed → spoken. A reranker that protects against ASR errors would lose a smaller share than BM25.\n",
          "| Test set | System | Typed | Spoken | Change |", "|---|---|---|---|---|"]
    beir = {f"BEIR {c.split('_')[1]} (macro)": [(d, c) for d in data.DATASETS] for c in ("asr_clean", "asr_phone")}
    for g, pairs in {**{k: groups[k] for k in ("SD-QA-Whisper (11 dialects)", "HeySQuAD-dev", "SLUE-SQA-5 test")}, **beir}.items():
        for sysname, key in (("BM25", "BM25"), ("Hybrid", "Hybrid RRF"), ("BM25 → ce-l6", "BM25 > ce-l6 @25"),
                             ("BM25 → bge-base", "BM25 > bge-base @25"), ("BM25 → bge-m3", "BM25 > bge-m3 @25"),
                             ("Hybrid → bge-m3", "Hybrid > bge-m3 @25")):
            if g.startswith("BEIR"):           # macro over datasets: mean of per-dataset means
                t = [scores([(d, "text")], key) for d, _ in pairs]
                s = [scores([p], key) for p in pairs]
                if any(x is None for x in t + s):
                    continue
                tm, sm = np.mean([x.mean() for x in t]), np.mean([x.mean() for x in s])
            else:
                tv, sv = typed(pairs, key), scores(pairs, key)
                if tv is None or sv is None:
                    continue
                tm, sm = tv.mean(), sv.mean()
            L.append(f"| {g} | {sysname} | {tm:.4f} | {sm:.4f} | {(sm - tm) / tm * 100:+.1f}% |")

    # ---------------------------------------------------------------- C2
    L += ["\n## C2. Fine-tuning a modern reranker (bge-reranker-base) on BEIR (confirmatory)\n",
          "Macro over SciFact, NFCorpus, FiQA. † p < 0.05 vs zero-shot bge-base (Holm over the 3 variants within the cell).\n"]
    variants = [("zero", "bge-base"), ("clean", "ft-bgeb-clean-s{s}-last"), ("noisy", "ft-bgeb-noisy-s{s}-last"),
                ("distill", "ft-bgeb-distill-s{s}-last")]
    cells, worse = {}, {}
    for d in data.DATASETS:
        for c in ("text", "asr_clean", "asr_phone"):
            vec = {v: scores([(d, c)], f"BM25 > {tag} @25") for v, tag in variants}
            if any(x is None for x in vec.values()):
                continue
            tests = {v: stats.randomization_test(vec[v], vec["zero"]) for v in vec if v != "zero"}
            adj = stats.holm(tests)
            cells[(d, c)] = {v: (vec[v].mean(), adj.get(v, 1.0)) for v in vec}
            for v in tests:
                if adj[v] < 0.05 and vec[v].mean() < vec["zero"].mean():
                    worse.setdefault(v, []).append(f"{d}/{c}")
    if cells:
        L += ["| Variant | Typed | Wideband | Telephone | Typed → telephone | 95% CI of drop |", "|---|---|---|---|---|---|"]
        rng = np.random.default_rng(0)
        for v, tag in variants:
            m = {c: np.mean([cells[(d, c)][v][0] for d in data.DATASETS if (d, c) in cells]) for c in ("text", "asr_clean", "asr_phone")}
            # bootstrap CI of the macro relative drop: resample queries within each dataset
            per = {d: (scores([(d, "text")], f"BM25 > {tag} @25"), scores([(d, "asr_phone")], f"BM25 > {tag} @25")) for d in data.DATASETS}
            boots = []
            for _ in range(2000):
                tt, pp = [], []
                for d, (a, b) in per.items():
                    i = rng.integers(0, len(a), len(a)); tt.append(a[i].mean()); pp.append(b[i].mean())
                boots.append((np.mean(pp) - np.mean(tt)) / np.mean(tt) * 100)
            L.append(f"| {v} | {m['text']:.4f} | {m['asr_clean']:.4f} | {m['asr_phone']:.4f} | "
                     f"{(m['asr_phone'] - m['text']) / m['text'] * 100:+.1f}% | [{np.percentile(boots, 2.5):+.1f}%, {np.percentile(boots, 97.5):+.1f}%] |")
        # paired bootstrap: (drop of variant) - (drop of zero-shot), same resampled queries
        def macro_drop(tag, idx):
            tt, pp = [], []
            for d in data.DATASETS:
                a, b = scores([(d, "text")], f"BM25 > {tag} @25"), scores([(d, "asr_phone")], f"BM25 > {tag} @25")
                tt.append(a[idx[d]].mean()); pp.append(b[idx[d]].mean())
            return (np.mean(pp) - np.mean(tt)) / np.mean(tt) * 100
        n_q = {d: len(scores([(d, "text")], "BM25 > bge-base @25")) for d in data.DATASETS}
        full = {d: np.arange(n) for d, n in n_q.items()}
        L += ["\nDifference in typed → telephone drop vs zero-shot (percentage points; paired bootstrap over queries, 2,000 resamples):\n",
              "| Variant | Δ drop | 95% CI |", "|---|---|---|"]
        for v, tag in variants[1:]:
            diffs = []
            for _ in range(2000):
                idx = {d: rng.integers(0, n, n) for d, n in n_q.items()}
                diffs.append(macro_drop(tag, idx) - macro_drop("bge-base", idx))
            L.append(f"| {v} | {macro_drop(tag, full) - macro_drop('bge-base', full):+.1f} | "
                     f"[{np.percentile(diffs, 2.5):+.1f}, {np.percentile(diffs, 97.5):+.1f}] |")
        L += ["\nPer cell (nDCG@10):\n", "| Dataset / condition | zero-shot | FT-clean | FT-noisy | FT-distill |", "|---|---|---|---|---|"]
        for (d, c), cv in cells.items():
            L.append(f"| {d} / {c} | {cv['zero'][0]:.4f} | " + " | ".join(
                f"{cv[v][0]:.4f}{'†' if cv[v][1] < 0.05 else ''}" for v in ("clean", "noisy", "distill")) + " |")
        L.append(f"\nVariants significantly worse than zero-shot: { {v: cs for v, cs in worse.items()} or 'none' }\n")

    # ---------------------------------------------------------------- C3
    L += ["\n## C3. Real vs synthetic training noise (HeySQuAD-trained ce-l6; confirmatory)\n",
          "| Test set | zero-shot | FT-clean | FT-noisy-real | FT-noisy-synth | FT-distill-real | FT-distill-synth |", "|---|---|---|---|---|---|---|"]
    h4 = {"zero": "ce-l6", "clean": "ft-hq-clean-s{s}-last", "real": "ft-hq-real-s{s}-last", "synth": "ft-hq-synth-s{s}-last",
          "dreal": "ft-hq-dreal-s{s}-last", "dsynth": "ft-hq-dsynth-s{s}-last"}
    c3_tests = {}
    beir_phone = [(d, "asr_phone") for d in data.DATASETS]
    for g, pairs in {**groups, "BEIR telephone (pooled)": beir_phone}.items():
        vec = {v: (np.concatenate([scores([p], f"BM25 > {t} @25") for p in pairs]) if g.startswith("BEIR")
                   and all(scores([p], f"BM25 > {t} @25") is not None for p in pairs) else
                   (None if g.startswith("BEIR") else scores(pairs, f"BM25 > {t} @25"))) for v, t in h4.items()}
        if all(x is None for x in vec.values()):
            continue
        L.append(f"| {g} | " + " | ".join(f"{vec[v].mean():.4f}" if vec[v] is not None else "—" for v in h4) + " |")
        if g in ("SD-QA-Whisper (11 dialects)", "SD-QA-Whisper, Indian", "SLUE-SQA-5 test"):
            for a, b in (("real", "synth"), ("dreal", "dsynth")):
                if vec[a] is not None and vec[b] is not None:
                    c3_tests[f"{g}: {a} vs {b}"] = (vec[a].mean() - vec[b].mean(), stats.randomization_test(vec[a], vec[b]))
    if c3_tests:
        adj = stats.holm({k: v[1] for k, v in c3_tests.items()})
        L += ["\n| Comparison | Δ | p | p (Holm, 6 tests) |", "|---|---|---|---|"]
        for k, (dlt, p) in c3_tests.items():
            L.append(f"| {k} | {dlt:+.4f} | {fmt_p(p)} | {fmt_p(adj[k])} |")
    tr = data.load_json(data.CACHE / "trainset_heysquad.json")
    if tr:
        ex = tr["examples"]
        L.append(f"\nTraining transcripts: mean WER real {np.mean([e['wer_real'] for e in ex]):.3f}, "
                 f"synthetic {np.mean([e['wer_synth'] for e in ex]):.3f} ({len(ex)} questions).\n")

    # ---------------------------------------------------------------- C6
    sel = data.load_json(HERE / "results" / "training" / "distill_selection.json")
    L += ["\n## C6. Stronger consistency distillation (confirmatory)\n"]
    if sel:
        L += [f"Selected on validation: **{sel['selected']}**.\n", "| Candidate | " + " | ".join(next(iter(sel["candidates"].values())).keys()) + " |",
              "|---|" + "---|" * len(next(iter(sel["candidates"].values())))]
        for n, r in sel["candidates"].items():
            L.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in r.values()) + " |")
    c6 = {}
    for g, pairs in (("BEIR spoken (pooled, 3 datasets × 2 conditions)", [(d, c) for d in data.DATASETS for c in ("asr_clean", "asr_phone")]),
                     ("SD-QA-Whisper (11 dialects)", groups["SD-QA-Whisper (11 dialects)"]),
                     ("SLUE-SQA-5 test", groups["SLUE-SQA-5 test"])):
        if g.startswith("BEIR"):
            parts = [(scores([p], "BM25 > ft-dbest-s{s}-last @25"), scores([p], "BM25 > ce-l6 @25")) for p in pairs]
            if any(a is None or b is None for a, b in parts):
                continue
            a, b = np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts])
        else:
            a, b = scores(pairs, "BM25 > ft-dbest-s{s}-last @25"), scores(pairs, "BM25 > ce-l6 @25")
            if a is None or b is None:
                continue
        c6[g] = (a.mean(), b.mean(), stats.randomization_test(a, b))
    if c6:
        adj = stats.holm({g: v[2] for g, v in c6.items()})
        L += ["\n| Test set | zero-shot ce-l6 | FT-distill (selected) | Δ | p | p (Holm, 3 tests) |", "|---|---|---|---|---|---|"]
        for g, (a, b, p) in c6.items():
            L.append(f"| {g} | {b:.4f} | {a:.4f} | {a - b:+.4f} | {fmt_p(p)} | {fmt_p(adj[g])} |")

    # ---------------------------------------------------------------- Whisper base vs turbo
    ds = data.load("sdqa")
    L += ["\n## ASR strength: Whisper base.en vs large-v3-turbo on SD-QA (descriptive)\n",
          "| Dialect | WER base | WER turbo | BM25 → ce-l6 base | BM25 → ce-l6 turbo |", "|---|---|---|---|---|"]
    for dkey, dname in DIALECT.items():
        row = []
        for tag in ("wb", "wt"):
            tr = data.load_json(data.cache_dir("sdqa") / f"transcripts_asr_{tag}_{dkey}.json")
            row.append(np.mean([speech.wer(ds.queries[q], tr.get(q, "")) for q in ds.queries if q not in MISSING]) if tr else np.nan)
        nd = [scores([("sdqa", f"asr_{tag}_{dkey}")], "BM25 > ce-l6 @25") for tag in ("wb", "wt")]
        L.append(f"| {dname} | {row[0]:.3f} | {row[1]:.3f} | " + " | ".join(f"{x.mean():.4f}" if x is not None else "—" for x in nd) + " |")
    tv = typed([("sdqa", "x")], "BM25 > ce-l6 @25")
    if tv is not None:
        L.append(f"\nTyped questions: {tv.mean():.4f}.\n")

    OUT.write_text("\n".join(L) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
