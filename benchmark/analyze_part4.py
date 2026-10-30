"""Part 4 analysis: pre-registered H1–H4 (paper/PREREGISTRATION_PART4.md) and secondary analyses.

Reads cache/<dataset>/p4res*_<system>.json, transcripts and prompts written by part4.py.
Writes results/part4_results.md and results/part4_results.json.
Usage: python -I analyze_part4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech, stats  # noqa: E402
from part4 import POOL, is_rare, units  # noqa: E402

DATASETS = {"sdqa": "SD-QA (11 dialects)", "heysquad": "HeySQuAD", "slue": "SLUE-SQA-5"}
KEY = f"Hybrid > ce-l6 @{POOL}"
BIG_KEY = f"Hybrid > bge-m3 @{POOL}"
M = "ndcg@10"
TESTS = (("H1", "lt", "base", "Listen Twice > pass 1"), ("H2", "lt", "rnd", "Listen Twice > random-passage prompt"),
         ("H3", "mc", "base", "Mishear the Corpus > plain index"), ("H4", "mc", "ph", "Mishear the Corpus > phonetic index"))
SYSTEMS = (("typed", "Typed question (reference)"), ("typed_mc", "Typed, MC index"), ("typed_ph", "Typed, phonetic index"),
           ("base", "Spoken, pass 1 (baseline)"), ("ph", "Phonetic index (classic baseline)"),
           ("mc", "**Mishear the Corpus (MC)**"), ("rnd", "Pass 2, random-passage prompt (control)"),
           ("lt", "**Listen Twice (LT)**"), ("ltfuse", "LT-fuse (pass 1 + pass 2 runs)"), ("ltmc", "**LT + MC**"),
           ("orc", "Pass 2, relevant-passage prompt (oracle bound)"))
TRANSCRIPT_OF = {"base": None, "lt": "p4lt", "rnd": "p4rnd", "orc": "p4orc", "ltmc": "p4ltmc"}


def missing(name: str) -> set[str]:
    return set(data.load_json(data.cache_dir("sdqa") / "missing_audio.json", [])) if name == "sdqa" else set()


def result(name: str, suffix: str, system: str):
    return data.load_json(data.cache_dir(name) / f"p4res{suffix}_{system}.json")


def per_question(name: str, system: str, key: str = KEY, metric: str = M) -> dict[str, float] | None:
    """Score per question; on SD-QA the mean over the 11 dialects. Typed systems have one unit."""
    drop = missing(name)
    sfx = [""] if system.startswith("typed") else [s for s, _ in units(name)]
    per: dict[str, list[float]] = {}
    for s in sfx:
        r = result(name, s, system)
        if r is None or key not in r["rows"]:
            return None
        for q, m in r["rows"][key].items():
            if q not in drop:
                per.setdefault(q, []).append(m[metric])
    return {q: float(np.mean(v)) for q, v in per.items()}


def paired(a: dict, b: dict):
    qs = sorted(set(a) & set(b))
    return np.array([a[q] for q in qs]), np.array([b[q] for q in qs]), qs


def transcript_stats(name: str) -> dict[str, dict]:
    ds = data.load(name)
    drop = missing(name)
    out = {}
    for system, prefix in TRANSCRIPT_OF.items():
        wers, rare_err, inj_any, inj_n = [], [], [], []
        for suffix, p1 in units(name):
            cache = data.cache_dir(name)
            t = data.load_json(cache / f"transcripts_{p1 if prefix is None else prefix + suffix}.json", {})
            t1 = data.load_json(cache / f"transcripts_{p1}.json", {})
            prompts = data.load_json(cache / f"p4_prompts{suffix}.json", {}).get(system, {})
            for q, ref in ds.queries.items():
                if q in drop or q not in t:
                    continue
                hyp = speech._words(t[q])
                wers.append(speech.wer(ref, t[q]))
                rare = [w for w in speech._words(ref) if is_rare(w)]
                if rare:
                    rare_err.append(np.mean([w not in hyp for w in rare]))
                if prefix is not None:
                    pw = {w for p in prompts.get(q, "").split(",") for w in speech._words(p)}
                    injected = [w for w in hyp if w in pw and w not in speech._words(ref) and w not in speech._words(t1.get(q, ""))]
                    inj_any.append(float(bool(injected))); inj_n.append(len(injected))
        out[system] = {"wer": float(np.mean(wers)), "rare_word_error": float(np.mean(rare_err)) if rare_err else None,
                       "prompt_injection_share": float(np.mean(inj_any)) if inj_any else None,
                       "injected_words_per_query": float(np.mean(inj_n)) if inj_n else None}
    return out


def main() -> None:
    report: dict = {"tests": {}, "means": {}, "secondary": {}}
    pvals = {}
    for name in DATASETS:
        for tid, a, b, _ in TESTS:
            sa, sb = per_question(name, a), per_question(name, b)
            if sa is None or sb is None:
                continue
            x, y, qs = paired(sa, sb)
            d = x - y
            report["tests"][f"{tid}/{name}"] = {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(d.mean()),
                                               "ci": stats.bootstrap_ci(d), "p": stats.randomization_test(x, y), "n": len(qs)}
            pvals[f"{tid}/{name}"] = report["tests"][f"{tid}/{name}"]["p"]
    for k, p in stats.holm(pvals).items():
        t = report["tests"][k]
        t["p_holm"] = p
        t["supported"] = bool(p < 0.05 and t["diff"] > 0)

    lines = ["# Part 4 results: Listen Twice and Mishear the Corpus\n",
             f"Primary metric: nDCG@10 of {KEY} (pre-registered in paper/PREREGISTRATION_PART4.md). "
             "SD-QA scores are per-question means over 11 dialects (43 Kenyan questions without audio dropped).\n",
             "## Pre-registered tests (paired randomization, Holm over all tests)\n",
             "| Test | Dataset | A | B | Δ (A − B) [95% CI] | p | p (Holm) | Supported |", "|---|---|---|---|---|---|---|---|"]
    for tid, a, b, label in TESTS:
        for name, nice in DATASETS.items():
            t = report["tests"].get(f"{tid}/{name}")
            if t:
                lines.append(f"| {tid}: {label} | {nice} | {t['mean_a']:.4f} | {t['mean_b']:.4f} | {t['diff']:+.4f} "
                             f"[{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] | {t['p']:.4f} | {t['p_holm']:.4f} | "
                             f"{'yes' if t['supported'] else 'no'} |")

    lines += ["\n## All systems (mean nDCG@10; Recall@25 of the hybrid pool)\n",
              "| System | " + " | ".join(f"{n} ce-l6 | {n} bge-m3 | {n} R@25" for n in DATASETS.values()) + " |",
              "|---|" + "---|" * (3 * len(DATASETS))]
    for system, label in SYSTEMS:
        cells = []
        for name in DATASETS:
            v = [per_question(name, system), per_question(name, system, BIG_KEY), per_question(name, system, "Hybrid", "recall@25")]
            report["means"].setdefault(name, {})[system] = [None if x is None else float(np.mean(list(x.values()))) for x in v]
            cells += ["—" if x is None else f"{np.mean(list(x.values())):.4f}" for x in v]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["\n## Transcripts: WER, rare-word error, prompt injection\n",
              "| Dataset | Transcript | WER | Rare-word error | Queries with injected prompt words | Injected words / query |",
              "|---|---|---|---|---|---|"]
    for name, nice in DATASETS.items():
        try:
            ts = transcript_stats(name)
        except Exception as exc:                                 # noqa: BLE001 — dataset not run yet
            print(f"transcripts {name}: {exc!r}"); continue
        report["secondary"].setdefault(name, {})["transcripts"] = ts
        for system, r in ts.items():
            f = lambda v, p="{:.3f}": "—" if v is None else p.format(v)   # noqa: E731
            lines.append(f"| {nice} | {system} | {f(r['wer'])} | {f(r['rare_word_error'])} | "
                         f"{f(r['prompt_injection_share'], '{:.1%}')} | {f(r['injected_words_per_query'], '{:.2f}')} |")

    lines += ["\n## Confirmation-bias check: LT − pass 1, split by whether pass-1 hybrid top-5 held a relevant passage\n",
              "| Dataset | Pass-1 top-5 had it: n, Δ | Pass-1 top-5 missed it: n, Δ | LT helped / hurt (share of questions) | MC helped / hurt |",
              "|---|---|---|---|---|"]
    for name, nice in DATASETS.items():
        lt, base, mc = per_question(name, "lt"), per_question(name, "base"), per_question(name, "mc")
        hit = per_question(name, "base", "Hybrid", "recall@5")
        if None in (lt, base, mc, hit):
            continue
        qs = sorted(set(lt) & set(base) & set(hit))
        d = np.array([lt[q] - base[q] for q in qs]); h = np.array([hit[q] > 0 for q in qs])
        dm = np.array([mc[q] - base[q] for q in qs])
        sec = {"in_top5": [int(h.sum()), float(d[h].mean()) if h.any() else None],
               "not_in_top5": [int((~h).sum()), float(d[~h].mean()) if (~h).any() else None],
               "lt_helped": float((d > 1e-9).mean()), "lt_hurt": float((d < -1e-9).mean()),
               "mc_helped": float((dm > 1e-9).mean()), "mc_hurt": float((dm < -1e-9).mean())}
        report["secondary"].setdefault(name, {})["confirmation"] = sec
        lines.append(f"| {nice} | {sec['in_top5'][0]}, {sec['in_top5'][1]:+.4f} | {sec['not_in_top5'][0]}, "
                     f"{sec['not_in_top5'][1]:+.4f} | {sec['lt_helped']:.1%} / {sec['lt_hurt']:.1%} | "
                     f"{sec['mc_helped']:.1%} / {sec['mc_hurt']:.1%} |")

    lines += ["\n## Typed queries on the MC index (non-inferiority, margin −0.005)\n",
              "| Dataset | Typed, plain | Typed, MC | Δ [95% CI] | Non-inferior |", "|---|---|---|---|---|"]
    for name, nice in DATASETS.items():
        a, b = per_question(name, "typed_mc"), per_question(name, "typed")
        if a is None or b is None:
            continue
        x, y, _ = paired(a, b)
        ci = stats.bootstrap_ci(x - y)
        report["secondary"].setdefault(name, {})["typed_noninferiority"] = {"diff": float((x - y).mean()), "ci": ci}
        lines.append(f"| {nice} | {y.mean():.4f} | {x.mean():.4f} | {(x - y).mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{'yes' if ci[0] > -0.005 else 'no'} |")

    lines += ["\n## LT + MC against each method alone (secondary)\n",
              "| Dataset | LT+MC − LT [95% CI], p | LT+MC − MC [95% CI], p |", "|---|---|---|"]
    for name, nice in DATASETS.items():
        c, lt, mc = per_question(name, "ltmc"), per_question(name, "lt"), per_question(name, "mc")
        if None in (c, lt, mc):
            continue
        cells = []
        for other in (lt, mc):
            x, y, _ = paired(c, other)
            ci = stats.bootstrap_ci(x - y)
            cells.append(f"{(x - y).mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], p={stats.randomization_test(x, y):.4f}")
        lines.append(f"| {nice} | " + " | ".join(cells) + " |")

    sd = [s for s, _ in units("sdqa")]
    if all(result("sdqa", s, "ltmc") for s in sd):
        lines += ["\n## SD-QA per dialect (nDCG@10, Hybrid > ce-l6 @25)\n",
                  "| Dialect | Pass 1 | LT | MC | LT+MC | Oracle prompt |", "|---|---|---|---|---|---|"]
        drop = missing("sdqa")
        for s in sd:
            row = [np.mean([m[M] for q, m in result("sdqa", s, sys_)["rows"][KEY].items() if q not in drop])
                   for sys_ in ("base", "lt", "mc", "ltmc", "orc")]
            lines.append(f"| {s[1:]} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    if result("heysquad", "", "lt_k1"):
        lines += ["\n## HeySQuAD ablation: number of passages K behind the LT prompt (post hoc)\n",
                  "| K | nDCG@10 |", "|---|---|"]
        for k, system in ((1, "lt_k1"), (3, "lt_k3"), (5, "lt"), (10, "lt_k10")):
            v = per_question("heysquad", system)
            lines.append(f"| {k}{' (pre-registered)' if k == 5 else ''} | {np.mean(list(v.values())):.4f} |")

    lines += ["\n## Cost\n", "| Dataset | MC: words misheard / vocabulary | MC: index growth | "
              "LT: added Whisper time per query (s) | Pass-1 re-decode identical |", "|---|---|---|---|---|"]
    examples = []
    for name, nice in DATASETS.items():
        st = data.load_json(data.cache_dir(name) / "p4_mc_stats.json")
        tm = data.load_json(data.cache_dir(name) / "p4_timing.json")
        if not st or not tm:
            continue
        lt_s = np.mean([u["seconds_per_decode"]["lt"] for u in tm.values()])
        ident = np.mean([u["pass1_redecode_identical"] for u in tm.values()])
        report["secondary"].setdefault(name, {})["cost"] = {"mc": st, "timing": tm}
        lines.append(f"| {nice} | {st['words_with_variants']:,} / {st['vocabulary']:,} ({st['share_misheard']:.0%}) | "
                     f"+{st['index_growth']:.1%} tokens | {lt_s:.3f} | {ident:.0%} |")
        examples.append(f"- {nice}: " + "; ".join(f"{w} → {' / '.join(v)}" for w, v in list(st["examples"].items())[:8]))

    lines += ["\nMisheard-word examples (most frequent corpus words first):\n"] + examples
    out = HERE / "results" / "part4_results.md"
    out.write_text("\n".join(lines) + "\n")
    data.save_json(HERE / "results" / "part4_results.json", report)
    print(out.read_text())


if __name__ == "__main__":
    main()
