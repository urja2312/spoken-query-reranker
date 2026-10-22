"""Turn cached runs into the paper's tables and figures.

Reads cache/<dataset>/results_<cond>.json (+ ood_*.json, transcripts, latency.json) and writes
results/*.md, results/*.csv and results/figures/*.png. Pure analysis: no model is run here.
Usage: python -I analyze.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech, stats  # noqa: E402

OUT = HERE / "results"
FIG = OUT / "figures"
CONDS = ("text", "asr_clean", "asr_phone")
COND_LABEL = {"text": "Text (oracle)", "asr_clean": "ASR wideband", "asr_phone": "ASR telephone"}
TAG = "ce-l6"
PROD = f"BM25 > {TAG} @25"
MAIN_SYSTEMS = ["BM25", "Dense (bge-small)", "Hybrid RRF", PROD, f"BM25 > {TAG} @100",
                f"Hybrid > {TAG} @25", f"Hybrid > {TAG} @100", f"{PROD} + alpha(CV)"]
METRICS = ("ndcg@10", "mrr@10", "hit@1", "recall@100")
FOLDS = 5


def load_results(name: str, cond: str):
    """Results with checkpoint aliases resolved (see select_models.py): an alias is a tag whose
    weights are identical to another evaluated tag, so its rows are that tag's rows."""
    res = data.load_json(data.cache_dir(name) / f"results_{cond}.json")
    aliases = data.load_json(OUT / "training" / "aliases.json", {})
    if res:
        rows = res["rows"]
        for alias, source in aliases.items():
            for key in [k for k in rows if f" > {source} @" in k]:
                rows.setdefault(key.replace(f" > {source} @", f" > {alias} @"), rows[key])
    return res


def per_query(rows: dict, system: str, metric: str, qids: list[str]) -> np.ndarray:
    return np.array([rows[system][q][metric] for q in qids], dtype=float)


def alpha_cv(rows: dict, depth: int, qids: list[str], seed: int = 13):
    """5-fold CV: pick alpha on 4 folds, score the held-out fold. No test-set peeking."""
    alphas = sorted({float(k.rsplit("=", 1)[1]) for k in rows if k.startswith(f"BM25 > {TAG} @{depth} | alpha=")})
    if not alphas:
        return None
    grid = {a: {q: rows[f"BM25 > {TAG} @{depth} | alpha={a}"][q] for q in qids} for a in alphas}
    order = np.random.default_rng(seed).permutation(len(qids))
    folds = np.array_split(order, FOLDS)
    picked, out = [], {}
    for k in range(FOLDS):
        test = {qids[i] for i in folds[k]}
        train = [q for q in qids if q not in test]
        best = max(alphas, key=lambda a: np.mean([grid[a][q]["ndcg@10"] for q in train]))
        picked.append(best)
        for q in test:
            out[q] = grid[best][q]
    curve = {a: float(np.mean([grid[a][q]["ndcg@10"] for q in qids])) for a in alphas}
    return out, picked, curve


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main_tables():
    lines, csv_rows, robust, alpha_info = [], [], {}, {}
    for name in data.DATASETS:
        for cond in CONDS:
            res = load_results(name, cond)
            if not res:
                continue
            rows = res["rows"]
            qids = sorted(rows["BM25"])
            cv = alpha_cv(rows, 25, qids)
            if cv:
                rows[f"{PROD} + alpha(CV)"] = cv[0]
                alpha_info[(name, cond)] = {"picked": cv[1], "curve": cv[2]}
            systems = [s for s in MAIN_SYSTEMS if s in rows]
            base = per_query(rows, "BM25", "ndcg@10", qids)
            pvals = {s: stats.randomization_test(per_query(rows, s, "ndcg@10", qids), base)
                     for s in systems if s != "BM25"}
            adj = stats.holm(pvals)
            lines.append(f"\n### {name} — {COND_LABEL[cond]} ({len(qids)} queries)\n")
            lines.append("| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for s in systems:
                v = {m: float(per_query(rows, s, m, qids).mean()) for m in METRICS}
                lo, hi = stats.bootstrap_ci(per_query(rows, s, "ndcg@10", qids))
                delta = v["ndcg@10"] - float(base.mean())
                p = adj.get(s)
                mark = "" if p is None else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                ptxt = "—" if p is None else (f"{p:.3f}" if p >= 0.001 else "<0.001")
                lines.append(f"| {s} | {fmt(v['ndcg@10'])}{mark} | [{lo:.3f}, {hi:.3f}] | {fmt(v['mrr@10'])} | "
                             f"{fmt(v['hit@1'])} | {fmt(v['recall@100'])} | {delta:+.4f} | {ptxt} |")
                csv_rows.append({"dataset": name, "condition": cond, "system": s, **{m: v[m] for m in METRICS},
                                 "ndcg_ci_lo": lo, "ndcg_ci_hi": hi, "delta_vs_bm25": delta,
                                 "p_holm": "" if p is None else p})
                robust.setdefault((name, s), {})[cond] = v["ndcg@10"]
    return lines, csv_rows, robust, alpha_info


def robustness_table(robust):
    lines = ["\n## Robustness to spoken queries (nDCG@10; relative change vs text)\n",
             "| Dataset | System | Text | ASR wideband | ASR telephone | Δ wideband | Δ telephone |",
             "|---|---|---|---|---|---|---|"]
    for (name, system), by in robust.items():
        if system not in MAIN_SYSTEMS[:5] or "text" not in by:
            continue
        t = by["text"]
        cells = [fmt(by[c]) if c in by else "—" for c in CONDS]
        rel = [f"{(by[c] - t) / t * 100:+.1f}%" if c in by and t else "—" for c in ("asr_clean", "asr_phone")]
        lines.append(f"| {name} | {system} | " + " | ".join(cells) + " | " + " | ".join(rel) + " |")
    return lines


def wer_table():
    lines = ["\n## ASR word error rate of the spoken queries\n",
             "Mean / median WER, and the share of transcripts that are hallucinations (no word in common "
             "with the query, WER > 1, and character similarity < 0.35 — an invention rather than a "
             "sound-alike mishearing; a known Whisper failure on very short noisy clips).\n",
             "| Dataset | Median query words | ASR wideband | ASR telephone |", "|---|---|---|---|"]
    for name in data.DATASETS:
        ds = data.load(name)
        cells = []
        for cond in ("asr_clean", "asr_phone"):
            tr = data.load_json(data.cache_dir(name) / f"transcripts_{cond}.json", {})
            if all(q in tr for q in ds.queries):
                w = np.array([speech.wer(ds.queries[q], tr[q]) for q in ds.queries])
                hall = np.mean([speech.is_hallucination(ds.queries[q], tr[q]) for q in ds.queries])
                cells.append(f"{w.mean():.3f} / {np.median(w):.3f} ({hall * 100:.0f}% halluc.)")
            else:
                cells.append("—")
        words = int(np.median([len(t.split()) for t in ds.queries.values()]))
        lines.append(f"| {name} | {words} | " + " | ".join(cells) + " |")
    return lines


def depth_table():
    lat = data.load_json(OUT / "latency.json", {})
    hw = lat.get("_hardware", {})
    lines = ["\n## Re-ranking depth vs effectiveness vs latency (text queries)\n",
             f"Latency hardware: CPU = {hw.get('cpu', 'unknown')}; GPU = {hw.get('gpu', 'unknown')}. "
             "Batch = depth (one forward pass per voice turn), warm model, 40 queries per dataset.\n",
             "| Dataset | Depth | nDCG@10 | Hit@1 | CE latency CPU p50 / p95 (ms) | CE latency GPU p50 (ms) |",
             "|---|---|---|---|---|---|"]
    curves = {}
    for name in data.DATASETS:
        res = load_results(name, "text")
        if not res:
            continue
        rows = res["rows"]
        qids = sorted(rows["BM25"])
        pts = [(0, float(per_query(rows, "BM25", "ndcg@10", qids).mean()),
                float(per_query(rows, "BM25", "hit@1", qids).mean()))]
        for depth in (10, 25, 50, 100):
            key = f"BM25 > {TAG} @{depth}"
            if key in rows:
                pts.append((depth, float(per_query(rows, key, "ndcg@10", qids).mean()),
                            float(per_query(rows, key, "hit@1", qids).mean())))
        curves[name] = pts
        for depth, nd, h1 in pts:
            cpu = lat.get(name, {}).get("cpu", {}).get(TAG, {}).get(str(depth))
            acc_key = next((k for k in ("cuda", "mps") if k in lat.get(name, {})), "")
            mps = lat.get(name, {}).get(acc_key, {}).get(TAG, {}).get(str(depth))
            ctxt = f"{cpu['p50']:.0f} / {cpu['p95']:.0f}" if cpu else ("0" if depth == 0 else "—")
            mtxt = f"{mps['p50']:.0f}" if mps else ("0" if depth == 0 else "—")
            lines.append(f"| {name} | {depth or 'BM25 only'} | {fmt(nd)} | {fmt(h1)} | {ctxt} | {mtxt} |")
    return lines, curves


def ablation_table():
    lines = ["\n## Ablations at the production depth (BM25 top-25, text queries)\n",
             "| Dataset | Reranker | Params | Max tokens | nDCG@10 | Hit@1 | p vs ce-l6 |", "|---|---|---|---|---|---|---|"]
    spec = {"ce-l6": ("22.7M", 256), "ce-l12": ("33.4M", 256), "ce-l6-512": ("22.7M", 512)}
    for name in data.DATASETS:
        res = load_results(name, "text")
        if not res:
            continue
        rows = res["rows"]
        qids = sorted(rows["BM25"])
        base = f"BM25 > ce-l6 @25"
        for tag, (params, mx) in spec.items():
            key = f"BM25 > {tag} @25"
            if key not in rows:
                continue
            v = per_query(rows, key, "ndcg@10", qids)
            p = "—" if tag == "ce-l6" else f"{stats.randomization_test(v, per_query(rows, base, 'ndcg@10', qids)):.3f}"
            lines.append(f"| {name} | {tag} | {params} | {mx} | {fmt(v.mean())} | "
                         f"{fmt(per_query(rows, key, 'hit@1', qids).mean())} | {p} |")
    return lines


def abstention_table():
    lines = ["\n## Abstention: can the system tell when the knowledge base has no answer?\n",
             "AUROC separates in-domain answerable queries from out-of-domain queries (no answer exists) "
             "using the top score of the BM25 top-25 pool. AURC is the area under the risk–coverage curve "
             "for Hit@1 on in-domain queries (lower is better).\n",
             "| Dataset | AUROC (BM25 score) | AUROC (CE score) | AURC (BM25 conf.) | AURC (CE conf.) | "
             "Keep @ 90% OOD rejected | Hit@1 when kept | CE threshold (logit) |",
             "|---|---|---|---|---|---|---|---|"]
    curves = {}
    for name in data.DATASETS:
        res = load_results(name, "text")
        ood = data.load_json(data.cache_dir(name) / f"ood_{TAG}.json")
        if not res or TAG not in res.get("abstain", {}) or not ood:
            continue
        ind = list(res["abstain"][TAG].values())
        ans = [r for r in ind if r["answerable"]]
        o_ce = [r["ce_top"] for r in ood.values() if np.isfinite(r["ce_top"])]
        o_bm = [r["bm25_top"] for r in ood.values()]
        au_ce = stats.auroc([r["ce_top"] for r in ans], o_ce)
        au_bm = stats.auroc([r["bm25_top"] for r in ans], o_bm)

        def aurc(conf_key, hit_key):
            order = sorted(ind, key=lambda r: -r[conf_key])
            hits = np.array([r[hit_key] for r in order])
            risk = 1 - np.cumsum(hits) / np.arange(1, len(hits) + 1)
            cov = np.arange(1, len(hits) + 1) / len(hits)
            return float(np.trapezoid(risk, cov)), cov, risk

        a_bm, _, _ = aurc("bm25_top", "bm25_hit1")
        a_ce, cov, risk = aurc("ce_top", "ce_hit1")
        curves[name] = (cov, risk)
        thr = float(np.quantile(o_ce, 0.90))
        kept = [r for r in ind if r["ce_top"] > thr]
        keep = len(kept) / len(ind)
        hit = np.mean([r["ce_hit1"] for r in kept]) if kept else float("nan")
        lines.append(f"| {name} | {au_bm:.3f} | {au_ce:.3f} | {a_bm:.3f} | {a_ce:.3f} | "
                     f"{keep * 100:.1f}% | {hit:.3f} | {thr:.2f} |")
    return lines, curves


def alpha_lines(alpha_info):
    lines = ["\n## Score interpolation (BM25 ⊕ CE, depth 25): alpha chosen by 5-fold CV\n",
             "| Dataset | Condition | alphas picked per fold | nDCG@10 at alpha=0 | best alpha in hindsight |",
             "|---|---|---|---|---|"]
    for (name, cond), info in alpha_info.items():
        curve = info["curve"]
        best = max(curve, key=curve.get)
        lines.append(f"| {name} | {COND_LABEL[cond]} | {', '.join(str(a) for a in info['picked'])} | "
                     f"{fmt(curve[0.0])} | {best} ({fmt(curve[best])}) |")
    return lines


def figures(curves, robust, risk_curves, alpha_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    lat = data.load_json(OUT / "latency.json", {})

    fig, axes = plt.subplots(1, max(1, len(curves)), figsize=(4.2 * max(1, len(curves)), 3.4), squeeze=False)
    for ax, (name, pts) in zip(axes[0], curves.items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label="nDCG@10")
        ax.set_title(name); ax.set_xlabel("re-rank depth (0 = BM25 only)"); ax.set_ylabel("nDCG@10")
        cpu = lat.get(name, {}).get("cpu", {}).get(TAG, {})
        if cpu:
            ax2 = ax.twinx()
            ds_ = sorted(int(d) for d in cpu)
            ax2.plot(ds_, [cpu[str(d)]["p50"] for d in ds_], "s--", color="tab:red", label="CPU p50 ms")
            ax2.set_ylabel("CE latency, CPU p50 (ms)", color="tab:red")
    fig.tight_layout(); fig.savefig(FIG / "depth_vs_ndcg_latency.png", dpi=160); plt.close(fig)

    sys_show = MAIN_SYSTEMS[:5]
    fig, axes = plt.subplots(1, len(data.DATASETS), figsize=(4.6 * len(data.DATASETS), 3.6), squeeze=False)
    for ax, name in zip(axes[0], data.DATASETS):
        width = 0.26
        for i, cond in enumerate(CONDS):
            vals = [robust.get((name, s), {}).get(cond, np.nan) for s in sys_show]
            ax.bar(np.arange(len(sys_show)) + (i - 1) * width, vals, width, label=COND_LABEL[cond])
        ax.set_xticks(range(len(sys_show)))
        ax.set_xticklabels([s.replace(" (bge-small)", "").replace(f" > {TAG} ", "→CE") for s in sys_show],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(name); ax.set_ylabel("nDCG@10")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "robustness_spoken_queries.png", dpi=160); plt.close(fig)

    if risk_curves:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for name, (cov, risk) in risk_curves.items():
            ax.plot(cov, risk, label=name)
        ax.set_xlabel("coverage (fraction of queries answered)"); ax.set_ylabel("risk (1 − Hit@1)")
        ax.set_title("Selective answering with CE confidence"); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "risk_coverage.png", dpi=160); plt.close(fig)

    if alpha_info:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for (name, cond), info in alpha_info.items():
            if cond != "text":
                continue
            a = sorted(info["curve"]); ax.plot(a, [info["curve"][x] for x in a], "o-", label=name)
        ax.set_xlabel("alpha (0 = CE only, 1 = BM25 only)"); ax.set_ylabel("nDCG@10")
        ax.set_title("BM25 ⊕ CE interpolation, depth 25"); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "alpha_sweep.png", dpi=160); plt.close(fig)


# Checkpoint tag pattern per protocol; {v} = variant, {s} = seed.
PROTOCOLS = {"A. early stopping on validation": "ft-{v}-s{s}", "B. fixed budget (3 epochs)": "ft-{v}-s{s}-last"}
SEEDS = (0, 1, 2)


def seed_scores(rows, pattern, variant, metric, qids):
    """Per-query scores averaged over the seeds available, plus each seed's dataset mean."""
    runs = [per_query(rows, f"BM25 > {pattern.format(v=variant, s=seed)} @25", metric, qids)
            for seed in SEEDS if f"BM25 > {pattern.format(v=variant, s=seed)} @25" in rows]
    if not runs:
        return None, []
    return np.mean(runs, axis=0), [float(r.mean()) for r in runs]
FT_LABEL = {"zero": "zero-shot (MS MARCO)", "clean": "FT clean (control)", "noisy": "FT noisy",
            "mixed": "FT mixed", "distill": "FT distill (Part 2)"}
VARIANTS = ("clean", "noisy", "mixed", "distill")


def finetune_table():
    """The paper's main result: does noise-aware fine-tuning beat plain in-domain fine-tuning
    on spoken queries, without giving up typed-query quality? Reported under both checkpoint-
    selection protocols, which were fixed before any test-set result was seen, and averaged over
    independent training seeds (per-query scores averaged across seeds, then tested)."""
    lines = ["\n## Noise-aware fine-tuning (re-rank BM25 top-25 — the production setting)\n",
             "nDCG@10 is the mean over seeds of per-query scores; ± is the standard deviation of the "
             "dataset-level score across seeds (n = seeds). Significance: paired randomization test on "
             "seed-averaged per-query nDCG@10, Holm-corrected within protocol × dataset × condition. "
             "† p<0.05 vs zero-shot; ‡ p<0.05 vs FT clean (the domain-adaptation control). Under "
             "protocol A, a seed whose best epoch was 0 *is* the zero-shot model.\n"]
    macro: dict[tuple[str, str, str], list[float]] = {}
    for proto, pattern in PROTOCOLS.items():
        lines.append(f"\n### Protocol {proto}\n")
        for cond in CONDS:
            lines.append(f"\n**{COND_LABEL[cond]}**\n")
            lines.append("| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for name in data.DATASETS:
                res = load_results(name, cond)
                if not res or "BM25 > ce-l6 @25" not in res["rows"]:
                    continue
                rows = res["rows"]
                qids = sorted(rows["BM25"])
                nd = {"zero": per_query(rows, "BM25 > ce-l6 @25", "ndcg@10", qids)}
                hit = {"zero": per_query(rows, "BM25 > ce-l6 @25", "hit@1", qids)}
                spread = {"zero": [float(nd["zero"].mean())]}
                for v in VARIANTS:
                    score, per_seed = seed_scores(rows, pattern, v, "ndcg@10", qids)
                    if score is not None:
                        nd[v], spread[v] = score, per_seed
                        hit[v] = seed_scores(rows, pattern, v, "hit@1", qids)[0]
                if len(nd) < 2:
                    continue
                tests = {f"{v}|zero": stats.randomization_test(nd[v], nd["zero"]) for v in nd if v != "zero"}
                tests.update({f"{v}|clean": stats.randomization_test(nd[v], nd["clean"])
                              for v in ("noisy", "mixed", "distill") if v in nd and "clean" in nd})
                adj = stats.holm(tests)
                for v in nd:
                    marks = ("†" if adj.get(f"{v}|zero", 1) < 0.05 else "") + \
                            ("‡" if adj.get(f"{v}|clean", 1) < 0.05 else "")
                    sd = f"{np.std(spread[v], ddof=1):.4f}" if len(spread[v]) > 1 else "—"
                    d_zero = "—" if v == "zero" else f"{nd[v].mean() - nd['zero'].mean():+.4f}"
                    d_clean = (f"{nd[v].mean() - nd['clean'].mean():+.4f}"
                               if v in ("noisy", "mixed", "distill") and "clean" in nd else "—")
                    n = "—" if v == "zero" else str(len(spread[v]))
                    lines.append(f"| {name} | {FT_LABEL[v]} | {n} | {nd[v].mean():.4f}{marks} | {sd} | "
                                 f"{hit[v].mean():.4f} | {d_zero} | {d_clean} |")
                    macro.setdefault((proto, cond, v), []).append(float(nd[v].mean()))
        lines += [f"\n**Macro-average over datasets, protocol {proto[:1]} (nDCG@10)**\n",
                  "| Reranker | " + " | ".join(COND_LABEL[c] for c in CONDS) + " | Text→telephone drop |",
                  "|---|" + "---|" * (len(CONDS) + 1)]
        for v in ("zero",) + VARIANTS:
            if not any((proto, c, v) in macro for c in CONDS):
                continue
            vals = [np.mean(macro[(proto, c, v)]) if (proto, c, v) in macro else float("nan") for c in CONDS]
            drop = (vals[2] - vals[0]) / vals[0] * 100 if all(np.isfinite([vals[0], vals[2]])) else float("nan")
            lines.append(f"| {FT_LABEL[v]} | " + " | ".join(f"{x:.4f}" for x in vals) + f" | {drop:+.1f}% |")
    return lines


def voice_table():
    """Seen vs held-out TTS voice: Moira (en_IE) was excluded from every training example."""
    lines = ["\n## Seen vs unseen speaker voice (ASR conditions, BM25 top-25, nDCG@10)\n",
             "| Condition | Voice group | Queries | zero-shot | FT mixed (protocol B, seed mean) | Δ |",
             "|---|---|---|---|---|---|"]
    for cond in ("asr_clean", "asr_phone"):
        groups: dict[str, dict[str, list[float]]] = {"seen voices": {}, "unseen voice (Moira)": {}}
        for name in data.DATASETS:
            res = load_results(name, cond)
            if not res or "BM25 > ce-l6 @25" not in res["rows"]:
                continue
            rows = res["rows"]
            qids = sorted(rows["BM25"])
            mixed, _ = seed_scores(rows, PROTOCOLS["B. fixed budget (3 epochs)"], "mixed", "ndcg@10", qids)
            if mixed is None:
                continue
            zero = per_query(rows, "BM25 > ce-l6 @25", "ndcg@10", qids)
            for i, q in enumerate(qids):
                g = "unseen voice (Moira)" if speech.voice_for(q) == "Moira" else "seen voices"
                groups[g].setdefault("zero", []).append(zero[i])
                groups[g].setdefault("mixed", []).append(mixed[i])
        for g, by in groups.items():
            if by:
                a, b = np.mean(by["zero"]), np.mean(by["mixed"])
                lines.append(f"| {COND_LABEL[cond]} | {g} | {len(by['zero'])} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    return lines


def training_figure():
    logs = sorted((OUT / "training").glob("ft-*-s[0-9].json")) if (OUT / "training").exists() else []
    if not logs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for path in logs:
        log = json.loads(path.read_text())
        color = {"clean": "tab:blue", "noisy": "tab:red", "mixed": "tab:green"}[log["variant"]]
        tag = f"{log['variant']} s{log['seed']}"
        a1.plot([p["step"] for p in log["loss"]], [p["loss"] for p in log["loss"]], color=color, alpha=.6, label=tag)
        a2.plot([v["epoch"] for v in log["val"]], [v["mean"] for v in log["val"]], "o-", color=color, alpha=.6, label=tag)
    a1.set_xlabel("step"); a1.set_ylabel("training BCE loss (20-step mean)"); a1.legend(fontsize=6)
    a2.set_xlabel("epoch (0 = base model)"); a2.set_ylabel("val nDCG@10 (clean+ASR mean)"); a2.legend()
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "training_curves.png", dpi=160); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_lines, csv_rows, robust, alpha_info = main_tables()
    depth_lines, curves = depth_table()
    abst_lines, risk_curves = abstention_table()
    doc = ["# Reranker benchmark — results\n",
           "Significance: two-sided paired randomization test on per-query nDCG@10 vs BM25, "
           "Holm-corrected within each dataset × condition; * p<0.05, ** p<0.01. "
           "CI: 95% bootstrap over queries.\n"]
    doc += main_lines + robustness_table(robust) + wer_table() + depth_lines + ablation_table() \
        + alpha_lines(alpha_info) + abst_lines + finetune_table() + voice_table()
    (OUT / "results.md").write_text("\n".join(doc) + "\n")
    with open(OUT / "main_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    figures(curves, robust, risk_curves, alpha_info)
    training_figure()
    print((OUT / "results.md").read_text())


if __name__ == "__main__":
    main()
