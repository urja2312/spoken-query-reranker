"""Figures for the paper, written to ../paper/figures (reads cached results only; no model is run)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
from bench import data  # noqa: E402

OUT = HERE.parent / "paper" / "figures"
CONDS = ("text", "asr_clean", "asr_phone")
LABEL = {"text": "Typed", "asr_clean": "Spoken (wideband)", "asr_phone": "Spoken (telephone)"}
COLORS = {"text": "#2b6cb0", "asr_clean": "#dd8a1f", "asr_phone": "#c0392b"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: robustness of each pipeline
systems = [("BM25", "BM25"), ("Dense (bge-small)", "Dense"), ("Hybrid RRF", "Hybrid"), ("BM25 > ce-l6 @25", "BM25→CE")]
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
for ax, name in zip(axes, data.DATASETS):
    for i, cond in enumerate(CONDS):
        res = analyze.load_results(name, cond)
        rows = res["rows"]; qids = sorted(rows["BM25"])
        vals = [analyze.per_query(rows, key, "ndcg@10", qids).mean() for key, _ in systems]
        ax.bar(np.arange(len(systems)) + (i - 1) * 0.27, vals, 0.27, color=COLORS[cond], label=LABEL[cond])
    ax.set_xticks(range(len(systems))); ax.set_xticklabels([s for _, s in systems], rotation=20, ha="right")
    ax.set_title({"scifact": "SciFact (science)", "nfcorpus": "NFCorpus (medical)", "fiqa": "FiQA (finance)"}[name])
    ax.set_ylabel("nDCG@10")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig1_robustness.png", dpi=220); plt.close(fig)

# Figure 2: fine-tuning variants, macro-average over datasets, mean ± sd across seeds (protocol B)
pattern = analyze.PROTOCOLS["B. fixed budget (3 epochs)"]
variants = [("zero", "Zero-shot"), ("clean", "FT clean"), ("noisy", "FT noisy"), ("mixed", "FT mixed"),
            ("distill", "FT distill")]
fig, ax = plt.subplots(figsize=(6.8, 2.9))
for j, (v, lab) in enumerate(variants):
    means, sds = [], []
    for cond in CONDS:
        per_seed = []
        for seed in analyze.SEEDS if v != "zero" else (None,):
            vals = []
            for name in data.DATASETS:
                rows = analyze.load_results(name, cond)["rows"]; qids = sorted(rows["BM25"])
                key = "BM25 > ce-l6 @25" if v == "zero" else f"BM25 > {pattern.format(v=v, s=seed)} @25"
                vals.append(analyze.per_query(rows, key, "ndcg@10", qids).mean())
            per_seed.append(np.mean(vals))
        means.append(np.mean(per_seed)); sds.append(np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0)
    ax.bar(np.arange(3) + (j - 2) * 0.16, means, 0.16, yerr=sds, capsize=2,
           label=lab, color=["#7f8c8d", "#2b6cb0", "#c0392b", "#27ae60", "#8e44ad"][j])
ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[c] for c in CONDS]); ax.set_ylabel("nDCG@10 (macro avg.)")
ax.set_ylim(0.2, 0.46); ax.legend(frameon=False, ncol=5, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig2_finetuning.png", dpi=220); plt.close(fig)

# Figures 4-5: SD-QA real speech (Part 2)
import analyze_sdqa as S  # noqa: E402
cache = data.cache_dir("sdqa")
sconds = data.load_json(cache / "conditions.json", [])
sres = {c: data.load_json(cache / f"results_{c}.json") for c in ["text"] + sconds}
sq = sorted(sres["text"]["rows"]["BM25"])
nd = lambda c, k: S.per_query(sres[c]["rows"], k, "ndcg@10", sq)
typed = nd("text", "BM25 > ce-l6 @25").mean()
by = {}
for c in sconds:
    d, m = S.parse(c)
    by.setdefault(d, {})["US" if m == "en-US" else "loc"] = nd(c, "BM25 > ce-l6 @25").mean()
order = sorted(by, key=lambda d: -max(by[d].values()))
fig, ax = plt.subplots(figsize=(7.4, 2.9))
x = np.arange(len(order))
ax.bar(x - 0.2, [by[d].get("loc", np.nan) for d in order], 0.4, color="#2b6cb0", label="Localised ASR model")
ax.bar(x + 0.2, [by[d]["US"] for d in order], 0.4, color="#dd8a1f", label="en-US ASR model")
ax.axhline(typed, color="#333", ls="--", lw=1); ax.text(-0.45, typed + 0.003, "typed questions", ha="left", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([S.DIALECT[d] for d in order], rotation=25, ha="right")
for i, d in enumerate(order):
    if d.startswith("ind"):
        ax.get_xticklabels()[i].set_color("#c0392b"); ax.get_xticklabels()[i].set_fontweight("bold")
ax.set_ylim(0.4, 0.62); ax.set_ylabel("nDCG@10 (BM25→CE)"); ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig(OUT / "fig4_sdqa_dialects.png", dpi=220); plt.close(fig)

groups = {"Typed": ["text"], "All real speech\n(21 conditions)": sconds,
          "Indian English\n(4 conditions)": [c for c in sconds if S.parse(c)[0].startswith("ind")],
          "Indian English,\nen-IN ASR": [c for c in sconds if S.parse(c)[0].startswith("ind") and S.parse(c)[1] == "en-IN"]}
pipes = [("BM25 > ce-l6 @25", "BM25→CE (production)", "#7f8c8d"), ("Dense (bge-small)", "Dense only", "#dd8a1f"),
         ("Hybrid > ce-l6 @25", "Hybrid→CE", "#2b6cb0"), ("Hybrid > ft-distill-s{s}-last @25", "Hybrid→FT-distill", "#8e44ad")]
fig, ax = plt.subplots(figsize=(7.4, 2.9))
for j, (key, lab, col) in enumerate(pipes):
    keys = [key.format(s=s) for s in (0, 1, 2)] if "{s}" in key else [key]
    vals = [np.mean([np.mean([nd(c, k) for k in keys], axis=0) for c in gc]) for gc in groups.values()]
    ax.bar(np.arange(len(groups)) + (j - 1.5) * 0.2, vals, 0.2, color=col, label=lab)
ax.set_xticks(range(len(groups))); ax.set_xticklabels(list(groups)); ax.set_ylim(0.4, 0.72)
ax.set_ylabel("nDCG@10"); ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig5_sdqa_pipelines.png", dpi=220); plt.close(fig)
# Figure 0 (report): voice agent architecture and where the reranker sits
from matplotlib.patches import FancyBboxPatch  # noqa: E402
fig, ax = plt.subplots(figsize=(9.6, 3.15)); ax.set_xlim(0, 9.6); ax.set_ylim(0, 3.15); ax.axis("off")
def box(x, y, w, h, title, sub, fc, tc="#16324F"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=9, weight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.2, color=tc)
def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#6B7A8F", lw=1.2))
y = 1.9; h = 0.85
steps = [(0.05, "Caller", "phone / app audio", "#EEF3F7"), (1.35, "Speech-to-text", "ASR transcript", "#EEF3F7"),
         (2.65, "BM25 search", "FTS5, top-25", "#EEF3F7"), (3.95, "Reranker", "cross-encoder seam", "#E4572E"),
         (5.25, "Brain (LLM)", "grounded answer", "#EEF3F7"), (6.55, "Text-to-speech", "spoken reply", "#EEF3F7"),
         (7.85, "Caller", "hears the answer", "#EEF3F7")]
for i, (x, a, b, fc) in enumerate(steps):
    box(x, y, 1.15, h, a, b, fc, "white" if fc == "#E4572E" else "#16324F")
    if i < len(steps) - 1:
        arrow(x + 1.17, y + h / 2, x + 1.33, y + h / 2)
box(2.65, 0.35, 2.45, 0.9, "Knowledge base", "SQLite FTS5 index of documents", "#F7F9FB")
arrow(3.2, 1.27, 3.2, 1.88)
box(5.25, 0.35, 1.15, 0.9, "Safety check", "runs before the model", "#F7F9FB")
arrow(5.83, 1.27, 5.83, 1.88)
box(6.55, 0.35, 2.45, 0.9, "Tools", "orders, tickets, escalation (approval)", "#F7F9FB")
arrow(6.0, 1.88, 7.2, 1.27)
ax.text(4.52, 2.95, "fallback to BM25 order on any model error", ha="center", fontsize=7, style="italic", color="#E4572E")
fig.tight_layout(pad=0.2); fig.savefig(OUT / "fig0_architecture.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------------ Part 3 figures (drawn when data exist)
import json  # noqa: E402
import analyze_part3 as P3  # noqa: E402

COND3 = ("text", "asr_clean", "asr_phone")
# Figure 6: bge-reranker-base fine-tuning on BEIR (C2), macro over datasets, mean ± sd over seeds
v6 = [("bge-base", "Zero-shot", "#7f8c8d"), ("ft-bgeb-clean-s{s}-last", "FT clean", "#2b6cb0"),
      ("ft-bgeb-noisy-s{s}-last", "FT noisy", "#c0392b"), ("ft-bgeb-distill-s{s}-last", "FT distill", "#8e44ad")]
if P3.scores([("scifact", "text")], "BM25 > ft-bgeb-distill-s{s}-last @25") is not None:
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for j, (tag, lab, col) in enumerate(v6):
        means, sds = [], []
        for cond in COND3:
            seeds = (None,) if "{s}" not in tag else P3.SEEDS
            per = [np.mean([P3.scores([(d, cond)], f"BM25 > {tag.format(s=sd)} @25").mean() for d in data.DATASETS]) for sd in seeds]
            means.append(np.mean(per)); sds.append(np.std(per, ddof=1) if len(per) > 1 else 0)
        ax.bar(np.arange(3) + (j - 1.5) * 0.2, means, 0.2, yerr=sds, capsize=2, label=lab, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[c] for c in COND3]); ax.set_ylabel("nDCG@10 (macro avg.)")
    ax.set_ylim(0.2, 0.46); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig6_bge_finetuning.png", dpi=220); plt.close(fig)

# Figure 7: which words break (C4) — OLS coefficients with query-cluster bootstrap CIs
ea = HERE / "results" / "error_analysis.json"
if ea.exists():
    rep = json.loads(ea.read_text())
    names = list(rep)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(names))
    for off, key, lab, col in ((-0.17, "beta_content", "Content words (nouns, names, numbers…)", "#c0392b"),
                               (0.17, "beta_function", "Function words (the, of, is…)", "#7f8c8d")):
        vals = [rep[n][key] for n in names]; ci = np.array([rep[n][key + "_ci"] for n in names])
        ax.barh(y + off, vals, 0.32, color=col, label=lab,
                xerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], capsize=2)
    ax.axvline(0, color="#333", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Change in nDCG@10 per unit word-error rate"); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(OUT / "fig7_word_errors.png", dpi=220); plt.close(fig)

# Figure 8: real vs synthetic training noise (C3)
g3 = P3.test_groups()
h4 = [("ce-l6", "Zero-shot", "#7f8c8d"), ("ft-hq-clean-s{s}-last", "FT clean", "#2b6cb0"),
      ("ft-hq-synth-s{s}-last", "FT noisy (synthetic)", "#f2a541"), ("ft-hq-real-s{s}-last", "FT noisy (real)", "#c0392b"),
      ("ft-hq-dsynth-s{s}-last", "Distill (synthetic)", "#b39ddb"), ("ft-hq-dreal-s{s}-last", "Distill (real)", "#6b4e9b")]
sets8 = ["SD-QA-Whisper (11 dialects)", "SD-QA-Whisper, Indian", "SLUE-SQA-5 test", "HeySQuAD-dev"]
if all(P3.scores(g3[s8], f"BM25 > {t} @25") is not None for s8 in sets8 for t, _, _ in h4):
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for j, (tag, lab, col) in enumerate(h4):
        ax.bar(np.arange(len(sets8)) + (j - 2.5) * 0.13, [P3.scores(g3[s8], f"BM25 > {tag} @25").mean() for s8 in sets8],
               0.13, label=lab, color=col)
    ax.set_xticks(range(len(sets8))); ax.set_xticklabels([s8.replace(" (11 dialects)", "").replace(", ", ",\n") for s8 in sets8])
    ax.set_ylabel("nDCG@10"); ax.set_ylim(0.3, None); ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig8_real_vs_synth.png", dpi=220); plt.close(fig)

# Figure 9: first stage on the new real-speech test sets (C1) and the stronger rerankers
sets9 = ["SD-QA-Whisper (11 dialects)", "HeySQuAD-dev", "SLUE-SQA-5 test"]
p9 = [("BM25 > ce-l6 @25", "BM25 → ce-l6", "#7f8c8d"), ("BM25 > bge-m3 @25", "BM25 → bge-m3", "#f2a541"),
      ("Hybrid > ce-l6 @25", "Hybrid → ce-l6", "#2b6cb0"), ("Hybrid > bge-m3 @25", "Hybrid → bge-m3", "#6b4e9b")]
if all(P3.scores(g3[s9], k) is not None for s9 in sets9 for k, _, _ in p9):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for j, (key, lab, col) in enumerate(p9):
        ax.bar(np.arange(len(sets9)) + (j - 1.5) * 0.2, [P3.scores(g3[s9], key).mean() for s9 in sets9], 0.2, label=lab, color=col)
    ax.set_xticks(range(len(sets9))); ax.set_xticklabels([s9.replace(" (11 dialects)", "") for s9 in sets9])
    ax.set_ylabel("nDCG@10 (spoken)"); ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.set_ylim(0, 1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig9_first_stage_new_sets.png", dpi=220); plt.close(fig)
print("figures written")

