"""Figures for the IEEE version of the paper (reads cached results only; no model is run).

Same data as make_figures.py, redrawn for IEEEtran: vector PDF, 8 pt type, single-column (3.5 in)
or double-column (7.16 in) width, and short axis and legend labels.
Writes ../paper/ieee/figures/*.pdf.  Usage (from benchmark/): python -I make_figures_ieee.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import analyze_part3 as P3  # noqa: E402
import analyze_sdqa as S  # noqa: E402
from bench import data  # noqa: E402

OUT = HERE.parent / "paper" / "ieee" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ONE, TWO = 3.5, 7.16                       # IEEE column and page widths (inches)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
                     "legend.frameon": False, "legend.handlelength": 1.0, "legend.columnspacing": 0.9,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
GREY, BLUE, ORANGE, RED, GREEN, PURPLE = "#7f8c8d", "#2b6cb0", "#dd8a1f", "#c0392b", "#27ae60", "#8e44ad"
CONDS = ("text", "asr_clean", "asr_phone")
CLABEL = {"text": "Typed", "asr_clean": "Wideband", "asr_phone": "Telephone"}
CCOL = {"text": BLUE, "asr_clean": ORANGE, "asr_phone": RED}
DNAME = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}
NDCG = "nDCG@10"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)


def legend_top(ax_or_fig, ncol, y=1.0, **kw):
    target = ax_or_fig
    return target.legend(ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, y), **kw)


# Fig. 1 — robustness of each pipeline (double column)
systems = [("BM25", "BM25"), ("Dense (bge-small)", "Dense"), ("Hybrid RRF", "Hybrid"), ("BM25 > ce-l6 @25", "+CE")]
fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.4))
for ax, name in zip(axes, data.DATASETS):
    for i, cond in enumerate(CONDS):
        rows = analyze.load_results(name, cond)["rows"]; qids = sorted(rows["BM25"])
        vals = [analyze.per_query(rows, key, "ndcg@10", qids).mean() for key, _ in systems]
        ax.bar(np.arange(len(systems)) + (i - 1) * 0.27, vals, 0.27, color=CCOL[cond], label=CLABEL[cond])
    ax.set_xticks(range(len(systems))); ax.set_xticklabels([s for _, s in systems])
    ax.set_title(DNAME[name], pad=2)
axes[0].set_ylabel(NDCG)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.97))
fig.tight_layout(pad=0.3, w_pad=0.8); save(fig, "fig1_robustness")


def variant_bars(ax, variants, score, width):
    for j, (lab, col, per_cond) in enumerate(variants):
        means = [np.mean(per_cond(c)) for c in CONDS]
        sds = [np.std(per_cond(c), ddof=1) if len(per_cond(c)) > 1 else 0 for c in CONDS]
        ax.bar(np.arange(3) + (j - (len(variants) - 1) / 2) * width, means, width, yerr=sds, capsize=1.2,
               error_kw={"elinewidth": 0.6, "capthick": 0.6}, label=lab, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels([CLABEL[c] for c in CONDS]); ax.set_ylabel(score)


# Fig. 2 — MiniLM fine-tuning variants, macro over collections, mean ± s.d. over seeds (protocol B)
pattern = analyze.PROTOCOLS["B. fixed budget (3 epochs)"]


def ft_macro(v):
    def per_cond(cond):
        out = []
        for seed in (analyze.SEEDS if v != "zero" else (None,)):
            vals = []
            for name in data.DATASETS:
                rows = analyze.load_results(name, cond)["rows"]; qids = sorted(rows["BM25"])
                key = "BM25 > ce-l6 @25" if v == "zero" else f"BM25 > {pattern.format(v=v, s=seed)} @25"
                vals.append(analyze.per_query(rows, key, "ndcg@10", qids).mean())
            out.append(np.mean(vals))
        return out
    return per_cond


fig, ax = plt.subplots(figsize=(ONE, 1.75))
variant_bars(ax, [("Zero-shot", GREY, ft_macro("zero")), ("Clean", BLUE, ft_macro("clean")),
                  ("Noisy", RED, ft_macro("noisy")), ("Mixed", GREEN, ft_macro("mixed")),
                  ("Distill", PURPLE, ft_macro("distill"))], NDCG, 0.16)
ax.set_ylim(0.2, 0.45); legend_top(ax, 5)
fig.tight_layout(pad=0.3); save(fig, "fig2_finetuning")

# Fig. 3 — re-ranking depth vs nDCG@10 and CPU latency (double column)
_, curves = analyze.depth_table()
lat = data.load_json(analyze.OUT / "latency.json", {})
fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.35))
for ax, (name, pts) in zip(axes, curves.items()):
    a, = ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=2.5, color=BLUE, label=NDCG)
    ax.set_title(DNAME.get(name, name), pad=2); ax.set_xlabel("Re-rank depth")
    cpu = lat.get(name, {}).get("cpu", {}).get(analyze.TAG, {})
    if cpu:
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
        ds_ = sorted(int(d) for d in cpu)
        b, = ax2.plot(ds_, [cpu[str(d)]["p50"] / 1000 for d in ds_], "s--", ms=2.5, color=RED, label="CPU latency (s)")
        ax2.tick_params(axis="y", colors=RED, labelsize=7)
        if name == list(curves)[-1]:
            ax2.set_ylabel("Latency (s)", color=RED)
axes[0].set_ylabel(NDCG)
fig.legend([a, b], [NDCG, "CPU latency (s)"], ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
fig.tight_layout(pad=0.3, w_pad=1.2); save(fig, "fig3_depth_latency")

# Figs. 4-5 — SD-QA real speech
cache = data.cache_dir("sdqa")
sconds = data.load_json(cache / "conditions.json", [])
sres = {c: data.load_json(cache / f"results_{c}.json") for c in ["text"] + sconds}
sq = sorted(sres["text"]["rows"]["BM25"])
nd = lambda c, k: S.per_query(sres[c]["rows"], k, "ndcg@10", sq)  # noqa: E731
typed = nd("text", "BM25 > ce-l6 @25").mean()
by = {}
for c in sconds:
    d, m = S.parse(c)
    by.setdefault(d, {})["US" if m == "en-US" else "loc"] = nd(c, "BM25 > ce-l6 @25").mean()
order = sorted(by, key=lambda d: -max(by[d].values()))
SHORT = {"South Africa": "S. Africa", "New Zealand": "NZ", "Philippines": "Phil.", "India-North": "India-N",
         "India-South": "India-S"}
fig, ax = plt.subplots(figsize=(TWO, 1.4))
x = np.arange(len(order))
ax.bar(x - 0.2, [by[d].get("loc", np.nan) for d in order], 0.4, color=BLUE, label="Local ASR")
ax.bar(x + 0.2, [by[d]["US"] for d in order], 0.4, color=ORANGE, label="en-US ASR")
ax.axhline(typed, color="#333", ls="--", lw=0.7, label="Typed")
ax.set_xticks(x); ax.set_xticklabels([SHORT.get(S.DIALECT[d], S.DIALECT[d]) for d in order])
for i, d in enumerate(order):
    if d.startswith("ind"):
        ax.get_xticklabels()[i].set_color(RED); ax.get_xticklabels()[i].set_fontweight("bold")
ax.set_ylim(0.4, 0.6); ax.set_ylabel(NDCG); legend_top(ax, 3)
fig.tight_layout(pad=0.3); save(fig, "fig4_sdqa_dialects")

groups = {"Typed": ["text"], "Speech": sconds,
          "Indian": [c for c in sconds if S.parse(c)[0].startswith("ind")],
          "Indian\n(en-IN)": [c for c in sconds if S.parse(c)[0].startswith("ind") and S.parse(c)[1] == "en-IN"]}
pipes = [("BM25 > ce-l6 @25", "BM25+CE", GREY), ("Dense (bge-small)", "Dense", ORANGE),
         ("Hybrid > ce-l6 @25", "Hybrid+CE", BLUE), ("Hybrid > ft-distill-s{s}-last @25", "Hybrid+Distill", PURPLE)]
fig, ax = plt.subplots(figsize=(ONE, 1.75))
for j, (key, lab, col) in enumerate(pipes):
    keys = [key.format(s=s) for s in (0, 1, 2)] if "{s}" in key else [key]
    vals = [np.mean([np.mean([nd(c, k) for k in keys], axis=0) for c in gc]) for gc in groups.values()]
    ax.bar(np.arange(len(groups)) + (j - 1.5) * 0.2, vals, 0.2, color=col, label=lab)
ax.set_xticks(range(len(groups))); ax.set_xticklabels(list(groups)); ax.set_ylim(0.4, 0.7)
ax.set_ylabel(NDCG); legend_top(ax, 4)
fig.tight_layout(pad=0.3); save(fig, "fig5_sdqa_pipelines")

# Fig. 6 — bge-reranker-base fine-tuning on BEIR (C2)


def bge_macro(tag):
    def per_cond(cond):
        seeds = (None,) if "{s}" not in tag else P3.SEEDS
        return [np.mean([P3.scores([(d, cond)], f"BM25 > {tag.format(s=sd)} @25").mean() for d in data.DATASETS])
                for sd in seeds]
    return per_cond


fig, ax = plt.subplots(figsize=(ONE, 1.75))
variant_bars(ax, [("Zero-shot", GREY, bge_macro("bge-base")), ("Clean", BLUE, bge_macro("ft-bgeb-clean-s{s}-last")),
                  ("Noisy", RED, bge_macro("ft-bgeb-noisy-s{s}-last")),
                  ("Distill", PURPLE, bge_macro("ft-bgeb-distill-s{s}-last"))], NDCG, 0.2)
ax.set_ylim(0.2, 0.45); legend_top(ax, 4)
fig.tight_layout(pad=0.3); save(fig, "fig6_bge_finetuning")

# Fig. 7 — which words break (C4)
rep = json.loads((HERE / "results" / "error_analysis.json").read_text())
GNAME = {"SD-QA, Whisper base.en": "SD-QA (Whisper)", "SD-QA, Google ASR": "SD-QA (Google)",
         "BEIR, wideband": "BEIR wideband", "BEIR, telephone": "BEIR telephone"}
names = list(rep)
fig, ax = plt.subplots(figsize=(ONE, 1.75))
y = np.arange(len(names))
for off, key, lab, col in ((-0.18, "beta_content", "Content words", RED), (0.18, "beta_function", "Function words", GREY)):
    vals = np.array([rep[n][key] for n in names]); ci = np.array([rep[n][key + "_ci"] for n in names])
    ax.barh(y + off, vals, 0.34, color=col, label=lab, xerr=[vals - ci[:, 0], ci[:, 1] - vals], capsize=1.2,
            error_kw={"elinewidth": 0.6, "capthick": 0.6})
ax.axvline(0, color="#333", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([GNAME.get(n, n) for n in names])
ax.invert_yaxis(); ax.set_xlabel(r"$\Delta$ " + f"{NDCG} per unit WER"); legend_top(ax, 2)
fig.tight_layout(pad=0.3); save(fig, "fig7_word_errors")

# Fig. 8 — real vs synthetic training noise (C3), double column
g3 = P3.test_groups()
h4 = [("ce-l6", "Zero-shot", GREY), ("ft-hq-clean-s{s}-last", "Clean", BLUE),
      ("ft-hq-synth-s{s}-last", "Noisy (TTS)", "#f2a541"), ("ft-hq-real-s{s}-last", "Noisy (real)", RED),
      ("ft-hq-dsynth-s{s}-last", "Distill (TTS)", "#b39ddb"), ("ft-hq-dreal-s{s}-last", "Distill (real)", "#6b4e9b")]
sets8 = [("SD-QA-Whisper (11 dialects)", "SD-QA"), ("SD-QA-Whisper, Indian", "SD-QA (Indian)"),
         ("SLUE-SQA-5 test", "SLUE-SQA-5"), ("HeySQuAD-dev", "HeySQuAD")]
fig, ax = plt.subplots(figsize=(TWO, 1.35))
for j, (tag, lab, col) in enumerate(h4):
    ax.bar(np.arange(len(sets8)) + (j - 2.5) * 0.13, [P3.scores(g3[s], f"BM25 > {tag} @25").mean() for s, _ in sets8],
           0.13, label=lab, color=col)
ax.set_xticks(range(len(sets8))); ax.set_xticklabels([s for _, s in sets8])
ax.set_ylabel(NDCG); ax.set_ylim(0.3, 0.95); legend_top(ax, 6)
fig.tight_layout(pad=0.3); save(fig, "fig8_real_vs_synth")

# Fig. 9 — first stage on the new real-speech test sets (C1)
sets9 = [("SD-QA-Whisper (11 dialects)", "SD-QA"), ("HeySQuAD-dev", "HeySQuAD"), ("SLUE-SQA-5 test", "SLUE-SQA-5")]
p9 = [("BM25 > ce-l6 @25", "BM25+MiniLM", GREY), ("BM25 > bge-m3 @25", "BM25+M3", "#f2a541"),
      ("Hybrid > ce-l6 @25", "Hybrid+MiniLM", BLUE), ("Hybrid > bge-m3 @25", "Hybrid+M3", "#6b4e9b")]
fig, ax = plt.subplots(figsize=(ONE, 1.75))
for j, (key, lab, col) in enumerate(p9):
    ax.bar(np.arange(len(sets9)) + (j - 1.5) * 0.2, [P3.scores(g3[s], key).mean() for s, _ in sets9], 0.2,
           label=lab, color=col)
ax.set_xticks(range(len(sets9))); ax.set_xticklabels([s for _, s in sets9])
ax.set_ylabel(NDCG); ax.set_ylim(0, 1.0); legend_top(ax, 2)
fig.tight_layout(pad=0.3); save(fig, "fig9_first_stage")
print("IEEE figures written to", OUT)


# Fig. 10 — Part 4 per SD-QA dialect: pass 1, Listen Twice, LT + MC, oracle prompt (double column)
import part4 as P4  # noqa: E402
drop = set(data.load_json(data.cache_dir("sdqa") / "missing_audio.json", []))
DIAL = {"aus": "AUS", "gbr": "GBR", "ind_n": "IND-N", "ind_s": "IND-S", "irl": "IRL", "kenya": "KEN",
        "nga": "NGA", "nzl": "NZL", "phl": "PHL", "usa": "USA", "zaf": "ZAF"}
bars = (("base", "Pass 1", GREY), ("mc", "MC", ORANGE), ("lt", "Listen Twice", BLUE),
        ("ltmc", "LT + MC", GREEN), ("orc", "Oracle prompt", PURPLE))
fig, ax = plt.subplots(figsize=(TWO, 1.5))
for i, (sysn, label, col) in enumerate(bars):
    vals = []
    for suffix, _ in P4.units("sdqa"):
        rows = data.load_json(data.cache_dir("sdqa") / f"p4res{suffix}_{sysn}.json")["rows"]["Hybrid > ce-l6 @25"]
        vals.append(np.mean([m["ndcg@10"] for q, m in rows.items() if q not in drop]))
    ax.bar(np.arange(len(DIAL)) + (i - 2) * 0.16, vals, 0.16, color=col, label=label)
ax.set_xticks(range(len(DIAL))); ax.set_xticklabels(DIAL.values())
ax.set_ylim(0.4, 0.68); ax.set_ylabel(NDCG)
legend_top(ax, 5)
save(fig, "fig10_listen_twice")
