"""Part 3 error analysis (pre-registered C4 and C5). Pure analysis of cached transcripts and results.

C4 — which words break. Each question is tagged with spaCy (en_core_web_sm); its words are aligned
     to the transcript with a Levenshtein alignment, and a reference word counts as an error when it
     is substituted or deleted. Content words: NOUN, PROPN, NUM, ADJ, VERB (not auxiliary); all other
     words are function words. Per (query, condition), Δ = nDCG@10(spoken) − nDCG@10(typed) for
     BM25 → ce-l6 at depth 25, and OLS fits Δ ~ content error rate + function error rate. The
     difference of the two coefficients gets a 95% CI from a query-cluster bootstrap.
C5 — where the loss happens. Among (query, condition) pairs where BM25 → ce-l6 loses nDCG@10, the
     share in which no relevant passage is left in the BM25 top-25 pool of the spoken query.

Writes results/error_analysis.md and results/error_analysis.json.
Usage: python -I error_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech  # noqa: E402

KEY = "BM25 > ce-l6 @25"
POOL = 25
CONTENT = {"NOUN", "PROPN", "NUM", "ADJ", "VERB"}
BOOT = 2000
OUT_MD = HERE / "results" / "error_analysis.md"
OUT_JSON = HERE / "results" / "error_analysis.json"


def groups() -> dict[str, list[tuple[str, str]]]:
    """Analysis groups: name -> [(dataset, condition)], only where results and transcripts exist."""
    sd = data.cache_dir("sdqa")
    google = data.load_json(sd / "conditions.json", [])
    whisper = [c for c in data.load_json(sd / "conditions_whisper.json", []) if c.startswith("asr_wb_")]
    out = {"SD-QA, Whisper base.en": [("sdqa", c) for c in whisper],
           "SD-QA, Google ASR": [("sdqa", c) for c in google],
           "BEIR, wideband": [(d, "asr_clean") for d in data.DATASETS],
           "BEIR, telephone": [(d, "asr_phone") for d in data.DATASETS]}
    keep = {}
    for name, pairs in out.items():
        ok = [(d, c) for d, c in pairs
              if (data.cache_dir(d) / f"results_{c}.json").exists() and (data.cache_dir(d) / f"transcripts_{c}.json").exists()]
        if ok:
            keep[name] = ok
    return keep


def align_errors(ref: list[str], hyp: list[str]) -> list[bool]:
    """True for each reference word that is substituted or deleted in the best alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    err, i, j = [False] * n, n, m
    while i > 0:
        if j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            err[i - 1] = ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j] + 1:
            err[i - 1] = True; i -= 1
        else:
            j -= 1
    return err


def tag_question(nlp, text: str) -> list[tuple[str, bool, bool]]:
    """(normalised word, is_content, in_named_entity) for each word of the question."""
    doc = nlp(text)
    words = []
    for tok in doc:
        content = tok.pos_ in CONTENT and not (tok.pos_ == "VERB" and tok.dep_.startswith("aux"))
        for w in speech._words(tok.text):
            words.append((w, content, tok.ent_type_ != ""))
    return words


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]


def main() -> None:
    import spacy
    from bench.first_stage import BM25
    nlp = spacy.load("en_core_web_sm")
    report, lines = {}, ["# Part 3 error analysis (pre-registered C4 and C5)\n",
                         f"Δ = nDCG@10(spoken) − nDCG@10(typed) for {KEY}. Content words: NOUN, PROPN, NUM, ADJ, "
                         "non-auxiliary VERB. Bootstrap: 2,000 resamples of queries (clusters).\n"]
    tags_cache: dict[tuple[str, str], list] = {}
    for gname, pairs in groups().items():
        rows = []          # (cluster id, delta, content rate, function rate, entity error, recall failure)
        for ds_name, cond in pairs:
            ds = data.load(ds_name)
            bm25 = BM25(ds)
            typed = data.load_json(data.cache_dir(ds_name) / "results_text.json")["rows"][KEY]
            spoken = data.load_json(data.cache_dir(ds_name) / f"results_{cond}.json")["rows"][KEY]
            trans = data.load_json(data.cache_dir(ds_name) / f"transcripts_{cond}.json")
            missing = set(data.load_json(data.cache_dir("sdqa") / "missing_audio.json", [])) if ds_name == "sdqa" else set()
            for qid, text in ds.queries.items():
                if qid in missing:
                    continue                      # no recording for one dialect: dropped everywhere
                if qid not in spoken or qid not in typed or qid not in trans:
                    continue
                key = (ds_name, qid)
                if key not in tags_cache:
                    tags_cache[key] = tag_question(nlp, text)
                tagged = tags_cache[key]
                if not tagged:
                    continue
                err = align_errors([w for w, _, _ in tagged], speech._words(trans[qid]))
                c = [e for e, (_, ct, _) in zip(err, tagged) if ct]
                f = [e for e, (_, ct, _) in zip(err, tagged) if not ct]
                ent = [e for e, (_, _, en) in zip(err, tagged) if en]
                pool = [d for d, _ in bm25.search(trans[qid], POOL)]
                rows.append((f"{ds_name}:{qid}", spoken[qid]["ndcg@10"] - typed[qid]["ndcg@10"],
                             np.mean(c) if c else np.nan, np.mean(f) if f else np.nan,
                             float(any(ent)) if ent else np.nan,
                             float(not any(d in ds.qrels[qid] for d in pool))))
        clusters = np.array([r[0] for r in rows])
        delta, crate, frate, eerr, rfail = (np.array([r[i] for r in rows], dtype=float) for i in range(1, 6))
        ok = ~np.isnan(crate) & ~np.isnan(frate)
        beta = ols(delta[ok], np.column_stack([crate[ok], frate[ok]]))
        uniq = np.unique(clusters[ok])
        idx_by = {u: np.where((clusters == u) & ok)[0] for u in uniq}
        rng = np.random.default_rng(0)
        diffs, bc, bf = [], [], []
        for _ in range(BOOT):
            take = np.concatenate([idx_by[u] for u in rng.choice(uniq, len(uniq))])
            b = ols(delta[take], np.column_stack([crate[take], frate[take]]))
            bc.append(b[1]); bf.append(b[2]); diffs.append(b[1] - b[2])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]   # noqa: E731
        lost = delta < 0
        lost_idx = np.where(lost)[0]
        boot_share = []
        lost_clusters = np.unique(clusters[lost_idx])
        lidx = {u: lost_idx[clusters[lost_idx] == u] for u in lost_clusters}
        for _ in range(BOOT):
            take = np.concatenate([lidx[u] for u in rng.choice(lost_clusters, len(lost_clusters))])
            boot_share.append(rfail[take].mean())
        e_ok = ~np.isnan(eerr)
        rep = {"pairs": int(len(rows)), "pairs_with_both_word_types": int(ok.sum()),
               "mean_content_error": float(np.nanmean(crate)), "mean_function_error": float(np.nanmean(frate)),
               "beta_content": float(beta[1]), "beta_content_ci": ci(bc),
               "beta_function": float(beta[2]), "beta_function_ci": ci(bf),
               "beta_diff": float(beta[1] - beta[2]), "beta_diff_ci": ci(diffs),
               "entity_error_mean_delta": float(delta[e_ok & (eerr == 1)].mean()) if (e_ok & (eerr == 1)).any() else None,
               "entity_ok_mean_delta": float(delta[e_ok & (eerr == 0)].mean()) if (e_ok & (eerr == 0)).any() else None,
               "lost_pairs": int(lost.sum()), "recall_failure_share": float(rfail[lost].mean()) if lost.any() else None,
               "recall_failure_share_ci": ci(boot_share) if lost.any() else None}
        report[gname] = rep
        print(gname, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}), flush=True)
    lines += ["\n## C4. Which words break\n",
              "| Group | Pairs | Content error rate | Function error rate | β content [95% CI] | β function [95% CI] | β content − β function [95% CI] |",
              "|---|---|---|---|---|---|---|"]
    f2 = lambda v, c: f"{v:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"   # noqa: E731
    for g, r in report.items():
        lines.append(f"| {g} | {r['pairs_with_both_word_types']} | {r['mean_content_error']:.3f} | {r['mean_function_error']:.3f} | "
                     f"{f2(r['beta_content'], r['beta_content_ci'])} | {f2(r['beta_function'], r['beta_function_ci'])} | "
                     f"{f2(r['beta_diff'], r['beta_diff_ci'])} |")
    lines += ["\nNamed entities (secondary): mean Δ when any entity word is misrecognised vs when none is.\n",
              "| Group | Δ, entity word wrong | Δ, entity words right |", "|---|---|---|"]
    for g, r in report.items():
        a, b = r["entity_error_mean_delta"], r["entity_ok_mean_delta"]
        lines.append(f"| {g} | {a:+.3f} | {b:+.3f} |" if a is not None and b is not None else f"| {g} | — | — |")
    lines += ["\n## C5. Where the loss happens\n",
              "Share of (query, condition) pairs that lose nDCG@10 in which no relevant passage is left in the "
              "BM25 top-25 of the spoken query (the reranker never sees it).\n",
              "| Group | Pairs that lose | Recall-failure share [95% CI] |", "|---|---|---|"]
    for g, r in report.items():
        if r["recall_failure_share"] is not None:
            c = r["recall_failure_share_ci"]
            lines.append(f"| {g} | {r['lost_pairs']} | {r['recall_failure_share']:.3f} [{c[0]:.3f}, {c[1]:.3f}] |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    data.save_json(OUT_JSON, report)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
