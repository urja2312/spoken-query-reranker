"""Part 4: Listen Twice (retrieval-guided re-transcription) and Mishear the Corpus (ASR-simulated
document expansion). Settings and tests fixed in paper/PREREGISTRATION_PART4.md.

Stages, per test set (each cached and resumable; run on a GPU machine):

  mc       vocabulary of corpus rare words -> Piper TTS (2 voices) -> Whisper base.en -> misheard
           variants; builds the MC-expanded BM25 index and the Metaphone (PH) baseline index.
  prompts  Whisper prompts from pass-1 retrieval (LT), random passages (RND), the relevant passages
           (ORC) and MC-expanded retrieval (LT+MC); K ablations on HeySQuAD.
  decode   pass-2 Whisper decodes of the real recordings with each prompt, plus a pass-1 re-decode
           identity check on the first 100 recordings of each unit.
  eval     every system through lexical/hybrid first stage and the reranker seam.

Usage: python -I part4.py <sdqa|heysquad|slue> [stage ...]      (default: all four stages)
       P4_SMOKE=1 limits every stage to a handful of queries and words.
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
import sqlite3
import sys
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, metrics, speech  # noqa: E402
from bench.first_stage import BM25, DEPTH, Dense, oncall_kb, rrf  # noqa: E402

SMOKE = os.environ.get("P4_SMOKE") == "1"
K_PASSAGES, N_WORDS = 5, 30
RARE_ZIPF, MIN_LEN = 4.0, 4
VOCAB_CAP = 200 if SMOKE else 40000
MC_VOICES = ("en_US-lessac-medium", "en_GB-alan-medium")
POOL = 25
ABLATION_K = (1, 3, 10)
CHECK_N = 100
BIG = "bge-m3"
BIG_SYSTEMS = {"typed", "base", "lt", "mc", "ltmc"}
SDQA_DIALECTS = ("aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf")
_WORD = re.compile(r"[A-Za-z]+")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def units(name: str) -> list[tuple[str, str]]:
    """(suffix, pass-1 condition) per recording set: one per dialect on SD-QA, one otherwise."""
    if name == "sdqa":
        return [(f"_{d}", f"asr_wb_{d}") for d in SDQA_DIALECTS]
    return [("", "asr_wb")]


def load_ds(name: str) -> data.Dataset:
    ds = data.load(name)
    if SMOKE:
        keep = sorted(ds.queries)[:12]
        ds.queries = {q: ds.queries[q] for q in keep}
        ds.qrels = {q: ds.qrels[q] for q in keep}
    return ds


# ----------------------------------------------------------------------------------- rare words


@lru_cache(maxsize=None)
def is_rare(word: str) -> bool:
    from wordfreq import zipf_frequency
    return (len(word) >= MIN_LEN and word not in oncall_kb._STOP
            and zipf_frequency(word, "en") < RARE_ZIPF)


def rare_words(text: str) -> dict[str, str]:
    """lowercase rare word -> surface form (a capitalised form wins, since names matter)."""
    out: dict[str, str] = {}
    for w in _WORD.findall(text):
        low = w.lower()
        if is_rare(low) and (low not in out or (w[0].isupper() and not out[low][0].isupper())):
            out[low] = w
    return out


def doc_frequency(ds: data.Dataset) -> dict[str, int]:
    path = data.cache_dir(ds.name) / "p4_df.json"
    df = data.load_json(path)
    if df is None:
        df = {}
        for doc_id in ds.corpus:
            for w in rare_words(ds.doc_text(doc_id)):
                df[w] = df.get(w, 0) + 1
        data.save_json(path, df)
    return df


def bias_prompt(ds: data.Dataset, doc_ids: list[str], df: dict[str, int], n: int = N_WORDS) -> str:
    """The N rarest content words of the passages, ranked by (passages containing it) x IDF."""
    count: dict[str, int] = {}
    surface: dict[str, str] = {}
    order: dict[str, int] = {}
    for doc_id in doc_ids:
        for low, w in rare_words(ds.doc_text(doc_id)).items():
            count[low] = count.get(low, 0) + 1
            order.setdefault(low, len(order))
            if low not in surface or (w[0].isupper() and not surface[low][0].isupper()):
                surface[low] = w
    total = len(ds.corpus)
    ranked = sorted(count, key=lambda w: (-count[w] * math.log(total / df.get(w, 1)), order[w]))
    return ", ".join(surface[w] for w in ranked[:n])


# ----------------------------------------------------------------------------------- lexical indexes


class IndexBM25(BM25):
    """Production BM25 search (same SQL, same query builder) over a different FTS5 file."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)


def _write_index(path: Path, rows) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as db:
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, section, doc_title, doc_id UNINDEXED)")
        db.executemany("INSERT INTO docs(body, section, doc_title, doc_id) VALUES (?, ?, ?, ?)", rows)
        db.commit()
    tmp.replace(path)


@lru_cache(maxsize=None)
def code(word: str) -> str:
    import jellyfish
    c = jellyfish.metaphone(word).lower()
    return f"q{c}" if c else ""


class Phonetic:
    """Classic phonetic indexing baseline: BM25 over Metaphone codes."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)

    def search(self, query: str, k: int = DEPTH) -> list[tuple[str, float]]:
        codes = list(dict.fromkeys(c for c in (code(t) for t in oncall_kb._query_tokens(query) if t.isalpha()) if c))
        if not codes:
            return []
        rows = self.db.execute("SELECT doc_id, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                               (oncall_kb._match_expr(codes), k)).fetchall()
        return [(str(d), -float(s)) for d, s in rows]


def _tts_job(job):
    word, voice = job
    try:
        return word, voice, (speech.synthesize_piper(word, voice) * 32767).astype(np.int16)
    except Exception as exc:                                   # noqa: BLE001 — logged, word skipped
        return word, voice, repr(exc)


def variants_of(word: str, heard: str) -> list[str]:
    """Misheard tokens for one ASR transcript of an isolated word (pre-registered filter)."""
    toks = speech._words(heard)
    if not toks or toks == [word] or speech.is_hallucination(word, heard):
        return []
    out = [t.replace("'", "") for t in toks]
    out = [t for t in out if len(t) >= 2 and t not in oncall_kb._STOP and t != word]
    if len(toks) > 1 and "".join(toks) != word:
        out.append("".join(t.replace("'", "") for t in toks))
    return list(dict.fromkeys(out))


def stage_mc(name: str) -> None:
    ds = data.load(name)
    cache = data.cache_dir(name)
    df = doc_frequency(ds)
    vocab = sorted(df, key=lambda w: (-df[w], w))[:VOCAB_CAP]
    heard_path = cache / "p4_heard.json"
    heard: dict[str, dict[str, str]] = data.load_json(heard_path, {})
    jobs = [(w, v) for w in vocab for v in MC_VOICES if v not in heard.get(w, {})]
    log(f"mc {name}: {len(df)} rare words in corpus, vocabulary {len(vocab)}, {len(jobs)} TTS->ASR jobs to run")
    if jobs:
        import multiprocessing as mp
        from huggingface_hub import hf_hub_download
        for v in MC_VOICES:                       # download only; each worker loads its own voice
            lang, voice, quality = v.split("-")
            sub = f"{lang.split('_')[0]}/{lang}/{voice}/{quality}/{v}.onnx"
            hf_hub_download("rhasspy/piper-voices", sub); hf_hub_download("rhasspy/piper-voices", sub + ".json")
        pool = mp.get_context("fork").Pool(4)     # forked before CUDA is touched in this process
        rec = speech.Recognizer("base.en", device=data.best_device())
        started, failed = time.time(), 0
        for i, (w, v, audio) in enumerate(pool.imap(_tts_job, jobs, chunksize=16), 1):
            if isinstance(audio, str):
                failed += 1
                heard.setdefault(w, {})[v] = ""
            else:
                heard.setdefault(w, {})[v] = rec(audio.astype(np.float32) / 32767)
            if i % 5000 == 0:
                data.save_json(heard_path, heard)
                log(f"  mc {name}: {i}/{len(jobs)} ({time.time() - started:.0f}s, {failed} TTS failures)")
        pool.close()
        data.save_json(heard_path, heard)
        log(f"  mc {name}: TTS->ASR done in {time.time() - started:.0f}s, {failed} TTS failures")
    var = {w: list(dict.fromkeys(t for v in MC_VOICES for t in variants_of(w, heard.get(w, {}).get(v, ""))))
           for w in vocab}
    var = {w: t for w, t in var.items() if t}
    rows, added = [], 0
    for doc_id, doc in ds.corpus.items():
        exp = list(dict.fromkeys(t for w in rare_words(ds.doc_text(doc_id)) for t in var.get(w, ())))
        added += len(exp)
        rows.append((doc["text"], " ".join(exp), doc["title"], doc_id))
    _write_index(cache / "bm25_mc.sqlite", rows)
    ph_rows = [(" ".join(c for c in (code(w.lower()) for w in _WORD.findall(ds.doc_text(d))) if c), "", "", d)
               for d in ds.corpus]
    _write_index(cache / "bm25_ph.sqlite", ph_rows)
    words = sum(len(_WORD.findall(ds.doc_text(d))) for d in ds.corpus)
    stats = {"rare_words_in_corpus": len(df), "vocabulary": len(vocab), "words_with_variants": len(var),
             "share_misheard": len(var) / max(1, len(vocab)), "expansion_tokens": added,
             "expansion_tokens_per_doc": added / len(ds.corpus), "corpus_words": words,
             "index_growth": added / max(1, words),
             "examples": {w: var[w] for w in sorted(var, key=lambda w: (-df[w], w))[:40]}}
    data.save_json(cache / "p4_mc_stats.json", stats)
    log(f"mc {name}: {len(var)}/{len(vocab)} words misheard by ASR; +{added} expansion tokens "
        f"({stats['index_growth']:.1%} of corpus words)")


# ----------------------------------------------------------------------------------- prompts


class Retrieval:
    def __init__(self, ds: data.Dataset):
        cache = data.cache_dir(ds.name)
        self.ds = ds
        self.bm25 = BM25(ds)
        self.mc = IndexBM25(cache / "bm25_mc.sqlite") if (cache / "bm25_mc.sqlite").exists() else None
        self.ph = Phonetic(cache / "bm25_ph.sqlite") if (cache / "bm25_ph.sqlite").exists() else None
        self.dense = Dense(ds)
        self._dense: dict[str, dict] = {}

    def lexical(self, kind: str, query: str) -> list[tuple[str, float]]:
        if kind == "bm25":
            return self.bm25.search(query, DEPTH)
        if kind == "mc":
            return self.mc.search(query, DEPTH)
        return rrf(self.bm25.search(query, DEPTH), self.ph.search(query, DEPTH))

    def dense_runs(self, cond: str, queries: dict[str, str]) -> dict:
        if cond not in self._dense:
            self._dense[cond] = self.dense.search_many(queries, DEPTH)
        return self._dense[cond]


def transcripts(name: str, cond: str) -> dict[str, str]:
    return data.load_json(data.cache_dir(name) / f"transcripts_{cond}.json", {})


def stage_prompts(name: str) -> None:
    ds = load_ds(name)
    df = doc_frequency(ds)
    R = Retrieval(ds)
    ids = sorted(ds.corpus)
    for suffix, p1 in units(name):
        t1 = transcripts(name, p1)
        q1 = {q: t1.get(q, "") for q in ds.queries}
        dn = R.dense_runs(p1, q1)
        out: dict[str, dict[str, str]] = {"lt": {}, "rnd": {}, "orc": {}, "ltmc": {}}
        if name == "heysquad":
            out.update({f"lt_k{k}": {} for k in ABLATION_K})
        for qid, text in q1.items():
            hy = [d for d, _ in rrf(R.lexical("bm25", text), dn[qid])]
            out["lt"][qid] = bias_prompt(ds, hy[:K_PASSAGES], df)
            for k in ABLATION_K:
                if f"lt_k{k}" in out:
                    out[f"lt_k{k}"][qid] = bias_prompt(ds, hy[:k], df)
            rng = random.Random(f"rnd|{name}|{suffix}|{qid}")
            rnd: list[str] = []
            while len(rnd) < K_PASSAGES:
                d = ids[rng.randrange(len(ids))]
                if d not in ds.qrels[qid] and d not in rnd:
                    rnd.append(d)
            out["rnd"][qid] = bias_prompt(ds, rnd, df)
            out["orc"][qid] = bias_prompt(ds, sorted(ds.qrels[qid]), df)
            if R.mc is not None:
                hy_mc = [d for d, _ in rrf(R.lexical("mc", text), dn[qid])]
                out["ltmc"][qid] = bias_prompt(ds, hy_mc[:K_PASSAGES], df)
        data.save_json(data.cache_dir(name) / f"p4_prompts{suffix}.json", out)
        log(f"prompts {name}{suffix}: {', '.join(f'{k}={len(v)}' for k, v in out.items())}; "
            f"example LT prompt: {next(iter(out['lt'].values()), '')[:160]!r}")


# ----------------------------------------------------------------------------------- audio + decode


def audio_items(name: str, suffix: str, qids: set[str]):
    """(qid, mono float32 16 kHz) for each real recording of the unit's questions."""
    import part3_data as p3
    if name == "sdqa":
        repo = HERE / "data" / "sdqa_audio"
        if not (repo / "test" / "eng").is_dir():
            import subprocess
            log("sparse-cloning SD-QA English test audio")
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", p3.SDQA_REPO, str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "test/eng"], check=True)
        for p in sorted(glob.glob(str(repo / "test" / "eng" / suffix[1:] / "wav_eng" / "*.wav"))):
            if Path(p).stem in qids:
                yield Path(p).stem, speech.load_audio(Path(p).read_bytes())
    elif name == "heysquad":
        meta = data.load_json(data.cache_dir("heysquad") / "meta.json")
        groups: dict[tuple, set] = {}
        for r in meta["dev"]:
            if r["id"] in qids:
                groups.setdefault((r["file"], r["rg"]), set()).add(r["id"])
        for (f, g), want in sorted(groups.items()):
            for row in p3._read(f, ["id", "audio"], [g]):
                if row["id"] in want:
                    yield row["id"], speech.load_audio(row["audio"]["bytes"])
    else:
        files = sorted(p3._hf_files(p3.SLUE_REPO, "sqa5/test-*.parquet"))
        for f in files:
            for row in p3._read(f, ["question_id", "question_audio"]):
                if row["question_id"] in qids:
                    yield row["question_id"], speech.load_audio(row["question_audio"]["bytes"])


def stage_decode(name: str) -> None:
    ds = load_ds(name)
    rec = speech.Recognizer("base.en", device=data.best_device())
    cache = data.cache_dir(name)
    timing = data.load_json(cache / "p4_timing.json", {})
    for suffix, p1 in units(name):
        prompts = data.load_json(cache / f"p4_prompts{suffix}.json")
        systems = [s for s, v in prompts.items() if v]
        out = {s: transcripts(name, f"p4{s}{suffix}") for s in systems}
        check = transcripts(name, f"p4check{suffix}")
        check_ids = set(sorted(ds.queries)[:CHECK_N])
        todo = {q for q in ds.queries if any(q not in out[s] for s in systems) or (q in check_ids and q not in check)}
        spent = {s: 0.0 for s in systems}
        seen = set()
        started = time.time()
        for n, (qid, audio) in enumerate(audio_items(name, suffix, todo), 1):
            seen.add(qid)
            for s in systems:
                if qid not in out[s]:
                    t = time.perf_counter()
                    out[s][qid] = rec(audio, prompt=prompts[s][qid] or None)
                    spent[s] += time.perf_counter() - t
            if qid in check_ids and qid not in check:
                check[qid] = rec(audio)
            if n % 250 == 0:
                for s in systems:
                    data.save_json(cache / f"transcripts_p4{s}{suffix}.json", out[s])
                log(f"  decode {name}{suffix}: {n} recordings ({time.time() - started:.0f}s)")
        for qid in ds.queries:                           # no recording: empty transcript, as in pass 1
            for s in systems:
                out[s].setdefault(qid, "")
        for s in systems:
            data.save_json(cache / f"transcripts_p4{s}{suffix}.json", out[s])
        data.save_json(cache / f"transcripts_p4check{suffix}.json", check)
        t1 = transcripts(name, p1)
        same = [check[q] == t1.get(q) for q in check]
        timing[suffix or "all"] = {"recordings": len(seen), "seconds_per_decode": {s: spent[s] / max(1, len(seen)) for s in systems},
                                   "pass1_redecode_identical": sum(same) / max(1, len(same))}
        data.save_json(cache / "p4_timing.json", timing)
        log(f"decode {name}{suffix}: {len(seen)} recordings, pass-1 re-decode identical on "
            f"{sum(same)}/{len(same)}; WER " + ", ".join(
                f"{s} {np.mean([speech.wer(ds.queries[q], out[s][q]) for q in seen]):.3f}" for s in systems)
            + f", pass1 {np.mean([speech.wer(ds.queries[q], t1.get(q, '')) for q in seen]):.3f}")


# ----------------------------------------------------------------------------------- evaluation


def systems_for(name: str, suffix: str, p1: str) -> dict[str, tuple[list[str], str]]:
    """system -> (transcript conditions fused in the first stage, lexical index). The reranker
    reads the last condition's text."""
    s = {"base": ([p1], "bm25"), "mc": ([p1], "mc"), "ph": ([p1], "ph"),
         "lt": ([f"p4lt{suffix}"], "bm25"), "rnd": ([f"p4rnd{suffix}"], "bm25"),
         "orc": ([f"p4orc{suffix}"], "bm25"), "ltmc": ([f"p4ltmc{suffix}"], "mc"),
         "ltfuse": ([p1, f"p4lt{suffix}"], "bm25")}
    if name == "heysquad":
        s.update({f"lt_k{k}": ([f"p4lt_k{k}{suffix}"], "bm25") for k in ABLATION_K})
    return s


def _metrics(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    m = metrics.evaluate(ranked, rel)
    for k in (5, 25):
        m[f"recall@{k}"] = len(set(ranked[:k]) & set(rel)) / len(rel)
    return m


def eval_system(R: Retrieval, scorers: dict, conds: list[str], kind: str, out_path: Path) -> None:
    from bench.rerankers import score_pool, seam_rerank
    ds = R.ds
    qsets = []
    for c in conds:
        t = dict(ds.queries) if c == "text" else transcripts(ds.name, c)
        qsets.append((c, {q: t.get(q, "") for q in ds.queries}))
    lex_runs, hy_runs = {}, {}
    for qid in ds.queries:
        lex = [R.lexical(kind, qs[qid]) for _, qs in qsets]
        den = [R.dense_runs(c, qs)[qid] for c, qs in qsets]
        lex_runs[qid] = lex[0] if len(lex) == 1 else rrf(*lex)
        hy_runs[qid] = rrf(*lex, *den)
    cond, queries = qsets[-1]
    pools = {q: list(dict.fromkeys([d for d, _ in lex_runs[q][:POOL]] + [d for d, _ in hy_runs[q][:POOL]])) for q in queries}
    rows: dict[str, dict[str, dict]] = {"Lexical": {}, "Hybrid": {}}
    for q in queries:
        rows["Lexical"][q] = _metrics([d for d, _ in lex_runs[q]], ds.qrels[q])
        rows["Hybrid"][q] = _metrics([d for d, _ in hy_runs[q]], ds.qrels[q])
    for tag, scorer in scorers.items():
        sc = score_pool(ds, scorer, cond, queries, pools)
        for first, runs in (("Lexical", lex_runs), ("Hybrid", hy_runs)):
            rows[f"{first} > {tag} @{POOL}"] = {
                q: _metrics(seam_rerank(queries[q], runs[q], sc.get(q, {}), depth=POOL, tag=tag), ds.qrels[q]) for q in queries}
    data.save_json(out_path, {"conds": conds, "lexical": kind, "rows": rows})
    log(f"  eval {out_path.stem}: Hybrid > ce-l6 nDCG@10 = "
        f"{np.mean([r['ndcg@10'] for r in rows[f'Hybrid > ce-l6 @{POOL}'].values()]):.4f}")


def stage_eval(name: str) -> None:
    from bench.rerankers import CrossEncoderScorer
    os.environ["BENCH_FP16"] = "1"
    ds = load_ds(name)
    R = Retrieval(ds)
    cache = data.cache_dir(name)
    small = {"ce-l6": CrossEncoderScorer("ce-l6")}
    both = {**small, BIG: CrossEncoderScorer(BIG)}
    jobs = [("", "typed", (["text"], "bm25")), ("", "typed_mc", (["text"], "mc")), ("", "typed_ph", (["text"], "ph"))]
    for suffix, p1 in units(name):
        jobs += [(suffix, s, spec) for s, spec in systems_for(name, suffix, p1).items()]
    for suffix, system, (conds, kind) in jobs:
        path = cache / f"p4res{suffix}_{system}.json"
        if path.exists():
            continue
        eval_system(R, both if system in BIG_SYSTEMS else small, conds, kind, path)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    name, stages = sys.argv[1], sys.argv[2:] or ["mc", "prompts", "decode", "eval"]
    for stage in stages:
        log(f">>> part4 {name}: {stage}")
        {"mc": stage_mc, "prompts": stage_prompts, "decode": stage_decode, "eval": stage_eval}[stage](name)
