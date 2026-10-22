"""Part 3 data: three real-speech test sets transcribed by Whisper, and a real-vs-synthetic training set.

Stages (each cached and resumable; run on a GPU machine):

  sdqa      SD-QA English test audio (11 dialects, from the SD-QA GitHub repo) transcribed with
            Whisper base.en (conditions asr_wb_<dialect>) and large-v3-turbo (asr_wt_<dialect>).
            Requires build_sdqa.py to have built the SD-QA corpus/queries first.
  heysquad  HeySQuAD human-spoken SQuAD questions (HF WillHeld/HeySQuAD_distill).
            Test set: answerable dev questions over a corpus of every SQuAD context paragraph,
            transcribed by Whisper base.en (asr_wb) — the dataset's own transcripts kept as asr_orig.
            Training set: 2,000 + 200 answerable train questions in three forms — typed (clean),
            the human recording via Whisper (real), Piper TTS via Whisper (synth) — with BM25 and
            hybrid hard negatives mined per form. Written to cache/trainset_heysquad.json.
  slue      SLUE-SQA-5 test questions (real speech) over the corpus of its dev+test documents,
            transcribed by Whisper base.en (asr_wb).
  beirhyb   adds hybrid (BM25 + dense, RRF) negatives to the Part 1 BEIR training set.

Data hygiene follows paper/PREREGISTRATION_PART3.md: recordings whose Whisper transcript fails the
hallucination rule against the question are treated as misaligned audio and dropped.
Usage: python -I part3_data.py sdqa heysquad slue beirhyb
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech  # noqa: E402

SDQA_REPO = "https://github.com/ffaisal93/SD-QA.git"
SDQA_DIALECTS = ("aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf")
ASR_MODELS = {"wb": "base.en", "wt": "large-v3-turbo"}
HEY_REPO = "datasets/WillHeld/HeySQuAD_distill"
SLUE_REPO = "datasets/asapp/slue-phase-2"
HEY_TRAIN, HEY_VAL = 2000, 200
HEY_ROW_GROUPS = 80                      # random train row groups (~100 rows each) to sample from
NEG_POOL = 30


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_recognizers: dict[str, speech.Recognizer] = {}


def transcribe(model: str, audio) -> str:
    if model not in _recognizers:
        _recognizers[model] = speech.Recognizer(model, device=data.best_device())
    return _recognizers[model](audio)


def write_beir(name: str, corpus: dict, queries: dict, qrels: dict) -> None:
    out = data.DATA / name / name
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for doc_id, doc in corpus.items():
            fh.write(json.dumps({"_id": doc_id, **doc}, ensure_ascii=False) + "\n")
    with open(out / "queries.jsonl", "w", encoding="utf-8") as fh:
        for qid, text in queries.items():
            fh.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
    with open(out / "qrels" / "test.tsv", "w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for qid, rel in qrels.items():
            for doc_id, grade in rel.items():
                fh.write(f"{qid}\t{doc_id}\t{grade}\n")


# ----------------------------------------------------------------------------------- SD-QA


def stage_sdqa() -> None:
    ds = data.load("sdqa")                       # built by build_sdqa.py
    cache = data.cache_dir("sdqa")
    repo = HERE / "data" / "sdqa_audio"
    if not (repo / "test" / "eng").is_dir():
        log("sparse-cloning SD-QA English test audio")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", SDQA_REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "test/eng"], check=True)
    conds = []
    for tag, model in ASR_MODELS.items():
        for dialect in SDQA_DIALECTS:
            cond = f"asr_{tag}_{dialect}"
            path = cache / f"transcripts_{cond}.json"
            have = data.load_json(path, {})
            todo = [q for q in ds.queries if q not in have]
            wavs = {Path(p).stem: p for p in glob.glob(str(repo / "test" / "eng" / dialect / "wav_eng" / "*.wav"))}
            started = time.time()
            for i, qid in enumerate(todo, 1):
                have[qid] = transcribe(model, speech.load_audio(Path(wavs[qid]).read_bytes())) if qid in wavs else ""
                if i % 200 == 0:
                    data.save_json(path, have)
            data.save_json(path, have)
            missing = sum(q not in wavs for q in ds.queries)
            log(f"sdqa {cond}: {len(todo)} transcribed in {time.time() - started:.0f}s; "
                f"{missing} questions without audio; mean WER "
                f"{sum(speech.wer(ds.queries[q], have[q]) for q in ds.queries) / len(ds.queries):.3f}")
            conds.append(cond)
    data.save_json(cache / "conditions_whisper.json", conds)


# ----------------------------------------------------------------------------------- HeySQuAD


def _hf_files(repo: str, pattern: str) -> list[str]:
    from huggingface_hub import HfFileSystem
    return sorted(HfFileSystem().glob(f"{repo}/{pattern}"))


def _read(path: str, columns: list[str], row_groups: list[int] | None = None) -> list[dict]:
    """Rows of the chosen row groups (default: all), reading only `columns`; each row gets `_rg`."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(path, block_size=8 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        groups = range(pf.num_row_groups) if row_groups is None else row_groups
        return [{**row, "_rg": g} for g in groups for row in pf.read_row_group(g, columns=columns).to_pylist()]


def stage_heysquad() -> None:
    cache = data.cache_dir("heysquad")
    meta_path = cache / "meta.json"
    meta = data.load_json(meta_path)
    text_cols = ["id", "question", "context", "is_impossible", "transcription"]
    if meta is None:
        log("reading HeySQuAD text columns (train + dev)")
        train_files = _hf_files(HEY_REPO, "data/train-*.parquet")
        dev_files = _hf_files(HEY_REPO, "data/validation-*.parquet")
        contexts: dict[str, str] = {}
        def ctx_id(text: str) -> str:
            return contexts.setdefault(text, f"c{len(contexts)}")
        train_rows = [{**r, "ctx": ctx_id(r["context"]), "file": f, "rg": r["_rg"]}
                      for f in train_files for r in _read(f, text_cols)]
        dev_rows = [{**r, "ctx": ctx_id(r["context"]), "file": f, "rg": r["_rg"]}
                    for f in dev_files for r in _read(f, text_cols)]
        dev = [r for r in dev_rows if not r["is_impossible"]]
        dev_text = {norm(r["question"]) for r in dev}
        rng = random.Random("heysquad-train")
        groups = sorted({(r["file"], r["rg"]) for r in train_rows})
        picked_groups = set(rng.sample(groups, HEY_ROW_GROUPS))
        pool = [r for r in train_rows if (r["file"], r["rg"]) in picked_groups and not r["is_impossible"]
                and norm(r["question"]) not in dev_text]
        seen, uniq = set(), []
        for r in pool:
            if norm(r["question"]) not in seen:
                seen.add(norm(r["question"])); uniq.append(r)
        picked = rng.sample(uniq, HEY_TRAIN + HEY_VAL)
        roles = {r["id"]: ("val" if i < HEY_VAL else "train") for i, r in enumerate(picked)}
        meta = {"corpus": {cid: text for text, cid in contexts.items()},
                "dev": [{k: r[k] for k in ("id", "question", "ctx", "transcription", "file", "rg")} for r in dev],
                "train": [{**{k: r[k] for k in ("id", "question", "ctx", "transcription", "file", "rg")},
                           "role": roles[r["id"]]} for r in picked],
                "counts": {"train_rows": len(train_rows), "dev_rows": len(dev_rows), "dev_answerable": len(dev),
                           "contexts": len(contexts), "train_pool": len(uniq)}}
        data.save_json(meta_path, meta)
    log(f"heysquad meta: {meta['counts']}")

    # audio -> Whisper base.en, for dev (test) and the sampled train/val questions
    trans_path = cache / "whisper_real.json"
    real = data.load_json(trans_path, {})
    need = [r for r in meta["dev"] + meta["train"] if r["id"] not in real]
    by_group: dict[tuple, set] = {}
    for r in need:
        by_group.setdefault((r["file"], r["rg"]), set()).add(r["id"])
    started = time.time()
    for n, ((f, g), ids) in enumerate(sorted(by_group.items()), 1):
        for row in _read(f, ["id", "audio"], [g]):
            if row["id"] in ids and row["id"] not in real:
                real[row["id"]] = transcribe("base.en", speech.load_audio(row["audio"]["bytes"]))
        if n % 10 == 0:
            data.save_json(trans_path, real)
            log(f"heysquad audio: {n}/{len(by_group)} row groups ({time.time() - started:.0f}s)")
    data.save_json(trans_path, real)

    synth_path = cache / "whisper_synth.json"
    synth = data.load_json(synth_path, {})
    for i, r in enumerate([r for r in meta["train"] if r["id"] not in synth], 1):
        voice = speech.piper_voice_for(r["id"])
        synth[r["id"]] = transcribe("base.en", speech.synthesize_piper(r["question"], voice))
        if i % 200 == 0:
            data.save_json(synth_path, synth)
            log(f"heysquad synth: {i} done")
    data.save_json(synth_path, synth)

    # hygiene: misaligned recordings (pre-registered rule) are dropped everywhere
    misaligned = {r["id"] for r in meta["dev"] + meta["train"] if speech.is_hallucination(r["question"], real.get(r["id"], ""))}
    corpus = {cid: {"title": "", "text": text} for cid, text in meta["corpus"].items()}
    dev = [r for r in meta["dev"] if r["id"] not in misaligned]
    write_beir("heysquad", corpus, {r["id"]: r["question"] for r in dev}, {r["id"]: {r["ctx"]: 1} for r in dev})
    data.save_json(cache / "transcripts_asr_wb.json", {r["id"]: real[r["id"]] for r in dev})
    data.save_json(cache / "transcripts_asr_orig.json", {r["id"]: r["transcription"] or "" for r in dev})
    log(f"heysquad test: {len(dev)} questions ({sum(r['id'] in misaligned for r in meta['dev'])} misaligned dropped); "
        f"WER wb {sum(speech.wer(r['question'], real[r['id']]) for r in dev) / len(dev):.3f}")

    # training set: three forms, BM25 + hybrid negatives per form
    from bench.first_stage import BM25, Dense, rrf
    ds = data.load("heysquad")
    bm25, dense = BM25(ds), Dense(ds)
    train = [r for r in meta["train"] if r["id"] not in misaligned]
    forms = {r["id"]: {"clean": r["question"], "real": real[r["id"]], "synth": synth[r["id"]]} for r in train}
    flat = {f"{qid}|{form}": text for qid, fs in forms.items() for form, text in fs.items()}
    dense_runs = dense.search_many(flat, NEG_POOL)
    examples = []
    for r in train:
        qid, pos = r["id"], r["ctx"]
        negs, negs_h = {}, {}
        for form, text in forms[qid].items():
            bm = bm25.search(text, NEG_POOL)
            negs[form] = [d for d, _ in bm if d != pos]
            negs_h[form] = [d for d, _ in rrf(bm, dense_runs[f"{qid}|{form}"]) if d != pos][:NEG_POOL]
        examples.append({"dataset": "heysquad", "qid": qid, "role": r["role"], **forms[qid],
                         "wer_real": speech.wer(r["question"], forms[qid]["real"]),
                         "wer_synth": speech.wer(r["question"], forms[qid]["synth"]),
                         "voice_synth": speech.piper_voice_for(qid),
                         "positives": {pos: 1}, "negatives": negs, "negatives_hybrid": negs_h})
    data.save_json(data.CACHE / "trainset_heysquad.json", {"examples": examples})
    tr = [e for e in examples if e["role"] == "train"]
    log(f"heysquad trainset: {len(tr)} train / {len(examples) - len(tr)} val "
        f"({sum(r['id'] in misaligned for r in meta['train'])} misaligned dropped); "
        f"mean WER real {sum(e['wer_real'] for e in examples) / len(examples):.3f}, "
        f"synth {sum(e['wer_synth'] for e in examples) / len(examples):.3f}")


# ----------------------------------------------------------------------------------- SLUE-SQA-5


def stage_slue() -> None:
    cache = data.cache_dir("slue")
    rows_path = cache / "rows.json"
    rows = data.load_json(rows_path)
    cols = ["question_id", "raw_question_text", "document_id", "raw_document_text"]
    if rows is None:
        rows = {"test": [], "validation": []}
        for split in rows:
            for f in _hf_files(SLUE_REPO, f"sqa5/{split}-*.parquet"):
                rows[split] += _read(f, cols) if split == "validation" else [
                    {**r, "file": f} for r in _read(f, cols)]
        data.save_json(rows_path, rows)
    corpus = {}
    for r in rows["validation"] + rows["test"]:
        corpus.setdefault(r["document_id"], {"title": "", "text": r["raw_document_text"]})
    trans_path = cache / "whisper_real.json"
    real = data.load_json(trans_path, {})
    files = sorted({r["file"] for r in rows["test"]})
    started = time.time()
    for n, f in enumerate(files, 1):
        if all(r["question_id"] in real for r in rows["test"] if r["file"] == f):
            continue
        for row in _read(f, ["question_id", "question_audio"]):
            if row["question_id"] not in real:
                real[row["question_id"]] = transcribe("base.en", speech.load_audio(row["question_audio"]["bytes"]))
        data.save_json(trans_path, real)
        log(f"slue audio: {n}/{len(files)} files ({time.time() - started:.0f}s)")
    test = [r for r in rows["test"] if not speech.is_hallucination(r["raw_question_text"], real.get(r["question_id"], ""))]
    write_beir("slue", corpus, {r["question_id"]: r["raw_question_text"] for r in test},
               {r["question_id"]: {r["document_id"]: 1} for r in test})
    data.save_json(cache / "transcripts_asr_wb.json", {r["question_id"]: real[r["question_id"]] for r in test})
    log(f"slue test: {len(test)} questions ({len(rows['test']) - len(test)} misaligned dropped), "
        f"{len(corpus)} documents; WER {sum(speech.wer(r['raw_question_text'], real[r['question_id']]) for r in test) / len(test):.3f}")


# ----------------------------------------------------------------------------------- BEIR hybrid negatives


def stage_beirhyb() -> None:
    from bench.first_stage import BM25, Dense, rrf
    path = data.CACHE / "trainset.json"
    built = data.load_json(path)
    for name in data.DATASETS:
        ds = data.load(name)
        bm25, dense = BM25(ds), Dense(ds)
        exs = [e for e in built["examples"] if e["dataset"] == name]
        flat = {f"{e['qid']}|{form}": e[form] for e in exs for form in ("clean", "noisy")}
        dense_runs = dense.search_many(flat, NEG_POOL)
        for e in exs:
            e["negatives_hybrid"] = {
                form: [d for d, _ in rrf(bm25.search(e[form], NEG_POOL), dense_runs[f"{e['qid']}|{form}"])
                       if d not in e["positives"]][:NEG_POOL] for form in ("clean", "noisy")}
        log(f"beirhyb {name}: {len(exs)} examples")
    data.save_json(path, built)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for stage in sys.argv[1:]:
        log(f">>> stage {stage}")
        {"sdqa": stage_sdqa, "heysquad": stage_heysquad, "slue": stage_slue, "beirhyb": stage_beirhyb}[stage]()
