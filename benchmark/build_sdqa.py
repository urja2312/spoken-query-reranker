"""Build the SD-QA real-speech retrieval benchmark in BEIR layout.

SD-QA (Faisal et al., Findings of EMNLP 2021) recorded the 1,031 English TyDi-QA dev questions read
aloud by real speakers in eleven English dialects, and released ASR transcripts of every recording
from Google's recogniser with a localised model (e.g. en-IN) and the en-US model. We pair those
transcripts with TyDi-QA's passage annotations to get a passage-retrieval task on real speech:

  corpus   every passage candidate of every English TyDi-QA dev article (closed corpus);
  qrels    passages chosen by TyDi annotators, graded by how many of the 3 chose them;
  queries  questions with at least one annotated passage; typed form = SD-QA's English gold text;
  conds    one transcript condition per (dialect, ASR model) column, e.g. asr_ind_s_en_in.

Questions are joined to TyDi by normalised question text (the HF TyDi release has no example id).
Writes data/sdqa/sdqa/{corpus.jsonl, queries.jsonl, qrels/test.tsv} and
cache/sdqa/transcripts_<cond>.json. Prints the condition list for run_bench.py.
Usage: python -I build_sdqa.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data  # noqa: E402

SDQA_CSV = "https://raw.githubusercontent.com/ffaisal93/SD-QA/master/test/eng/english.test.csv"
TYDI_REPO = "google-research-datasets/tydiqa"
TYDI_FILE = "primary_task/validation-00000-of-00001.parquet"
MAX_PASSAGE_CHARS = 2000


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out = data.DATA / "sdqa" / "sdqa"
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    cache = data.cache_dir("sdqa")

    with urllib.request.urlopen(SDQA_CSV, timeout=120) as resp:
        sdqa = list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))
    tydi = pd.read_parquet(hf_hub_download(TYDI_REPO, TYDI_FILE, repo_type="dataset"))
    tydi = tydi[tydi["language"] == "english"].reset_index(drop=True)
    print(f"SD-QA test rows: {len(sdqa)}; TyDi English dev rows: {len(tydi)}", flush=True)

    article_ids: dict[str, int] = {}
    corpus: dict[str, dict] = {}
    by_question: dict[str, list[tuple[dict[str, int], int]]] = {}
    for _, row in tydi.iterrows():
        url = str(row["document_url"])
        art = article_ids.setdefault(url, len(article_ids))
        raw = str(row["document_plaintext"]).encode("utf-8")
        cand = row["passage_answer_candidates"]
        for i, (s, e) in enumerate(zip(cand["plaintext_start_byte"], cand["plaintext_end_byte"])):
            doc_id = f"a{art}_p{i}"
            if doc_id not in corpus:
                text = raw[int(s):int(e)].decode("utf-8", "ignore").strip()[:MAX_PASSAGE_CHARS]
                if text:
                    corpus[doc_id] = {"title": str(row["document_title"]), "text": text}
        votes = [int(v) for v in row["annotations"]["passage_answer_candidate_index"] if int(v) >= 0]
        graded = {}
        for v in votes:
            graded[f"a{art}_p{v}"] = graded.get(f"a{art}_p{v}", 0) + 1
        by_question.setdefault(norm(row["question_text"]), []).append((graded, art))

    conds = [c for c in sdqa[0] if c not in ("", "ID", "in XOR test", "english-gold")]
    queries, qrels, transcripts = {}, {}, {c: {} for c in conds}
    unmatched = 0
    for r in sdqa:
        hits = by_question.get(norm(r["english-gold"]))
        if not hits:
            unmatched += 1
            continue
        graded = {d: g for d, g in hits[0][0].items() if d in corpus}
        if not graded:
            continue                                  # TyDi has no passage answer for this question
        qid = str(r["ID"])
        queries[qid], qrels[qid] = r["english-gold"], graded
        for c in conds:
            transcripts[c][qid] = r[c] or ""

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

    cond_names = []
    for c in conds:
        name = "asr_" + re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
        data.save_json(cache / f"transcripts_{name}.json", transcripts[c])
        cond_names.append(name)
    data.save_json(cache / "conditions.json", cond_names)
    print(f"articles {len(article_ids)}, passages {len(corpus)}, answerable queries {len(queries)}, "
          f"unmatched {unmatched}, conditions {len(cond_names)}", flush=True)
    print("CONDS " + " ".join(cond_names), flush=True)


if __name__ == "__main__":
    main()
