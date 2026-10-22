"""BEIR dataset loading. Test split only; graded qrels are kept as graded."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
DATASETS = ("scifact", "nfcorpus", "fiqa")


@dataclass
class Dataset:
    name: str
    corpus: dict[str, dict[str, str]]          # doc_id -> {"title", "text"}
    queries: dict[str, str]                    # qid -> text (test queries only)
    qrels: dict[str, dict[str, int]]           # qid -> {doc_id: grade}

    def doc_text(self, doc_id: str) -> str:
        doc = self.corpus[doc_id]
        return f"{doc['title']}. {doc['text']}".strip(". ") if doc["title"] else doc["text"]


def load(name: str) -> Dataset:
    base = DATA / name / name
    corpus: dict[str, dict[str, str]] = {}
    with open(base / "corpus.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            corpus[str(row["_id"])] = {"title": row.get("title") or "", "text": row.get("text") or ""}
    qrels: dict[str, dict[str, int]] = {}
    with open(base / "qrels" / "test.tsv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grade = int(row["score"])
            if grade > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = grade
    queries: dict[str, str] = {}
    with open(base / "queries.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = row["text"]
    qrels = {qid: rel for qid, rel in qrels.items() if qid in queries}
    return Dataset(name, corpus, queries, qrels)


def cache_dir(name: str) -> Path:
    path = CACHE / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


def load_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def load_split(name: str, split: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Queries and graded qrels for a non-test split (train/dev). Corpus is shared with `load`."""
    base = DATA / name / name
    qrels: dict[str, dict[str, int]] = {}
    with open(base / "qrels" / f"{split}.tsv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries: dict[str, str] = {}
    with open(base / "queries.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = row["text"]
    return queries, {q: r for q, r in qrels.items() if q in queries}


def best_device() -> str:
    """cuda (Kaggle/Colab), else mps (Apple silicon), else cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
