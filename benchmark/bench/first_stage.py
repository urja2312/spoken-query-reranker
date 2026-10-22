"""First-stage retrievers: production BM25 (SQLite FTS5), a dense bi-encoder, and RRF hybrid.

The BM25 retriever is not a reimplementation. It uses the production engine and the production
query builder from `src/oncall_kb.py` — the same FTS5 virtual table layout
(`body, section, doc_title`), the same unicode61 tokenizer, the same `_query_tokens` stopword
filtering and term cap, and the same quoted OR-match expression — so a number measured here is
a number about the shipped retriever. The one deliberate change is that the printer-domain
stopwords are disabled, because this benchmark is domain-general.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import oncall_kb  # noqa: E402  (production retrieval code)

oncall_kb._DOMAIN_STOP = set()          # domain-general benchmark: no printer-specific stopwords

from .data import Dataset, best_device, cache_dir   # noqa: E402

DEPTH = 100
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60


# --------------------------------------------------------------------------- BM25 (production)


def build_bm25(ds: Dataset) -> Path:
    path = cache_dir(ds.name) / "bm25.sqlite"
    if path.exists():
        return path
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as db:
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, section, doc_title, doc_id UNINDEXED)")
        db.executemany("INSERT INTO docs(body, section, doc_title, doc_id) VALUES (?, '', ?, ?)",
                       ((doc["text"], doc["title"], doc_id) for doc_id, doc in ds.corpus.items()))
        db.commit()
    tmp.replace(path)
    return path


class BM25:
    def __init__(self, ds: Dataset):
        self.db = sqlite3.connect(build_bm25(ds), check_same_thread=False)

    def search(self, query: str, k: int = DEPTH) -> list[tuple[str, float]]:
        tokens = oncall_kb._query_tokens(query)
        if not tokens:
            return []
        rows = self.db.execute(
            "SELECT doc_id, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
            (oncall_kb._match_expr(tokens), k)).fetchall()
        return [(str(doc_id), -float(raw)) for doc_id, raw in rows]


# --------------------------------------------------------------------------- dense


class Dense:
    def __init__(self, ds: Dataset, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(DENSE_MODEL, device=device or best_device())
        self.model.max_seq_length = 256
        cache = cache_dir(ds.name)
        emb_path, ids_path = cache / "dense_emb.npy", cache / "dense_ids.npy"
        if emb_path.exists():
            self.emb = np.load(emb_path)
            self.ids = np.load(ids_path, allow_pickle=False).astype(str)
        else:
            ids = list(ds.corpus)
            texts = [ds.doc_text(doc_id) for doc_id in ids]
            self.emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            self.ids = np.array(ids)
            np.save(emb_path, self.emb)
            np.save(ids_path, self.ids)

    def search_many(self, queries: dict[str, str], k: int = DEPTH) -> dict[str, list[tuple[str, float]]]:
        qids = list(queries)
        q = self.model.encode([DENSE_QUERY_PREFIX + queries[qid] for qid in qids], batch_size=64,
                              normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = q @ self.emb.T
        out = {}
        for row, qid in enumerate(qids):
            top = np.argpartition(-sims[row], k)[:k]
            top = top[np.argsort(-sims[row][top])]
            out[qid] = [(str(self.ids[i]), float(sims[row][i])) for i in top]
        return out


# --------------------------------------------------------------------------- hybrid


def rrf(*runs: list[tuple[str, float]], k: int = DEPTH) -> list[tuple[str, float]]:
    """Reciprocal rank fusion (Cormack et al., 2009): score = sum 1 / (RRF_K + rank)."""
    fused: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: -item[1])[:k]
