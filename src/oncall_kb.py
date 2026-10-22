"""Knowledge base for the AI On-Call Engineer — ingest manuals, retrieve cited passages.

The agent's first rule (from `agent_think.py`'s meta-prompt) is that it may never state a
step, part code or number that did not come from a tool result. This module is that tool.
It ships **empty**: until a client's printer manuals and cartridge spec sheets are ingested,
`kb_search` returns "not in the manuals I have", and the agent must relay that rather than
invent an answer. An empty KB is a correct KB with nothing in it, not a broken one.

Design decisions, and why
-------------------------
1. **Chunks, not documents.** A retrieval must return *the E-045 section*, not a 300-page
   manual. Text is split into passages on blank-line/heading boundaries and each is indexed
   on its own, carrying the section and a page reference so every spoken claim is traceable.
2. **SQLite FTS5, no new dependency.** Full-text search with BM25 ranking ships inside the
   standard library's sqlite3 here (verified: SQLite 3.53.1). No vector store, no embedding
   API, nothing to run on the client's box. If a build lacks FTS5, retrieval degrades to a
   token-overlap scorer and *says so* — slower, still grounded, never silently wrong.
3. **Device-family scoping.** A Canon answer must never leak into an HP call. Every chunk
   carries a `device_family`; retrieval filters on it before ranking.
4. **Re-ingest is idempotent.** Ingesting the same source path again replaces its chunks
   rather than doubling them, so correcting a manual is safe.

PDF text extraction is deliberately out of scope for a stdlib module: `ingest_text` takes
already-extracted text and records the source path. A PDF->text step is a named future task,
not a silent gap.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import oncall_reranker

MAX_CHUNK_WORDS = 180
MIN_CHUNK_WORDS = 12
MAX_RESULTS = 5
BODY_PREVIEW = 1200

DOC_KINDS = {"manual", "spec_sheet", "faq", "bulletin", "note"}

# Common English words carry no retrieval signal, and letting them through is not harmless:
# a live run had "my printer is grinding" match the E-045 page on the word "is" ("unit is
# overheating") and the agent read out the wrong fix. Error codes and part numbers survive
# because they contain digits or are single letters filtered by length, never by this set.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "off",
    "on", "or", "our", "out", "over", "she", "so", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "up", "was", "we", "were", "what", "when",
    "where", "which", "while", "will", "with", "would", "you", "your", "please", "press",
}

# Domain stopwords: words so common on a printer support line that they discriminate nothing.
# Without this, "my printer is grinding" matches the E-045 page on the word "printer" alone,
# and the agent confidently reads the wrong procedure — a live-run bug this exists to kill.
# When a query has no term left after this, retrieval misses honestly and the agent asks for
# the model or offers an engineer, which is the correct behaviour for a symptom no manual
# covers.
_DOMAIN_STOP = {"printer", "printers", "machine", "unit", "device", "print"}


class KnowledgeError(ValueError):
    """A caller supplied something the knowledge base cannot accept."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path):
    connection = sqlite3.connect(Path(db_path), timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


# --------------------------------------------------------------------------- schema


def fts5_available(db: sqlite3.Connection) -> bool:
    """Whether this SQLite build can create an FTS5 index.

    Checked once at schema init and recorded, rather than assumed. A build without FTS5 is
    unusual but real (some minimal Python distributions omit it), and finding out at first
    search instead of at boot is the kind of surprise this codebase refuses to ship.
    """
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        db.execute("DROP TABLE IF EXISTS _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def initialize_kb_schema(db_path: Path) -> None:
    with connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS oncall_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                device_family TEXT NOT NULL,
                doc_title TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'manual',
                source_path TEXT NOT NULL DEFAULT '',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                ingested_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oncall_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                device_family TEXT NOT NULL,
                doc_title TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                page_ref TEXT NOT NULL DEFAULT '',
                ord INTEGER NOT NULL DEFAULT 0,
                body TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES oncall_documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_oncall_chunks_family
                ON oncall_chunks(device_family);
            CREATE INDEX IF NOT EXISTS idx_oncall_chunks_doc
                ON oncall_chunks(document_id);
            CREATE TABLE IF NOT EXISTS oncall_devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                model_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                device_family TEXT NOT NULL,
                consumables_json TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                registered_at TEXT NOT NULL,
                UNIQUE(tenant_id, model_key)
            );
            """
        )
        # Safe additive migration for databases created before tenant isolation existed.
        for table in ("oncall_documents", "oncall_chunks", "oncall_devices"):
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if "tenant_id" not in columns:
                db.execute(
                    f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
        # The pre-tenant device table made model_key globally UNIQUE. Rebuild it once so two
        # tenants may register the same model without either seeing or blocking the other.
        device_sql = str(db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='oncall_devices'"
        ).fetchone()[0] or "")
        if "MODEL_KEY TEXT NOT NULL UNIQUE" in device_sql.upper():
            db.executescript(
                """
                CREATE TABLE oncall_devices_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    model_key TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    device_family TEXT NOT NULL,
                    consumables_json TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '',
                    registered_at TEXT NOT NULL,
                    UNIQUE(tenant_id, model_key)
                );
                INSERT INTO oncall_devices_v2
                    (id, tenant_id, model_key, display_name, device_family,
                     consumables_json, notes, registered_at)
                SELECT id, tenant_id, model_key, display_name, device_family,
                       consumables_json, notes, registered_at FROM oncall_devices;
                DROP TABLE oncall_devices;
                ALTER TABLE oncall_devices_v2 RENAME TO oncall_devices;
                """
            )
        db.execute("CREATE INDEX IF NOT EXISTS idx_oncall_documents_tenant ON oncall_documents(tenant_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_oncall_chunks_tenant_family ON oncall_chunks(tenant_id, device_family)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_oncall_devices_tenant ON oncall_devices(tenant_id, model_key)")
        if fts5_available(db):
            # Standalone FTS5 whose rowid we pin to the chunk id, so a match joins straight
            # back to oncall_chunks. Contentless external tables need triggers; at this scale
            # an explicit-rowid table we populate on ingest is simpler and just as fast.
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS oncall_fts "
                "USING fts5(body, section, doc_title)"
            )
        db.commit()


def _has_fts(db: sqlite3.Connection) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='oncall_fts'"
    ).fetchone()
    return row is not None


# --------------------------------------------------------------------------- chunking


_PAGE_MARK = re.compile(r"(?:page|pg|p\.)\s*(\d{1,4})", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
# Words that carry no capitalisation signal in a title, so "Connecting over USB" and
# "Replacing the Toner Cartridge" both read as headings despite the lowercase joiner.
_TITLE_JOINERS = {"a", "an", "and", "the", "of", "to", "for", "in", "on", "over", "with",
                  "by", "or", "via", "from", "at"}


def _looks_like_heading(line: str) -> bool:
    """A short line that reads as a section title, not a sentence.

    Manuals title their sections three ways and this catches all three: ALL CAPS, a line
    ending in a colon, and — the one an earlier version missed and a live ingest exposed —
    Title Case ("Replacing the Toner Cartridge"). A section boundary is what turns a whole
    manual into a retrievable E-045 passage, so getting this right is the difference between
    a citation and a 300-page dump. A wrong guess only mislabels a section, never an answer.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    if stripped[-1] in ".!?,;":
        return False
    if stripped.endswith(":"):
        return True
    words = stripped.split()
    if not 1 <= len(words) <= 8:
        return False

    lettered = [w for w in words if any(c.isalpha() for c in w)]
    if not lettered:
        return False
    if all(w.upper() == w for w in lettered):        # ALL CAPS heading
        return True

    # Title Case: every word that is not a small joiner starts with a capital.
    significant = [w for w in lettered if w.lower() not in _TITLE_JOINERS]
    if not significant:
        return False
    capitalised = sum(1 for w in significant if w[0].isupper())
    return capitalised == len(significant)


def chunk_text(text: str) -> list[dict[str, str]]:
    """Split a document into retrieval passages with a section label and page hint.

    Paragraphs are accumulated up to `MAX_CHUNK_WORDS`, breaking on the boundary rather than
    mid-sentence. The most recent heading-like line labels the chunks beneath it, and any
    'page N' marker seen carries into `page_ref`.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[dict[str, str]] = []
    section = ""
    page = ""
    buffer: list[str] = []
    word_count = 0

    def flush() -> None:
        nonlocal buffer, word_count
        body = " ".join(" ".join(buffer).split()).strip()
        if len(_WORD.findall(body)) >= MIN_CHUNK_WORDS:
            chunks.append({"section": section, "page_ref": page, "body": body})
        buffer = []
        word_count = 0

    for line in lines:
        mark = _PAGE_MARK.search(line)
        if mark:
            page = f"p.{mark.group(1)}"
        if _looks_like_heading(line):
            flush()
            section = line.strip().rstrip(":")
            continue
        if not line.strip():
            if word_count >= MAX_CHUNK_WORDS:
                flush()
            continue
        buffer.append(line.strip())
        word_count += len(line.split())
        if word_count >= MAX_CHUNK_WORDS:
            flush()
    flush()

    # A short document with no blank lines still deserves one chunk.
    if not chunks:
        body = " ".join(text.split()).strip()
        if body:
            chunks.append({"section": section, "page_ref": page, "body": body})
    return chunks


# --------------------------------------------------------------------------- ingest


def _norm_family(value: str) -> str:
    family = str(value or "").strip().lower()
    if not family:
        raise KnowledgeError("device_family is required; a chunk with no family cannot be scoped")
    return re.sub(r"\s+", "-", family)[:60]


def _norm_tenant(value: str) -> str:
    tenant = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "default").strip())[:80]
    if not tenant:
        raise KnowledgeError("tenant_id is required")
    return tenant


def ingest_text(db_path: Path, *, device_family: str, doc_title: str, text: str,
                tenant_id: str = "default",
                kind: str = "manual", source_path: str = "") -> dict[str, Any]:
    """Ingest one document's text. Re-ingesting the same source_path replaces its chunks.

    Returns a receipt the console can show: how many chunks landed, under which family.
    """
    initialize_kb_schema(db_path)
    family = _norm_family(device_family)
    tenant = _norm_tenant(tenant_id)
    title = str(doc_title or "").strip()
    if not title:
        raise KnowledgeError("doc_title is required")
    if kind not in DOC_KINDS:
        raise KnowledgeError(f"unknown document kind {kind!r}; known: {sorted(DOC_KINDS)}")
    body = str(text or "")
    if not body.strip():
        raise KnowledgeError("cannot ingest an empty document")

    passages = chunk_text(body)
    if not passages:
        raise KnowledgeError("document produced no usable passages")

    with connect(db_path) as db:
        has_fts = _has_fts(db)
        # Idempotent replace: a re-ingest of the same source must not double the corpus.
        if source_path:
            stale = db.execute(
                "SELECT id FROM oncall_documents WHERE tenant_id = ? AND source_path = ? AND doc_title = ?",
                (tenant, source_path, title),
            ).fetchall()
            for row in stale:
                _delete_document(db, int(row["id"]), has_fts)

        cursor = db.execute(
            "INSERT INTO oncall_documents "
            "(tenant_id, device_family, doc_title, kind, source_path, chunk_count, ingested_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (tenant, family, title, kind, source_path, len(passages), utc_now()),
        )
        document_id = int(cursor.lastrowid)

        for ordinal, passage in enumerate(passages):
            chunk_cursor = db.execute(
                "INSERT INTO oncall_chunks "
                "(document_id, tenant_id, device_family, doc_title, section, page_ref, ord, body) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (document_id, tenant, family, title, passage["section"],
                 passage["page_ref"] or f"{title}", ordinal, passage["body"]),
            )
            if has_fts:
                db.execute(
                    "INSERT INTO oncall_fts (rowid, body, section, doc_title) VALUES (?,?,?,?)",
                    (int(chunk_cursor.lastrowid), passage["body"], passage["section"], title),
                )
        db.commit()

    return {"document_id": document_id, "tenant_id": tenant, "device_family": family, "doc_title": title,
            "chunks": len(passages), "indexed": has_fts, "ingested_at": utc_now()}


def ingest_file(db_path: Path, path: Path, *, device_family: str,
                kind: str = "manual", tenant_id: str = "default") -> dict[str, Any]:
    """Ingest a .txt or .md file. Non-text formats are refused with a clear reason."""
    path = Path(path)
    if not path.is_file():
        raise KnowledgeError(f"file not found: {path}")
    if path.suffix.lower() not in {".txt", ".md"}:
        raise KnowledgeError(
            f"{path.suffix or 'this file'} is not text; extract it to .txt first "
            "(PDF->text is a separate step, out of scope for this module)"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    return ingest_text(db_path, tenant_id=tenant_id, device_family=device_family, doc_title=path.stem,
                       text=text, kind=kind, source_path=str(path))


def ingest_directory(db_path: Path, root: Path, *, tenant_id: str = "default") -> dict[str, Any]:
    """Walk uploads/knowledge/<device_family>/*.txt|.md and ingest each.

    The first path segment under `root` is the device family, so a client drops files into
    named folders and the scoping is automatic.
    """
    root = Path(root)
    if not root.is_dir():
        raise KnowledgeError(f"knowledge root not found: {root}")
    receipts = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
            continue
        try:
            family = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            family = "general"
        # A file sitting directly in root has itself as parts[0]; treat that as general.
        if family == path.name:
            family = "general"
        receipts.append(ingest_file(db_path, path, device_family=family, tenant_id=tenant_id))
    return {"root": str(root), "documents": len(receipts), "receipts": receipts}


def _delete_document(db: sqlite3.Connection, document_id: int, has_fts: bool) -> None:
    if has_fts:
        for row in db.execute(
            "SELECT id FROM oncall_chunks WHERE document_id = ?", (document_id,)
        ):
            db.execute("DELETE FROM oncall_fts WHERE rowid = ?", (int(row["id"]),))
    db.execute("DELETE FROM oncall_chunks WHERE document_id = ?", (document_id,))
    db.execute("DELETE FROM oncall_documents WHERE id = ?", (document_id,))


# --------------------------------------------------------------------------- retrieval


def _query_tokens(query: str, error_code: str = "") -> list[str]:
    """Alphanumeric tokens for the search, error code first so bm25 weights it.

    Building the FTS match string from tokens we control, each quoted, keeps FTS5 query
    syntax (a hyphen in 'E-045', a stray quote) out of the parser. A rare token like an
    error code needs no manual boost: bm25 already rewards it for being rare.
    """
    terms: list[str] = []
    if error_code:
        terms += _WORD.findall(str(error_code).lower())
    for token in _WORD.findall(str(query).lower()):
        if token in _STOP or token in _DOMAIN_STOP or len(token) < 2:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:24]


def _match_expr(tokens: list[str]) -> str:
    return " OR ".join(f'"{token}"' for token in tokens)


def kb_search(db_path: Path, query: str, *, device_family: str = "", tenant_id: str = "default",
              error_code: str = "", limit: int = 3) -> dict[str, Any]:
    """Retrieve the most relevant cited passages, or report an honest miss.

    The return shape is what the agent reads and what the Call Simulator renders. `found`
    is the flag the agent must obey: a false `found` means "not in the manuals I have",
    which is a finding to relay, never a gap to fill with a guess.
    """
    initialize_kb_schema(db_path)
    limit = max(1, min(int(limit), MAX_RESULTS))
    family = re.sub(r"\s+", "-", str(device_family or "").strip().lower())[:60]
    tenant = _norm_tenant(tenant_id)
    tokens = _query_tokens(query, error_code)

    if not tokens:
        return {"found": False, "reason": "no searchable terms in the query",
                "query": query, "device_family": family, "results": [], "mode": "none"}

    # Two-stage retrieval. BM25 is the fast recall stage: when a re-ranker is active it
    # shortlists a larger candidate pool, and the model decides the final order. With the seam
    # closed this fetches exactly `limit` rows, as it always did.
    pool = oncall_reranker.candidate_pool(limit)

    with connect(db_path) as db:
        mode = "fts5" if _has_fts(db) else "token_overlap"
        if mode == "fts5":
            rows = _search_fts(db, tokens, tenant, family, pool)
        else:
            rows = _search_overlap(db, tokens, tenant, family, pool)

    candidates = [
        {
            "doc_title": row["doc_title"],
            "section": row["section"],
            "page_ref": row["page_ref"],
            "device_family": row["device_family"],
            "citation": _citation(row["doc_title"], row["section"], row["page_ref"]),
            "score": round(float(row["score"]), 3),
            "body": row["body"][:BODY_PREVIEW],
        }
        for row in rows
    ]
    started = time.perf_counter()
    results, rerank_mode = oncall_reranker.rerank(query, candidates, limit=limit)
    rerank_ms = round((time.perf_counter() - started) * 1000.0, 2)
    if results:
        reason = ""
    elif rerank_mode.endswith(":abstained"):
        reason = "the best passage scored below the re-ranker's relevance threshold"
    else:
        reason = "no passage in the ingested manuals matched"
    return {
        "found": bool(results),
        "query": query,
        "tenant_id": tenant,
        "device_family": family or "any",
        "mode": mode,
        "rerank": rerank_mode,
        "rerank_ms": rerank_ms,
        "candidates_considered": len(candidates),
        "reason": reason,
        "results": results,
    }


def _citation(doc_title: str, section: str, page_ref: str) -> str:
    parts = [doc_title]
    if section:
        parts.append(section)
    if page_ref and page_ref != doc_title:
        parts.append(page_ref)
    return ", ".join(parts)


def _search_fts(db: sqlite3.Connection, tokens: list[str], tenant: str, family: str,
                limit: int) -> list[dict[str, Any]]:
    # bm25() returns a score where more negative is a better match; ascending order gives
    # the strongest passages first. We flip the sign on the way out so the console shows a
    # positive "higher is better" number, which is what a human expects to read.
    sql = (
        "SELECT c.doc_title, c.section, c.page_ref, c.device_family, c.body, "
        "       bm25(oncall_fts) AS raw "
        "FROM oncall_fts "
        "JOIN oncall_chunks c ON c.id = oncall_fts.rowid "
        "WHERE oncall_fts MATCH ? AND c.tenant_id = ? "
    )
    params: list[Any] = [_match_expr(tokens), tenant]
    if family:
        sql += "AND c.device_family = ? "
        params.append(family)
    sql += "ORDER BY raw LIMIT ?"
    params.append(limit)
    out = []
    for row in db.execute(sql, params):
        item = dict(row)
        item["score"] = -float(item.pop("raw"))
        out.append(item)
    return out


def _search_overlap(db: sqlite3.Connection, tokens: list[str], tenant: str, family: str,
                    limit: int) -> list[dict[str, Any]]:
    """Fallback ranking when FTS5 is absent: token overlap, section-weighted.

    The same shape as `service_ops.search_knowledge`, kept here so the KB has one honest
    behaviour in a degraded build rather than none.
    """
    token_set = set(tokens)
    sql = "SELECT doc_title, section, page_ref, device_family, body FROM oncall_chunks WHERE tenant_id = ?"
    params: list[Any] = [tenant]
    if family:
        sql += " AND device_family = ?"
        params.append(family)
    scored = []
    for row in db.execute(sql, params):
        body_terms = set(_WORD.findall(row["body"].lower()))
        section_terms = set(_WORD.findall((row["section"] or "").lower()))
        score = len(token_set & body_terms) + 2 * len(token_set & section_terms)
        if score:
            item = dict(row)
            item["score"] = float(score)
            scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


# --------------------------------------------------------------------------- inspection


def knowledge_stats(db_path: Path, *, tenant_id: str = "default") -> dict[str, Any]:
    """What the KB currently holds. Shown on the On-Call desk so 'empty' is never a mystery."""
    with connect(db_path) as db:
        mode = "fts5" if _has_fts(db) else "token_overlap"
        tenant = _norm_tenant(tenant_id)
        documents = int(db.execute("SELECT COUNT(*) FROM oncall_documents WHERE tenant_id = ?", (tenant,)).fetchone()[0])
        chunks = int(db.execute("SELECT COUNT(*) FROM oncall_chunks WHERE tenant_id = ?", (tenant,)).fetchone()[0])
        families = [
            {"device_family": row["device_family"], "documents": int(row["n"])}
            for row in db.execute(
                "SELECT device_family, COUNT(*) AS n FROM oncall_documents "
                "WHERE tenant_id = ? GROUP BY device_family ORDER BY device_family", (tenant,)
            )
        ]
    return {
        "documents": documents,
        "chunks": chunks,
        "families": families,
        "retrieval_mode": mode,
        "empty": documents == 0,
        "note": ("No manuals ingested yet — the agent will say it has nothing to go on."
                 if documents == 0 else
                 f"{documents} document(s), {chunks} passages, ranked by "
                 f"{'BM25 full-text' if mode == 'fts5' else 'token overlap (FTS5 unavailable)'}."),
    }


# --------------------------------------------------------------------------- device registry
#
# Structured facts a client gives us about their fleet: which model takes which cartridge.
# This is configured data, not invented data — `check_compatibility` answers "unknown" when
# the client has not told us, and the agent must relay that rather than guess a part number.


def _model_key(model: str) -> str:
    """Normalise a messy model string to a comparison key.

    'Canon LBP 2900', 'lbp2900', 'LBP-2900' all collapse to the same key, so a caller who
    reads a model off a label the way a human would still resolves it.
    """
    return re.sub(r"[^a-z0-9]+", "", str(model or "").lower())


def register_device(db_path: Path, *, model: str, display_name: str, device_family: str,
                    tenant_id: str = "default",
                    consumables: list[str] | None = None, notes: str = "") -> dict[str, Any]:
    key = _model_key(model)
    if not key:
        raise KnowledgeError("a device needs a model string with letters or digits")
    family = _norm_family(device_family)
    tenant = _norm_tenant(tenant_id)
    codes = [str(c).strip() for c in (consumables or []) if str(c).strip()]
    with connect(db_path) as db:
        db.execute(
            "INSERT INTO oncall_devices "
            "(tenant_id, model_key, display_name, device_family, consumables_json, notes, registered_at) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(tenant_id, model_key) DO UPDATE SET "
            "display_name=excluded.display_name, device_family=excluded.device_family, "
            "consumables_json=excluded.consumables_json, notes=excluded.notes",
            (tenant, key, str(display_name).strip() or model, family, json.dumps(codes), str(notes),
             utc_now()),
        )
        db.commit()
    return {"model_key": key, "display_name": display_name, "device_family": family,
            "consumables": codes}


def resolve_device(db_path: Path, model: str, *, tenant_id: str = "default") -> dict[str, Any]:
    """Match a spoken/typed model to a registered device, or report it unknown.

    An unknown model is a finding: the agent must ask the caller to read the label, not
    guess a fault tree for a device it cannot identify.
    """
    key = _model_key(model)
    if not key:
        return {"known": False, "reason": "no model was given", "query": model}
    with connect(db_path) as db:
        rows = [dict(r) for r in db.execute("SELECT * FROM oncall_devices WHERE tenant_id = ?", (_norm_tenant(tenant_id),))]
    for row in rows:                                    # exact, then containment either way
        if row["model_key"] == key:
            return _device_dict(row, model)
    for row in rows:
        if key in row["model_key"] or row["model_key"] in key:
            return _device_dict(row, model)
    return {"known": False, "reason": "model not in the registered fleet", "query": model}


def _device_dict(row: dict, query: str) -> dict[str, Any]:
    return {
        "known": True,
        "query": query,
        "model_key": row["model_key"],
        "display_name": row["display_name"],
        "device_family": row["device_family"],
        "consumables": json.loads(row["consumables_json"] or "[]"),
        "notes": row["notes"],
    }


def check_compatibility(db_path: Path, model: str, cartridge: str, *, tenant_id: str = "default") -> dict[str, Any]:
    """Is `cartridge` valid for `model`? A grounded yes / no / unknown, never a guess."""
    device = resolve_device(db_path, model, tenant_id=tenant_id)
    if not device.get("known"):
        return {"compatible": None, "reason": device.get("reason", "unknown device"),
                "model": model, "cartridge": cartridge}
    consumables = device.get("consumables", [])
    if not consumables:
        return {"compatible": None, "model": device["display_name"], "cartridge": cartridge,
                "reason": "no cartridge list is registered for this model"}
    wanted = _model_key(cartridge)
    match = any(_model_key(c) == wanted or wanted in _model_key(c) for c in consumables)
    return {
        "compatible": bool(match),
        "model": device["display_name"],
        "cartridge": cartridge,
        "supported_cartridges": consumables,
        "reason": ("listed for this model" if match
                   else "not in this model's registered cartridge list"),
    }


def list_documents(db_path: Path, device_family: str = "", *, tenant_id: str = "default") -> list[dict[str, Any]]:
    with connect(db_path) as db:
        sql = ("SELECT id, device_family, doc_title, kind, source_path, chunk_count, "
               "ingested_at FROM oncall_documents")
        sql += " WHERE tenant_id = ?"
        params: tuple = (_norm_tenant(tenant_id),)
        if device_family:
            sql += " AND device_family = ?"
            params += (re.sub(r"\s+", "-", device_family.strip().lower())[:60],)
        sql += " ORDER BY device_family, doc_title"
        return [dict(row) for row in db.execute(sql, params)]
