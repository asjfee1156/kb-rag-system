"""
SQLite 資料存取層。

使用 Python 內建 sqlite3，零安裝、單檔案，符合「完全離線」限制。
設計三張表：
  documents : 上傳文件的中繼資料（檔名、類型、雜湊、切片數）
  chunks    : 每個文字切片 + 向量（以 JSON 文字儲存，避免二進制相容問題）
  history   : 使用者問答歷史與检索到的引用來源

每一個公開函式都「自行開啟連線、用完即關」，避免跨執行緒共用連線的問題
（FastAPI 以執行緒池處理同步端點）。
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT    NOT NULL,
    file_type     TEXT    NOT NULL,
    file_hash     TEXT    NOT NULL,
    content_length INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (DATETIME('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content    TEXT    NOT NULL,
    embedding  TEXT    NOT NULL,          -- JSON 化的向量 [0.0123, -0.045, ...]
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    citations  TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (DATETIME('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """建立資料表（若不存在）。啟動時呼叫一次。"""
    with _get_conn() as conn:
        conn.executescript(_SCHEMA)


# ------------------------------------------------------------------ 文件
def file_exists_by_hash(file_hash: str) -> Optional[sqlite3.Row]:
    """以 內容雜湊 判斷檔案是否已上傳過（重複上傳的快速檢查）。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, filename FROM documents WHERE file_hash = ? LIMIT 1", (file_hash,)
        ).fetchone()
        return row


def insert_document(
    filename: str,
    file_type: str,
    file_hash: str,
    content_length: int,
    chunks: Iterable[Dict[str, Any]],
) -> int:
    """
    寫入一份文件與其所有切片（單一交易，全部成功或全部回滾）。

    chunks: 每個元素為 {"chunk_index": int, "content": str, "embedding": List[float]}
    回傳新文件的 id。
    """
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, file_type, file_hash, content_length, chunk_count)"
            " VALUES (?, ?, ?, ?, ?)",
            (filename, file_type, file_hash, content_length, 0),
        )
        doc_id = cur.lastrowid

        rows = [
            (
                doc_id,
                c["chunk_index"],
                c["content"],
                json.dumps(c["embedding"], ensure_ascii=False),
            )
            for c in chunks
        ]
        conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, content, embedding)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (len(rows), doc_id))
        return doc_id


def delete_document(doc_id: int) -> bool:
    """刪除文件與其切片（FK ON DELETE CASCADE 連動刪除）。"""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0


def list_documents() -> List[Dict[str, Any]]:
    """列出所有已上傳文件（最新在前）。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, file_type, content_length, chunk_count, created_at"
            " FROM documents ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def list_chunks() -> List[Dict[str, Any]]:
    """讀取全部切片（含向量），供檢索引擎建立記憶體索引。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.doc_id, c.chunk_index, c.content, c.embedding, d.filename"
            " FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " ORDER BY c.doc_id, c.chunk_index"
        ).fetchall()
        results = []
        for r in rows:
            results.append(
                {
                    "chunk_id": r["id"],
                    "doc_id": r["doc_id"],
                    "chunk_index": r["chunk_index"],
                    "content": r["content"],
                    "embedding": json.loads(r["embedding"]),
                    "doc_name": r["filename"],
                }
            )
        return results


def count_stats() -> Dict[str, int]:
    """系統狀態：文件數、切片數。"""
    with _get_conn() as conn:
        docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": docs, "chunks": chunks}


# ------------------------------------------------------------------ 歷史
def insert_history(question: str, answer: str, citations: List[Dict[str, Any]]) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO history (question, answer, citations) VALUES (?, ?, ?)",
            (question, answer, json.dumps(citations, ensure_ascii=False)),
        )
        return cur.lastrowid


def list_history(limit: int = 100) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, question, answer, citations, created_at FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "citations": json.loads(r["citations"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        out.reverse()  # 由舊到新，便於前端逐条渲染
        return out


def clear_history() -> int:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM history")
        return cur.rowcount


def compute_file_hash(content: bytes) -> str:
    """SHA-256 內容雜湊，用於重複上傳偵測。"""
    return hashlib.sha256(content).hexdigest()
