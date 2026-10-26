"""
FastAPI 應用入口：RESTful API 與前端單頁服務。

API 總覽
  GET  /                     前端單頁（完全內聯的 index.html）
  GET  /api/health           系統狀態（模型、向量數）
  POST /api/upload           上傳 .txt/.md → 切片 → 向量化 → 入庫
  GET  /api/documents        已上傳文件清單
  DELETE /api/documents/{id} 刪除某份文件（含其切片）
  POST /api/query            向量檢索問答（回傳答案 + 引用來源）
  GET  /api/history          歷史問答紀錄
  DELETE /api/history        清空歷史
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import database
from .chunker import chunk_text
from .config import (
    ALLOWED_EXTENSIONS,
    HOST,
    INDEX_FILE,
    MAX_UPLOAD_MB,
    MODEL_DIR,
    PORT,
    TOP_K,
    UPLOAD_DIR,
    ensure_directories,
)
from .corpus import refresh_local_corpus
from .embedding import embedding_service
from .retriever import retrieval_engine

app = FastAPI(
    title="企業級離線智能知識庫與 RAG 檢索系統",
    version="1.0.0",
    docs_url=None,      # 離線環境關閉 Swagger 文件頁（減少依賴資源）
    redoc_url=None,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """應用生命週期：初始化資料庫、引擎、索引；空庫時自動匯入預設知識文件。"""
    ensure_directories()
    database.init_db()
    embedding_service.load()          # 自動分層挑選向量引擎（Tier 1 MiniLM / Tier 2 本機向量）
    retrieval_engine.rebuild()        # 從 SQLite 重建記憶體向量索引

    # 首次開箱體驗：知識庫為空時自動匯入 samples/ 的預設文件（保持冪等）
    if database.count_stats()["documents"] == 0:
        try:
            from .seed import DEFAULT_SEEDS, seed_many

            for name, status in seed_many(DEFAULT_SEEDS):
                print(f"[SEED] {name}: {status}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 種子失敗不應阻擋啟動
            print(f"[SEED] 自動匯入預設文件失敗：{exc}", flush=True)

    # 語料一致性：Tier 1 重建索引；Tier 2 以現有語料重算 IDF + 全庫向量
    refresh_local_corpus()

    app.state.start_time = time.time()
    yield


app.router.lifespan_context = lifespan


# ==================================================================
# 前端
# ==================================================================
@app.get("/")
def index():
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE), media_type="text/html; charset=utf-8")
    raise HTTPException(status_code=404, detail="前端檔案 app/static/index.html 遺失")


@app.get("/api/health")
def health():
    stats = database.count_stats()
    return {
        "status": "ok",
        "engine": embedding_service.engine_name,
        "engine_tier": embedding_service.engine_tier,
        "model_loaded": embedding_service.is_loaded(),
        "engine_note": embedding_service.engine_note or None,
        "similarity_threshold": embedding_service.similarity_threshold,
        "vector_dim": embedding_service.vector_dim,
        "dim_mismatch": retrieval_engine.dim_mismatch_count,
        "documents": stats["documents"],
        "vectors": stats["chunks"],
        "uptime_sec": round(time.time() - getattr(app.state, "start_time", time.time()), 1),
    }


# ==================================================================
# 文件上傳 → 切片 → 向量化 → 入庫
# ==================================================================
def _repair_filename(name: str) -> str:
    """
    修復 multipart 上傳的中文檔名亂碼。
    瀏覽器以「原始 UTF-8 位元組」發送 Content-Disposition 檔名，
    而 Starlette 以 latin-1 解碼，導致中文變成亂碼（Ã¥Â¬…）。
    這裡反向編解碼還原：先試 UTF-8（瀏覽器），再試 Big5/CP950（部分舊版工具,
    例如繁中 Windows 下的 curl），都失敗才原樣回傳。
    """
    try:
        raw = name.encode("latin-1")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "big5", "cp950"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):  # noqa: B008 - FastAPI 慣例
    # 1) 副檔名校驗
    filename = _repair_filename(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"僅支援 {', '.join(sorted(ALLOWED_EXTENSIONS))} 格式，收到 {suffix or '無副檔名'}",
        )

    # 2) 讀取內容（大小限制）
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"檔案不得超過 {MAX_UPLOAD_MB} MB")
    if not raw.strip():
        raise HTTPException(status_code=400, detail="檔案內容為空")

    # 3) 編碼容錯：UTF-8 → Big5 → Latin-1（應對不同來源的中文純文字）
    text = None
    for enc in ("utf-8", "utf-8-sig", "big5", "cp950", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    # 4) 內容雜湊：重複內容直接拒絕（避免向量被重複計算）
    file_hash = database.compute_file_hash(raw)
    duplicate = database.file_exists_by_hash(file_hash)
    if duplicate:
        raise HTTPException(
            status_code=409,
            detail=f"文件《{duplicate['filename']}》內容完全相同，已存在於知識庫中",
        )

    # 5) 儲存原始檔案副本（data/uploads/），方便日後審核或重新匯入
    try:
        (UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        (UPLOAD_DIR / safe_name).write_bytes(raw)
    except OSError:
        pass  # 副本儲存失敗不影響主要流程

    # 6) 模型必須就緒才能算向量
    if not embedding_service.is_loaded():
        embedding_service.load()
    if not embedding_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail=embedding_service.engine_note or "向量引擎未就緒",
        )

    # 7) 切片（Chunking）
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="切片失敗：文件無有效文字內容")

    # 8) 向量化（批量編碼，CPU 推理）
    vectors = embedding_service.encode(chunks, show_progress=False)

    # 9) 寫入 SQLite（向量以 JSON 字串保存）
    payload = [
        {
            "chunk_index": i,
            "content": c,
            "embedding": [round(float(v), 6) for v in vectors[i]],
        }
        for i, c in enumerate(chunks)
    ]
    doc_id = database.insert_document(
        filename=filename,
        file_type=suffix.lstrip("."),
        file_hash=file_hash,
        content_length=len(text),
        chunks=payload,
    )

    # 10) 同步語料與索引：Tier 2 會以新語料重算全域 IDF 並重編全部向量
    refresh_local_corpus()

    return {
        "document_id": doc_id,
        "filename": filename,
        "file_type": suffix.lstrip("."),
        "content_length": len(text),
        "chunk_count": len(chunks),
        "vector_dim": int(vectors.shape[1]),
        "message": f"已切為 {len(chunks)} 個片段並完成向量化",
    }


@app.get("/api/documents")
def list_docs():
    return {"documents": database.list_documents(), "total": len(database.list_documents())}


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: int):
    if not database.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="文件不存在")
    refresh_local_corpus()   # 語料縮小 → 重算 IDF 與全部向量（Tier 2）
    return {"message": f"已刪除文件 id={doc_id} 及其全部切片"}


# ==================================================================
# 檢索問答
# ==================================================================
def _compose_answer(question: str, hits: List[dict]) -> str:
    """無生成式 LLM 時的答案組裝：萃取最相關切片，標註來源文件。"""
    if not hits:
        return (
            "在目前知識庫中找不到足夠相關的內容。\n"
            "可以嘗試：\n1. 上傳與問題相關的文件\n2. 換用更精確的關鍵詞（例如「VPN」「特休」「加班費」）"
        )
    best = hits[0]
    answer = f"依據《{best['doc_name']}》第 {best['chunk_index'] + 1} 個片段：\n\n{best['content']}"
    if len(hits) > 1:
        others = "、".join(
            f"《{h['doc_name']}》片段{h['chunk_index'] + 1}" for h in hits[1:]
        )
        answer += f"\n\n（另有參考來源：{others}）"
    return answer


@app.post("/api/query")
async def query(payload: dict):
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="問題不可為空")
    top_k = int(payload.get("top_k") or TOP_K)

    if not embedding_service.is_loaded():
        embedding_service.load()
    if not embedding_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail=embedding_service.engine_note or "向量引擎未就緒",
        )
    if retrieval_engine.size == 0:
        raise HTTPException(status_code=409, detail="知識庫為空，請先上傳文件")

    t0 = time.perf_counter()
    # 檢索：Tier 1 走向量餘弦；Tier 2 走 BM25（門檻隨引擎分層而異）
    hits = retrieval_engine.search(
        question, top_k=top_k, threshold=embedding_service.similarity_threshold
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = _compose_answer(question, hits)

    # 寫入歷史
    citation_records = [
        {
            "doc_name": h["doc_name"],
            "chunk_index": h["chunk_index"] + 1,
            "score": h["score"],
            "content": h["content"],
        }
        for h in hits
    ]
    database.insert_history(question, answer, citation_records)

    return JSONResponse(
        {
            "question": question,
            "answer": answer,
            "citations": citation_records,
            "latency_ms": elapsed_ms,
            "retrieved_count": len(hits),
            "engine": embedding_service.engine_name,
            "threshold": embedding_service.similarity_threshold,
        }
    )


# ==================================================================
# 歷史紀錄
# ==================================================================
@app.get("/api/history")
def get_history():
    return {"history": database.list_history(limit=100)}


@app.delete("/api/history")
def clear_history():
    n = database.clear_history()
    return {"message": f"已清空 {n} 筆歷史紀錄"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
