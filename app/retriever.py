"""
向量檢索引擎：雙評分模型 + Top-K 檢索。

資料流向：
  SQLite(chunks 表) --載入--> 記憶體索引 --查詢--> Top-K 切片

評分模型（依引擎分層）：
  Tier 1（MiniLM）：向量餘弦相似度——入庫與查詢向量皆已 L2 正規化，
      故 cosine = 內積，檢索只需一次矩陣乘法。
  Tier 2（本機引擎）：字串鍵倒排索引 + BM25（k1=1.5, b=0.5），
      並加入「文件級脈絡平滑」（chunk 分數 + 0.3 × 所屬文件整體分數，
      BM25F 思路）校正極短文件的詞頻集中偏置。顯示分數以
      raw/(raw+4) 壓縮至 0~1；過濾門檻使用原始分數。

兩種引擎的分數分布不同，門檻由 embedding_service 依分層提供。
"""

import threading
from typing import Any, Dict, List

import numpy as np

from . import database
from .config import EMBEDDING_DIM, SIM_THRESHOLD, TOP_K
from .embedding import embedding_service


class RetrievalEngine:
    """記憶體向量索引 + Top-K 檢索（依引擎分層評分）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._matrix: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._meta: List[Dict[str, Any]] = []
        # 維度不一致的向量數（切換引擎層級後，舊向量維度不符 → 不納入索引）
        self.dim_mismatch_count = 0

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._meta)

    def rebuild(self) -> int:
        """
        從 SQLite 重建完整索引（上傳/刪除文件後呼叫）。回傳有效切片數。

        維度校驗：只有與「當前引擎維度」一致的向量才納入索引。
        若使用者切換機器導致引擎分層改變（384 ↔ 8192），舊維度的向量
        會被標記為 dim_mismatch_count 並排除，避免內積形狀錯誤；
        此時請執行 reset.bat 重新建立知識庫。
        """
        rows = database.list_chunks()
        expected = embedding_service.vector_dim
        valid = [r for r in rows if len(r["embedding"]) == expected]
        with self._lock:
            self.dim_mismatch_count = len(rows) - len(valid)
            if not valid:
                self._matrix = np.zeros((0, expected), dtype=np.float32)
                self._meta = []
                return 0
            # Tier 2 在此同時建立倒排索引（BM25 檢索使用）
            local = embedding_service.local_engine
            if local is not None:
                local.build_index(
                    [r["content"] for r in valid],
                    [r["doc_id"] for r in valid],
                )
            self._matrix = np.vstack([np.asarray(r["embedding"], dtype=np.float32) for r in valid])
            self._meta = valid
            return len(valid)

    def invalidate(self) -> None:
        """清空索引（等候下次 rebuild）。"""
        with self._lock:
            self._matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._meta = []

    # ------------------------------------------------------------------
    def search(
        self,
        query_text: str,
        top_k: int = TOP_K,
        threshold: float = SIM_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Top-K 檢索。評分依引擎分層（見模組 docstring）。

        參數:
            query_text:  使用者的問題文字
            top_k:       回傳筆數
            threshold:   分數門檻，低於者過濾掉（由呼叫端依引擎分層傳入）
        回傳: 依分數排序的切片清單，每筆含 doc_name / content / score
        """
        with self._lock:
            if self._matrix.shape[0] == 0:
                return []

            local = embedding_service.local_engine
            if local is not None:
                # ---------- Tier 2：倒排索引 + BM25 ----------
                raw = local.bm25_scores(query_text)
                if raw.size == 0:
                    return []
                scores = raw
                # 顯示用壓縮：raw/(raw+K)，將 BM25 分數映射到 0~1
                K = 4.0
                display = scores / (scores + K)
            else:
                # ---------- Tier 1：餘弦相似度 ----------
                q = np.asarray(embedding_service.encode([query_text])[0], dtype=np.float32).reshape(-1)
                qnorm = float(np.linalg.norm(q))
                q = q / qnorm if qnorm > 1e-9 else q
                scores = self._matrix @ q
                display = scores

            k = min(top_k, len(scores))
            # argpartition 取 Top-K，再精確排序（O(N) 而非 O(N log N)）
            cand_idx = np.argpartition(-scores, k - 1)[:k]
            cand_idx = cand_idx[np.argsort(-scores[cand_idx])]

            results = []
            for idx in cand_idx:
                score = float(scores[idx])
                if score < threshold:
                    continue
                meta = self._meta[int(idx)]
                results.append(
                    {
                        "doc_name": meta["doc_name"],
                        "doc_id": meta["doc_id"],
                        "chunk_index": meta["chunk_index"],
                        "score": round(float(display[idx]), 4),
                        "content": meta["content"],
                    }
                )
            return results


# 模組級單例
retrieval_engine = RetrievalEngine()
