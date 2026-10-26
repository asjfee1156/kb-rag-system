"""
Embedding / 檢索服務：雙層自動分層（Tiered Fallback）。

  Tier 1  sentence-transformers + torch + all-MiniLM-L6-v2
          完整神經語意能力（同義改寫泛化）；需要可載入 torch 的 CPU（具備 AVX2）。
  Tier 2  純本機檢索引擎（numpy：字元 n-gram 倒排索引 + BM25）
          零神經網路依賴，保證無 AVX2 的 VM/舊 CPU 上依然完整離線運作。

分層策略讓同一份程式碼「能在哪種機器就用哪種能力」：
展示用筆電走 Tier 1 獲得最佳語意匹配；內網封閉 VM 走 Tier 2 不靠任何神經框架。
Tier 1 模型一律只從本地 models/ 載入（local_files_only=True），運作時完全斷網。

評分規格（詳見 retriever.py）：
  Tier 1：向量餘弦相似度（0~1），門檻 0.30
  Tier 2：BM25 相關度（顯示時壓縮至 0~1），門檻 1.20（原始分數）
不同引擎的分數分布不同，故每層配一組門檻值（similarity_threshold）。
"""

import threading
from typing import List

import numpy as np

from .config import EMBEDDING_DIM, LOCAL_ENGINE_DIM, MODEL_DIR, MODEL_ID

# 各引擎的相關度門檻（低於者視為「不相關」）
TIER1_THRESHOLD = 0.30  # MiniLM 餘弦分布（0~1）
TIER2_THRESHOLD = 1.20  # 本機引擎 BM25 原始分數（典型相關命中 2~30，離題 ≈ 0）


class EmbeddingService:
    """向量服務（單例 + 執行緒鎖 + 延遲載入）。"""

    def __init__(self) -> None:
        self._impl = None          # ("minilm-st", model) 或 ("local", engine)
        self._lock = threading.Lock()
        self.engine_tier = 0
        self.engine_name = "未初始化"
        self.similarity_threshold = TIER2_THRESHOLD
        self.vector_dim = EMBEDDING_DIM   # 當前引擎的向量維度（Tier 1=384 / Tier 2=8192）
        self.engine_note = ""      # 分層決策的原因（診斷用，非錯誤）

    # ------------------------------------------------------------------
    @property
    def local_engine(self):
        """Tier 2 時回傳 LocalNgramEngine 實例，否則 None。"""
        kind, impl = self._impl or (None, None)
        return impl if kind == "local" else None

    def is_loaded(self) -> bool:
        return self._impl is not None

    def load(self) -> None:
        """依序嘗試 Tier 1 → Tier 2。Tier 2 恆可成功（僅需 numpy）。"""
        with self._lock:
            if self._impl is not None:
                return

            # ---------- Tier 1：sentence-transformers（需 torch 可載入）----------
            try:
                from sentence_transformers import SentenceTransformer

                # local_files_only=True：禁止任何網路存取，純離線載入
                model = SentenceTransformer(
                    str(MODEL_DIR), device="cpu", local_files_only=True
                )
                self._impl = ("minilm-st", model)
                self.engine_tier = 1
                self.engine_name = f"{MODEL_ID} (sentence-transformers + torch)"
                self.similarity_threshold = TIER1_THRESHOLD
                self.vector_dim = EMBEDDING_DIM
                self.engine_note = "Tier 1 神經語意模型已啟用"
                return
            except Exception as exc:  # noqa: BLE001 - 原因需透傳給 health 端點
                tier1_reason = f"{type(exc).__name__}: {exc}"

            # ---------- Tier 2：純本機向量引擎 ----------
            from .local_engine import LocalNgramEngine

            self._impl = ("local", LocalNgramEngine())
            self.engine_tier = 2
            self.engine_name = "本機檢索引擎 (numpy n-gram 倒排索引 + BM25)"
            self.similarity_threshold = TIER2_THRESHOLD
            self.vector_dim = LOCAL_ENGINE_DIM
            # 記錄未採用 Tier 1 的原因（供 /api/health 診斷）
            short = tier1_reason[: 160].replace("\n", " ")
            self.engine_note = (
                "Tier 1 未啟用（通常因 CPU 無 AVX2，torch/onnxruntime 無法載入），"
                f"自動切換至本機向量引擎。詳情：{short}"
            )

    # ------------------------------------------------------------------
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """將文字清單編碼為 (N, 384) 的 L2 正規化向量矩陣。"""
        if self._impl is None:
            self.load()
        kind, impl = self._impl
        if kind == "minilm-st":
            vectors = impl.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            return vectors.astype(np.float32)
        # local engine
        return impl.encode(texts)


# 模組級單例
embedding_service = EmbeddingService()
