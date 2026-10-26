"""
語料一致性管理：檢索索引與 SQLite 內容保持同步。

資料流向：
  upload / seed / delete / 啟動  →  refresh_local_corpus()
      →  retrieval_engine.rebuild()
            - Tier 1（MiniLM）：從 chunks 表載入向量矩陣（餘弦檢索用）
            - Tier 2（本機引擎）：以全部切片文字重建 BM25 倒排索引

倒排索引只存在記憶體，統計來源永遠是資料庫本身，因此不需持久化、
也不需重算任何向量即可保持一致。
"""

from .embedding import embedding_service
from .retriever import retrieval_engine


def refresh_local_corpus() -> bool:
    """
    語料變動後呼叫，保持「檢索索引」與資料庫內容一致。

      Tier 1（MiniLM）：向量語意由模型決定，與語料規模無關 → 重建索引即可。
      Tier 2（本機引擎）：retriever.rebuild() 內部會以切片文字重建
                         BM25 倒排索引（統計來源永遠是資料庫本身）。

    回傳 True 表示目前為 Tier 2（本機引擎）。
    """
    retrieval_engine.rebuild()
    return embedding_service.engine_tier == 2
