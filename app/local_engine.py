"""
純本機檢索引擎（零神經網路依賴）。

為什麼需要它：PyTorch（CPU 版）與 ONNX Runtime 在「不支援 AVX2 指令集」的
CPU 上皆無法運作——torch 會丟出 c10.dll 初始化失敗（WinError 1114），
ONNX Runtime 更會直接 segfault。舊型 CPU 與許多虛擬機器都屬於此類環境。

此引擎改以「字元 n-gram 字串鍵倒排索引 + BM25」運作：
中文以 2~3 字 n-gram 捕捉詞形（「特休」「加班費」「vpn」），英文同樣覆蓋。
刻意不使用雜湊維度做檢索——8192 維的雜湊仍會讓「氣壓力」與「間享有」
碰撞，使離題查詢得到虛假高分；字串鍵索引零碰撞，
且 BM25 的 IDF、詞頻飽和（k1）與長度正規化（b）都是資訊檢索的成熟工具。

與 Tier 1（MiniLM 餘弦）的分工：
  - 檢索品質：兩者都能精準定位文件與章節；Tier 1 另有同義改寫的語意泛化。
  - 離線保底：Tier 2 只需 numpy，在任何 x86 CPU 上都能運作。
"""

import hashlib
import math
import unicodedata

import numpy as np

from .config import LOCAL_ENGINE_DIM


class LocalNgramEngine:
    """字元 n-gram 倒排索引 + BM25 檢索引擎（執行緒安全、完全離線）。"""

    # 入庫特徵向量維度（TF 雜湊表徵，供維度哨兵與資料分析；
    # 實際檢索不走向量，見模組 docstring）。
    # 維度刻意高於 MiniLM 的 384：n-gram 詞表達數萬，壓在 384 維會讓
    # 「上的」與「與測」這類無關 gram 大量碰撞。
    DIM = LOCAL_ENGINE_DIM
    # (n, 權重)：2-gram 對應中文二字詞（特休/婚假/報修），
    # 3-gram 覆蓋三字詞（加班費/會議室）與英文片語，給予較高權重。
    # 不採用 1-gram：單字高頻詞（的、是、一）會嚴重稀釋檢索訊號。
    NGRAMS = ((2, 1.0), (3, 1.5))

    # CJK 虛詞（功能詞）字元表：用於過濾 2-gram。
    # 原因：「火星上的大氣壓力？」這類離題查詢，可僅靠「上的」一個慣用
    # 虛詞 2-gram 就越過門檻。3-gram 保留不過濾，避免傷害短詞查詢的召回。
    # （中日韓 IR 引擎的標準做法，如 Lucene 的 CJK 分析器停用詞。）
    # 注意：刻意不放入「以」「怎」「麼」「多」「少」等字——
    # 「可以」「怎麼」「多少」是提問句的重要語意載體。
    STOPCHARS = frozenset("的而是在了上下不有這那及或之其也我你他嗎呢和與一")

    def __init__(self) -> None:
        # 倒排索引狀態（build_index 建立，僅存在記憶體）
        self._postings: dict = {}
        self._doc_postings: dict = {}
        self._doc_len: dict = {}
        self._doc_chunk_len: list = []
        self._doc_of: list = []
        self._n_docs = 0
        self._avgdl = 1.0
        self._doc_avgdl = 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _keep(gram: str) -> bool:
        """過濾含標點（P*）或空白（Z*）的 n-gram，避免「，的」「。2」等噪音。"""
        for ch in gram:
            if unicodedata.category(ch)[0] in ("P", "Z"):
                return False
        return True

    @staticmethod
    def _bucket(gram: str) -> int:
        """n-gram → 維度槽位（僅 encode() 入庫表徵使用）。"""
        return int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16) % LOCAL_ENGINE_DIM

    # ------------------------------------------------------------------
    def _iter_grams(self, text):
        """产出 (gram, 權重)，已過濾標點、空白與（2-gram 中的）虛詞。"""
        s = (text or "").lower()
        for n, weight in self.NGRAMS:
            for j in range(max(0, len(s) - n + 1)):
                gram = s[j : j + n]
                if not self._keep(gram):
                    continue
                if n == 2 and any(ch in self.STOPCHARS for ch in gram):
                    continue
                yield gram, weight

    def build_index(self, texts, doc_ids=None) -> None:
        """
        建立字串鍵倒排索引（gram → [(chunk_idx, tf 加權)]）。

        刻意不使用雜湊：8192 維雜湊仍會讓「氣壓力」與「間享有」碰撞，
        離題查詢因此得到虛假的高分。字串鍵倒排索引零碰撞，
        且 BM25 的 IDF / 飽和 / 長度正規化都能直接施用。

        doc_ids 為每個切片所屬的文件 id（同一文件的多個切片同一值），
        用於「文件級脈絡平滑」（BM25F 思路）：極短文件（如門禁密碼條）
        容易因 tf 集中而壓過資訊更完整的長文件，加入文件級分數校正。
        """
        if doc_ids is None:
            doc_ids = list(range(len(texts)))
        self._postings: dict = {}          # gram → [(chunk_idx, tf)]
        self._doc_of: list = []            # chunk_idx → doc_group_id
        self._doc_postings: dict = {}      # gram → {doc_group_id: tf 合計}
        self._doc_len: dict = {}           # doc_group_id → gram 總數
        self._doc_chunk_len: list = []
        for i, t in enumerate(texts):
            g = doc_ids[i] if i < len(doc_ids) else i
            self._doc_of.append(g)
            tf: dict = {}
            grams = 0
            for gram, w in self._iter_grams(t):
                tf[gram] = tf.get(gram, 0) + w
                grams += 1
            self._doc_chunk_len.append(max(grams, 1))
            self._doc_len[g] = self._doc_len.get(g, 0) + grams
            for gram, w in tf.items():
                self._postings.setdefault(gram, []).append((i, w))
                self._doc_postings.setdefault(gram, {})
                self._doc_postings[gram][g] = self._doc_postings[gram].get(g, 0) + w
        self._n_docs = len(texts)
        avg = sum(self._doc_chunk_len) / len(self._doc_chunk_len) if self._doc_chunk_len else 1.0
        self._avgdl = avg
        self._doc_avgdl = (
            sum(self._doc_len.values()) / len(self._doc_len) if self._doc_len else 1.0
        )

    def _bm25_core(self, query, postings, doc_len, avgdl, n_docs, k1, b):
        """BM25 主計算（供 chunk 層與文件層共用）。"""
        scores = {}
        qtf: dict = {}
        for gram, w in self._iter_grams(query):
            qtf[gram] = qtf.get(gram, 0) + w
        for gram, qw in qtf.items():
            plist = postings.get(gram)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            if isinstance(plist, dict):
                items = plist.items()
            else:
                items = plist
            for key, tf in items:
                dl = doc_len.get(key, avgdl)
                sat = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
                scores[key] = scores.get(key, 0.0) + idf * sat * (1.0 + math.log(qw))
        return scores

    def bm25_scores(self, query, k1: float = 1.5, b: float = 0.5, doc_alpha: float = 0.3):
        """
        BM25 相關度（k1=1.5, b=0.5）。回傳 (N,) chunk 分數陣列。

        分數 = chunk 層 BM25 + doc_alpha × 文件層 BM25（脈絡平滑）：
        同一文件的其他切片若也命中查詢詞，代表該文件整體高度相關，
        可校正極短文件的 tf 集中偏置（BM25F 的核心想法）。
        """
        if self._n_docs == 0 or not getattr(self, "_postings", None):
            return np.zeros(max(self._n_docs, 0), dtype=np.float32)

        chunk_scores = self._bm25_core(
            query, self._postings, {i: l for i, l in enumerate(self._doc_chunk_len)},
            self._avgdl, self._n_docs, k1, b,
        )
        doc_scores = self._bm25_core(
            query, self._doc_postings, self._doc_len, self._doc_avgdl,
            len(self._doc_len), k1, b,
        )
        out = np.zeros(self._n_docs, dtype=np.float32)
        for i in range(self._n_docs):
            g = self._doc_of[i]
            out[i] = chunk_scores.get(i, 0.0) + doc_alpha * doc_scores.get(g, 0.0)
        return out

    # ------------------------------------------------------------------
    def encode(self, texts):
        """子線性 TF 特徵向量（雜湊維度，寫入 SQLite 的 chunks.embedding）。

        用途有二：
          1. 維度哨兵——retriever 以向量長度（8192 ≠ 384）偵測
             「跨引擎分層」造成的舊資料，提示執行 reset.bat。
          2. 資料分析——chunks 表在兩種引擎下都有實質的表徵內容。
        Tier 2 的實際檢索由倒排索引 + BM25 負責，不讀取本向量。
        """
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            feats = {}
            for gram, weight in self._iter_grams(t):
                h = self._bucket(gram)
                feats[h] = feats.get(h, 0) + weight
            for h, wsum in feats.items():
                out[i, h] = 1.0 + math.log(wsum)  # 子線性 TF
        return out.astype(np.float32)
