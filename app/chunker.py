"""
文本切片（Chunking）模組。

RAG 流程的第一步：把長文件切成適合 Embedding 的語意單元。
切得太短會丟失上下文、太長會稀釋關鍵詞的向量訊號，
這裡採用「標題感知 + 段落感知 + 滑動重疊視窗」策略：

1. 先以 Markdown 標題（# ~ ######）切出區塊——不同章節絕不相連續
   （避免「婚假」切片開頭出現「事假」的內容）。.txt 等無標題文件
   則整篇視為單一區塊，行為與舊版相同。
2. 區塊內以行為單位，逐句拆解（中西文句點皆支援）。
3. 以滑動視窗把句子組裝成 ~CHUNK_SIZE 字元的切片，
   相鄰切片保留 CHUNK_OVERLAP 重疊，保持語意連貫。
"""

import re
from typing import List

from .config import CHUNK_OVERLAP, CHUNK_SIZE

# 中英文句末標點：。！？；。!?;
_SENTENCE_END = re.compile(r"(?<=[。！？；!?])\s*")
# Markdown ATX 標題：# ~ ######（行首 1-6 個 # 加空白）
_HEADING = re.compile(r"^#{1,6}\s+")


def _clean_text(text: str) -> str:
    """統一換行與空白，去除無意義的裝飾。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Markdown 標題符號適度保留（標題本身有語意），但移除連續裝飾線
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(paragraph: str) -> List[str]:
    """將單一段落切成句子。"""
    parts = [p.strip() for p in _SENTENCE_END.split(paragraph) if p.strip()]
    return parts or [paragraph.strip()]


def _merge_units(units: List[str], max_size: int, overlap: int) -> List[str]:
    """滑動視窗組裝：每個切片不超過 max_size，相鄰切片重疊約 overlap 字元。"""
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    i = 0
    while i < len(units):
        unit = units[i]
        if buf and buf_len + len(unit) > max_size:
            chunks.append("\n".join(buf))
            # 保留尾部內容作為下一個切片的開頭（重疊）
            tail: List[str] = []
            tail_len = 0
            for u in reversed(buf):
                if tail_len + len(u) > overlap:
                    break
                tail.insert(0, u)
                tail_len += len(u)
            buf, buf_len = tail, tail_len
        buf.append(unit)
        buf_len += len(unit)
        i += 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _split_by_headings(text: str) -> List[str]:
    """以 Markdown 標題列為硬邊界切出區塊（標題本身隸屬其後內容）。"""
    blocks: List[str] = []
    current: List[str] = []
    for line in text.split("\n"):
        if _HEADING.match(line.strip()):
            if current:
                blocks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def chunk_text(text: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """將完整文件內容切成 chunk 清單（標題優先、段落其次、句子為最小單元）。"""
    text = _clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    for block in _split_by_headings(text):
        if not block.strip():
            continue

        # 區塊內以行為單位，逐句拆解成語意單元
        units: List[str] = []
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            units.extend(_split_sentences(line))

        # 單一句子本身就超過上限（例如長表格、網址）：硬切
        final_units: List[str] = []
        for u in units:
            if len(u) <= max_size:
                final_units.append(u)
            else:
                final_units.extend(u[k : k + max_size] for k in range(0, len(u), max_size))

        chunks.extend(_merge_units(final_units, max_size, overlap))

    return [c for c in chunks if c.strip()]
