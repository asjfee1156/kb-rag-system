"""
預設知識庫 seeding：直接以 app 模組把 samples/ 的預設文件寫入 SQLite。

用途：
  - reset.bat 呼叫（還原到預設狀態）
  - 初次使用時若想預先載入文件，也能手動執行： python -m app.seed

不需要伺服器運行，全程離線；重複執行時已存在的文件會自動跳過（依內容雜湊）。
"""

import shutil
import sys

from . import database
from .chunker import chunk_text
from .config import BASE_DIR, UPLOAD_DIR, ensure_directories
from .corpus import refresh_local_corpus
from .embedding import embedding_service

# 預設知識文件（reset.bat 會還原成這三份；samples/ 另外兩份保留給手動示範）
DEFAULT_SEEDS = [
    "samples/公司內部IT與網路安全規範.md",
    "samples/企業員工福利與休假制度指南.md",
    "samples/公司大門密碼.txt",
]


def seed_many(seed_files=None, quiet=False):
    """把指定檔案清單寫入知識庫；回傳 [(檔名, 狀態)]。"""
    ensure_directories()
    database.init_db()
    embedding_service.load()

    results = []
    for rel in seed_files if seed_files is not None else DEFAULT_SEEDS:
        path = BASE_DIR / rel
        if not path.exists():
            results.append((path.name, "檔案不存在，略過"))
            continue
        raw = path.read_bytes()

        # 編碼容錯（與 /api/upload 相同策略）
        text = None
        for enc in ("utf-8", "utf-8-sig", "big5", "cp950", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="ignore")

        # 重複偵測：內容雜湊已存在則跳過（冪等）
        file_hash = database.compute_file_hash(raw)
        if database.file_exists_by_hash(file_hash):
            results.append((path.name, "已存在，略過"))
            continue

        chunks = chunk_text(text)
        if not chunks:
            results.append((path.name, "無有效文字，略過"))
            continue

        vectors = embedding_service.encode(chunks)
        payload = [
            {
                "chunk_index": i,
                "content": c,
                "embedding": [round(float(v), 6) for v in vectors[i]],
            }
            for i, c in enumerate(chunks)
        ]
        database.insert_document(
            filename=path.name,
            file_type=path.suffix.lstrip("."),
            file_hash=file_hash,
            content_length=len(text),
            chunks=payload,
        )
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, UPLOAD_DIR / path.name)
        results.append((path.name, f"已寫入 {len(chunks)} 個片段"))

    # 語料一致性：Tier 2 需以最新語料重建索引，Tier 1 則重建索引
    if refresh_local_corpus() and not quiet:
        print("[SEED] 已依最新語料重建檢索索引")
    return results


if __name__ == "__main__":
    files = sys.argv[1:] or None
    for name, status in seed_many(files):
        print(f"[SEED] {name}: {status}")
