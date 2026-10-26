"""
全域配置：所有路徑、模型與檢索參數集中管理。
專案採「相對於專案根目錄」的定位方式，整個資料夾搬到任何路徑都能運作，
完全不需要改設定，滿足「免安裝、可攜帶、離線執行」的部署需求。
"""

from pathlib import Path

# 專案根目錄：app/config.py 的上一層
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ 路徑
# 本地 SentenceTransformer 模型（離線載入，不連網）
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"

# 資料目錄：SQLite 資料庫 + 上傳檔案
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "knowledge.db"

# 前端靜態資源（單一 HTML 檔，完全內聯）
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# ------------------------------------------------------------------ 模型
EMBEDDING_DIM = 384      # all-MiniLM-L6-v2 輸出維度（Tier 1）
LOCAL_ENGINE_DIM = 8192  # 本機 n-gram 引擎維度（Tier 2）
MODEL_ID = "all-MiniLM-L6-v2"

# ------------------------------------------------------------------ 切片與檢索
CHUNK_SIZE = 480         # 每個切片目標字元數
CHUNK_OVERLAP = 120      # 相鄰切片重疊字元數（避免語意被硬切斷）
TOP_K = 3                # 檢索回傳的片段數
SIM_THRESHOLD = 0.30     # 餘弦相似度門檻：低於此值視為「找不到相關內容」

# ------------------------------------------------------------------ 服務
HOST = "127.0.0.1"
PORT = 10086
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".txt", ".md"}


def ensure_directories() -> None:
    """啟動時建立必要的資料夾（data/、data/uploads/）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
