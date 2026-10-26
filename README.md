# 企業級離線智能知識庫與 RAG 檢索系統

> Local Offline Knowledge Base & RAG Retrieval System · **v1.1.1**
> 一份文件看懂整套系統「每一個零件」的運作原理

---

## 目錄

1. [專案簡介](#1-專案簡介)
2. [目錄結構](#2-目錄結構)
3. [快速開始](#3-快速開始)
4. [系統架構與資料流](#4-系統架構與資料流)
5. [雙層檢索引擎（核心設計）](#5-雙層檢索引擎核心設計)
6. [各模組原理詳解](#6-各模組原理詳解)
7. [RESTful API 參考](#7-restful-api-參考)
8. [打包與移機](#8-打包與移機)
9. [維運腳本](#9-維運腳本)
10. [展示流程建議](#10-展示流程建議)
11. [常見問題](#11-常見問題)
12. [測試](#12-測試)
13. [進階擴充方向](#13-進階擴充方向)
14. [版本歷史](#14-版本歷史)

---

## 1. 專案簡介

這是一套**完全離線、零雲端依賴、解壓即用**的企業知識庫檢索系統：
把企業文件丟進去，系統自動完成「切片 → 特徵化 → 入庫」，
提問時從向量庫 / 倒排索引中找出最相關的片段、組成答案並**附上來源標註（Citation）**。

| 特色 | 說明 |
| --- | --- |
| 完全離線 | 引擎、資料庫、前端全部在專案資料夾內，拔掉網線也能完整運作 |
| 免安裝 | 內建 Python 3.12 執行環境（`runtime/`），解壓後雙擊 `start.bat` 即用 |
| 不改變本機環境 | 不寫入登錄檔、不安裝任何套件到系統 Python，關閉後不留痕跡 |
| 跨機器相容 | Win10/Win11；偵測到無 AVX2 的 VM/舊 CPU 會自動降級到純 numpy 引擎 |
| 零外部服務 | 不需要 OpenAI、不需要向量資料庫、不需要 Docker、不需要 Node.js |
| 來源可追溯 | 每個答案都標注出處文件、片段編號與相關度分數 |
| 單檔前端 | 一個 `index.html` 完成全部 UI，無 CDN、無外部引用 |

---

## 2. 目錄結構

```
 kb-server/                      （本資料夾整包就是可分發單位）
 ├── start.bat                   # 一鍵啟動：啟動後端 + 延遲開啟瀏覽器
 ├── stop.bat                    # 強制關閉服務（釋放 port 10086）
 ├── reset.bat                   # 清空知識庫並還原為預設 3 份文件
 ├── README.md                   # 本文件
 ├── requirements.txt            # 依賴清單（開發/移機重建用；運作時不需要）
 │
 ├── app/                        # 後端主程式（FastAPI 套件）
 │   ├── main.py                 #   入口：生命週期、路由、答案組裝
 │   ├── config.py               #   全域設定：路徑、port、切片參數、門檻
 │   ├── database.py             #   SQLite：documents / chunks / history 三表
 │   ├── chunker.py              #   文本切片：標題感知 + 段落 + 句子滑動視窗
 │   ├── embedding.py            #   引擎分層服務（Tier 1 / Tier 2 偵測與門檻）
 │   ├── local_engine.py         #   Tier 2 引擎：n-gram 倒排索引 + BM25
 │   ├── retriever.py            #   檢索引擎：向量矩陣 + Top-K + 分層評分
 │   ├── corpus.py               #   語料一致性：索引與資料庫同步
 │   ├── seed.py                 #   預設知識導入（首次啟動 / reset.bat）
 │   └── static/
 │       └── index.html          #   單頁前端（HTML + CSS + JS 全內聯）
 │
 ├── runtime/                    # 內建 Python 3.12.8 embeddable + 依賴套件
 │   ├── python.exe              #   解壓即用，不需本機已安裝 Python
 │   ├── python312._pth          #   模組搜尋路徑（含 ..\ 指向專案根）
 │   └── Lib/site-packages/      #   fastapi / uvicorn / numpy / python-multipart
 │
 ├── models/
 │   └── all-MiniLM-L6-v2/       # Tier 1 神經模型權重（離線，~90MB）
 │
 ├── data/                       # 運行時資料（可隨時刪除重建）
 │   ├── knowledge.db            #   SQLite 資料庫
 │   └── uploads/                #   上傳檔案副本
 │
 ├── samples/                    # 知識文件樣本
 │   ├── 公司內部IT與網路安全規範.md      ┐
 │   ├── 企業員工福利與休假制度指南.md    ├ 預設 3 份（首次啟動自動匯入）
 │   ├── 公司大門密碼.txt                ┘
 │   ├── 產品研發流程與專案管理規範.md    ┐ 手動示範「上傳擴充知識」用
 │   ├── 辦公室環境與設備使用須知.md      ┘
 │   └── 演示提問清單.txt                # 13 題 demo 腳本（含預期答案）
 │
 └── tests/
     └── test_system.py          # 端到端驗收測試（30 項，純標準庫）
```

> `data/` 與 `runtime/Lib/site-packages` 都在資料夾內——
> 把整包壓成 zip 丟到任何 Win10/Win11 機器解壓即可使用，真正「zip 解壓即用」。

---

## 3. 快速開始

### 3.1 一般使用者

1. 解壓資料夾（放哪裡都行，路徑含中文也無妨）。
2. 雙擊 **`start.bat`**：
   - 設定 UTF-8 碼頁（`chcp 65001`）
   - 優先使用 `runtime\python.exe`（免本機 Python）；若 runtime 缺失才退回系統 Python
   - 啟動 FastAPI 後端於 `http://127.0.0.1:10086`
   - 延遲數秒後自動開啟預設瀏覽器
3. **首次啟動**會自動把 `samples/` 的預設 3 份文件匯入知識庫（之後可隨時用 `reset.bat` 還原）。
4. 用完雙擊 **`stop.bat`** 關閉服務。

### 3.2 操作介面

- **左側**：拖放 `.txt` / `.md` 上傳（**支援一次拖入或選擇多個檔案**，逐檔上傳並顯示整體進度）、
  文件清單與刪除、引擎狀態卡；文件過多時清單與快速體驗所在的中段會**獨立捲動**，不擠壓其他區塊
- **右側**：檢索對話視窗（內容過長時對話框**獨立上下捲動**，輸入列永遠可見），
  答案下方可展開**引用來源**卡片，並附**「全部展開 / 全部收合」**切換鈕
- 文件刪除為**兩段式切換**：點一下 ✕ 進入紅色「確認?」待命態，3 秒內再點才真正刪除（防止誤刪）
- 底部：清除歷史、快速體驗按鈕

### 3.3 想動手改的開發者

```bash
# 用本機 Python 開發（需 Python 3.10+）
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 10086
# 或直接用內建 runtime（不需要任何 pip）
runtime\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 10086
```

---

## 4. 系統架構與資料流

```
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  上傳檔案   │ →  │ chunker  │ →  │ engine   │ →  │ SQLite  │
│ .txt / .md  │    │ 標題/段落 │    │ encode   │    │ chunks  │
└─────────────┘    └──────────┘    └──────────┘    └─────────┘
                                                            │
提問 ──→ engine 評分（Tier 1 餘弦 / Tier 2 BM25）──→ Top-K 門檻過濾
                                                        │
      ┌─────────────────────────────────────────────────┘
      ▼
 答案組裝（首選片段 + 參考來源）+ 引用清單 ──→ 前端對話視窗
      │
      └──→ SQLite history 表（提問、答案、引用一併留存）
```

上傳一篇文章後，`/api/upload` 的處理步驟：
**multipart 解析（含中文檔名還原）→ 副檔名與大小檢查 → 編碼容錯解碼
→ 標題感知切片 → 引擎編碼 → 雜湊去重（重複上傳回 409）→ 入庫 → 語料索引同步**。

---

## 5. 雙層檢索引擎（核心設計）

### 5.1 為什麼要分層

`all-MiniLM-L6-v2` 透過 `sentence-transformers + torch` 載入，
而 **torch 的 CPU 版與 ONNX Runtime 在「不支援 AVX2 指令集」的 CPU 上無法執行**：
torch 會在 `c10.dll` 初始化時失敗（`WinError 1114`），ONNX Runtime 直接 segfault。
舊型 CPU 與多數虛擬機器都屬於此類環境——若只有單層引擎，系統在這些機器上會完全無法啟動。

因此系統在啟動時自動偵測並分層（`app/embedding.py`）：

| | Tier 1（優先） | Tier 2（保底） |
| --- | --- | --- |
| 引擎 | sentence-transformers + torch + all-MiniLM-L6-v2 | 純本機：numpy n-gram 倒排索引 + BM25 |
| 評分 | 384 維 L2 正規化向量的**餘弦相似度** | **BM25**（k1=1.5, b=0.5）+ 文件級脈絡平滑 |
| 門檻 | 0.30 | 1.20（原始分） |
| 顯示分數 | 0~1 直接顯示 | raw/(raw+4) 壓縮至 0~1 |
| 需要 AVX2 | 是 | 否（只需 numpy） |
| 語意泛化 | 強（同義改寫能命中） | 弱（倚賴詞形重疊） |

兩層的**檢索結果與 API 介面完全一致**，門檻各自獨立；
`/api/health` 與前端狀態卡透明顯示目前使用哪一層。

### 5.2 Tier 1：MiniLM + 餘弦相似度

- 模型只從本地 `models/all-MiniLM-L6-v2` 載入（`local_files_only=True`），**禁止任何網路存取**。
- 入庫與查詢向量皆已 L2 正規化 → 餘弦相似度 = 內積 → 檢索只需一次矩陣乘法。

### 5.3 Tier 2：倒排索引 + BM25（本機引擎）

設計過程踩過的坑（全部寫在 `app/local_engine.py` 註解，避免後人重蹈）：

1. **雜湊向量不可行**：把 n-gram 雜湊到 384/8192 維做餘弦，
   「氣壓力」與「間享有」會碰撞，離題查詢（火星大氣）憑空得到 0.15~0.5 的餘弦。
2. **純餘弦的長文件偏置**：短查詢 vs 長切片，分母被切片範數主導，真實匹配全部壓在 0.1 以下。
3. **最終方案**：**字串鍵倒排索引（零碰撞）+ BM25**：
   - 字元 2-gram（權重 1.0）+ 3-gram（權重 1.5），中文二字詞與英文片語皆能覆蓋
   - 2-gram 過濾 CJK 虛詞（的/是/在/了/…；刻意保留「可以」「怎麼」「多少」等提問語意載體）
   - 標點與空白 n-gram 一律丟棄（`unicodedata` 分類判斷）
   - BM25 的 IDF（`log(1+(N-df+0.5)/(df+0.5))`）、詞頻飽和（k1）、長度正規化（b）
   - **文件級脈絡平滑**：chunk 分數 + 0.3 × 所屬文件整體分數（BM25F 思路），
     校正極短文件（如門禁密碼條）的詞頻集中偏置

效果（13 題實測，詳見 `測試報告.md`）：真實問題分數 2~30，離題問題 **0**，
分離度足夠讓 1.20 的門檻穩定運作。

### 5.4 語料一致性（`app/corpus.py`）

Tier 2 的索引統計（IDF、文件長度）只存在記憶體，**來源永遠是資料庫本身**：
上傳 / 刪除 / 還原 / 啟動都會呼叫 `refresh_local_corpus()` 重建，不需持久化、不可能不一致。
Tier 1 的向量語意由模型決定，與語料規模無關，僅需重建記憶體矩陣。

---

## 6. 各模組原理詳解

### `app/config.py`
集中所有常數：host/port（`127.0.0.1:10086`）、`CHUNK_SIZE=480` / `CHUNK_OVERLAP=120`、
`TOP_K=3`、`MAX_UPLOAD_MB`、模型與資料路徑。
路徑一律以 `BASE_DIR` 反推（`__file__` 往上一層），所以專案資料夾搬到哪裡都能跑。

### `app/database.py`
SQLite 三張表：
- `documents`：文件中繼資料 + 內容雜湊（重複上傳偵測）
- `chunks`：切片內容 + 向量（JSON 字串；Tier 1 為 384 維正規化向量，Tier 2 為 8192 維特徵表徵）
- `history`：提問 / 答案 / 引用清單

連線逐次建立、關閉，避免跨執行緒共享問題；PRAGMA 開啟 WAL 提升並發讀寫。

### `app/chunker.py`
「標題感知 + 段落感知 + 滑動重疊視窗」三層策略：
先以 Markdown 標題（`#`~`######`）切硬邊界，避免不同章節併入同一切片；
區塊內逐行逐句（中英文句點）拆解，再以滑動視窗組裝成 ~480 字元的切片，
相鄰切片重疊 120 字元保持語意連貫。`.txt` 無標題時整篇視為單一區塊。

### `app/embedding.py`
`EmbeddingService` 單例：延遲載入 + 執行緒鎖，Tier 1 失敗時記錄原因並降級 Tier 2。
對外統一 `encode()` / `similarity_threshold` / `engine_tier` / `vector_dim` / `local_engine`。

### `app/retriever.py`
`RetrievalEngine` 單例：啟動與語料變動時從 SQLite 重建索引；
`search()` 依分層評分（餘弦 / BM25），以 `argpartition` 取 Top-K 再精確排序（O(N)）。
**維度哨兵**：入庫向量長度必須等於當前引擎維度（384/8192），
若把資料夾從有 AVX2 的機器搬到無 AVX2 的機器（或反向），舊向量會被排除並在
`/api/health` 的 `dim_mismatch` 欄位計數——此時執行 `reset.bat` 重建即可。

### `app/seed.py`
首次啟動（空庫）時把 `samples/` 預設 3 份文件寫入：編碼容錯、雜湊去重、冪等。
`reset.bat` 就是呼叫它來還原狀態。

### `app/main.py`
FastAPI 應用：`lifespan` 負責「初始化 → 引擎載入 → 空庫自動匯入 → 語料同步」，
之後提供 8 個端點（見 [§7](#7-restful-api-參考)）。
答案組裝採**擷取式**：首選片段為答案主體，其餘片段列為參考來源——
不呼叫任何 LLM，過程完全斷網，結果可驗證。

### `app/static/index.html`
單檔前端：深色科技風、無外部資源。狀態卡顯示引擎分層與向量數；
對話視窗支援歷史載入、引用展開、相似度徽章；上傳區支援拖放與進度。

---

## 7. RESTful API 參考

| 方法 | 路徑 | 說明 |
| --- | --- | --- |
| GET | `/` | 單頁前端 |
| GET | `/api/health` | 引擎分層、門檻、維度、文件/向量數、啟動時間、Tier 1 未啟用原因 |
| POST | `/api/upload` | multipart 上傳 `.txt` / `.md`：解析 → 切片 → 入庫 → 索引同步 |
| GET | `/api/documents` | 文件清單（名稱、類型、字數、切片數、上傳時間） |
| DELETE | `/api/documents/{id}` | 刪除文件及其全部切片 |
| POST | `/api/query` | 檢索問答：`{"question": "...", "top_k": 3}` |
| GET | `/api/history` | 歷史對話（含引用） |
| DELETE | `/api/history` | 清空歷史 |

`/api/query` 回應範例：

```json
{
  "question": "公司大門密碼是多少？",
  "answer": "依據《公司大門密碼.txt》第 1 個片段：\n\n公司大門門禁密碼：10086...",
  "citations": [
    {"doc_name": "公司大門密碼.txt", "chunk_index": 1, "score": 0.79, "content": "..."}
  ],
  "latency_ms": 0.23,
  "engine": "...",
  "threshold": 1.2
}
```

---

## 8. 打包與移機

- **分發就是複製資料夾**：整包壓成 zip（含 `runtime/` 與 `models/`），
  在任何 Win10/Win11 解壓即可雙擊 `start.bat` 使用。
- **runtime/ 的由來**：Python 官方 [embeddable distribution](https://www.python.org/downloads/windows/)
  （3.12.8）解壓至此，再用 pip `--target` 把 fastapi / uvicorn / numpy / python-multipart
  裝進 `runtime/Lib/site-packages`。`python312._pth` 加入 `..\` 讓它找得到 `app` 套件。
- **為什麼 runtime 裡沒有 torch**：刻意不裝——torch 是唯一需要 AVX2 的大件依賴；
  不裝它，Tier 2 路徑在任何 CPU 上都穩定。在有 AVX2 的機器上，
  若想啟用 Tier 1，把 `sentence-transformers` 與 `torch`（CPU 版）裝進
  `runtime/Lib/site-packages` 即可（開發機也可直接裝到系統 Python）。
- `requirements.txt` 僅供開發者重建環境用，**運作時不需要**。

---

## 9. 維運腳本

| 腳本 | 作用 |
| --- | --- |
| `start.bat` | 啟動服務（優先 bundled runtime）+ 延遲開瀏覽器；首次自動匯入預設知識 |
| `stop.bat` | 以 `netstat` 定位 port 10086 的行程並 `taskkill /F`，關閉乾淨 |
| `reset.bat` | 停止服務 → 清空 `data/knowledge.db` 與 `data/uploads/` → 重新導入預設 3 份 |

備份：直接複製 `data/knowledge.db` 即可；移機：整包資料夾複製即可。

---

## 10. 展示流程建議

1. 雙擊 `start.bat`，瀏覽器自動開啟；指出左下角狀態卡目前使用的引擎。
2. 打開 `samples/演示提問清單.txt`，逐題提問（清單含預期答案與展示重點）。
3. 提到知識範圍外（如「火星上的大氣壓力」）展示系統**誠實拒答**。
4. 拖入 `samples/產品研發流程與專案管理規範.md` 與 `辦公室環境與設備使用須知.md`
   （可一次拖兩個進去），立刻提問——展示知識擴充能力。
5. 展開引用卡片說明來源可追溯；最後 `reset.bat` 還原狀態。

---

## 11. 常見問題

**Q1：啟動時看見 `[WARN] sentence-transformers import failed`？**
屬正常降級：本機（或 VM）CPU 無 AVX2，torch/onnxruntime 無法載入，
系統自動切換 Tier 2 本機引擎，功能完全不受影響。
（歷史背景：早期版本在此環境會直接報 `WinError 1114 c10.dll` 失敗，v1.1.0 起改為分層降級。）

**Q2：`/api/health` 顯示 `dim_mismatch` 不為 0？**
資料夾在「AVX2 有無不同」的機器之間搬移過，舊向量維度與現役引擎不符。
執行 `reset.bat` 重建知識庫即可。

**Q3：port 10086 被佔用 / 啟動失敗？**
先雙擊 `stop.bat` 清掉舊行程；或檢查是否有其他程式佔用（`netstat -aon -p tcp ^| findstr 10086`）。

**Q4：上傳中文檔名變亂碼？**
已處理（multipart 中文檔名三段還原）；若以 curl 測試，記得用 UTF-8 編碼的檔名。

**Q5：首次啟動比較久？**
首次需建立 SQLite、導入預設文件、建立索引；之後啟動是毫秒級。

**Q6：怎麼確認現在用哪層引擎？**
前端左下狀態卡，或 `GET /api/health` 的 `engine` / `engine_tier`。

---

## 12. 測試

```bash
# 服務啟動後執行（用內建 runtime 或本機 Python 皆可）
runtime\python.exe tests\test_system.py
```

`tests/test_system.py` 是 30 項端到端驗收（純標準庫，無需 pytest）：
服務健康、引擎分層、預設 3 文件、13 題中 10 題命中比對、離題拒答、
上傳擴充、重複上傳 409、空問題 400、歷史累積、刪除同步、延遲基準。
完整執行紀錄見同目錄 **`測試報告.md`**。

---

## 13. 進階擴充方向

- **接入生成式 LLM**：`_compose_answer` 是擷取式；
  接上本地 Ollama / llama.cpp 即可把檢索片段餵入生成模型（補齊 RAG 的生成環節）。
- **更多文件格式**：在 `ALLOWED_EXTENSIONS` 加入 docx/pdf，配上對應解析器即可。
- **持久化向量索引**：目前索引為記憶體重建；切片量大時可考慮磁碟快取或輕量向量庫。
- **權限與多用戶**：SQLite 加 user_id 欄位，API 加 token 驗證。

---

## 14. 版本歷史

| 版本 | 重點 |
| --- | --- |
| v1.0.0 | FastAPI + sentence-transformers 單引擎版本 |
| v1.1.0 | 內建 Python 3.12 runtime（解壓即用、不理會本機環境）；雙層引擎自動分層（Tier 1 MiniLM / Tier 2 倒排索引 + BM25）；新增 stop.bat / reset.bat；標題感知切片；samples 擴充至 5 份文件 + 13 題演示清單；30 項驗收測試 |
| **v1.1.1** | 前端體驗修正：對話框恢復正常上下捲動（flex 溢出修復）、文件過多時側邊欄中段獨立捲動、文件刪除改為兩段式切換、引用來源新增「全部展開/收合」、支援多檔同時上傳 |
