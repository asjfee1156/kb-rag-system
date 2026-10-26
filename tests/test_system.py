# -*- coding: utf-8 -*-
r"""
自動化驗收測試：對執行中的離線知識庫服務跑完整流程。

用法：
  1. 雙擊 start.bat（或 python -m uvicorn app.main:app）
  2. 執行：  python tests/test_system.py
     （用 bundled runtime 亦可： runtime\python.exe tests/test_system.py）

僅使用 Python 標準庫（urllib），無需安裝 pytest 或 requests。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ.get("KB_BASE", "http://127.0.0.1:10086")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post_json(path, obj):
    body = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def post_file(path, abs_path):
    boundary = uuid.uuid4().hex
    filename = os.path.basename(abs_path)
    with open(abs_path, "rb") as f:
        content = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def delete(path):
    req = urllib.request.Request(BASE + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def ask(question):
    _, data = post_json("/api/query", {"question": question})
    return data


def top_doc(resp):
    if resp["citations"]:
        return resp["citations"][0]["doc_name"]
    return None


def main():
    print("=" * 72)
    print(f"目標服務：{BASE}   專案：{PROJECT}")
    print("=" * 72)

    # ---- 0. 服務與引擎 ----
    t0 = time.time()
    try:
        h = get("/api/health")
    except Exception as e:  # noqa: BLE001
        print("服務無法連線:", e)
        sys.exit(1)
    check("health 回傳 ok", h.get("status") == "ok", str(h))
    check("向量引擎已分層選定", bool(h.get("engine")) and h.get("engine_tier") in (1, 2), str(h.get("engine")))
    check(
        "預設知識已匯入（3 份）",
        h.get("documents") == 3,
        f"documents={h.get('documents')}",
    )
    check("向量已建立", h.get("vectors") >= 10, f"vectors={h.get('vectors')}")

    docs = get("/api/documents")["documents"]
    names = {d["filename"] for d in docs}
    check("預設文件齊全（IT/福利/大門密碼）", {
        "公司內部IT與網路安全規範.md",
        "企業員工福利與休假制度指南.md",
        "公司大門密碼.txt",
    }.issubset(names), str(sorted(names)))

    # ---- 1. 預設知識的 10 題 ----
    cases = [
        ("VPN 登入命中 IT 規範", "請問 VPN 要怎麼登入？", ("公司內部IT與網路安全規範.md",)),
        # 以下兩題語料中存在兩個合理來源（大門門禁密碼 / 系統帳號密碼），
        # 系統應命中其中之一，並在參考來源同時列出另一個
        ("密碼週期命中任一密碼來源", "密碼多久需要更換一次？", ("公司內部IT與網路安全規範.md", "公司大門密碼.txt")),
        ("報修流程命中 IT 規範", "電腦壞了要怎麼報修？", ("公司內部IT與網路安全規範.md",)),
        ("忘記密碼命中任一密碼來源", "忘記密碼怎麼辦？", ("公司內部IT與網路安全規範.md", "公司大門密碼.txt")),
        ("公共 WiFi 同義改寫命中", "在咖啡廳可以直接連公司系統嗎？", ("公司內部IT與網路安全規範.md",)),
        ("試用期病假命中福利指南", "試用期員工可以請病假嗎？", ("企業員工福利與休假制度指南.md",)),
        ("特休天數命中福利指南", "到職滿一年的特休有幾天？", ("企業員工福利與休假制度指南.md",)),
        ("加班費命中福利指南", "加班費怎麼計算？", ("企業員工福利與休假制度指南.md",)),
        ("婚假命中福利指南", "婚假有幾天？", ("企業員工福利與休假制度指南.md",)),
    ]
    for name, q, expect_doc in cases:
        r = ask(q)
        ok = top_doc(r) in expect_doc
        score = r["citations"][0]["score"] if r["citations"] else 0
        check(
            name,
            ok,
            f"top={top_doc(r)} expected={expect_doc} score={score}",
        )

    r = ask("公司大門密碼是多少？")
    check("大門密碼答案包含 10086", "10086" in r["answer"], r["answer"][:80])
    check("大門密碼命中來源文件", top_doc(r) == "公司大門密碼.txt", str(top_doc(r)))

    # ---- 2. 難不倒系統的問題（知識庫外）----
    r = ask("火星上的大氣壓力是多少？")
    check("知識範圍外問題不崩潰", "找不到" in r["answer"] or r["retrieved_count"] == 0, r["answer"][:60])

    # ---- 3. 擴充知識庫 ----
    status, r4 = post_file("/api/upload", os.path.join(PROJECT, "samples", "產品研發流程與專案管理規範.md"))
    check("上傳研發流程文件 201", status == 200, f"http={status}")
    check("研發文件切片 > 0", r4.get("chunk_count", 0) > 0, str(r4))

    status, r5 = post_file("/api/upload", os.path.join(PROJECT, "samples", "辦公室環境與設備使用須知.md"))
    check("上傳辦公室須知 200", status == 200, f"http={status}")

    for name, q, expect_doc in [
        ("發版窗口命中研發流程", "發版通常排在星期幾？", "產品研發流程與專案管理規範.md"),
        ("PR 審核命中研發流程", "Pull Request 需要幾個人審核？", "產品研發流程與專案管理規範.md"),
        ("門禁卡補辦命中辦公室須知", "員工門禁卡遺失了怎麼補辦？", "辦公室環境與設備使用須知.md"),
        ("會議室預約命中辦公室須知", "會議室要怎麼預約？", "辦公室環境與設備使用須知.md"),
    ]:
        r = ask(q)
        check(name, top_doc(r) == expect_doc, f"top={top_doc(r)}")

    # ---- 4. 異常與邊界 ----
    status, dup = post_file("/api/upload", os.path.join(PROJECT, "samples", "公司大門密碼.txt"))
    check("重複上傳回 409", status == 409, f"http={status} detail={dup.get('detail', '')[:60]}")

    status, bad = post_json("/api/query", {"question": "   "})
    check("空問題被拒（400）", status == 400, f"http={status}")

    hist = get("/api/history")["history"]
    check("歷史紀錄持續累積", len(hist) >= 13, f"count={len(hist)}")

    before = get("/api/health")
    status, _ = delete(f"/api/documents/{r5['document_id']}")
    after = get("/api/health")
    check("刪除文件成功（200）", status == 200, f"http={status}")
    check("刪除後向量同步減少", after["vectors"] < before["vectors"], f"{before['vectors']} -> {after['vectors']}")

    # ---- 5. 效能 ----
    r = ask("VPN 登入方式")
    check("單次檢索延遲 < 200ms", r["latency_ms"] < 200, f"latency={r['latency_ms']}ms")

    # ---- 總結 ----
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    elapsed = round(time.time() - t0, 1)
    print("=" * 72)
    print(f"結果：{passed}/{total} 通過（{total - passed} 失敗）　耗時 {elapsed}s")
    print(f"引擎：{h.get('engine')}（threshold={h.get('similarity_threshold')}）")
    print("=" * 72)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
