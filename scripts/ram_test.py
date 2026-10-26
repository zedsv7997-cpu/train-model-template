"""Kiểm tra RAM ngay trong Windows, không cần khởi động lại (2026-10-22, sau hai lần crash do bộ nhớ hỏng dữ liệu).

Nhiều tiến trình cùng ghi rồi đọc lại các mẫu bit trên phần lớn RAM còn trống; ô nhớ đọc ra khác giá trị vừa ghi là lỗi.
Chạy song song để ép bộ điều khiển bộ nhớ như lúc tải nặng (lỗi DDR5 chập chờn hay lộ ra khi đó). Mỗi lượt gồm:
  const  6 mẫu cố định (0, toàn 1, 0xAA…, 0x55…, 0x0F…, 0xF0…): ghi cả vùng rồi đọc lại
  addr   giá trị = chỉ số ô × hằng số lẻ + seed (mỗi ô một giá trị): bắt lỗi đường địa chỉ / ghi nhầm chỗ
  inv    moving inversions: đọc mẫu, ghi phần bù, đọc lại phần bù
  rand   số ngẫu nhiên PCG64 theo seed của lượt, sinh lại để so
Giới hạn: chỉ thử phần RAM Windows cấp cho tiến trình (không thử RAM hệ điều hành / ứng dụng khác đang giữ), không biết lỗi
nằm ở thanh nào (ô nhớ xen kẽ hai kênh). Có lỗi = RAM (hoặc thiết lập RAM / BIOS) không ổn định; 0 lỗi chưa chứng minh RAM
tốt — MemTest86 / Windows Memory Diagnostic (Extended) khởi động riêng thử được toàn bộ RAM.

  python scripts/ram_test.py --minutes 40 --log outputs/ram-test.log
  python scripts/ram_test.py --gb 1 --minutes 0.5 --self-test   # lật một bit sau khi ghi: phải báo đúng 1 lỗi
Ghi log từng dòng (flush + fsync) để máy có crash giữa chừng vẫn còn dấu vết; cuối cùng ghi <log>.json tóm tắt.
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

BLOCK = 8 << 20  # số ô uint64 mỗi lần so (64 MB) -> bộ nhớ tạm của phép so nhỏ
CONSTS = [0, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0]
MUL = np.uint64(0x9E3779B97F4A7C15)


class MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem_status():
    s = MemStatus()
    s.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s


def check(q, wid, test, arr, s, expected, stats):
    """So arr[s:s+len] với expected (số hoặc mảng); gửi tối đa 4 ô sai đầu tiên của khối."""
    bad = np.flatnonzero(arr[s:s + (expected.size if isinstance(expected, np.ndarray) else BLOCK)] != expected)
    if bad.size:
        stats["errors"] += int(bad.size)
        for i in bad[:4]:
            exp = int(expected[i] if isinstance(expected, np.ndarray) else expected)
            got = int(arr[s + i])
            q.put(("error", wid, test, int(s + i), exp, got, bin(exp ^ got).count("1")))


def worker(wid, n, deadline, seed, self_test, q):
    arr = np.empty(n, dtype=np.uint64)
    arr.fill(0)  # chạm hết các trang để Windows cấp RAM thật ngay
    stats = {"errors": 0, "passes": 0}
    p = 0
    while time.time() < deadline:
        t0 = time.time()
        for v in CONSTS:
            arr.fill(v)
            if self_test and p == 0 and wid == 0 and v == CONSTS[0]:
                arr[n // 2] ^= np.uint64(1 << 17)  # lỗi giả: phải bị phát hiện đúng 1 ô
            for s in range(0, n, BLOCK):
                check(q, wid, "const", arr, s, np.uint64(v), stats)
        key = np.uint64((seed + p) * 2654435761 & 0xFFFFFFFFFFFFFFFF)
        for s in range(0, n, BLOCK):
            arr[s:s + BLOCK] = np.arange(s, min(s + BLOCK, n), dtype=np.uint64) * MUL + key
        for s in range(0, n, BLOCK):
            check(q, wid, "addr", arr, s, np.arange(s, min(s + BLOCK, n), dtype=np.uint64) * MUL + key, stats)
        v = np.uint64(CONSTS[2 + p % 4])
        arr.fill(v)
        for s in range(0, n, BLOCK):  # đi lên: đọc mẫu, ghi phần bù
            check(q, wid, "inv", arr, s, v, stats)
            arr[s:s + BLOCK] = ~v
        for s in reversed(range(0, n, BLOCK)):  # đi xuống: đọc phần bù
            check(q, wid, "inv", arr, s, ~v, stats)
        rs = (seed, wid, p)
        g = np.random.Generator(np.random.PCG64(rs))
        for s in range(0, n, BLOCK):
            arr[s:s + BLOCK] = g.integers(0, 2**64 - 1, size=min(BLOCK, n - s), dtype=np.uint64, endpoint=True)
        g = np.random.Generator(np.random.PCG64(rs))
        for s in range(0, n, BLOCK):
            check(q, wid, "rand", arr, s, g.integers(0, 2**64 - 1, size=min(BLOCK, n - s), dtype=np.uint64, endpoint=True), stats)
        p += 1
        stats["passes"] = p
        q.put(("pass", wid, p, round(time.time() - t0, 1), stats["errors"]))
    q.put(("done", wid, stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=0, help="tổng GB thử; 0 = 70%% RAM còn trống lúc bắt đầu")
    ap.add_argument("--workers", type=int, default=os.cpu_count() // 2, help="số tiến trình (mặc định = số nhân thật)")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--self-test", action="store_true", help="lật một bit sau khi ghi ở lượt đầu: phải báo đúng 1 lỗi")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()
            os.fsync(logf.fileno())

    st = mem_status()
    gb = args.gb or 0.7 * st.ullAvailPhys / 2**30
    n = int(gb * 2**30 / 8 / args.workers)
    log(f"RAM {st.ullTotalPhys / 2**30:.1f} GB, trống {st.ullAvailPhys / 2**30:.1f} GB -> thử {n * 8 * args.workers / 2**30:.1f} GB "
        f"bằng {args.workers} tiến trình, {args.minutes} phút, seed {args.seed}" + (", SELF-TEST" if args.self_test else ""))
    q = mp.Queue()
    deadline = time.time() + args.minutes * 60
    procs = [mp.Process(target=worker, args=(w, n, deadline, args.seed, args.self_test, q)) for w in range(args.workers)]
    for pr in procs:
        pr.start()
    errors, done, passes, shown = 0, {}, {}, 0
    t_start = time.time()
    while len(done) < len(procs):
        try:
            msg = q.get(timeout=30)
        except Exception:
            if all(not pr.is_alive() for pr in procs):
                log("LỖI: mọi tiến trình đã dừng mà chưa báo xong")
                break
            continue
        if msg[0] == "error":
            _, wid, test, idx, exp, got, nbits = msg
            errors += 1
            if shown < 50:
                shown += 1
                log(f"  LỖI tiến trình {wid} [{test}] ô {idx}: ghi {exp:#018x}, đọc {got:#018x} ({nbits} bit sai)")
        elif msg[0] == "pass":
            _, wid, p, sec, err = msg
            passes[wid] = p
            if wid == 0:
                st = mem_status()
                log(f"lượt {p} xong ({sec} s/lượt), tổng lượt các tiến trình {sum(passes.values())}, lỗi {errors}, "
                    f"RAM trống {st.ullAvailPhys / 2**30:.1f} GB")
        elif msg[0] == "done":
            done[msg[1]] = msg[2]
    for pr in procs:
        pr.join(timeout=60)
    total_err = sum(d["errors"] for d in done.values())
    total_pass = sum(d["passes"] for d in done.values())
    tested = n * 8 * args.workers
    log(f"XONG sau {(time.time() - t_start) / 60:.1f} phút: {total_pass} lượt (mỗi tiến trình {n * 8 / 2**30:.2f} GB), "
        f"đã thử ~{total_pass * n * 8 * 22 / 2**40:.2f} TB ghi+đọc, LỖI {total_err}"
        + (" -> RAM KHÔNG ỔN ĐỊNH" if total_err and not args.self_test else ""))
    if args.log:
        with open(args.log + ".json", "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "workers": args.workers, "gb": tested / 2**30, "minutes": args.minutes,
                       "passes": total_pass, "errors": total_err, "self_test": args.self_test,
                       "per_worker": done}, f, ensure_ascii=False, indent=1)
    return 1 if total_err and not args.self_test else 0


if __name__ == "__main__":
    sys.exit(main())
