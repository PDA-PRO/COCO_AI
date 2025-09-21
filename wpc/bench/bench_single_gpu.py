# bench_single_gpu.py
import os, sys, time, json, csv, argparse, signal
from datetime import datetime
from reference.wpc import wpc  # GPU 1대라면 바인딩 생략 가능

def run_once(code: str, p_id: str):
    t0 = time.perf_counter()
    res = wpc.process(code, p_id)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    return lat_ms

def percentile(sorted_list, p):
    """p in [0,100], list는 정렬되어 있다고 가정"""
    n = len(sorted_list)
    if n == 0: return 0.0
    k = (n - 1) * (p / 100.0)
    f = int(k); c = min(f + 1, n - 1)
    if f == c: return round(sorted_list[f], 2)
    return round(sorted_list[f] + (sorted_list[c] - sorted_list[f]) * (k - f), 2)

def fmt_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="test_code_fairs.json 경로")
    ap.add_argument("--warmup", type=int, default=500, help="워밍업 개수(최소 수백 권장)")

    # 진행 로그 옵션
    ap.add_argument("--log_every", type=int, default=50, help="이 건마다 진행 로그 출력")
    ap.add_argument("--status_interval", type=float, default=5.0, help="이 초마다 상태 로그 출력")
    ap.add_argument("--log_file", default="", help="진행 로그를 파일에도 남김(옵션)")

    # per-item 상세 CSV (옵션)
    ap.add_argument("--detail_csv", default="", help="per-item 상세 지연 CSV 경로(미지정시 비활성)")
    ap.add_argument("--flush_every", type=int, default=200, help="상세 CSV를 이 건마다 flush")
    args = ap.parse_args()

    # 입력 로드 (test_code_fairs.json 가정)
    with open(args.inputs, encoding="utf-8") as f:
        items = json.load(f)
    total = len(items)
    if total == 0:
        print("입력이 비어있습니다.", file=sys.stderr); sys.exit(1)

    # 로깅 헬퍼
    def log(msg):
        line = f"[{fmt_ts()}] {msg}"
        print(line, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as lf:
                lf.write(line + "\n")

    # 스냅샷 함수
    lats = []           # 누적 지연(정렬 전)
    errs = 0
    start_t = time.perf_counter()
    last_status_t = start_t

    def snapshot(prefix="SNAPSHOT"):
        elapsed = time.perf_counter() - start_t
        done = len(lats) + errs
        rps = (done / elapsed) if elapsed > 0 else 0.0
        l_sorted = sorted(lats)
        p50 = percentile(l_sorted, 50)
        p90 = percentile(l_sorted, 90)
        p95 = percentile(l_sorted, 95)
        p99 = percentile(l_sorted, 99)
        eta = (elapsed / done * (total - done)) if done > 0 else float("inf")
        log(f"{prefix} {done}/{total} ({done/total*100:.1f}%) | "
            f"R={rps:.2f} jobs/s | p50={p50}ms p90={p90}ms p95={p95}ms p99={p99}ms | "
            f"errors={errs} | "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    # 신호 핸들러: USR1 스냅샷, INT/TERM 요약 후 종료
    def on_usr1(signum, frame): snapshot("USR1")
    def on_term(signum, frame):
        snapshot("FINAL")
        sys.exit(0)

    # 리눅스 컨테이너에서만 동작(윈도우 PowerShell→Linux 컨테이너로 실행되므로 OK)
    try:
        signal.signal(signal.SIGUSR1, on_usr1)
        signal.signal(signal.SIGINT,  on_term)
        signal.signal(signal.SIGTERM, on_term)
    except Exception:
        pass  # 환경에 따라 신호 등록이 실패할 수 있음

    # 배너
    log(f"Start bench | items={total} warmup={args.warmup} "
        f"| gpu_visible={os.environ.get('CUDA_VISIBLE_DEVICES','all')}")

    # 워밍업 (버그 수정: items[i] 사용)
    warm_n = min(args.warmup, total)
    for idx in range(warm_n):
        try:
            it = items[idx]
            _lat = run_once(it["code"]["buggy_code"], it["p_name"])
        except Exception:
            pass
    log(f"Warmup done: {warm_n} iters")

    # 상세 CSV 준비(옵션)
    detail_buf = []
    def flush_detail(force=False):
        nonlocal detail_buf
        if args.detail_csv and (force or len(detail_buf) >= args.flush_every):
            newfile = not os.path.exists(args.detail_csv)
            with open(args.detail_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["idx","lat_ms","ok"])
                if newfile: w.writeheader()
                w.writerows(detail_buf)
            detail_buf = []

    # 본 측정
    for idx, it in enumerate(items):
        try:
            t_ms = run_once(it["code"]["buggy_code"], it["p_name"])
            lats.append(t_ms)
        except Exception:
            errs += 1
            t_ms = 0.0  # 상세 CSV 채우기용

        # 상세 CSV 버퍼링
        if args.detail_csv:
            detail_buf.append({"idx": idx, "lat_ms": round(t_ms,2)})
            flush_detail(force=False)

        # 진행 로그: 개수 기반
        done = idx + 1
        if (done % args.log_every) == 0:
            snapshot("PROG")

        # 진행 로그: 시간 기반
        now = time.perf_counter()
        if (now - last_status_t) >= args.status_interval:
            snapshot("STAT")
            last_status_t = now

    # 종료 처리
    flush_detail(force=True)

    # 요약 및 CSV 기록
    elapsed = time.perf_counter() - start_t
    l_sorted = sorted(lats)
    p50 = percentile(l_sorted, 50)
    p90 = percentile(l_sorted, 90)
    p95 = percentile(l_sorted, 95)
    p99 = percentile(l_sorted, 99)
    jobs_per_sec = round((total / elapsed) if elapsed > 0 else 0.0, 2)

    summary = {
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "total": total,
        "elapsed_sec": round(elapsed, 3),
        "jobs_per_sec": jobs_per_sec,
        "errors": errs,
        "p50_ms": p50, "p90_ms": p90, "p95_ms": p95, "p99_ms": p99,
    }

    log(f"FINAL total={total} | jobs/s={jobs_per_sec} | p95={p95}ms p99={p99}ms | "
        f"errors={errs} | elapsed={elapsed:.1f}s")

    # 요약 JSON도 표준출력
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
