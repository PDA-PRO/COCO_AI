import argparse
import os
import json
import glob
import tempfile
import subprocess
import platform
import resource
import sys

print("🔧 채점 스크립트 시작", file=sys.stderr)
sys.stderr.flush()

# ========== 유틸 ==========

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# ========== 테스트 케이스 실행 ==========

def run_testcase(pred_code, input_data, expected_output, timeout_sec, memory_kb):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(pred_code)
        code_path = tmp_file.name

    def set_limits():
        if platform.system() != "Windows":
            mem_bytes = memory_kb * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    try:
        result = subprocess.run(
            ["python3", code_path],
            input=input_data.encode('utf-8'),
            capture_output=True,
            timeout=timeout_sec,
            preexec_fn=set_limits if platform.system() != "Windows" else None
        )
        output = result.stdout.decode().strip()
        if result.returncode != 0:
            return False, "[ERROR] Runtime Error", expected_output, output
        elif output != expected_output:
            return False, "Wrong Answer", expected_output, output
        else:
            return True, "Passed", "", ""
    except subprocess.TimeoutExpired:
        return False, "[ERROR] Timeout", expected_output, "[Timeout]"
    except Exception as e:
        return False, f"[ERROR] {str(e)}", expected_output, "[Exception]"
    finally:
        os.remove(code_path)

# ========== 전체 테스트 케이스 로딩 ==========

def load_all_testcases(base_task_dir, task_detail):
    testcases_per_problem = {}
    for pname, detail in task_detail.items():

        task_dir = os.path.join(base_task_dir, pname)
        input_dir = os.path.join(task_dir, "input")
        output_dir = os.path.join(task_dir, "output")

        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            continue

        inputs = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
        outputs = sorted(glob.glob(os.path.join(output_dir, "*.txt")))

        if len(inputs) != len(outputs):
            print(f"[WARN] {pname}: input/output count mismatch ({len(inputs)} vs {len(outputs)})")
            continue

        testcases = []
        for idx, (in_f, out_f) in enumerate(zip(inputs, outputs)):
            input_data = load_file(in_f)
            output_data = load_file(out_f)
            testcases.append((idx, input_data, output_data))

        testcases_per_problem[pname] = testcases
    return testcases_per_problem

# ========== 하나의 ref 코드에 대해 모든 테스트 실행 ==========

def run_all_tests_for_ref(pname, pred_code, testcases, timeout_sec, memory_kb):
    failures = []

    for idx, input_data, expected_output in testcases:
        passed, reason, expected, actual = run_testcase(
            pred_code, input_data, expected_output, timeout_sec, memory_kb
        )
        if not passed:
            failures.append({
                "test_id": idx,
                "reason": reason
            })

    result = {
        "p_name": pname,
        "result": "pass" if not failures else "fail",
    }
    if failures:
        result["failures"] = failures
    return result

# ========== 메인 실행 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_jsonl_path", default=None, type=str, required=True)
    parser.add_argument("--base_task_dir", default="base_task_set/TC", type=str)
    parser.add_argument("--task_detail_path", default="base_task_set/task_detail.json", type=str)
    parser.add_argument("--result_path", default="./result.json", type=str)
    args = parser.parse_args()

    task_detail = load_json(args.task_detail_path)
    records = load_jsonl(args.output_jsonl_path)
    testcases_per_problem = load_all_testcases(args.base_task_dir, task_detail)

    BATCH_SIZE = 100
    total_passed = 0
    total_failed = 0
    results = []

    # ✅ 파일 초기화
    with open(args.result_path, "w", encoding="utf-8") as f:
        f.write("[\n")

    for i, rec in enumerate(records, start=1):
        pname = rec["p_name"]
        pred_code = rec["pred"]
        limit = task_detail.get(pname, {}).get("limit", {})
        timeout_sec = int(limit.get("time", 1000)) / 1000
        memory_kb = int(limit.get("memory", 256 * 1024))

        testcases = testcases_per_problem.get(pname, [])
        if not testcases:
            print(f"[{i}][{pname}] ⚠️ No test cases available. Skipping.")
            continue

        result = run_all_tests_for_ref(pname, pred_code, testcases, timeout_sec, memory_kb)
        result.update({
            "pred": rec["pred"],
            "target": rec["target"],
            "source": rec["source"],
        })
        results.append(result)

        # 통계 갱신
        if result["result"] == "pass":
            total_passed += 1
        else:
            total_failed += 1

        status = "✅ Passed" if result["result"] == "pass" else f"❌ Failed ({len(result['failures'])} failed)"
        print(f"[{i}][{pname}] {status} | ✅ {total_passed} / ❌ {total_failed}")

        # ✅ 일정 수마다 flush + clear
        if len(results) >= BATCH_SIZE or i == len(records):
            with open(args.result_path, "a", encoding="utf-8") as f:
                for j, item in enumerate(results):
                    json.dump(item, f, indent=2, ensure_ascii=False)
                    if not (i == len(records) and j == len(results) - 1):  # 마지막 항목이 아니면 쉼표
                        f.write(",\n")
                    else:
                        f.write("\n")
            results.clear()
            print(f"💾 저장됨: {i}개까지 완료")

    # ✅ JSON 닫기
    with open(args.result_path, "a", encoding="utf-8") as f:
        f.write("]\n")

    # ✅ 최종 통계
    print("\n🎯 전체 평가 완료")
    print(f"총 {len(records)}개 코드 평가됨: ✅ {total_passed}개 pass, ❌ {total_failed}개 fail")


if __name__ == "__main__":
    main()
