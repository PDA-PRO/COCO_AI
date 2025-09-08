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

def load_jsonl(path, skip=0):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for idx, line in enumerate(f, 1) if idx > skip]

# ========== 테스트 케이스 실행 ==========

def run_testcase(pname, pred_code, input_data, expected_output, timeout_sec, memory_kb):
    """하나의 테스트 케이스를 실행하여 결과를 반환한다."""
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
        actual = result.stdout.decode().strip()
        expected_output = expected_output.strip()

        if result.returncode != 0:
            return False, "[ERROR] Runtime Error", expected_output, actual

        # 1) 특수 채점 대상인 경우
        if pname in SPECIAL_CHECKERS:
            is_ok = SPECIAL_CHECKERS[pname](input_data, expected_output, actual)
            if not is_ok:
                return False, "Wrong Answer (custom check)", expected_output, actual
            else:
                return True, "Passed", "", ""

        # 2) 실수 오차 허용 문제
        if pname in FLOAT_TOLERANCE:
            tol = FLOAT_TOLERANCE[pname]
            try:
                exp_vals = [float(x) for x in expected_output.split()]
                act_vals = [float(x) for x in actual.split()]
            except ValueError:
                return False, "Wrong Answer (invalid float)", expected_output, actual
            if len(exp_vals) != len(act_vals):
                return False, "Wrong Answer (length mismatch)", expected_output, actual
            for ev, av in zip(exp_vals, act_vals):
                diff = abs(ev - av)
                rel_err = diff / abs(ev) if abs(ev) > 1e-12 else diff
                if diff > tol and rel_err > tol:
                    return False, f"Wrong Answer (error>tol: {diff})", expected_output, actual
            return True, "Passed", "", ""

        # 3) 그 밖의 문제는 문자열 비교
        if actual != expected_output:
            return False, "Wrong Answer", expected_output, actual
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
            pname, pred_code, input_data, expected_output, timeout_sec, memory_kb
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


# 특수한 채점 처리를 위한 설정들
# 실수 출력의 허용 오차 (abs 또는 rel 비교) 정의
FLOAT_TOLERANCE = {
    "p02380": 1e-4,
    "p02399": 1e-5,
    "p02400": 1e-5,
    "p02677": 1e-9,
    "p02705": 1e-2,
    "p02731": 1e-6,
    "p02897": 1e-6,
    "p02934": 1e-5,
    "p02935": 1e-5,
    "p03001": 1e-9,
    "p03043": 1e-9,
    "p03304": 1e-6,
}

# 여러 정답을 허용하는 문제에 대한 검증 함수들
def check_p02664(input_data: str, expected_output: str, actual_output: str) -> bool:
    """?를 P/D로 치환하는 문제: PD점수가 최대인지 확인"""
    if expected_output == actual_output:
        return True
    T = input_data.strip()
    pred = actual_output.strip()
    exp = expected_output.strip()
    # 길이와 T의 고정 문자 확인
    if len(pred) != len(T):
        return False
    for c_t, c_p in zip(T, pred):
        if c_t in ("P", "D") and c_t != c_p:
            return False
        if c_p not in ("P", "D"):
            return False

    def pd_score(s: str) -> int:
        return s.count("D") + sum(1 for i in range(len(s) - 1) if s[i:i+2] == "PD")

    # 예상 출력의 PD 점수와 비교하여 동일해야 함
    return pd_score(pred) == pd_score(exp)

def check_p02842(input_data: str, expected_output: str, actual_output: str) -> bool:
    """세전 가격 찾기 문제: 출력이 올바른지 검증"""
    if expected_output == actual_output:
        return True
    N = int(input_data.strip())
    pred = actual_output.strip()

    try:
        x = int(pred)
    except ValueError:
        return False
    # 세금 계산식에 부합해야 함
    return x > 0 and (x * 108) // 100 == N

def check_p03471(input_data: str, expected_output: str, actual_output: str) -> bool:
    """지폐 조합 문제: 올바른 조합인지 확인"""
    if expected_output == actual_output:
        return True
    
    N, Y = map(int, input_data.split())
    xs = actual_output.strip().split()
    if len(xs) != 3:
        return False
    try:
        x, y, z = map(int, xs)
    except ValueError:
        return False
    return x >= 0 and y >= 0 and z >= 0 \
        and x + y + z == N \
        and 10000 * x + 5000 * y + 1000 * z == Y

def check_p03545(input_data: str, expected_output: str, actual_output: str) -> bool:
    """+/-를 넣어 7을 만드는 문제: 식이 맞는지 확인"""
    if expected_output == actual_output:
        return True
    
    expr = actual_output.strip()
    if not expr.endswith("=7"):
        return False
    expr = expr[:-2]  # '=7' 제거
    # 기대되는 형식: A op B op C op D
    if len(expr) != 7:
        return False
    try:
        A = int(expr[0]); B = int(expr[2]); C = int(expr[4]); D = int(expr[6])
    except ValueError:
        return False
    if not (A == int(input_data[0]) and B == int(input_data[1]) and C == int(input_data[2]) and D == int(input_data[3])):
        return False
    ops = [expr[1], expr[3], expr[5]]
    if not all(op in ('+', '-') for op in ops):
        return False
    # 값 계산
    val = A
    for op, n in zip(ops, (B, C, D)):
        if op == '+':
            val += n
        else:
            val -= n
    return val == 7

def check_p03583(input_data: str, expected_output: str, actual_output: str) -> bool:
    """4/N = 1/h + 1/n + 1/w을 만족하는지 확인"""
    if expected_output == actual_output:
        return True

    N = int(input_data.strip())
    xs = actual_output.strip().split()
    if len(xs) != 3:
        return False
    try:
        h, n, w = map(int, xs)
    except ValueError:
        return False
    if h <= 0 or n <= 0 or w <= 0:
        return False
    # 4/h*n*w == N*(n*w + h*w + h*n)
    lhs = 4 * h * n * w
    rhs = N * (n * w + h * w + h * n)
    return lhs == rhs

def check_p03910(input_data: str, expected_output: str, actual_output: str) -> bool:
    """점수합이 N이 되도록 문제를 선택하는지 확인"""
    N = int(input_data.strip())
    # 예측 출력 각 줄에 하나의 정수
    try:
        selected = [int(x) for x in actual_output.strip().split('\n')]
    except ValueError:
        return False
    # 모든 번호는 1 이상 중복 없이
    if len(selected) != len(set(selected)):
        return False
    if any(x <= 0 for x in selected):
        return False
    # 합이 N인지
    if sum(selected) != N:
        return False
    # 최소 가능한 최대값 계산: M*(M+1)/2 >= N
    return max(selected) == max(int(x) for x in expected_output.strip().split('\n'))

def check_p03437(input_data: str, expected_output: str, actual_output: str) -> bool:
    """X의 배수이며 Y의 배수가 아닌 수를 찾는 문제"""
    if expected_output == actual_output:
        return True
    try:
        X, Y = map(int, input_data.strip().split())
        actual = actual_output.strip()

        val = int(actual)
        if val > 10**18:
            return False
        return val > 0 and val % X == 0 and val % Y != 0
    except Exception:
        return False

# 문제 이름을 키로 하여 검증 함수를 저장
SPECIAL_CHECKERS = {
    "p02664": check_p02664,
    "p02842": check_p02842,
    "p03471": check_p03471,
    "p03545": check_p03545,
    "p03583": check_p03583,
    "p03910": check_p03910,
    "p03437": check_p03437
}

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
