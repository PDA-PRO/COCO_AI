import difflib

def compute_line_edit_ratio(code_before: str, code_after: str) -> int:
    """
    두 코드 간의 줄 단위 변경 비율을 계산합니다.
    공백/빈 줄은 무시하며, 삽입/삭제/수정된 줄 수를 기반으로 비교합니다.
    반환값은 유사도(1.0에 가까울수록 유사)입니다.
    """
    def clean_lines(code: str):
        return [line for line in code.strip().splitlines() if line.strip() != ""]

    before_lines = clean_lines(code_before)
    after_lines = clean_lines(code_after)

    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    opcodes = sm.get_opcodes()

    changed_lines = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "replace":
            changed_lines += max(i2 - i1, j2 - j1)  # 교체는 양쪽 최대 줄 수
        elif tag == "insert":
            changed_lines += j2 - j1  # after에 삽입된 줄 수
        elif tag == "delete":
            changed_lines += i2 - i1  # before에서 삭제된 줄 수

    base_line_count = max(len(before_lines), 1)  # 0줄 대비 방지
    ratio = min(changed_lines / base_line_count, 1.0)
    return round(1 - ratio, 4)