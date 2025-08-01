import argparse
import json
from util.syntax_match import compute_syntax_match
from util.line_diff import compute_line_edit_ratio

def process_json_for_change_ratios():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q","--quiet", default=False, type=bool, required=False)
    parser.add_argument("--result_path", default=None, type=str, required=True)
    parser.add_argument("--output_path", default=None, type=str, required=False)
    args = parser.parse_args()
    
    if args.output_path:
        print(f"{args.output_path} 경로에 결과가 저장됩니다.")

    with open(args.result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))
    results = []
    processed_count = 0

    for idx, item in enumerate(data):
        if item.get("result") != "pass":
            continue

        pred = item.get("pred", "")
        src = item.get("source", "")

        if not pred.strip() or not src.strip():
            continue

        line_ratio = compute_line_edit_ratio(src, pred)
        ast_ratio = compute_syntax_match([src], [pred])

        results.append({
            "idx": idx,
            "p_name": item.get("p_name", ""),
            "LineDiffRatio": round(line_ratio, 4),
            "ASTNodeDiffRatio": round(ast_ratio, 4),
        })

        processed_count += 1

        if not args.quiet and processed_count % 2000 == 0:
            print(f"[INFO] Processed {processed_count} idx: {idx} LineDiffRatio: {round(line_ratio, 4)} ASTNodeDiffRatio: {round(ast_ratio, 4)}")

    print(f"[DONE] Total processed: {processed_count}")

        # ===== 저장 =====
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Results saved to JSON: {args.output_path}")

    return results

if __name__ == "__main__":
    process_json_for_change_ratios()