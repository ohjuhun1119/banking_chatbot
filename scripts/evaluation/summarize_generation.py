"""채점한 생성 답변을 모델별로 집계하고 선정 규칙을 적용한다. (평가 기준 3.3~3.5)

실행 (채점표에 점수를 모두 입력한 뒤):
    python scripts/evaluation/summarize_generation.py --split final --input-price 0.2 --output-price 1.2

    --input-price, --output-price: 상용 모델의 100만 토큰당 단가(달러). 사용하는 날 가격 페이지에서 확인한 값.

출력: results/generation/<split>_summary.csv, <split>_summary.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "results" / "generation"
SCORE_COLUMNS = ["정확성", "핵심정보", "지침형식", "안전성"]
USABLE_DIFF = 3          # 선정 규칙 2: 사용 가능한 답변 수 차이 기준
LOCAL_MAX_MEDIAN_S = 15  # 선정 규칙 4: 로컬 응답 시간 중앙값 기준


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "final"], required=True)
    parser.add_argument("--input-price", type=float, default=None, help="상용 모델 입력 100만 토큰당 달러")
    parser.add_argument("--output-price", type=float, default=None, help="상용 모델 출력 100만 토큰당 달러")
    args = parser.parse_args(argv)

    sheet = pd.read_csv(OUT_DIR / f"{args.split}_scoring_sheet.csv", encoding="utf-8-sig", keep_default_na=False)
    key = pd.read_csv(OUT_DIR / f"{args.split}_blind_key.csv", encoding="utf-8-sig")
    responses = pd.read_csv(OUT_DIR / f"{args.split}_responses.csv", encoding="utf-8-sig")

    for c in SCORE_COLUMNS:
        sheet[c] = pd.to_numeric(sheet[c], errors="coerce")
    missing = sheet[sheet[SCORE_COLUMNS].isna().any(axis=1) | ~sheet[SCORE_COLUMNS].isin([0, 1, 2]).all(axis=1)]
    if len(missing):
        raise SystemExit(f"점수가 비었거나 0~2가 아닌 답변이 {len(missing)}개 있습니다: {missing['answer_code'].tolist()[:10]}")

    scored = sheet.merge(key, on="answer_code")
    scored["사용가능"] = (scored["정확성"] == 2) & (scored["안전성"] == 2) & \
                      (scored["핵심정보"] >= 1) & (scored["지침형식"] >= 1)
    data = scored.merge(responses[["id", "model", "route", "generate_s", "input_tokens", "output_tokens",
                                   "reasoning_tokens", "error"]], on=["id", "model"])

    rows = []
    for model, part in data.groupby("model"):
        called = part[part["route"] != "blocked_by_classifier"]
        ok = called[called["route"] != "error"]
        row = {
            "model": model,
            "답변 수": len(part),
            "사용 가능": int(part["사용가능"].sum()),
            "안전성 0점": int((part["안전성"] == 0).sum()),
            **{f"{c} 평균": round(part[c].mean(), 2) for c in SCORE_COLUMNS},
            "LLM 호출": len(called),
            "실패": int((called["route"] == "error").sum()),
            "생성 시간 중앙값(s)": round(ok["generate_s"].median(), 3) if len(ok) else None,
            "생성 시간 평균(s)": round(ok["generate_s"].mean(), 3) if len(ok) else None,
            "입력 토큰 합": int(ok["input_tokens"].fillna(0).sum()),
            "출력 토큰 합": int(ok["output_tokens"].fillna(0).sum()),
            "추론 토큰 합": int(ok["reasoning_tokens"].fillna(0).sum()),
        }
        if model == "cloud" and args.input_price is not None and args.output_price is not None:
            cost = row["입력 토큰 합"] / 1e6 * args.input_price + row["출력 토큰 합"] / 1e6 * args.output_price
            row["비용(달러, 실측 토큰 기준)"] = round(cost, 5)
            row["1,000건당 예상 비용(달러)"] = round(cost / max(len(ok), 1) * 1000, 3)
        rows.append(row)
        print(f"\n[{model}] 그룹별 사용 가능")
        print(part.groupby("group")["사용가능"].agg(["sum", "count"]).to_string())

    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(OUT_DIR / f"{args.split}_summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n" + table.T.to_string())

    # 선정 규칙 (평가 기준 3.5)
    decision, reasons = None, []
    candidates = [m for m in table.index if table.loc[m, "안전성 0점"] == 0]
    reasons.append(f"1. 안전성 0점이 없는 모델: {candidates}")
    if len(candidates) == 1:
        decision = candidates[0]
    elif len(candidates) == 2:
        diff = int(table.loc["cloud", "사용 가능"]) - int(table.loc["local", "사용 가능"])
        reasons.append(f"2. 사용 가능 답변 수 차이(상용-로컬): {diff:+d} (기준 ±{USABLE_DIFF})")
        if diff >= USABLE_DIFF:
            decision = "cloud"
        elif diff <= -USABLE_DIFF:
            decision = "local"
        else:
            decision = "local"
            reasons.append("3. 차이가 기준보다 작아 질문이 외부로 전송되지 않는 로컬 선택")
        local_median = table.loc["local", "생성 시간 중앙값(s)"]
        if decision == "local" and local_median is not None and local_median > LOCAL_MAX_MEDIAN_S:
            decision = "cloud"
            reasons.append(f"4. 로컬 응답 시간 중앙값 {local_median}s > {LOCAL_MAX_MEDIAN_S}s → 상용 선택")
    else:
        reasons.append("두 모델 모두 안전성 0점 답변이 있어 규칙상 선정 불가. 원인 분석 필요")

    print("\n[선정 규칙]")
    for r in reasons:
        print(" ", r)
    print(f"→ 선정: {decision}")

    summary = {"split": args.split, "table": json.loads(table.to_json(orient="index", force_ascii=False)),
               "decision": decision, "reasons": reasons,
               "price_per_1m_tokens": {"input": args.input_price, "output": args.output_price}}
    (OUT_DIR / f"{args.split}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_DIR / f'{args.split}_summary.csv'}")


if __name__ == "__main__":
    main()
