"""학습한 분류 모델들의 validation 결과를 비교하고, 평가 기준 2.3의 선정 규칙을 적용한다.

실행 (프로젝트 루트에서, 모든 모델 학습 후):
    python scripts/training/compare_models.py

입력: results/*/summary.json (시험 실행 결과 summary_smoke.json은 제외)
출력: results/classifier_comparison.csv

선정 규칙 (docs/evaluation_criteria.md 2.3)
1. validation Macro F1(11개 주제)이 가장 높은 모델을 찾는다.
2. 그 모델과 차이가 1%p 미만인 모델들 중 건당 추론 시간(CPU 중앙값)이 가장 짧은 모델을 선택한다.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "results"
THRESHOLD_PP = 1.0   # 선정 규칙의 차이 기준 (%p)


def load_summaries() -> list[dict]:
    summaries = []
    for path in sorted(RESULT_DIR.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "validation" in data:
            data["_dir"] = path.parent.name
            summaries.append(data)
    if len(summaries) < 2:
        raise FileNotFoundError("비교할 모델 결과가 2개 이상 필요합니다. 학습 스크립트를 먼저 실행하세요.")
    return summaries


def cpu_latency(summary: dict) -> float:
    lat = summary["inference_latency_batch1"]
    return lat["cpu"]["median_s"] if "cpu" in lat else lat["median_s"]


def main():
    rows = []
    for s in load_summaries():
        v = s["validation"]
        lat = s["inference_latency_batch1"]
        gpu = next((k for k in lat if k != "cpu"), None)
        rows.append({
            "모델": s["model"] + (f" ({s['selected_config']})" if "selected_config" in s else ""),
            "Macro F1 (11종)": v["macro_f1"],
            "Macro F1 (은행 10종)": v.get("macro_f1_bank"),
            "Accuracy": v["accuracy"],
            "업무 외 Recall": v.get("oos_recall"),
            "은행→업무 외 오차단": v.get("bank_blocked_as_oos"),
            "업무 외→은행 통과": v.get("oos_passed_as_bank"),
            "학습 시간(s)": s["train_time_s"],
            "학습 최대 RAM(MB)": s["peak_ram_mb_train"],
            "추론 중앙값 CPU(s)": cpu_latency(s),
            "추론 중앙값 GPU(s)": lat[gpu]["median_s"] if gpu else None,
            "모델 크기(MB)": s["model_size_mb"],
            "결과 폴더": s["_dir"],
        })
    table = pd.DataFrame(rows).sort_values("Macro F1 (11종)", ascending=False).reset_index(drop=True)
    table.to_csv(RESULT_DIR / "classifier_comparison.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(table.drop(columns=["결과 폴더"]).to_string(index=False))

    # 선정 규칙 적용
    best_f1 = table["Macro F1 (11종)"].max()
    table["최고와 차이(%p)"] = ((best_f1 - table["Macro F1 (11종)"]) * 100).round(2)
    candidates = table[table["최고와 차이(%p)"] < THRESHOLD_PP]
    selected = candidates.sort_values("추론 중앙값 CPU(s)").iloc[0]

    print(f"\n[선정 규칙] 최고 Macro F1 {best_f1:.4f}, 차이 {THRESHOLD_PP}%p 미만 후보:")
    for _, r in candidates.iterrows():
        print(f"  - {r['모델']}: Macro F1 {r['Macro F1 (11종)']:.4f} (차이 {r['최고와 차이(%p)']}%p),"
              f" CPU 추론 {r['추론 중앙값 CPU(s)']}s")
    print(f"→ 선정: {selected['모델']}")
    print(f"\n저장: {RESULT_DIR / 'classifier_comparison.csv'}")


if __name__ == "__main__":
    main()
