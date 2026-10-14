"""직접 작성한 확인용 질문으로 학습한 분류기들을 평가한다. (평가 기준 2.4)

템플릿으로 만든 업무 외 질문은 같은 방식으로 만든 validation에서 쉽게 맞힐 수 있어,
템플릿과 무관하게 작성한 질문으로 실제 업무 외 검출 능력과 정상 질문 통과 여부를 확인한다.

실행 (프로젝트 루트에서, 모델 학습 후):
    python scripts/evaluation/evaluate_probe.py

입력:
    tests/probe_questions.csv            group, question, expected("업무 외" 또는 "은행 업무")
    models/baseline/model.joblib         베이스라인
    models/*/best/                       파인튜닝한 사전학습 모델들
출력:
    results/probe/summary.csv            모델별·그룹별 정답 수
    results/probe/predictions.csv        질문별 예측 (직접 작성한 질문이라 커밋 가능)
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROBE_PATH = PROJECT_DIR / "tests" / "probe_questions.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUT_DIR = PROJECT_DIR / "results" / "probe"
OUT_OF_SCOPE = "업무 외"


def load_models() -> dict:
    """이름 → 예측 함수(질문 목록 → 라벨 목록)."""
    models = {}
    baseline_path = MODEL_DIR / "baseline" / "model.joblib"
    if baseline_path.exists():
        import joblib
        pipe = joblib.load(baseline_path)
        models["TF-IDF + LR"] = lambda texts, p=pipe: list(p.predict(texts))

    best_dirs = sorted(p for p in MODEL_DIR.glob("*/best") if (p / "config.json").exists())
    if best_dirs:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        for d in best_dirs:
            tok = AutoTokenizer.from_pretrained(str(d))
            model = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()  # CPU에서 예측

            def predict(texts, tok=tok, model=model):
                with torch.inference_mode():
                    enc = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
                    ids = model(**enc).logits.argmax(-1).tolist()
                return [model.config.id2label[i] for i in ids]

            models[f"{model.config._name_or_path or d.parent.name} ({d.parent.name})"] = predict

    if not models:
        raise FileNotFoundError(f"{MODEL_DIR}에서 학습한 모델을 찾지 못했습니다.")
    return models


def main():
    probe = pd.read_csv(PROBE_PATH, encoding="utf-8-sig")
    questions = probe["question"].tolist()
    print(f"확인용 질문 {len(probe)}개: {probe['group'].value_counts().to_dict()}")

    preds = probe.copy()
    rows = []
    for name, predict in load_models().items():
        labels = predict(questions)
        preds[name] = labels
        is_oos = pd.Series([l == OUT_OF_SCOPE for l in labels])
        # 업무 외 그룹은 업무 외로 예측해야 정답, 은행 업무 그룹은 업무 외가 아니어야 정답
        correct = (probe["expected"] == OUT_OF_SCOPE) == is_oos
        row = {"모델": name}
        for group in probe["group"].unique():
            mask = probe["group"] == group
            row[group] = f"{int(correct[mask].sum())}/{int(mask.sum())}"
        row["전체"] = f"{int(correct.sum())}/{len(probe)}"
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    print("\n그룹별 정답 수 (업무 외 그룹: 업무 외로 걸러냄 / 은행 업무 그룹: 통과시킴)")
    print(summary.to_string(index=False))

    model_names = [r["모델"] for r in rows]
    print("\n틀린 질문:")
    for name in model_names:
        wrong = preds[((preds["expected"] == OUT_OF_SCOPE) != (preds[name] == OUT_OF_SCOPE))]
        print(f"\n[{name}] {len(wrong)}개")
        for _, r in wrong.iterrows():
            print(f"  - ({r['group']}) {r['question']} → {r[name]}")
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
