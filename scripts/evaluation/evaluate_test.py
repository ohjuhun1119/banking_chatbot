"""선정을 마친 분류 모델들을 test 데이터로 1회 평가한다. (평가 기준 2.3의 4번)

- 모델과 설정은 validation으로 이미 확정된 상태여야 한다.
- 이 결과로 모델을 다시 고르거나 설정을 바꾸지 않는다.
- 실수로 여러 번 실행하지 않도록, 이미 평가한 기록이 있으면 멈춘다. (--force로 덮어쓰기)

실행 (프로젝트 루트에서, compare_models.py 실행 후):
    python scripts/evaluation/evaluate_test.py

출력: results/test/<모델 폴더>/test_report.csv, test_confusion_matrix.*, summary.json, results/test/comparison.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "scripts" / "training"))   # 학습 때와 같은 평가 함수 사용

import pandas as pd  # noqa: E402

from eval_utils import (LABEL_COL, MODEL_DIR, RESULT_DIR, TEXT_COL, load_split,  # noqa: E402
                        save_evaluation, save_json)

OUT_DIR = RESULT_DIR / "test"


def load_models() -> dict:
    models = {}
    baseline = MODEL_DIR / "baseline" / "model.joblib"
    if baseline.exists():
        import joblib
        pipe = joblib.load(baseline)
        models["baseline"] = ("TF-IDF + LogisticRegression", lambda texts, p=pipe: list(p.predict(texts)))

    for d in sorted(p for p in MODEL_DIR.glob("*/best") if (p / "config.json").exists()):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(d))
        model = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
        max_length = json.loads((d / "train_config.json").read_text(encoding="utf-8")).get("max_length", 128) \
            if (d / "train_config.json").exists() else 128

        def predict(texts, tok=tok, model=model, max_length=max_length, bs=64):
            out = []
            with torch.inference_mode():
                for i in range(0, len(texts), bs):
                    enc = tok(texts[i:i + bs], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
                    out += [model.config.id2label[j] for j in model(**enc).logits.argmax(-1).tolist()]
            return out

        models[d.parent.name] = (d.parent.name, predict)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="이미 평가한 기록이 있어도 다시 평가 (변경 기록에 이유를 남길 것)")
    args = parser.parse_args()

    if not (RESULT_DIR / "classifier_comparison.csv").exists():
        raise SystemExit("validation 비교가 없습니다. compare_models.py로 선정을 먼저 마치세요.")
    if (OUT_DIR / "comparison.csv").exists() and not args.force:
        raise SystemExit(f"이미 test를 평가했습니다: {OUT_DIR}\n다시 평가하려면 --force를 쓰고, 평가 기준 변경 기록에 이유를 남기세요.")

    test_df = load_split("test")
    labels = sorted(test_df[LABEL_COL].unique())
    print(f"test {len(test_df):,}건 / 라벨 {len(labels)}종")

    rows = []
    for folder, (name, predict) in load_models().items():
        pred = predict(test_df[TEXT_COL].tolist())
        metrics = save_evaluation(OUT_DIR / folder, test_df, pred, labels, name, split="test")
        save_json({"model": name, "test": metrics, "evaluated_at": datetime.now().isoformat(timespec="seconds")},
                  OUT_DIR / folder / "summary.json")
        rows.append({"모델": name, "Macro F1 (11종)": metrics["macro_f1"], "Macro F1 (은행 10종)": metrics["macro_f1_bank"],
                     "Accuracy": metrics["accuracy"], "업무 외 Recall": metrics["oos_recall"],
                     "은행→업무 외 오차단": metrics["bank_blocked_as_oos"], "업무 외→은행 통과": metrics["oos_passed_as_bank"]})

    table = pd.DataFrame(rows).sort_values("Macro F1 (11종)", ascending=False)
    table.to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    print("\n[test 결과] 이 결과로 모델을 다시 고르지 않습니다.")
    print(table.to_string(index=False))
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
