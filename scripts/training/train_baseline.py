"""TF-IDF + 로지스틱 회귀 베이스라인 학습과 validation 평가.

- 학습은 train만 사용한다. TF-IDF의 어휘·IDF도 train에서만 계산된다.
- 설정 후보는 validation Macro F1로 비교한다.
- test는 사용하지 않는다. (분류기 선정 후 별도로 1회 평가)

출력:
    models/baseline/model.joblib, labels.json, config.json
    results/baseline/summary.json, grid_results.csv, validation_*.csv/png
"""

import argparse
import time
from pathlib import Path
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from eval_utils import (LABEL_COL, MODEL_DIR, RESULT_DIR, SEED, TEXT_COL, PeakMemory, check_split,
                        compute_metrics, environment_info, load_split, measure_latency,
                        save_evaluation, save_json)


# validation으로 비교할 설정 후보 (문장 분할 단위 2가지 × 규제 강도 2가지)
CANDIDATES = [
    {"name": "char2-4_C1", "analyzer": "char_wb", "ngram_range": (2, 4), "C": 1.0},
    {"name": "char2-4_C4", "analyzer": "char_wb", "ngram_range": (2, 4), "C": 4.0},
    {"name": "word1-2_C1", "analyzer": "word", "ngram_range": (1, 2), "C": 1.0},
    {"name": "word1-2_C4", "analyzer": "word", "ngram_range": (1, 2), "C": 4.0},
]
COMMON = {"min_df": 2, "sublinear_tf": True, "max_features": 300_000,
          "class_weight": "balanced", "max_iter": 3000}


def build_pipeline(cfg: dict) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        min_df=COMMON["min_df"],
        sublinear_tf=COMMON["sublinear_tf"],
        max_features=COMMON["max_features"],
        token_pattern=r"(?u)\b\w+\b" if cfg["analyzer"] == "word" else None,
    )
    classifier = LogisticRegression(C=cfg["C"], class_weight=COMMON["class_weight"],
                                    max_iter=COMMON["max_iter"], random_state=SEED)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(MODEL_DIR / "baseline"))
    parser.add_argument("--result-dir", default=str(RESULT_DIR / "baseline"))
    args = parser.parse_args()
    model_dir, result_dir = Path(args.model_dir), Path(args.result_dir)

    train_df, val_df = load_split("train"), load_split("validation")
    check_split(train_df, val_df)
    labels = sorted(train_df[LABEL_COL].unique())

    # 1) 설정 후보 비교
    grid, fitted = [], {}
    for cfg in CANDIDATES:
        pipe = build_pipeline(cfg)
        with PeakMemory() as mem:
            start = time.perf_counter()
            pipe.fit(train_df[TEXT_COL], train_df[LABEL_COL])
            train_s = time.perf_counter() - start
        m = compute_metrics(val_df[LABEL_COL], pipe.predict(val_df[TEXT_COL]))
        row = {"name": cfg["name"], "analyzer": cfg["analyzer"], "ngram_range": str(cfg["ngram_range"]),
               "C": cfg["C"], "macro_f1": m["macro_f1"], "macro_f1_bank": m["macro_f1_bank"],
               "accuracy": m["accuracy"], "oos_recall": m["oos_recall"],
               "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
               "train_time_s": round(train_s, 2), "peak_ram_mb": mem.peak_mb}
        grid.append(row)
        fitted[cfg["name"]] = (pipe, cfg, row)
        print(f"  {cfg['name']:<12} Macro F1 {row['macro_f1']:.4f} (은행 {row['macro_f1_bank']:.4f})"
              f"  Acc {row['accuracy']:.4f}  학습 {row['train_time_s']}s")

    result_dir.mkdir(parents=True, exist_ok=True)
    grid_df = pd.DataFrame(grid).sort_values("macro_f1", ascending=False)
    grid_df.to_csv(result_dir / "grid_results.csv", index=False, encoding="utf-8-sig")
    best_name = grid_df.iloc[0]["name"]
    pipe, best_cfg, best_row = fitted[best_name]
    print(f"\n선택된 베이스라인 설정: {best_name}")

    # 2) 선택된 설정의 validation 상세 평가
    val_pred = pipe.predict(val_df[TEXT_COL])
    val_metrics = save_evaluation(result_dir, val_df, val_pred, labels, f"Baseline ({best_name})")

    # 3) 추론 시간 (배치 크기 1, CPU)
    latency = measure_latency(lambda t: pipe.predict([t]), val_df[TEXT_COL].tolist())

    # 4) 저장 → 다시 불러와 같은 예측이 나오는지 확인
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "model.joblib")
    config = {**best_cfg, "ngram_range": list(best_cfg["ngram_range"]), **COMMON, "seed": SEED}
    save_json(labels, model_dir / "labels.json")
    save_json(config, model_dir / "config.json")

    start = time.perf_counter()
    reloaded = joblib.load(model_dir / "model.joblib")
    load_s = time.perf_counter() - start
    same = bool((reloaded.predict(val_df[TEXT_COL]) == val_pred).all())
    model_mb = round((model_dir / "model.joblib").stat().st_size / 1024**2, 1)

    save_json({
        "model": "TF-IDF + LogisticRegression",
        "selected_config": best_name,
        "config": config,
        "data": {"train": len(train_df), "validation": len(val_df), "n_labels": len(labels)},
        "validation": val_metrics,
        "train_time_s": best_row["train_time_s"],
        "peak_ram_mb_train": best_row["peak_ram_mb"],
        "inference_latency_batch1": {"cpu": latency},
        "model_load_time_s": round(load_s, 3),
        "model_size_mb": model_mb,
        "reload_prediction_identical": same,
        "environment": environment_info(),
    }, result_dir / "summary.json")

    print(f"\n추론 시간 중앙값(CPU, 배치 1) {latency['median_s']}s / 모델 로딩 {load_s:.3f}s / 크기 {model_mb}MB")
    print(f"재로딩 후 예측 일치: {same}")
    print(f"저장: {model_dir}, {result_dir}")


if __name__ == "__main__":
    main()
