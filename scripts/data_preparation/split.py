"""정제한 데이터를 학습·검증·테스트로 약 80:10:10 분할한다.

실행 (프로젝트 루트에서):
    python scripts/data_preparation/split.py

입력: data/processed/cleaned.csv
출력:
    data/processed/train.csv, validation.csv, test.csv
    results/data/split_info.json, split_summary.csv   (건수 통계, 커밋 가능)

분할 원칙
- 묶음 단위 분할: 같은 source_id의 질문은 같은 분할에만 들어간다.
  (은행 질문은 같은 상담, 업무 외 질문은 같은 템플릿 묶음)
  → 비슷한 질문이 학습과 평가에 나뉘어 들어가 점수가 부풀려지는 것을 막는다.
- 주제 비율 유지: StratifiedGroupKFold로 세 분할의 주제 비율을 최대한 같게 맞춘다.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results" / "data"
SEED = 42
LABEL_COL, GROUP_COL = "qa_topic", "source_id"


def group_split(df: pd.DataFrame, n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """n_splits개 묶음 중 1개를 떼어낸다. (나머지, 떼어낸 것)"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest_idx, held_idx = next(sgkf.split(df, df[LABEL_COL], groups=df[GROUP_COL]))
    return df.iloc[rest_idx], df.iloc[held_idx]


def main():
    in_path = PROCESSED_DIR / "cleaned.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path}가 없습니다. clean.py를 먼저 실행하세요.")
    df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError(f"{in_path}가 비어 있습니다.")

    # 1) 전체의 1/10을 test로
    rest, test = group_split(df, n_splits=10)
    # 2) 나머지의 1/9을 validation으로 → 전체의 약 1/10
    train, validation = group_split(rest, n_splits=9)
    splits = {"train": train, "validation": validation, "test": test}

    # 검증: 모든 행이 정확히 한 분할에 들어갔는지, 묶음이 겹치지 않는지
    if sum(len(v) for v in splits.values()) != len(df):
        raise ValueError("분할 후 행 수 합계가 원본과 다릅니다.")
    overlaps = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}-{b}"] = len(set(splits[a][GROUP_COL]) & set(splits[b][GROUP_COL]))
    if any(overlaps.values()):
        raise ValueError(f"분할 사이에 같은 source_id가 있습니다: {overlaps}")

    # 저장
    for name, part in splits.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    labels = sorted(df[LABEL_COL].unique())
    counts = pd.DataFrame({k: v[LABEL_COL].value_counts() for k, v in splits.items()}).reindex(labels).fillna(0).astype(int)
    ratios = (counts / counts.sum() * 100).round(2).add_suffix("_%")
    summary = pd.concat([counts, ratios], axis=1)
    summary.loc["합계"] = list(counts.sum()) + [100.0] * len(splits)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / "split_summary.csv", encoding="utf-8-sig")

    total = len(df)
    info = {
        "seed": SEED,
        "method": "StratifiedGroupKFold (group=source_id), test 1/10 → validation 1/9 of rest",
        "rows": {k: len(v) for k, v in splits.items()},
        "ratio_%": {k: round(len(v) / total * 100, 2) for k, v in splits.items()},
        "groups": {k: int(v[GROUP_COL].nunique()) for k, v in splits.items()},
        "source_id_overlaps": overlaps,
        "n_labels": len(labels),
        "labels": labels,
    }
    (RESULT_DIR / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("분할 결과:")
    for k in splits:
        print(f"  {k:<10} {info['rows'][k]:>6,}행 ({info['ratio_%'][k]:>5.2f}%), 묶음 {info['groups'][k]:,}개")
    print("겹치는 source_id:", ", ".join(f"{k} {v}" for k, v in overlaps.items()))
    print("\n주제별 건수와 비율:")
    print(summary.to_string())
    print(f"\n저장: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
