"""추출한 질문 데이터를 정제한다.

실행 (프로젝트 루트에서):
    python scripts/data_preparation/clean.py

입력:
    data/processed/all.csv            은행 질문 (extract.py)
    data/processed/out_of_scope.csv   업무 외 질문 (build_out_of_scope.py, 있으면 합침)
출력:
    data/processed/cleaned.csv     학습에 사용할 데이터
    data/processed/excluded.csv    제외한 행과 제외 이유 (확인용)
    results/data/clean_info.json   단계별 건수 (커밋 가능, 질문 원문 없음)

정제 규칙
0. 업무 외 질문 합치기: out_of_scope.csv가 있으면 은행 질문과 합쳐 11개 주제로 만든다
1. 공백 정리: 연속 공백·줄바꿈을 공백 하나로 바꾸고 앞뒤 공백 제거 (개인정보 마스킹이 지워지며 생긴 공백 포함)
2. 빈 질문 제외
3. 정답 충돌 제외: 같은 질문에 정답 주제가 2개 이상이면 정답을 정할 수 없으므로 해당 질문의 모든 행 제외
4. 중복 제거: 질문과 정답이 모두 같은 행은 첫 번째만 유지
"""

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results" / "data"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main():
    in_path = PROCESSED_DIR / "all.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path}가 없습니다. extract.py를 먼저 실행하세요.")
    df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError(f"{in_path}가 비어 있습니다.")
    info = {"bank_rows": len(df)}

    # 0. 업무 외 질문 합치기
    ood_path = PROCESSED_DIR / "out_of_scope.csv"
    if ood_path.exists():
        ood = pd.read_csv(ood_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if list(ood.columns) != list(df.columns):
            raise ValueError("out_of_scope.csv의 컬럼이 all.csv와 다릅니다.")
        df = pd.concat([df, ood], ignore_index=True)
        info["out_of_scope_rows"] = len(ood)
    else:
        print("[참고] out_of_scope.csv 없음 → 은행 질문만 정제 (10개 주제)")
        info["out_of_scope_rows"] = 0
    info["input_rows"] = len(df)

    # 1. 공백 정리
    original = df["question"].copy()
    df["question"] = df["question"].map(normalize_space)
    info["whitespace_normalized_rows"] = int((original != df["question"]).sum())

    excluded = []

    # 2. 빈 질문 제외
    empty_mask = df["question"] == ""
    excluded.append(df[empty_mask].assign(reason="빈 질문"))
    df = df[~empty_mask]
    info["removed_empty"] = int(empty_mask.sum())

    # 3. 정답 충돌 제외
    topic_counts = df.groupby("question")["qa_topic"].nunique()
    conflict_questions = topic_counts[topic_counts > 1].index
    conflict_mask = df["question"].isin(conflict_questions)
    excluded.append(df[conflict_mask].assign(reason="정답 충돌"))
    df = df[~conflict_mask]
    info["conflict_questions"] = int(len(conflict_questions))
    info["removed_conflict_rows"] = int(conflict_mask.sum())

    # 4. 중복 제거
    dup_mask = df.duplicated(subset=["question", "qa_topic"], keep="first")
    excluded.append(df[dup_mask].assign(reason="중복"))
    df = df[~dup_mask].reset_index(drop=True)
    info["removed_duplicate_rows"] = int(dup_mask.sum())

    info["cleaned_rows"] = len(df)
    removed = info["removed_empty"] + info["removed_conflict_rows"] + info["removed_duplicate_rows"]
    if info["input_rows"] != removed + info["cleaned_rows"]:
        raise ValueError("정제 전후 건수가 맞지 않습니다.")

    # 참고 지표
    info["n_topics"] = int(df["qa_topic"].nunique())
    info["source_ids"] = int(df["source_id"].nunique())
    info["sources_with_multiple_questions"] = int((df["source_id"].value_counts() > 1).sum())
    info["qa_topic_consulting_topic_agreement"] = round(float((df["qa_topic"] == df["consulting_topic"]).mean()), 4)
    info["topic_counts"] = df["qa_topic"].value_counts().to_dict()

    # 저장
    df.to_csv(PROCESSED_DIR / "cleaned.csv", index=False, encoding="utf-8-sig")
    pd.concat(excluded, ignore_index=True).to_csv(
        PROCESSED_DIR / "excluded.csv", index=False, encoding="utf-8-sig")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "clean_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"입력: {info['input_rows']:,}행 (은행 {info['bank_rows']:,} + 업무 외 {info['out_of_scope_rows']:,})")
    print(f"  공백 정리된 질문: {info['whitespace_normalized_rows']:,}")
    print(f"  빈 질문 제외: {info['removed_empty']:,}")
    print(f"  정답 충돌 제외: {info['removed_conflict_rows']:,}행 (질문 {info['conflict_questions']:,}종)")
    print(f"  중복 제거: {info['removed_duplicate_rows']:,}")
    print(f"최종: {info['cleaned_rows']:,}행 / 주제 {info['n_topics']}종")
    print(f"\nsource_id {info['source_ids']:,}개 중 질문이 2개 이상인 상담: {info['sources_with_multiple_questions']:,}")
    print(f"qa_topic - consulting_topic 일치율: {info['qa_topic_consulting_topic_agreement']:.1%}")
    print("\n주제별 건수:")
    for topic, count in info["topic_counts"].items():
        print(f"  {topic}: {count:,} ({count / info['cleaned_rows']:.1%})")
    print(f"\n저장: {PROCESSED_DIR / 'cleaned.csv'}")


if __name__ == "__main__":
    main()
