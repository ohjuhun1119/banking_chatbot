"""AI Hub 은행 라벨링 JSON에서 질문 단위 데이터를 추출해 CSV로 저장한다.

실행 (프로젝트 루트에서):
    python scripts/data_preparation/extract.py

입력: data/raw/training_labeling_data_bank/, data/raw/validation_labeling_data_bank/ 아래 JSON
출력: data/processed/all.csv (질문 하나가 한 행)
"""

import json
from pathlib import Path

import pandas as pd

# 이 파일 위치 기준으로 프로젝트 폴더를 찾는다 (scripts/data_preparation/extract.py → 두 단계 위)
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "processed"
SUB_DIRS = ["training_labeling_data_bank", "validation_labeling_data_bank"]
COLUMNS = ["qa_id", "source_id", "question", "qa_topic", "consulting_topic", "reference_answer"]


def find_json_files() -> list[Path]:
    files = []
    for name in SUB_DIRS:
        folder = RAW_DIR / name
        # 폴더가 없으면 바로 멈춘다 (빈 CSV가 만들어지는 걸 막기 위해)
        if not folder.is_dir():
            raise FileNotFoundError(f"라벨링 데이터 폴더 없음: {folder}")
        # 하위 폴더까지 모든 JSON을 실행할 때마다 같은 순서로 읽도록 정렬
        found = sorted(folder.rglob("*.json"))
        print(f"{name}: JSON {len(found):,}개")
        files.extend(found)
    if not files:
        raise FileNotFoundError(f"JSON 파일이 하나도 없습니다: {RAW_DIR}")
    return files


def extract_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source_id = data["source"]["source_id"]
    consulting_topic = data.get("consulting", {}).get("consulting_topic", "")

    rows = []
    # qa_data는 리스트라 파일 하나에 질문이 여러 개일 수 있다
    for qa in data["qa_data"]:
        rows.append({
            "qa_id": qa.get("qa_id", ""),
            "source_id": source_id,
            "question": qa["input"]["question"],
            "qa_topic": qa["qa_topic"],
            "consulting_topic": consulting_topic,
            "reference_answer": qa.get("output", ""),
        })
    return rows


def main():
    files = find_json_files()
    rows = []
    for path in files:
        rows.extend(extract_rows(path))

    df = pd.DataFrame(rows, columns=COLUMNS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "all.csv"
    # utf-8-sig: Windows 엑셀에서 열어도 한글이 깨지지 않음
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n읽은 JSON 파일: {len(files):,}")
    print(f"추출 행: {len(df):,}")
    print(f"주제 수: {df['qa_topic'].nunique()}")
    print(f"빈 질문: {(df['question'].str.strip() == '').sum():,}")
    print(f"source_id 종류: {df['source_id'].nunique():,}")
    print("\n주제별 건수:")
    print(df["qa_topic"].value_counts().to_string())

    # 저장한 파일을 다시 읽어 행 수와 컬럼이 같은지 확인
    loaded = pd.read_csv(out_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if len(loaded) != len(df) or list(loaded.columns) != COLUMNS:
        raise ValueError("저장 전후 CSV의 행 수 또는 컬럼이 다릅니다.")
    print(f"\n저장 확인 완료: {out_path}")


if __name__ == "__main__":
    main()
