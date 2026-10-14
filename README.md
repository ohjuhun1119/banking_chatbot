# 은행 고객 문의 분류·답변 챗봇 

## 개요

| 항목 | 내용 |
| --- | --- |
| 문제 | 은행 고객 질문을 상담 주제로 분류하고, 주제별 지침에 맞는 안내 답변을 생성 |
| 입력 | 고객 질문 1개 |
| 출력 | 상담 주제, 확신도, 안내 답변 (질문 1개 → 응답 1개) |
| 분류 | 은행 업무 10종 + 업무 외 (총 11종) |
| 데이터 | AI Hub 금융분야 고객상담 데이터 중 은행 라벨링 데이터 + 직접 설계한 업무 외 질문 |

## 진행 상황

- [x] 데이터 준비: 추출, 업무 외 질문 생성, 정제, 묶음 단위 분할
- [ ] 평가 기준
- [ ] 분류 모델 학습·비교 (TF-IDF + 로지스틱 회귀, klue/bert-base)
- [ ] 응답 정책·프롬프트
- [ ] LangChain·FastAPI 서비스
- [ ] 생성 모델 비교 (로컬 vs 상용)

## 데이터

- 출처: [AI Hub 금융분야 고객상담 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71926&topMenu=100), 은행 라벨링 데이터 (Training 40,000 + Validation 5,000)
- 사용 조건: AI Hub 이용정책에 따라 원본·가공 데이터는 저장소에 포함하지 않음. 각자 AI Hub에서 승인받아 내려받아 사용
- 업무 외 질문: `build_out_of_scope.py`로 직접 설계한 템플릿에서 5,000건 생성 (일반 3,500, 금융 단어가 들어간 어려운 예시 1,500)

### 분류 입력·정답

| 구분 | 필드 |
| --- | --- |
| 입력 X | `qa_data[].input.question` |
| 정답 y | `qa_data[].qa_topic` |

## 데이터 준비

원본 JSON을 아래 위치에 넣은 뒤 프로젝트 루트에서 순서대로 실행합니다.

```text
data/raw/
├── training_labeling_data_bank/
└── validation_labeling_data_bank/
```

```bash
python scripts/data_preparation/extract.py
python scripts/data_preparation/build_out_of_scope.py
python scripts/data_preparation/clean.py
python scripts/data_preparation/split.py
```

| 단계 | 처리 | 결과 |
| --- | --- | --- |
| 추출 | 질문 단위로 6개 컬럼 추출 | 45,000행 |
| 업무 외 생성 | 템플릿 68묶음에서 서로 다른 질문 5,000개 | 5,000행 |
| 정제 | 공백 정리 → 빈 질문 제외 → 정답 충돌 제외 → 중복 제거 | 49,809행, 11개 주제 |
| 분할 | `source_id` 묶음 단위, 주제 비율 유지 80:10:10 | 39,844 / 4,983 / 4,982 |

### 설계 판단

- **공백 정리 후 정제:** 개인정보 마스킹이 지워지며 생긴 연속 공백 때문에, 같은 질문이 다른 질문으로 취급되는 경우를 함께 걸러냄
- **묶음 단위 분할:** 같은 상담에서 나온 질문, 같은 템플릿에서 나온 업무 외 질문이 학습과 평가에 나뉘어 들어가지 않도록 함. 분할 간 겹치는 `source_id` 0개
- **업무 외 질문 설계:** 길이와 말투를 다양하게 하고 금융 단어가 들어간 어려운 예시를 포함해, 모델이 말투나 단어만으로 업무 외를 구분하지 않도록 함

### 한계

- 기타(은행)는 검증·테스트에 각 18건으로 성능 평가가 불안정함
- 업무 외 질문은 템플릿 기반이라 실제 사용자 질문보다 다양성이 부족함. 템플릿과 무관하게 직접 작성한 확인용 질문으로 별도 검증 예정

## 폴더 구조

```text
scripts/data_preparation/   데이터 추출·생성·정제·분할
results/data/               단계별 건수 기록 (질문 원문 없음)
docs/                       평가 기준 등 문서
data/, models/              로컬 전용, Git 제외
```

## 실행 환경

- Python 3.12, 패키지 관리 `uv`
- 설치: `uv sync`
