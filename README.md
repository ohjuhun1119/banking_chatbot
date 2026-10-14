# 은행 고객 문의 분류·답변 챗봇 (개인 구현)

4인 팀 프로젝트([팀 저장소](https://github.com/YunsOps/banking-chatbot-mvp-main))에서 서비스 구현(FastAPI)을 맡은 뒤,
데이터 준비부터 서비스까지 전 과정을 혼자 처음부터 다시 구현한 저장소입니다.

고객 질문을 **klue/bert-base**로 상담 주제(은행 업무 10종 + 업무 외)로 분류하고,
주제별 답변 정책에 따라 **GPT-5.6 Luna**가 안내 답변을 생성하는 FastAPI 서비스입니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 문제 | 은행 고객 질문을 상담 주제로 분류하고, 주제별 정책에 맞는 1차 안내 답변을 생성 |
| 입력 | 고객 질문 1개 |
| 출력 | 상담 주제, 확신도, 안내 답변 (질문 1개 → 응답 1개) |
| 분류 | 은행 업무 10종 + 업무 외 (총 11종) |
| 분류 모델 | klue/bert-base 파인튜닝 (TF-IDF + 로지스틱 회귀, kakaobank/kf-deberta-base와 비교 후 선정) |
| 생성 모델 | OpenAI gpt-5.6-luna (로컬 Qwen 2.5 7B와 비교 후 선정) |
| 서비스 | FastAPI + LangChain |

## 진행 상황

- [x] 데이터 준비: 추출, 업무 외 질문 생성, 정제, 묶음 단위 분할
- [x] 평가 기준 수립 (실험 전)
- [x] 분류 모델 학습·비교 (TF-IDF + 로지스틱 회귀, klue/bert-base)
- [x] 추가 실험: 금융 도메인 사전학습 모델(kakaobank/kf-deberta-base)
- [x] 업무 외 확인용 질문 평가, test 1회 평가
- [x] 응답 정책·프롬프트
- [x] LangChain·FastAPI 서비스
- [x] 생성 모델 비교 (로컬 vs 상용)
- [x] 서비스 통합 확인

## 처리 흐름

```text
고객 질문
  → 입력 검증 (빈 질문, 500자 초과 → 422)
  → klue/bert-base 분류 (은행 업무 10종 또는 업무 외)
      ├─ 업무 외 & 확신도 0.9 이상 → 고정 안내 (LLM 호출 없음)        route: blocked_by_classifier
      └─ 그 외 → 생성 LLM (원문 + 분류 결과 + 주제별 정책 전체)
            ├─ 은행 업무로 확인 → 정책에 맞춘 답변                    route: answered
            ├─ 업무 외로 판단 → 고정 안내                              route: blocked_by_llm
            └─ 의도 판단 어려움 → 고정 안내                            route: unclear
  → 실패·시간 초과 → 고정 안내 + 원인 기록                            route: error
```

- **2단계 차단:** 분류기는 업무 외로 **확실할 때만** 막고, 애매하면 생성 LLM이 원문을 보고 다시 판단합니다. 분류기가 은행 질문을 업무 외로 잘못 막는 문제를 확인용 질문에서 발견해 추가한 설계입니다.
- **분류 결과는 참고 정보:** LLM에는 분류 결과를 넘기되, 원문을 기준으로 판단하도록 프롬프트에 명시했습니다.

## 데이터

- 출처: [AI Hub 금융분야 고객상담 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71926&topMenu=100), 은행 라벨링 데이터 (Training 40,000 + Validation 5,000)
- 사용 조건: AI Hub 이용정책에 따라 원본·가공 데이터는 저장소에 포함하지 않습니다. 각자 AI Hub에서 승인받아 내려받아 사용합니다.
- 외부 전송: AI Hub 데이터는 외부 API로 보내지 않았습니다. 상용 LLM 비교에는 직접 작성한 질문만 사용했습니다.
- 업무 외 질문: `build_out_of_scope.py`로 직접 설계한 템플릿에서 5,000건 생성 (일반 3,500, 금융 단어가 들어간 어려운 예시 1,500)

### 분류 입력·정답

| 구분 | 필드 |
| --- | --- |
| 입력 X | `qa_data[].input.question` |
| 정답 y | `qa_data[].qa_topic` |

### 데이터 준비

| 단계 | 처리 | 결과 |
| --- | --- | --- |
| 추출 | 질문 단위로 6개 컬럼 추출 | 45,000행 |
| 업무 외 생성 | 템플릿 68묶음에서 서로 다른 질문 5,000개 | 5,000행 |
| 정제 | 공백 정리 → 빈 질문 제외 → 정답 충돌 제외 → 중복 제거 | 49,809행, 11개 주제 |
| 분할 | `source_id` 묶음 단위, 주제 비율 유지 80:10:10 | 39,844 / 4,983 / 4,982 |

**설계 판단**

- **공백 정리 후 정제:** 개인정보 마스킹이 지워지며 생긴 연속 공백 때문에 같은 질문이 다른 질문으로 취급되는 경우를 함께 걸러냈습니다.
- **묶음 단위 분할:** 같은 상담에서 나온 질문, 같은 템플릿에서 나온 업무 외 질문이 학습과 평가에 나뉘어 들어가지 않게 했습니다. 분할 간 겹치는 `source_id`는 0개입니다. 질문 단위로 나눈 팀 프로젝트(겹침 1,995개)보다 은행 업무 Macro F1이 약 1%p 낮게 나와, 질문 단위 분할이 점수를 부풀릴 수 있음을 확인했습니다.
- **업무 외 질문 설계:** 길이와 말투를 다양하게 하고 금융 단어가 들어간 어려운 예시(은행나무, 도서관 대출 등)를 넣어, 모델이 말투나 단어만으로 업무 외를 구분하지 않도록 했습니다.

## 분류 모델 결과

| 모델 | validation Macro F1 | test Macro F1 | test Accuracy | CPU 추론 (배치 1) | 크기 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + 로지스틱 회귀 | 0.7040 | 0.7031 | 0.7782 | 0.0002초 | 7MB |
| **klue/bert-base (선정)** | **0.7274** | 0.7287 | **0.8143** | 0.014초 | 423MB |
| kakaobank/kf-deberta-base | 0.7243 | 0.7351 | 0.8113 | 0.028초 | 712MB |

- **선정:** validation Macro F1 최고(klue/bert-base)와 1%p 이내인 kf-deberta 중 CPU 추론이 빠른 klue/bert-base ([평가 기준](docs/evaluation_criteria.md) 2.3)
- **test:** 세 모델 모두 validation과 비슷해 묶음 단위 분할에서 평가가 안정적이었습니다. kf-deberta가 test에서 0.64%p 높았지만, 대부분 표본 18건인 기타(은행)에서 1건을 맞힌 효과이며 이를 제외하면 klue/bert-base가 근소하게 높습니다. 사전에 정한 규칙대로 test 결과로 선정을 바꾸지 않았습니다.
- **남은 약점:** 대출문의 ↔ 이자/연체금액 혼동(약 225건), 기타(은행) 미분류

### 업무 외 확인용 질문 (직접 작성 60개)

| 모델 | 업무 외 검출 (40) | 은행 업무 통과 (20) |
| --- | --- | --- |
| TF-IDF + 로지스틱 회귀 | 22 | 19 |
| klue/bert-base | 38 | 18 |
| kakaobank/kf-deberta-base | 39 | 19 |

validation에서는 세 모델 모두 업무 외를 거의 완벽하게 맞혔지만, 템플릿과 무관하게 작성한 질문에서는 베이스라인이 40개 중 22개만 걸러내 **단어 기반 분류의 한계**가 드러났습니다. 사전학습 모델도 "통장 비번 까먹었어요"처럼 기타(은행)에 해당하는 질문을 업무 외로 막아, 서비스에 **업무 외 확신도 기준**을 추가했습니다.

### 추가 실험: 금융 도메인 사전학습

klue/bert-base의 오분류 약 25%가 대출 ↔ 이자 혼동이라, 금융 말뭉치로 사전학습한 kf-deberta-base가 이를 줄이는지 확인했습니다. 혼동은 236건에서 232건으로 거의 줄지 않았고, 이 혼동은 금융 용어 이해보다 **라벨 경계의 모호성**에서 비롯된 것으로 판단했습니다. 자세한 내용은 [실험 기록](docs/experiments/kf_deberta.md)에 있습니다.

## 생성 모델 비교

| 항목 | **GPT-5.6 Luna (선정)** | Qwen 2.5 7B (로컬) |
| --- | --- | --- |
| 사용 가능한 답변 (30개) | **30** | 20 |
| └ 은행 업무 (24개) | **24** | 14 |
| └ 업무 외·의도 불명 (6개) | 6 | 6 |
| 안전성 0점 | 0 | 0 |
| 생성 시간 중앙값 | **2.6초** | 4.3초 |
| 1,000건당 비용 | 약 $0.45 | 장비·전력 |

- **조건:** 같은 프롬프트, 상용은 `reasoning_effort=none`, 로컬은 Ollama `qwen2.5:7b`(`temperature 0`)
- **질문:** 직접 작성한 30개(은행 24, 업무 외 3, 의도 불명 3). AI Hub 데이터를 외부로 보내지 않기 위해 test 데이터 대신 직접 작성했습니다.
- **프롬프트:** 개발용 질문 10개로 **최대 3회까지만** 수정한 뒤 확정하고, 최종 질문은 1회만 실행했습니다.
- **채점:** 모델을 가린 블라인드 채점표로 0~2점 4개 항목(정확성, 핵심정보, 지침·형식, 안전성). AI 1차 채점 후 본인 검토.
- **해석:** 두 모델 모두 업무 외·의도 불명은 정확히 처리했지만, 로컬 모델은 구어체 은행 질문을 의도 불명으로 판단하거나 정책의 "~할 수 있다"를 단정하는 경우가 있었습니다. 이 경향은 개발용 질문에서도 같게 관찰됐습니다.
- **선정:** 사용 가능한 답변 차이(10개)가 기준(3개)을 넘어 상용 모델을 선택했습니다. ([평가 기준](docs/evaluation_criteria.md) 3.5)

## 서비스 확인

실행 중인 서버에 실제 모델로 요청을 보내 확인했습니다. (`scripts/evaluation/check_service.py`)

| 항목 | 결과 |
| --- | --- |
| 시나리오 (정상, 업무 외 2종, 의도 불명, 입력 오류 3종) | (실행 결과 기입) |
| 확인용 은행 질문 20개 | 답변 (기입) / 의도 불명 (기입) / 업무 외 차단 (기입) |
| 전체 응답 시간 (요청~응답) | 중앙값 (기입)초, p95 (기입)초 |
| 자동 테스트 (가짜 모델) | 19개 통과 |

## 실행 방법

### 환경

- Python 3.12, 패키지 관리 `uv`
- Apple Silicon Mac에서는 MPS(GPU) 자동 사용
- 로컬 LLM: [Ollama](https://ollama.com) + `qwen2.5:7b`

```bash
uv sync
```

### 1. 데이터 준비

원본 JSON을 아래 위치에 넣고 순서대로 실행합니다.

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

### 2. 분류 모델 학습과 평가

```bash
python scripts/training/train_baseline.py
python scripts/training/train_bert.py --smoke-test        # 실행 가능 여부·예상 시간 확인
python scripts/training/train_bert.py                     # klue/bert-base
python scripts/training/train_bert.py --model-name kakaobank/kf-deberta-base --device cpu
python scripts/training/compare_models.py                 # 비교와 선정
python scripts/evaluation/evaluate_probe.py               # 확인용 질문 60개
python scripts/evaluation/evaluate_test.py                # test 1회 평가
```

kf-deberta-base(DeBERTa 구조)는 Mac의 MPS에서 일부 연산이 지원되지 않아 매우 느려(스텝당 약 19초), CPU로 학습했습니다(스텝당 약 2.5초).

### 3. 서비스 실행

`.env` (저장소에 올리지 않음):

```text
OLLAMA_MODEL=qwen2.5:7b
CLOUD_PROVIDER=openai
CLOUD_MODEL=gpt-5.6-luna
CLOUD_REASONING_EFFORT=none
OPENAI_API_KEY=발급받은 키
DEFAULT_LLM=cloud
```

```bash
uvicorn app.main:app
```

`http://localhost:8000/docs`에서 `/health`, `/classify`, `/chat`을 직접 호출할 수 있습니다.

```json
POST /chat
{ "question": "환전 신청은 어떻게 하나요?", "model": "cloud" }
```

### 4. 생성 모델 비교와 서비스 확인

```bash
python scripts/evaluation/run_generation.py --split dev
python scripts/evaluation/run_generation.py --split final
python scripts/evaluation/make_scoring_sheet.py --split final
python scripts/evaluation/summarize_generation.py --split final --input-price 0.2 --output-price 1.2
python scripts/evaluation/check_service.py --model cloud
```

### 5. 테스트

```bash
python -m pytest tests -v
```

## 폴더 구조

```text
scripts/
├── data_preparation/   추출, 업무 외 생성, 정제, 분할
├── training/           베이스라인·사전학습 모델 학습, 공통 평가 도구, 모델 비교
└── evaluation/         확인용 질문, test, 생성 모델 비교, 서비스 확인
app/                    FastAPI 서비스 (분류기, LangChain, 고정 문구, 실행 기록)
prompts/                시스템 프롬프트, 주제별 답변 정책, 고정 문구
tests/                  API·체인 테스트, 확인용 질문 60개, 생성 비교 질문 40개
docs/                   평가 기준, 추가 실험 기록
results/                단계별 결과 (질문 원문이 담긴 오분류 사례는 제외)
data/, models/, logs/   로컬 전용, Git 제외
```

## 한계와 개선 방향

| 한계 | 개선 방향 |
| --- | --- |
| 대출문의 ↔ 이자/연체금액 혼동이 금융 특화 모델로도 줄지 않음 | 라벨 정의 재검토, 두 주제에 걸친 질문을 위한 다중 라벨 |
| 기타(은행)은 표본이 적어(검증·테스트 각 18건) 거의 분류하지 못함 | 데이터 보강, 클래스 가중치 |
| 업무 외 학습 데이터가 템플릿 기반이라 실제 질문보다 다양성이 부족 | 실제 업무 외 질문 수집, 확인용 질문 확대 |
| 확인용 질문 60개 결과를 보고 확신도 기준을 추가해, 이 질문들이 개발에 쓰임 | 새로운 확인용 질문으로 최종 재확인 |
| 상용 LLM 선정으로 고객 질문이 외부로 전송됨 | 개인정보 마스킹, 데이터 처리 계약, 더 큰 로컬 모델이나 미세조정 검토 |
| 생성 비교는 질문 30개, 1회 실행, AI 1차 채점 후 본인 검토 | 질문 확대, 반복 실행, 복수 채점자 |
| 분류 모델은 시드 1회 학습이라 0.3%p 수준의 차이는 우연일 수 있음 | 여러 시드로 반복 학습 |

## 라이선스

- 코드: 개인 포트폴리오용
- 데이터: AI Hub 이용정책을 따르며 저장소에 포함하지 않음
- 모델: klue/bert-base(CC-BY-SA-4.0), kakaobank/kf-deberta-base(MIT), Qwen 2.5 7B(Apache 2.0), GPT-5.6 Luna(OpenAI API 이용약관)
