"""서비스 설정. 값은 환경 변수(.env)로 바꿀 수 있고, 없으면 아래 기본값을 사용한다.

기본값은 scripts/training/train_bert.py의 저장 형식(models/<실행 이름>/best, max_length 128)에 맞춘다.
분류기 선정 결과에 따라 .env의 BERT_MODEL_DIR 또는 CLASSIFIER_TYPE을 바꾼다.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_DIR / value


# 분류기
CLASSIFIER_TYPE = os.getenv("CLASSIFIER_TYPE", "bert")              # "bert" 또는 "baseline"
BERT_MODEL_DIR = _path("BERT_MODEL_DIR", "models/bert/best")
BERT_TOKENIZER_FALLBACK = os.getenv("BERT_TOKENIZER_FALLBACK", "klue/bert-base")
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "128"))         # 학습 스크립트와 같은 값
BASELINE_MODEL_PATH = _path("BASELINE_MODEL_PATH", "models/baseline/model.joblib")
OUT_OF_SCOPE_LABEL = os.getenv("OUT_OF_SCOPE_LABEL", "업무 외")
# 분류기가 업무 외로 예측해도 확신도가 이 값보다 낮으면 차단하지 않고 생성 LLM에 넘겨 2차 확인한다.
# 근거: 확인용 질문에서 은행 질문("통장 비번 까먹었어요" 등)이 업무 외로 오차단됨.
#       값은 확인용 질문 결과에 맞추지 않고 "확실할 때만 1차 차단"이라는 원칙으로 정함.
OUT_OF_SCOPE_BLOCK_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_BLOCK_THRESHOLD", "0.9"))

# 요청 제한 [합의 필요]
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "local")                    # "local" 또는 "cloud"

# 고정 문구 파일 (없으면 기본 문구 사용)
FIXED_MESSAGES_PATH = _path("FIXED_MESSAGES_PATH", "prompts/fixed_messages.json")

# 프롬프트·지침 파일
SYSTEM_PROMPT_PATH = _path("SYSTEM_PROMPT_PATH", "prompts/system_prompt.md")
TOPIC_POLICIES_PATH = _path("TOPIC_POLICIES_PATH", "prompts/topic_policies.json")

# 생성 LLM 공통 설정 (로컬·상용 비교를 위해 같은 값 사용)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# 로컬 LLM (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")                         # 예: ollama list에 나오는 모델 이름
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")          # 모델을 메모리에 유지할 시간

# 상용 LLM
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "openai")             # "openai", "anthropic", "google"
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "")                         # 제공업체의 정확한 모델 ID
# 추론(reasoning) 모델의 추론 강도. 예: OpenAI GPT-5 계열 "none" / "low" / "medium".
# 비워두면 일반 모델처럼 temperature를 사용한다. 추론 모델은 temperature를 지원하지 않을 수 있다.
CLOUD_REASONING_EFFORT = os.getenv("CLOUD_REASONING_EFFORT", "")
_DEFAULT_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}
CLOUD_API_KEY_ENV = os.getenv("CLOUD_API_KEY_ENV", _DEFAULT_KEY_ENV.get(CLOUD_PROVIDER, "OPENAI_API_KEY"))

# 실행 기록 (Git 제외: /logs/)
REQUEST_LOG_PATH = _path("REQUEST_LOG_PATH", "logs/requests.jsonl")

# 챗봇 화면에서 호출할 수 있도록 허용할 주소 (쉼표로 구분)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
