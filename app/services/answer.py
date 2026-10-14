"""생성 LLM 연결 지점. FastAPI는 이 파일의 generate_answer만 호출한다.

약속 (docs/api_spec.md)
- 입력: 사용자 원문, 분류기 예측 주제, 사용할 LLM("local" / "cloud")
- 출력: LLM 답변 문자열. 업무 외·의도 불명으로 판단하면 고정 문구 ①·②를 가공하지 않고 그대로 반환
- 실패: 예외를 발생시킨다. (FastAPI가 잡아서 고정 문구 ③으로 응답)
"""

import os
import urllib.request

from app import config
from app.services.chain import LLMNotConfigured, run_chain  # noqa: F401  (main.py가 LLMNotConfigured를 사용)


def generate_answer(question: str, label: str, model: str) -> str:
    return run_chain(question, label, model)


def is_local_available() -> bool:
    """Ollama 서버가 응답하고, 사용할 모델이 설정되어 있는지 확인."""
    if not config.OLLAMA_MODEL:
        return False
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_cloud_configured() -> bool:
    """상용 LLM 모델 이름과 API 키가 설정되어 있는지 확인. (키 값은 기록하지 않는다)"""
    return bool(config.CLOUD_MODEL and os.getenv(config.CLOUD_API_KEY_ENV))
