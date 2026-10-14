"""고정 안내 문구와, 생성 LLM 답변이 고정 문구인지 판별하는 함수.

문구 원문은 prompts/fixed_messages.json을 기준으로 하고, 파일이 없으면 응답 정책 문서의 문구를 사용한다.
"""

import json
import re

from app import config

DEFAULT_MESSAGES = {
    "out_of_scope": "은행 업무와 관련된 질문만 안내해드릴 수 있습니다. 예금·대출·이체·환전 등 은행 업무에 대해 질문해 주세요.",
    "unclear": "문의 내용을 명확히 파악하기 어려워 안내해드릴 수 없습니다.",
    "error": "현재 답변을 제공하기 어렵습니다. 잠시 후 다시 시도해 주세요.",
}


def load_messages() -> dict:
    path = config.FIXED_MESSAGES_PATH
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = set(DEFAULT_MESSAGES) - set(data)
        if missing:
            raise ValueError(f"{path}에 없는 고정 문구: {missing}")
        return {k: data[k] for k in DEFAULT_MESSAGES}
    print(f"[참고] {path} 없음 → 기본 고정 문구 사용")
    return dict(DEFAULT_MESSAGES)


MESSAGES = load_messages()


def normalize(text: str) -> str:
    """따옴표·공백·줄바꿈 차이를 없애 비교하기 쉬운 형태로 만든다."""
    text = re.sub(r"[\"'“”‘’「」『』]", "", text or "")
    return re.sub(r"\s+", "", text)


def match_fixed_message(answer: str) -> str | None:
    """LLM 답변이 고정 문구 ①(업무 외) 또는 ②(의도 불명)이면 그 이름을, 아니면 None을 반환."""
    norm_answer = normalize(answer)
    for key in ("out_of_scope", "unclear"):
        norm_msg = normalize(MESSAGES[key])
        # 문구만 그대로 돌려준 경우 (앞뒤에 짧은 군더더기가 붙은 경우까지 허용)
        if norm_answer == norm_msg or (norm_msg in norm_answer and len(norm_answer) <= len(norm_msg) * 1.3):
            return key
    return None
