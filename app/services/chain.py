"""LangChain 처리 흐름: 응답 정책 프롬프트 구성 → 생성 LLM 호출 → 답변 문자열.

응답 정책 (prompts/)
- LLM에 사용자 원문, 분류 결과, 주제별 답변 정책 전체를 전달한다.
- LLM은 원문을 다시 보고 업무 외·의도 불명이면 고정 문구만 반환한다.
- 고정 문구는 prompts/fixed_messages.json에서 가져와 프롬프트에 넣는다. (FastAPI 판별 문구와 항상 같음)
"""

import json
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app import config
from app.messages import MESSAGES

HUMAN_TEMPLATE = """[사용자 원문]
{question}

[분류 결과 (참고용)]
{label}

[주제별 답변 정책]
{guidelines}"""


class LLMNotConfigured(Exception):
    """요청한 LLM을 사용할 설정이 없음 (모델 이름 또는 API 키 없음)."""


@lru_cache
def load_guidelines_text() -> str:
    """prompts/topic_policies.json → 프롬프트에 넣을 글. 주제마다 안내할 내용, 하지 말 것, 연결할 곳."""
    policies = json.loads(config.TOPIC_POLICIES_PATH.read_text(encoding="utf-8"))
    blocks = []
    for label, policy in policies.items():
        blocks.append(
            f"### {label}\n"
            f"- 안내: {' / '.join(policy['안내'])}\n"
            f"- 하지 말 것: {' / '.join(policy['금지'])}\n"
            f"- 연결할 곳: {policy['연결']}"
        )
    return "\n\n".join(blocks)


@lru_cache
def build_prompt() -> ChatPromptTemplate:
    system_text = config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages([("system", system_text), ("human", HUMAN_TEMPLATE)])
    # 고정 문구와 지침은 요청마다 같으므로 미리 채워 둔다
    return prompt.partial(
        out_of_scope_message=MESSAGES["out_of_scope"],
        unclear_message=MESSAGES["unclear"],
        guidelines=load_guidelines_text(),
    )


def _local_llm():
    if not config.OLLAMA_MODEL:
        raise LLMNotConfigured("OLLAMA_MODEL이 설정되지 않았습니다.")
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        num_predict=config.LLM_MAX_TOKENS,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )


def _cloud_llm():
    import os
    if not config.CLOUD_MODEL:
        raise LLMNotConfigured("CLOUD_MODEL이 설정되지 않았습니다.")
    if not os.getenv(config.CLOUD_API_KEY_ENV):
        raise LLMNotConfigured(f"{config.CLOUD_API_KEY_ENV}가 설정되지 않았습니다.")

    common = {"temperature": config.LLM_TEMPERATURE, "max_retries": 1, "timeout": config.LLM_TIMEOUT_S}
    if config.CLOUD_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        if config.CLOUD_REASONING_EFFORT:
            # 추론 모델: temperature 대신 추론 강도를 지정한다.
            # 추론에 쓰인 토큰도 최대 출력 길이에 포함되므로, 답변이 잘리지 않게 여유를 둔다.
            return ChatOpenAI(model=config.CLOUD_MODEL, reasoning_effort=config.CLOUD_REASONING_EFFORT,
                              max_tokens=config.LLM_MAX_TOKENS * 4, max_retries=1, timeout=config.LLM_TIMEOUT_S)
        return ChatOpenAI(model=config.CLOUD_MODEL, max_tokens=config.LLM_MAX_TOKENS, **common)
    if config.CLOUD_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.CLOUD_MODEL, max_tokens=config.LLM_MAX_TOKENS, **common)
    if config.CLOUD_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.CLOUD_MODEL, max_output_tokens=config.LLM_MAX_TOKENS,
                                      temperature=config.LLM_TEMPERATURE, max_retries=1)
    raise LLMNotConfigured(f"지원하지 않는 CLOUD_PROVIDER: {config.CLOUD_PROVIDER}")


@lru_cache
def get_chain(model: str):
    """model: "local" 또는 "cloud". 같은 프롬프트에 LLM만 바꿔 연결한다."""
    llm = _local_llm() if model == "local" else _cloud_llm()
    return build_prompt() | llm | StrOutputParser()


def run_chain(question: str, label: str, model: str) -> str:
    return get_chain(model).invoke({"question": question, "label": label})
