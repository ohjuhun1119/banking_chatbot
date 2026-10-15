"""요청·응답 형식 (docs/api_spec.md 기준)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app import config

Route = Literal["answered", "blocked_by_classifier", "blocked_by_llm", "unclear", "error"]
LLMName = Literal["local", "cloud"]


class QuestionRequest(BaseModel):
    question: str = Field(..., description="고객 질문", examples=["잘못 보낸 돈 돌려받을 수 있나요?"])

    @field_validator("question")
    @classmethod
    def check_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("질문을 입력해 주세요.")
        if len(value) > config.MAX_QUESTION_CHARS:
            raise ValueError(f"질문은 {config.MAX_QUESTION_CHARS}자 이하로 입력해 주세요.")
        return value


class ChatRequest(QuestionRequest):
    model: LLMName = Field(default=config.DEFAULT_LLM, description="답변을 생성할 LLM")


class ClassifyLatency(BaseModel):
    classify: float | None


class ChatLatency(BaseModel):
    classify: float | None
    generate: float | None
    total: float


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    latency_s: ClassifyLatency


class ChatResponse(BaseModel):
    answer: str
    route: Route
    label: str | None
    confidence: float | None
    model: LLMName | None
    latency_s: ChatLatency
    error: str | None
    masked: dict[str, int] = Field(default_factory=dict, description="외부로 보내기 전에 가린 개인정보의 종류와 개수 (원문은 포함하지 않음)")


class ClassifierStatus(BaseModel):
    loaded: bool
    model: str | None
    labels: int | None


class LLMStatus(BaseModel):
    local: bool
    cloud: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    classifier: ClassifierStatus
    llm: LLMStatus
