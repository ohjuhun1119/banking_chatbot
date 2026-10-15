"""은행 상담 챗봇 FastAPI 서비스.

실행 (프로젝트 루트에서):
    uvicorn app.main:app --reload
문서 화면:
    http://localhost:8000/docs
"""

import asyncio
import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.classifier import load_classifier
from app.messages import MESSAGES, match_fixed_message
from app.privacy import mask_pii
from app.request_log import write_log
from app.schemas import (ChatRequest, ChatResponse, ClassifyResponse, HealthResponse, QuestionRequest)
from app.services import answer as answer_service

state = {"classifier": None, "classifier_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 분류기를 한 번만 불러온다
    state.update(classifier=None, classifier_error=None)
    try:
        state["classifier"] = load_classifier()
        state["classifier"].predict("서버 시작 확인용 질문입니다.")  # 워밍업: 첫 요청 지연 방지
        print(f"[시작] 분류기 로드 완료: {state['classifier'].name}")
    except Exception as e:
        state["classifier_error"] = str(e)
        print(f"[시작] 분류기 로드 실패: {e}")
    yield


app = FastAPI(title="은행 상담 챗봇 API", version="0.1.0", lifespan=lifespan,
              description="질문 분류(은행 업무 10종 + 업무 외) 후 주제별 정책으로 답변을 생성합니다.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# 챗봇 화면: 같은 서버에서 제공해 /chat을 같은 주소로 호출한다
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health", response_model=HealthResponse)
def health():
    clf = state["classifier"]
    local_ok = answer_service.is_local_available()
    cloud_ok = answer_service.is_cloud_configured()
    return {
        "status": "ok" if clf and (local_ok or cloud_ok) else "degraded",
        "classifier": {"loaded": clf is not None, "model": clf.name if clf else None,
                       "labels": clf.n_labels if clf else None},
        "llm": {"local": local_ok, "cloud": cloud_ok},
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: QuestionRequest):
    clf = state["classifier"]
    if clf is None:
        raise HTTPException(status_code=503, detail=f"분류기를 사용할 수 없습니다: {state['classifier_error']}")
    pred = clf.predict(req.question)
    result = {"label": pred.label, "confidence": pred.confidence, "latency_s": {"classify": pred.latency_s}}
    write_log("/classify", mask_pii(req.question)[0], result)   # 기록에는 가린 질문만 남긴다
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.perf_counter()
    # 분류기는 이 서버 안에서만 돌아가 원문을 쓰고, 외부 API로 가는 질문과 실행 기록에는 가린 질문을 쓴다
    safe_question, masked = mask_pii(req.question)
    result = {"answer": None, "route": None, "label": None, "confidence": None, "model": None,
              "latency_s": {"classify": None, "generate": None, "total": 0.0}, "error": None,
              "masked": masked}

    def finish(route: str, answer: str, error: str | None = None):
        result.update(route=route, answer=answer, error=error)
        result["latency_s"]["total"] = round(time.perf_counter() - start, 4)
        write_log("/chat", safe_question, result)
        return result

    # 1차 분류
    clf = state["classifier"]
    if clf is None:
        return finish("error", MESSAGES["error"], "classifier_unavailable")
    try:
        pred = clf.predict(req.question)
    except Exception as e:
        print(f"[오류] 분류 실패: {e}")
        return finish("error", MESSAGES["error"], "classifier_unavailable")
    result.update(label=pred.label, confidence=pred.confidence)
    result["latency_s"]["classify"] = pred.latency_s

    # 1차 차단: 업무 외로 확실할 때만 생성 LLM을 호출하지 않는다
    # 확신도가 낮으면 은행 질문을 잘못 막지 않도록 LLM에 넘겨 원문 기준으로 다시 판단한다
    if pred.label == config.OUT_OF_SCOPE_LABEL and pred.confidence >= config.OUT_OF_SCOPE_BLOCK_THRESHOLD:
        return finish("blocked_by_classifier", MESSAGES["out_of_scope"])

    # 2차 확인·답변: 생성 LLM
    result["model"] = req.model
    gen_start = time.perf_counter()
    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(answer_service.generate_answer, safe_question, pred.label, req.model),
            timeout=config.LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        result["latency_s"]["generate"] = round(time.perf_counter() - gen_start, 4)
        return finish("error", MESSAGES["error"], "llm_timeout")
    except answer_service.LLMNotConfigured:
        result["latency_s"]["generate"] = round(time.perf_counter() - gen_start, 4)
        return finish("error", MESSAGES["error"], "llm_not_configured")
    except Exception as e:
        print(f"[오류] 생성 LLM 호출 실패: {e}")
        result["latency_s"]["generate"] = round(time.perf_counter() - gen_start, 4)
        return finish("error", MESSAGES["error"], "llm_error")
    result["latency_s"]["generate"] = round(time.perf_counter() - gen_start, 4)

    # 2차 차단 판별: LLM이 고정 문구를 돌려줬는지
    matched = match_fixed_message(answer)
    if matched == "out_of_scope":
        return finish("blocked_by_llm", MESSAGES["out_of_scope"])
    if matched == "unclear":
        return finish("unclear", MESSAGES["unclear"])
    return finish("answered", answer.strip())
