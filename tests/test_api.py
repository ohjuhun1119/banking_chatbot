"""API 시나리오 테스트 (평가 기준 ④: 정상·예외 시나리오별 기대 동작).

실제 모델 없이 동작하도록 분류기와 생성 LLM을 가짜로 바꿔 흐름만 검증한다.
실행: python -m pytest tests -v
"""

import time

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.classifier import Prediction
from app.messages import MESSAGES
from app.services import answer as answer_service


class FakeClassifier:
    """질문에 들어 있는 단어로 주제를 정하는 가짜 분류기."""
    name, n_labels = "fake", 11

    def predict(self, text):
        if "무지개" in text and "은행" not in text:
            return Prediction(config.OUT_OF_SCOPE_LABEL, 0.99, 0.001)
        if "비번" in text:   # 은행 질문인데 분류기가 업무 외로 애매하게 예측한 경우
            return Prediction(config.OUT_OF_SCOPE_LABEL, 0.55, 0.001)
        if "환전" in text:
            return Prediction("환전문의", 0.9, 0.001)
        return Prediction("대출문의(만기/연장/조회등)", 0.5, 0.001)


def fake_generate(question, label, model):
    if "고장" in question:
        raise RuntimeError("연결 실패")
    if "느림" in question:
        time.sleep(2)
    if "고양이" in question:   # LLM이 업무 외로 2차 차단 (따옴표를 붙여 돌려주는 경우)
        return f"“{MESSAGES['out_of_scope']}”"
    if "그거" in question:     # LLM이 의도 불명으로 판단
        return MESSAGES["unclear"] + "\n"
    return f"{label} 관련 절차를 안내해 드립니다."


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(answer_service, "generate_answer", fake_generate)
    monkeypatch.setattr(config, "REQUEST_LOG_PATH", tmp_path / "requests.jsonl")
    monkeypatch.setattr(config, "LLM_TIMEOUT_S", 1)
    with TestClient(main.app) as c:
        yield c


def chat(client, question, model="local"):
    return client.post("/chat", json={"question": question, "model": model})


def test_정상_은행_질문(client):
    body = chat(client, "환전 신청은 어떻게 하나요?").json()
    assert body["route"] == "answered" and body["label"] == "환전문의" and body["model"] == "local"
    assert body["latency_s"]["generate"] is not None


def test_분류기_1차_차단은_LLM을_호출하지_않음(client):
    body = chat(client, "무지개는 왜 생겨?").json()
    assert body["route"] == "blocked_by_classifier" and body["answer"] == MESSAGES["out_of_scope"]
    assert body["model"] is None and body["latency_s"]["generate"] is None


def test_업무외_확신도_낮으면_차단하지_않고_LLM으로(client):
    body = chat(client, "통장 비번 까먹었어요").json()
    assert body["label"] == config.OUT_OF_SCOPE_LABEL and body["confidence"] < config.OUT_OF_SCOPE_BLOCK_THRESHOLD
    assert body["route"] == "answered" and body["model"] == "local"   # LLM이 원문을 보고 답변


def test_LLM_2차_차단_따옴표_붙어도_판별(client):
    body = chat(client, "은행 이름을 딴 고양이 장난감 추천해줘").json()
    assert body["route"] == "blocked_by_llm" and body["answer"] == MESSAGES["out_of_scope"]


def test_의도_불명(client):
    body = chat(client, "그거 어떻게 해요").json()
    assert body["route"] == "unclear" and body["answer"] == MESSAGES["unclear"]


def test_LLM_호출_실패(client):
    body = chat(client, "대출 고장 테스트").json()
    assert body["route"] == "error" and body["error"] == "llm_error" and body["answer"] == MESSAGES["error"]


def test_LLM_시간_초과(client):
    body = chat(client, "대출 느림 테스트").json()
    assert body["route"] == "error" and body["error"] == "llm_timeout"


@pytest.mark.parametrize("payload", [
    {},                                              # 질문 없음
    {"question": "   "},                             # 공백만
    {"question": "가" * (config.MAX_QUESTION_CHARS + 1)},  # 길이 초과
    {"question": "대출 문의", "model": "gpt"},        # 허용되지 않은 모델
])
def test_입력_오류는_422(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_분류만(client):
    body = client.post("/classify", json={"question": "환전 수수료 문의"}).json()
    assert body["label"] == "환전문의" and 0 <= body["confidence"] <= 1


def test_상태_확인(client):
    body = client.get("/health").json()
    assert body["classifier"]["loaded"] is True and body["classifier"]["labels"] == 11


def test_분류기_없으면_error_route(monkeypatch, tmp_path):
    def broken():
        raise FileNotFoundError("모델 없음")
    monkeypatch.setattr(main, "load_classifier", broken)
    monkeypatch.setattr(config, "REQUEST_LOG_PATH", tmp_path / "requests.jsonl")
    with TestClient(main.app) as c:
        body = c.post("/chat", json={"question": "환전 문의"}).json()
        assert body["route"] == "error" and body["error"] == "classifier_unavailable"
        assert c.post("/classify", json={"question": "환전 문의"}).status_code == 503


def test_실행_기록_저장(client):
    chat(client, "환전 신청은 어떻게 하나요?")
    assert config.REQUEST_LOG_PATH.exists()
