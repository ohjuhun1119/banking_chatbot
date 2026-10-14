"""LangChain 처리 흐름 테스트. 실제 LLM 대신 정해진 답을 돌려주는 가짜 모델을 사용한다."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import config
from app.messages import MESSAGES, match_fixed_message
from app.services import chain


def test_프롬프트에_원문_분류결과_지침_고정문구가_모두_들어감():
    messages = chain.build_prompt().format_messages(question="환전 신청 방법", label="환전문의")
    system, human = messages[0].content, messages[1].content
    assert MESSAGES["out_of_scope"] in system and MESSAGES["unclear"] in system
    assert "환전 신청 방법" in human and "환전문의" in human
    assert "착오송금 반환지원 제도" in human and "### 환전문의" in human   # 주제별 정책 전체가 전달됨
    assert "{" not in system and "{" not in human   # 채우지 않은 변수가 남지 않음


def test_체인이_LLM_답변을_문자열로_반환(monkeypatch):
    fake = FakeListChatModel(responses=["환전은 영업점 또는 앱에서 신청할 수 있습니다."])
    monkeypatch.setattr(chain, "get_chain", lambda model: chain.build_prompt() | fake | chain.StrOutputParser())
    assert chain.run_chain("환전 신청 방법", "환전문의", "local").startswith("환전은")


def test_LLM이_돌려준_고정문구를_판별(monkeypatch):
    fake = FakeListChatModel(responses=[f"“{MESSAGES['out_of_scope']}”"])
    monkeypatch.setattr(chain, "get_chain", lambda model: chain.build_prompt() | fake | chain.StrOutputParser())
    answer = chain.run_chain("무지개는 왜 생겨?", "대출문의(만기/연장/조회등)", "local")
    assert match_fixed_message(answer) == "out_of_scope"


def test_모델_설정이_없으면_LLMNotConfigured(monkeypatch):
    chain.get_chain.cache_clear()
    monkeypatch.setattr(config, "OLLAMA_MODEL", "")
    try:
        chain.get_chain("local")
        assert False, "예외가 발생해야 함"
    except chain.LLMNotConfigured:
        pass
    finally:
        chain.get_chain.cache_clear()
