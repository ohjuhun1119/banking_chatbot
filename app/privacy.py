"""생성 LLM(외부 API)과 실행 기록으로 보내기 전에 질문 속 개인정보를 가린다.

가리는 것: 주민등록번호, 카드번호, 전화번호, 이메일, 계좌번호(하이픈 형태·긴 숫자),
           '인증번호·OTP·비밀번호·PIN' 뒤에 적힌 숫자
가리지 못하는 것: 이름, 주소, 생년월일처럼 정해진 형태가 없는 정보 (정규식으로는 걸러낼 수 없음)

패턴은 위에서부터 순서대로 적용한다. (주민등록번호 → 카드 → 전화 → ... 순서가 바뀌면 서로 겹쳐 잘못 가려짐)
"""

import re
from collections import Counter

# (이름, 패턴, 바꿀 글자). 숫자 앞뒤에 다른 숫자가 붙어 있지 않을 때만 일치시킨다.
_D = r"(?<!\d)"      # 앞에 숫자가 없음
_E = r"(?!\d)"       # 뒤에 숫자가 없음

PATTERNS = [
    ("주민등록번호", re.compile(rf"{_D}\d{{6}}[-\s]?[1-4]\d{{6}}{_E}"), "[주민등록번호]"),
    ("카드번호", re.compile(rf"{_D}\d{{4}}[-\s]?\d{{4}}[-\s]?\d{{4}}[-\s]?\d{{4}}{_E}"), "[카드번호]"),
    ("전화번호", re.compile(rf"{_D}(?:01[016789]|0[2-6]\d?)[-\s]?\d{{3,4}}[-\s]?\d{{4}}{_E}"), "[전화번호]"),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[이메일]"),
]

# 하이픈으로 나뉜 계좌번호 후보. 날짜(2026-10-15)와 구분하려고 숫자 개수를 따로 검사한다.
_ACCOUNT_DASHED = re.compile(rf"{_D}\d{{2,6}}(?:-\d{{2,8}}){{1,3}}{_E}")
# 하이픈 없는 긴 숫자(금액 표기와 헷갈릴 수 있어 11자리 이상만)
_ACCOUNT_PLAIN = re.compile(rf"{_D}\d{{11,14}}{_E}")
# '인증번호 123456', 'OTP는 123456', '비밀번호: 1234' 처럼 말 뒤에 따라오는 숫자
_SECRET = re.compile(r"((?:인증\s?번호|OTP|otp|보안\s?카드|비밀\s?번호|비번|PIN|pin)\s*(?:번호)?\s*(?:은|는|이|가|:|=)?\s*)(\d{4,8})(?!\d)")


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """(가린 문장, {종류: 가린 개수})를 돌려준다. 가린 게 없으면 개수 딕셔너리는 비어 있다."""
    found: Counter = Counter()

    for name, pattern, replacement in PATTERNS:
        text, n = pattern.subn(replacement, text)
        if n:
            found[name] += n

    def _account(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 16:          # 날짜(8자리)·짧은 번호는 제외
            found["계좌번호"] += 1
            return "[계좌번호]"
        return match.group()

    text = _ACCOUNT_DASHED.sub(_account, text)
    text, n = _ACCOUNT_PLAIN.subn(lambda m: "[계좌번호]", text)
    if n:
        found["계좌번호"] += n

    text, n = _SECRET.subn(lambda m: f"{m.group(1)}[비밀정보]", text)
    if n:
        found["비밀정보"] += n

    return text, dict(found)
