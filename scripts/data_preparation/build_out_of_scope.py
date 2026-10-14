"""은행 업무와 관련 없는 '업무 외' 질문을 직접 설계한 템플릿으로 생성한다.

실행 (프로젝트 루트에서):
    python scripts/data_preparation/build_out_of_scope.py

출력: data/processed/out_of_scope.csv (all.csv와 같은 컬럼, qa_topic = "업무 외")

설계 원칙
- 일반 질문: 생활·과학·취미 등 은행과 무관한 주제
- 어려운 질문: '은행·대출·이자·환전·잔액' 같은 금융 단어가 들어 있지만 은행 업무가 아닌 질문
  (은행나무, 도서관 도서 대출, 소설 속 은행 강도 등) → 단어만 보고 분류하는 모델을 걸러내기 위해
- 말투·길이 다양화: 짧은 반말 질문부터 상황 설명이 붙은 긴 존댓말까지 섞어,
  모델이 '말투'나 '길이'로 업무 외를 구분하지 않도록 함
- source_id = 템플릿 묶음 ID: 같은 템플릿에서 나온 질문은 분할할 때 같은 쪽에 들어가도록
"""

import random
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_DIR / "data" / "processed" / "out_of_scope.csv"
SEED = 42
N_GENERAL = 3500   # 일반 질문 수
N_HARD = 1500      # 금융 단어가 들어간 어려운 질문 수
LABEL = "업무 외"

# 주제별 질문 템플릿: {x}에 같은 주제의 소재가 들어간다
GENERAL = {
    "과학상식": (["무지개", "번개", "밀물과 썰물", "낮과 밤", "계절 변화", "오로라", "지진", "블랙홀"],
              ["{x}는 왜 생기는 거야?", "{x}가 생기는 원리를 쉽게 설명해 줄 수 있나요?",
               "아이에게 {x}를 어떻게 설명하면 좋을까요?", "{x}에 대해 재미있는 사실 알려줘"]),
    "요리": (["김치찌개", "계란말이", "파스타", "된장국", "떡볶이", "샐러드", "카레"],
            ["{x} 맛있게 만드는 방법 알려줘", "{x}에 들어가는 재료가 뭐예요?",
             "{x} 남은 거 어떻게 보관해야 하나요?", "처음 해보는데 {x} 실패하지 않는 팁 있을까요?"]),
    "운동": (["스쿼트", "러닝", "요가", "수영", "스트레칭", "플랭크", "등산"],
            ["{x} 처음 시작할 때 주의할 점이 뭐예요?", "{x} 하루에 얼마나 하면 좋아?",
             "{x}할 때 필요한 준비물 알려주세요", "{x} 자세 교정하는 방법 있을까요?"]),
    "여행": (["제주도", "부산", "강릉", "경주", "전주", "속초"],
            ["{x} 1박 2일 여행 코스 짜줘", "{x}에서 꼭 먹어봐야 할 음식이 뭐예요?",
             "비 오는 날 {x}에서 갈 만한 곳 추천해 주세요", "{x} 가는 가장 빠른 방법이 뭐야?"]),
    "공부": (["영어 단어", "수학 공식", "한국사", "코딩", "글쓰기", "발표 준비"],
            ["{x} 효율적으로 공부하는 방법 알려줘", "{x} 하는데 집중이 안 될 때 어떻게 해요?",
             "{x} 초보자가 보기 좋은 자료 추천해 줄래?", "{x} 매일 30분씩 하면 효과 있을까요?"]),
    "반려동물": (["강아지", "고양이", "햄스터", "앵무새", "거북이"],
              ["{x}가 밥을 잘 안 먹는데 왜 그럴까요?", "{x} 처음 키울 때 준비할 게 뭐야?",
               "{x}랑 친해지는 방법 알려주세요", "{x} 산책이나 놀이는 얼마나 해줘야 해?"]),
    "생활": (["빨래 냄새", "방 습기", "곰팡이", "냉장고 정리", "분리수거", "수면 습관"],
            ["{x} 문제 해결하는 방법 있나요?", "{x} 관리 어떻게 하는 게 좋아?",
             "{x} 때문에 고민인데 간단한 팁 알려줄래?", "{x} 개선하려면 뭐부터 해야 할까요?"]),
    "취미창작": (["시", "짧은 소설", "노래 가사", "캐릭터 이름", "동아리 이름", "생일 축하 문구"],
              ["{x} 하나 지어줘", "봄을 주제로 {x} 써 줄 수 있어?",
               "친구에게 줄 {x} 아이디어 좀 주세요", "재미있는 {x} 몇 개 추천해 줘"]),
}

# 금융 단어가 들어 있지만 은행 업무가 아닌 질문 (어려운 예시)
HARD = {
    "은행나무": ["은행나무 열매에서 냄새가 나는 이유가 뭐야?", "은행 열매 구워 먹을 때 몇 개까지 괜찮아요?",
              "가을에 은행나무 단풍 보기 좋은 곳 추천해 줘", "은행나무가 오래 사는 이유가 궁금해요",
              "은행 열매 손질할 때 장갑을 껴야 하나요?", "은행나무 암나무와 수나무 구별하는 법 알려줘"],
    "도서관대출": ["도서관에서 책 대출 기간 연장하려면 어떻게 해요?", "학교 도서관 대출 권수가 몇 권까지야?",
               "전자책 대출 앱 사용법 알려주세요", "도서관 대출 연체되면 언제부터 다시 빌릴 수 있어?",
               "공공도서관 상호대차 서비스가 뭔가요?", "대출한 책을 잃어버리면 어떻게 해야 해?"],
    "창작소재": ["은행 강도가 나오는 영화 추천해 줘", "은행원이 주인공인 짧은 소설 줄거리 써 줄래?",
              "환전을 소재로 한 그림 아이디어 좀 주세요", "잔액이 0원이라는 말로 웃긴 짤 문구 만들어 줘",
              "대출 이자를 소재로 한 랩 가사 써 줘", "은행 창구를 배경으로 한 연극 대본 첫 장면 써 줄래?"],
    "단어뜻": ["'은행'이라는 단어의 어원이 뭐예요?", "대출이라는 말을 영어로 어떻게 표현해?",
             "이자라는 한자 뜻이 궁금해요", "환전이랑 환율 단어 차이를 국어 숙제로 설명해야 해요",
             "'잔액'과 '잔고'는 맞춤법상 어떻게 달라요?", "통장이라는 말은 언제부터 쓰였어?"],
    "게임": ["게임 속 은행에 아이템 맡기는 방법 알려줘", "보드게임에서 은행 역할은 뭘 해요?",
           "마인크래프트에서 경제 시스템 만들려면 어떻게 해?", "모노폴리에서 이기는 전략 알려주세요",
           "게임에서 대출 퀘스트 깨는 방법 알려 줘", "경영 시뮬레이션 게임에서 은행 건물 위치 어디가 좋아?"],
    "인간관계": ["은행에서 만난 직원한테 연락처 물어봐도 될까요?", "환전해 준 직원이랑 친해지고 싶은데 조언해 줘",
              "은행 다니는 친구 생일 선물 추천해 주세요", "대출 상담하다 알게 된 사람과 대화 주제 추천해 줘",
              "은행원 친구에게 보낼 응원 메시지 써 줘", "은행 입사 동기들과 갈 회식 장소 추천해 줄래?"],
}

# 말투와 길이를 다양하게 만드는 앞 문장 (빈 문자열 = 앞 문장 없음)
CONTEXTS = [
    "", "", "", "", "질문 하나만 할게요.", "궁금한 게 있어서요.", "요즘 고민이 있는데요,",
    "주말에 시간이 좀 생겨서 이것저것 해보려고 하는데요.", "친구랑 이야기하다가 갑자기 궁금해졌어요.",
    "아이가 물어봐서 대답을 해줘야 하는데 저도 잘 모르겠더라고요.",
    "회사 일 끝나고 집에 와서 쉬다가 문득 생각났는데,", "이번에 처음 해보는 거라 잘 몰라서 여쭤봐요.",
    "인터넷에서 찾아봐도 설명이 다 달라서 정확하게 알고 싶어서 질문드립니다.",
    "가족들이랑 저녁 먹으면서 이야기하다가 다들 의견이 달라서 확인하고 싶어요.",
]

# 끝맺음 (빈 문자열 = 덧붙이지 않음)
SUFFIXES = ["", "", " 궁금해요.", " 알려주시면 감사하겠습니다.", " 최대한 자세히 부탁해요.", " 짧게 답해 줘."]


def all_variants(core_by_group: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(묶음 ID, 질문) 조합을 중복 없이 모두 만든다: 앞 문장 × 핵심 질문 × 끝맺음."""
    variants = set()
    for group_id, cores in core_by_group.items():
        for core in cores:
            for context in set(CONTEXTS):
                for suffix in set(SUFFIXES):
                    variants.add((group_id, f"{context} {core}{suffix}".strip()))
    return sorted(variants)


def build_rows(rng: random.Random) -> list[dict]:
    general = {f"ood:general:{topic}:{t_idx}": [template.format(x=x) for x in items]
               for topic, (items, templates) in GENERAL.items()
               for t_idx, template in enumerate(templates)}
    hard = {f"ood:hard:{topic}:{q_idx}": [question]
            for topic, questions in HARD.items()
            for q_idx, question in enumerate(questions)}

    general_pool, hard_pool = all_variants(general), all_variants(hard)
    picked = (rng.sample(general_pool, min(N_GENERAL, len(general_pool)))
              + rng.sample(hard_pool, min(N_HARD, len(hard_pool))))

    return [{"qa_id": f"ood_{i:05d}", "source_id": group_id, "question": question,
             "qa_topic": LABEL, "consulting_topic": LABEL, "reference_answer": ""}
            for i, (group_id, question) in enumerate(picked)]


def main():
    rng = random.Random(SEED)
    df = pd.DataFrame(build_rows(rng))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    lengths = df["question"].str.len()
    print(f"업무 외 질문: {len(df):,}행 / 서로 다른 질문 {df['question'].nunique():,}종 / 템플릿 묶음 {df['source_id'].nunique()}개")
    print(f"일반 {df['source_id'].str.contains(':general:').sum():,} / 어려운 예시 {df['source_id'].str.contains(':hard:').sum():,}")
    print(f"길이: 평균 {lengths.mean():.1f}자, 최소 {lengths.min()}, 최대 {lengths.max()}")
    print("\n예시:")
    for q in df.sample(6, random_state=SEED)["question"]:
        print(f"  - {q}")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
