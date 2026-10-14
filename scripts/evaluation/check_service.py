"""실행 중인 서비스에 실제 요청을 보내 통합 동작과 전체 응답 시간을 확인한다. (평가 기준 4장)

- 시나리오: 정상 답변, 업무 외 1차 차단, 의도 불명, 입력 오류(422) 등이 기대대로 처리되는지
- 응답 시간: 은행 업무 질문을 여러 번 보내 전체 응답 시간(요청 접수~응답)의 중앙값·p95
- 업무 외 오차단 확인: 확인용 질문 중 은행 업무 20개가 막히지 않고 답변되는지

실행 (서버를 켠 상태에서, 다른 터미널):
    uvicorn app.main:app
    python scripts/evaluation/check_service.py --model cloud

출력: results/service/<model>_scenarios.csv, <model>_latency.csv, <model>_summary.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROBE_PATH = PROJECT_DIR / "tests" / "probe_questions.csv"
OUT_DIR = PROJECT_DIR / "results" / "service"

# (이름, 요청 본문, 기대 상태 코드, 기대 route 목록)
SCENARIOS = [
    ("은행 업무 질문", {"question": "적금 만기되면 어떻게 처리돼요?"}, 200, ["answered"]),
    ("업무 외 (분류기 차단 예상)", {"question": "오늘 저녁 메뉴 좀 골라줘"}, 200, ["blocked_by_classifier", "blocked_by_llm"]),
    ("업무 외 (금융 단어 포함)", {"question": "대출이라는 말로 삼행시 지어줘"}, 200, ["blocked_by_classifier", "blocked_by_llm"]),
    ("의도 불명", {"question": "그거요"}, 200, ["unclear"]),
    ("빈 질문", {"question": "   "}, 422, None),
    ("길이 초과", {"question": "가" * 501}, 422, None),
    ("허용되지 않은 모델", {"question": "환전 문의", "model": "gpt"}, 422, None),
]


def post_chat(client, body):
    start = time.perf_counter()
    resp = client.post("/chat", json=body)
    elapsed = time.perf_counter() - start
    return resp, round(elapsed, 3)


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--repeat", type=int, default=2, help="응답 시간 측정 시 질문당 반복 횟수")
    args = parser.parse_args(argv)
    client = client or httpx.Client(base_url=args.base_url, timeout=60)

    health = client.get("/health").json()
    print("상태:", json.dumps(health, ensure_ascii=False))
    if not health["classifier"]["loaded"]:
        raise SystemExit("분류기가 로드되지 않았습니다. 서버 터미널의 시작 메시지를 확인하세요.")

    # 1) 시나리오
    rows = []
    for name, body, want_status, want_routes in SCENARIOS:
        if "model" not in body:
            body = {**body, "model": args.model}
        resp, elapsed = post_chat(client, body)
        data = resp.json() if resp.status_code == 200 else {}
        route = data.get("route")
        ok = resp.status_code == want_status and (want_routes is None or route in want_routes)
        rows.append({"시나리오": name, "기대": want_routes or want_status, "상태 코드": resp.status_code,
                     "route": route, "label": data.get("label"), "통과": ok, "응답 시간(s)": elapsed})
        print(f"  {'✅' if ok else '❌'} {name}: {resp.status_code} {route or ''}")
    scenarios = pd.DataFrame(rows)

    # 2) 은행 업무 질문 오차단 확인 + 응답 시간 (확인용 질문 중 은행 업무 20개)
    probe = pd.read_csv(PROBE_PATH, encoding="utf-8-sig")
    bank = probe[probe["expected"] == "은행 업무"]["question"].tolist()
    client.post("/chat", json={"question": bank[0], "model": args.model})   # 첫 요청 제외
    lat_rows = []
    for _ in range(args.repeat):
        for q in bank:
            resp, elapsed = post_chat(client, {"question": q, "model": args.model})
            d = resp.json()
            lat_rows.append({"question": q, "route": d["route"], "label": d["label"], "confidence": d["confidence"],
                             "server_total_s": d["latency_s"]["total"], "client_total_s": elapsed})
    latency = pd.DataFrame(lat_rows)
    answered = latency[latency["route"] == "answered"]
    times = sorted(answered["client_total_s"])
    first_pass = latency.iloc[:len(bank)]

    summary = {
        "model": args.model,
        "health": health,
        "scenarios_passed": f"{int(scenarios['통과'].sum())}/{len(scenarios)}",
        "bank_questions": len(bank),
        "bank_blocked": int(first_pass["route"].isin(["blocked_by_classifier", "blocked_by_llm"]).sum()),
        "bank_unclear": int((first_pass["route"] == "unclear").sum()),
        "bank_answered": int((first_pass["route"] == "answered").sum()),
        "requests_timed": len(latency),
        "total_latency_median_s": round(statistics.median(times), 3) if times else None,
        "total_latency_p95_s": round(times[int(len(times) * 0.95) - 1], 3) if times else None,
        "total_latency_max_s": round(max(times), 3) if times else None,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(OUT_DIR / f"{args.model}_scenarios.csv", index=False, encoding="utf-8-sig")
    latency.to_csv(OUT_DIR / f"{args.model}_latency.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / f"{args.model}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n시나리오 통과: {summary['scenarios_passed']}")
    print(f"은행 업무 {summary['bank_questions']}개: 답변 {summary['bank_answered']} / 의도 불명 {summary['bank_unclear']}"
          f" / 업무 외로 차단 {summary['bank_blocked']}")
    print(f"전체 응답 시간 ({len(times)}건): 중앙값 {summary['total_latency_median_s']}s,"
          f" p95 {summary['total_latency_p95_s']}s, 최대 {summary['total_latency_max_s']}s")
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
