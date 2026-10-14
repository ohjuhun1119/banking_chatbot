"""생성 모델 비교: 같은 질문을 로컬·상용 LLM에 보내 답변과 측정값을 저장한다. (평가 기준 3장)

- 서비스와 같은 흐름: 분류기 예측 → (업무 외 확신 시 1차 차단) → 같은 프롬프트로 LLM 호출 → 고정 문구 판별
- 분류기는 한 번만 예측해 두 모델에 같은 분류 결과를 넘긴다. (비교 대상은 LLM뿐)
- 모델마다 첫 호출(모델 로딩)은 따로 재고, 비교에는 넣지 않는다.

실행 (프로젝트 루트에서, Ollama 실행 중, .env 설정 후):
    python scripts/evaluation/run_generation.py --split dev     # 프롬프트 확인용
    python scripts/evaluation/run_generation.py --split final   # 최종 비교 (1회)

입력: tests/generation_questions.csv (모델에는 question만 전달)
출력: results/generation/<split>_responses.csv, <split>_run_info.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd  # noqa: E402

from app import config  # noqa: E402
from app.classifier import load_classifier  # noqa: E402
from app.messages import MESSAGES, match_fixed_message  # noqa: E402
from app.services import chain  # noqa: E402

QUESTIONS_PATH = PROJECT_DIR / "tests" / "generation_questions.csv"
OUT_DIR = PROJECT_DIR / "results" / "generation"
WARMUP = {"question": "환전 신청 방법을 알려주세요.", "label": "환전문의"}


def message_text(msg) -> str:
    content = msg.content
    if isinstance(content, list):   # 일부 제공업체는 여러 조각으로 돌려준다
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content


def usage_of(msg) -> dict:
    usage = getattr(msg, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    return {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": details.get("reasoning")}


def build_runnable(model: str):
    llm = chain._local_llm() if model == "local" else chain._cloud_llm()
    return chain.build_prompt() | llm   # 문자열 변환 없이 받아 토큰 사용량을 읽는다


def model_id(model: str) -> str:
    if model == "local":
        return f"ollama/{config.OLLAMA_MODEL}"
    effort = f", reasoning_effort={config.CLOUD_REASONING_EFFORT}" if config.CLOUD_REASONING_EFFORT else ""
    return f"{config.CLOUD_PROVIDER}/{config.CLOUD_MODEL}{effort}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "final"], required=True)
    parser.add_argument("--models", nargs="+", default=["local", "cloud"], choices=["local", "cloud"])
    args = parser.parse_args(argv)

    questions = pd.read_csv(QUESTIONS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    questions = questions[questions["split"] == args.split].reset_index(drop=True)
    print(f"{args.split} 질문 {len(questions)}개 / 모델 {args.models}")

    classifier = load_classifier()
    preds = {row.id: classifier.predict(row.question) for row in questions.itertuples()}

    rows, run_info = [], {"split": args.split, "started_at": datetime.now().isoformat(timespec="seconds"),
                          "classifier": classifier.name, "oos_threshold": config.OUT_OF_SCOPE_BLOCK_THRESHOLD,
                          "llm_timeout_s": config.LLM_TIMEOUT_S, "max_tokens": config.LLM_MAX_TOKENS,
                          "temperature": config.LLM_TEMPERATURE, "models": {}}
    executor = ThreadPoolExecutor(max_workers=1)

    for model in args.models:
        runnable = build_runnable(model)
        start = time.perf_counter()
        runnable.invoke(WARMUP)                       # 첫 호출: 모델 로딩·연결 준비
        warmup_s = round(time.perf_counter() - start, 3)
        run_info["models"][model] = {"id": model_id(model), "warmup_s": warmup_s}
        print(f"\n[{model}] {model_id(model)} 준비 완료 (첫 호출 {warmup_s}s)")

        for row in questions.itertuples():
            pred = preds[row.id]
            record = {"id": row.id, "group": row.group, "question": row.question, "model": model,
                      "label": pred.label, "confidence": pred.confidence, "route": None, "answer": None,
                      "generate_s": None, "input_tokens": None, "output_tokens": None,
                      "reasoning_tokens": None, "error": None}

            if pred.label == config.OUT_OF_SCOPE_LABEL and pred.confidence >= config.OUT_OF_SCOPE_BLOCK_THRESHOLD:
                record.update(route="blocked_by_classifier", answer=MESSAGES["out_of_scope"])
            else:
                start = time.perf_counter()
                try:
                    future = executor.submit(runnable.invoke, {"question": row.question, "label": pred.label})
                    msg = future.result(timeout=config.LLM_TIMEOUT_S)
                    record["generate_s"] = round(time.perf_counter() - start, 3)
                    record.update(usage_of(msg))
                    answer = message_text(msg).strip()
                    matched = match_fixed_message(answer)
                    if matched == "out_of_scope":
                        record.update(route="blocked_by_llm", answer=MESSAGES["out_of_scope"])
                    elif matched == "unclear":
                        record.update(route="unclear", answer=MESSAGES["unclear"])
                    else:
                        record.update(route="answered", answer=answer)
                except FutureTimeout:
                    record.update(route="error", answer=MESSAGES["error"], error="llm_timeout",
                                  generate_s=round(time.perf_counter() - start, 3))
                except Exception as e:
                    record.update(route="error", answer=MESSAGES["error"], error=f"llm_error: {type(e).__name__}",
                                  generate_s=round(time.perf_counter() - start, 3))
            rows.append(record)
            gs = record['generate_s']
            print(f"  {row.id} {record['route']:<22} {'-' if gs is None else f'{gs:.2f}s':>7}  {record['answer'][:40]}")

    executor.shutdown(wait=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.split}_responses.csv", index=False, encoding="utf-8-sig")
    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (OUT_DIR / f"{args.split}_run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_DIR / f'{args.split}_responses.csv'}")


if __name__ == "__main__":
    main()
