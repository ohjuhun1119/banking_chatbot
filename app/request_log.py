"""요청마다 실행 기록을 JSONL로 남긴다. (평가 기준 ④: 전체 응답 시간, 실패 사례 분류의 근거)

기록 파일은 질문 원문을 포함하므로 저장소에 올리지 않는다. (.gitignore: /logs/)
"""

import json
import threading
from datetime import datetime, timezone

from app import config

_lock = threading.Lock()


def write_log(endpoint: str, question: str, result: dict):
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": endpoint,
        "question": question,
        **{k: result.get(k) for k in ("route", "label", "confidence", "model", "latency_s", "error")},
    }
    try:
        config.REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, config.REQUEST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[경고] 실행 기록 저장 실패: {e}")
