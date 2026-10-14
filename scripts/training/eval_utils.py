"""분류 모델 공통 평가 도구.

TF-IDF 베이스라인과 klue/bert-base가 같은 함수로 평가되도록 두 학습 스크립트에서 함께 사용한다.
"""

import json
import platform
import statistics
import sys
import threading
import time
import warnings
from pathlib import Path

import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

TEXT_COL, LABEL_COL, GROUP_COL = "question", "qa_topic", "source_id"
OUT_OF_SCOPE = "업무 외"
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"분할 파일 없음: {path}\n데이터 준비 스크립트를 먼저 실행하세요.")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError(f"분할 파일이 비어 있습니다: {path}")
    return df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """라벨 구성을 확인하고, train과 validation에 같은 source_id가 있으면 멈춘다."""
    unknown = set(val_df[LABEL_COL]) - set(train_df[LABEL_COL])
    if unknown:
        raise ValueError(f"validation에만 있는 라벨: {unknown}")
    overlap = len(set(train_df[GROUP_COL]) & set(val_df[GROUP_COL]))
    print(f"train {len(train_df):,}건 / validation {len(val_df):,}건 / 라벨 {train_df[LABEL_COL].nunique()}종")
    if overlap:
        raise ValueError(f"train과 validation에 같은 source_id가 {overlap:,}개 있습니다. split.py로 다시 분할하세요.")
    print("train-validation 겹치는 source_id: 0")


def compute_metrics(y_true, y_pred) -> dict:
    """평가 기준 2.2의 지표: 11개 주제·은행 10개 주제 Macro F1, Accuracy, 업무 외 관련 지표."""
    y_true, y_pred = list(y_true), list(y_pred)
    bank_labels = sorted(set(y_true) - {OUT_OF_SCOPE})
    true_oos = [t == OUT_OF_SCOPE for t in y_true]
    pred_oos = [p == OUT_OF_SCOPE for p in y_pred]
    n_oos = sum(true_oos)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "macro_f1_bank": round(float(f1_score(y_true, y_pred, labels=bank_labels, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # 업무 외 질문 중 업무 외로 걸러낸 비율
        "oos_recall": round(sum(t and p for t, p in zip(true_oos, pred_oos)) / n_oos, 4) if n_oos else None,
        # 은행 질문을 업무 외로 잘못 막은 수 (서비스에서 더 치명적)
        "bank_blocked_as_oos": int(sum((not t) and p for t, p in zip(true_oos, pred_oos))),
        # 업무 외 질문을 은행 주제로 잘못 통과시킨 수
        "oos_passed_as_bank": int(sum(t and (not p) for t, p in zip(true_oos, pred_oos))),
        "n": len(y_true),
    }


def save_evaluation(out_dir: Path, df: pd.DataFrame, y_pred, labels: list[str], title: str,
                    split: str = "validation") -> dict:
    """Macro F1·Accuracy, 주제별 P·R·F1, 혼동행렬(CSV·PNG), 오분류 사례를 저장하고 지표를 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = df[LABEL_COL].tolist()
    y_pred = list(y_pred)
    metrics = compute_metrics(y_true, y_pred)

    report = pd.DataFrame(classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0)).T.round(4)
    report.to_csv(out_dir / f"{split}_report.csv", encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"정답:{l}" for l in labels], columns=[f"예측:{l}" for l in labels]) \
        .to_csv(out_dir / f"{split}_confusion_matrix.csv", encoding="utf-8-sig")
    _save_confusion_png(cm, labels, out_dir / f"{split}_confusion_matrix.png", f"{title} - {split}")

    errors = df.assign(pred=y_pred)
    errors = errors[errors[LABEL_COL] != errors["pred"]]
    # 질문 원문 포함 → 커밋 금지 (.gitignore: /results/**/*_errors.csv)
    errors[[GROUP_COL, LABEL_COL, "pred", TEXT_COL]].to_csv(
        out_dir / f"{split}_errors.csv", index=False, encoding="utf-8-sig")

    pairs = errors.groupby([LABEL_COL, "pred"]).size().sort_values(ascending=False).head(10)
    metrics["n_errors"] = int(len(errors))
    metrics["top_confusions"] = [{"label": a, "pred": b, "count": int(c)} for (a, b), c in pairs.items()]

    print(f"\n[{title}] {split} Macro F1 {metrics['macro_f1']:.4f} (은행 10종 {metrics['macro_f1_bank']:.4f})"
          f" / Accuracy {metrics['accuracy']:.4f}")
    print(f"업무 외 Recall {metrics['oos_recall']} / 은행→업무 외 오차단 {metrics['bank_blocked_as_oos']}건"
          f" / 업무 외→은행 통과 {metrics['oos_passed_as_bank']}건")
    print(report.to_string())
    print("자주 헷갈린 주제 쌍:")
    for c in metrics["top_confusions"][:5]:
        print(f"  {c['label']} → {c['pred']}: {c['count']}건")
    return metrics


def _save_confusion_png(cm, labels, path: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        warnings.filterwarnings("ignore", message="Glyph .* missing")

        for font in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:  # Windows / Mac 한글 폰트
            if any(f.name == font for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = font
                break
        plt.rcParams["axes.unicode_minus"] = False

        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, label="정답 대비 비율")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[참고] 혼동행렬 이미지 저장 생략: {e}")


def measure_latency(predict_one, texts: list[str], n: int = 1000, warmup: int = 10) -> dict:
    """배치 크기 1로 질문 한 건씩 예측하는 시간(초). 전처리부터 예측 완료까지 포함."""
    sample = list(texts[:n])
    for t in sample[:warmup]:
        predict_one(t)
    times = []
    for t in sample:
        start = time.perf_counter()
        predict_one(t)
        times.append(time.perf_counter() - start)
    ts = sorted(times)
    return {"n": len(times), "mean_s": round(statistics.mean(times), 5),
            "median_s": round(statistics.median(times), 5), "p95_s": round(ts[int(len(ts) * 0.95) - 1], 5)}


class PeakMemory:
    """with 블록 동안 현재 프로세스의 최대 RAM(MB)을 측정."""

    def __init__(self, interval: float = 0.1):
        self.interval, self.peak_mb = interval, 0.0
        self._proc, self._stop = psutil.Process(), threading.Event()

    def _watch(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 1024**2)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_mb = self._proc.memory_info().rss / 1024**2
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = round(max(self.peak_mb, self._proc.memory_info().rss / 1024**2), 1)


def environment_info() -> dict:
    import sklearn
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
    }


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
