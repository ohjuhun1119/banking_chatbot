"""분류기 불러오기와 예측.

학습 담당 스크립트의 저장 형식에 맞춘다.
- BERT: scripts/training/fineture_bert.py → trainer.save_model (토크나이저·id2label 포함)
- 베이스라인: baseline_tfidf_lr.py → joblib으로 저장한 sklearn Pipeline
"""

import time
from dataclasses import dataclass
from pathlib import Path

from app import config


@dataclass
class Prediction:
    label: str
    confidence: float
    latency_s: float


class BertClassifier:
    def __init__(self, model_dir: Path, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_dir.exists():
            raise FileNotFoundError(f"BERT 모델 폴더 없음: {model_dir}")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        # 모델 폴더에 토크나이저가 있으면 사용, 없으면 사전학습 모델의 토크나이저 사용
        tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else config.BERT_TOKENIZER_FALLBACK
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length = max_length
        self.name = f"BERT ({model_dir.name}, {self.device})"
        self.n_labels = len(self.id2label)

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        with self.torch.inference_mode():
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[0]
            idx = int(probs.argmax())
            confidence = float(probs[idx])
        return Prediction(self.id2label[idx], round(confidence, 4), round(time.perf_counter() - start, 4))


class BaselineClassifier:
    def __init__(self, model_path: Path):
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(f"베이스라인 모델 파일 없음: {model_path}")
        self.model = joblib.load(model_path)
        self.name = f"TF-IDF + LR ({model_path.name})"
        self.n_labels = len(self.model.classes_)

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        probs = self.model.predict_proba([text])[0]
        idx = int(probs.argmax())
        return Prediction(str(self.model.classes_[idx]), round(float(probs[idx]), 4),
                          round(time.perf_counter() - start, 4))


def load_classifier():
    if config.CLASSIFIER_TYPE == "baseline":
        return BaselineClassifier(config.BASELINE_MODEL_PATH)
    if config.CLASSIFIER_TYPE == "bert":
        return BertClassifier(config.BERT_MODEL_DIR, config.BERT_MAX_LENGTH)
    raise ValueError(f"알 수 없는 CLASSIFIER_TYPE: {config.CLASSIFIER_TYPE}")
