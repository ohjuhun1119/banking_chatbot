"""사전학습 모델 파인튜닝과 validation 평가. 베이스라인과 같은 분할·같은 지표로 비교한다.

- 학습은 train만 사용하고, epoch마다 validation Macro F1로 가장 좋은 시점의 모델을 고른다.
- test는 사용하지 않는다. (분류기 선정 후 별도로 1회 평가)
- 먼저 --smoke-test로 짧게 실행해 실행 가능 여부와 본 학습 예상 시간을 확인한다.

실행:
    python scripts/training/train_bert.py --smoke-test                                   # klue/bert-base 시험 실행
    python scripts/training/train_bert.py                                                # klue/bert-base 본 학습
    python scripts/training/train_bert.py --model-name kakaobank/kf-deberta-base         # 금융 특화 모델 본 학습
    python scripts/training/train_bert.py --model-name kakaobank/kf-deberta-base --device cpu --smoke-test  # CPU 속도 확인

출력 (실행 이름은 모델 이름에서 자동 생성, 예: klue-bert-base, kakaobank-kf-deberta-base):
    models/<실행 이름>/best/            학습된 모델·토크나이저·라벨 매핑
    results/<실행 이름>/summary.json    지표·시간·자원 (시험 실행은 summary_smoke.json)

모든 모델을 학습한 뒤 compare_models.py로 비교표를 만든다.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac에서 지원 안 되는 연산은 CPU로 대신 처리

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from eval_utils import (LABEL_COL, MODEL_DIR, RESULT_DIR, SEED, TEXT_COL, PeakMemory, check_split,
                        environment_info, load_split, measure_latency, save_evaluation, save_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="klue/bert-base")
    parser.add_argument("--run-name", default=None, help="결과 폴더 이름. 기본값은 모델 이름에서 '/'를 '-'로 바꾼 것")
    parser.add_argument("--smoke-test", action="store_true", help="train 1,000건으로 30스텝만 학습")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128,
                        help="질문 토큰 길이 95%가 약 65, 128 초과는 0.1% 미만이라 128로 충분")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--keep-checkpoints", action="store_true", help="학습 중간 저장본을 지우지 않음")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="학습 장치. auto는 cuda → mps → cpu 순서로 자동 선택")
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id, max_length):
        self.enc = tokenizer(df[TEXT_COL].tolist(), truncation=True, max_length=max_length)
        self.labels = [label2id[l] for l in df[LABEL_COL]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.labels[i]}


def compute_metrics(eval_pred):
    logits, y = eval_pred
    pred = logits.argmax(axis=-1)
    return {"macro_f1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred)}


def make_predict_one(model, tokenizer, device, max_length):
    model = model.to(device).eval()

    @torch.inference_mode()
    def predict_one(text):
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc).logits.argmax(-1).item()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        return out

    return predict_one


@torch.inference_mode()
def predict_labels(model, tokenizer, texts, device, max_length, batch_size=64):
    model = model.to(device).eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(device)
        preds += model(**enc).logits.argmax(-1).tolist()
    return [model.config.id2label[i] for i in preds]


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device() if args.device == "auto" else args.device
    use_fp16 = device == "cuda"
    run_name = args.run_name or args.model_name.replace("/", "-")
    out_model = MODEL_DIR / run_name
    out_result = RESULT_DIR / run_name
    out_result.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__} / transformers {transformers.__version__} / 장치: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    elif device == "cpu":
        print("[주의] GPU가 없어 CPU로 학습합니다. 매우 오래 걸릴 수 있습니다.")

    # 1) 데이터
    train_df, val_df = load_split("train"), load_split("validation")
    check_split(train_df, val_df)
    labels = sorted(train_df[LABEL_COL].unique())  # 베이스라인과 같은 순서
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lengths = pd.Series([len(x) for x in tokenizer(train_df[TEXT_COL].tolist())["input_ids"]])
    print(f"토큰 길이: 중앙값 {lengths.median():.0f}, 95% {lengths.quantile(0.95):.0f}, 최대 {lengths.max()}"
          f" / max_length({args.max_length}) 초과 {(lengths > args.max_length).mean():.2%}")

    if args.smoke_test:
        train_part = train_df.sample(n=min(1000, len(train_df)), random_state=SEED)
        val_part = val_df.sample(n=min(300, len(val_df)), random_state=SEED)
        print(f"[시험 실행] train {len(train_part)}건 / validation {len(val_part)}건")
    else:
        train_part, val_part = train_df, val_df
    train_ds = QuestionDataset(train_part, tokenizer, label2id, args.max_length)
    val_ds = QuestionDataset(val_part, tokenizer, label2id, args.max_length)

    # 2) 학습
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

    total_steps = 30 if args.smoke_test else -(-len(train_ds) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)  # transformers 버전과 무관하게 스텝 수로 지정
    strategy = "steps" if args.smoke_test else "epoch"

    training_args = TrainingArguments(
        output_dir=str(out_model / "checkpoints"),
        num_train_epochs=1 if args.smoke_test else args.epochs,
        max_steps=30 if args.smoke_test else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy=strategy,
        save_strategy=strategy,
        eval_steps=30,
        save_steps=30,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_fp16,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),   # Trainer가 장치를 스스로 고르므로 CPU 지정 시 명시
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with PeakMemory() as mem:
        start = time.perf_counter()
        trainer.train()
        train_s = time.perf_counter() - start
    peak_vram = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None
    print(f"\n학습 {train_s:.1f}s / 최대 RAM {mem.peak_mb}MB / 최대 VRAM {peak_vram}MB")

    if args.smoke_test:
        per_step = train_s / trainer.state.global_step
        full_steps = -(-len(train_df) // args.batch_size) * args.epochs
        print(f"[예상] 본 학습 {full_steps}스텝 × {per_step:.2f}s ≈ {full_steps * per_step / 60:.0f}분 (평가 시간 제외)")

    log = pd.DataFrame(trainer.state.log_history)
    if "eval_macro_f1" in log:
        eval_log = log.dropna(subset=["eval_macro_f1"])[["epoch", "step", "eval_loss", "eval_macro_f1", "eval_accuracy"]]
        eval_log.to_csv(out_result / "training_log.csv", index=False, encoding="utf-8-sig")
        print(eval_log.round(4).to_string(index=False))

    # 3) validation 상세 평가
    pred_ids = trainer.predict(val_ds).predictions.argmax(axis=-1)
    val_pred = [id2label[i] for i in pred_ids]
    val_metrics = save_evaluation(out_result, val_part, val_pred, labels, args.model_name)

    # 4) 저장
    best_dir = out_model / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    train_config = {"model_name": args.model_name, "max_length": args.max_length, "batch_size": args.batch_size,
                    "learning_rate": args.lr, "epochs": args.epochs, "warmup_ratio": args.warmup_ratio,
                    "warmup_steps": warmup_steps, "weight_decay": args.weight_decay, "fp16": use_fp16,
                    "seed": SEED, "smoke_test": args.smoke_test, "device": device,
                    "best_checkpoint": trainer.state.best_model_checkpoint}
    save_json(train_config, best_dir / "train_config.json")
    if not args.keep_checkpoints:
        shutil.rmtree(out_model / "checkpoints", ignore_errors=True)  # 중간 저장본 정리 (용량 절약)

    # 5) 추론 시간 (배치 1): 학습 장치와 CPU 모두
    best = trainer.model
    texts = val_df[TEXT_COL].tolist()
    latency = {}
    if device != "cpu":
        latency[device] = measure_latency(make_predict_one(best, tokenizer, device, args.max_length), texts, n=1000)
    latency["cpu"] = measure_latency(make_predict_one(best, tokenizer, "cpu", args.max_length), texts, n=200)
    print("추론 시간(배치 1):", json.dumps(latency, ensure_ascii=False))

    # 6) 다시 불러와 같은 예측이 나오는지 확인
    start = time.perf_counter()
    re_tok = AutoTokenizer.from_pretrained(str(best_dir))
    re_model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    load_s = time.perf_counter() - start
    re_pred = predict_labels(re_model, re_tok, val_part[TEXT_COL].tolist(), device, args.max_length)
    same_ratio = float(np.mean(np.array(re_pred) == np.array(val_pred)))
    model_mb = round(sum(f.stat().st_size for f in best_dir.glob("*") if f.is_file()) / 1024**2, 1)
    print(f"모델 로딩 {load_s:.2f}s / 크기 {model_mb}MB / 재로딩 후 예측 일치율 {same_ratio:.2%}")

    summary = {
        "model": args.model_name,
        "run_name": run_name,
        "config": train_config,
        "data": {"train": len(train_part), "validation": len(val_part), "n_labels": len(labels)},
        "validation": val_metrics,
        "train_time_s": round(train_s, 1),
        "peak_ram_mb_train": mem.peak_mb,
        "peak_vram_mb_train": peak_vram,
        "inference_latency_batch1": latency,
        "model_load_time_s": round(load_s, 2),
        "model_size_mb": model_mb,
        "reload_prediction_identical_ratio": same_ratio,
        "environment": {**environment_info(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": device},
    }
    save_json(summary, out_result / ("summary_smoke.json" if args.smoke_test else "summary.json"))

    print(f"\n저장: {out_model / 'best'}, {out_result}")
    if not args.smoke_test:
        print("모든 모델 학습이 끝나면 python scripts/training/compare_models.py 로 비교하세요.")

if __name__ == "__main__":
    main()
