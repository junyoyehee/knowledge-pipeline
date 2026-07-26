"""
2단계: CPT (Continued Pretraining) — 지식 주입
- 도메인 원문 텍스트를 이어서 사전학습하여 모델에 새 지식을 축적
- unsloth 권장 CPT 설정 사용:
  * embed_tokens / lm_head 포함 학습 (새 지식 흡수에 핵심)
  * embedding_learning_rate를 본체 lr의 1/5~1/10로 낮게 설정
  * UnslothTrainer 사용 (embedding_learning_rate 지원)

사용법:
    python -m scripts.train.train_cpt [--config configs/config.yaml]

출력: outputs/cpt/ (LoRA 어댑터)
"""
import argparse
import os

# unsloth는 transformers/trl보다 먼저 import해야 최적화가 적용됨
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from unsloth import is_bfloat16_supported

from datasets import load_dataset

from scripts.lib.common import load_config, add_report_arg, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    add_report_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, ccfg = cfg["model"], cfg["cpt"]
    tcfg, lcfg = ccfg["train"], ccfg["lora"]

    # ---------- 모델 로드 ----------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg["dtype"],
        load_in_4bit=mcfg["load_in_4bit"],
    )

    # ---------- LoRA 어댑터 (embed_tokens/lm_head 포함) ----------
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        target_modules=lcfg["target_modules"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # VRAM 절약 (긴 시퀀스 필수)
        random_state=tcfg["seed"],
        use_rslora=True,  # rank가 클 때 안정적
    )

    # ---------- 데이터셋 ----------
    dataset = load_dataset("json", data_files=cfg["data"]["cpt_dataset"],
                           split="train")

    eos = tokenizer.eos_token

    def add_eos(examples):
        return {"text": [t + eos for t in examples["text"]]}

    # meta 컬럼은 학습에 쓰지 않으므로 제거 (text만 남김)
    dataset = dataset.map(add_eos, batched=True,
                          remove_columns=[c for c in dataset.column_names
                                          if c != "text"])
    print(f"[i] CPT 학습 샘플 수: {len(dataset)}")

    # ---------- 학습 ----------
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=mcfg["max_seq_length"],
        dataset_num_proc=2,
        packing=True,  # 짧은 청크를 묶어 시퀀스 활용도 향상
        args=UnslothTrainingArguments(
            output_dir=ccfg["output_dir"],
            num_train_epochs=tcfg["num_epochs"],
            per_device_train_batch_size=tcfg["per_device_batch_size"],
            gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
            learning_rate=float(tcfg["learning_rate"]),
            embedding_learning_rate=float(tcfg["embedding_learning_rate"]),
            warmup_ratio=tcfg["warmup_ratio"],
            lr_scheduler_type=tcfg["lr_scheduler_type"],
            weight_decay=tcfg["weight_decay"],
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=tcfg["logging_steps"],
            save_steps=tcfg["save_steps"],
            save_total_limit=2,
            seed=tcfg["seed"],
            report_to="none",
        ),
    )

    stats = trainer.train()
    print(f"[OK] CPT 완료 — loss: {stats.training_loss:.4f}")

    # ---------- 저장 ----------
    final_dir = os.path.join(ccfg["output_dir"], "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] CPT 어댑터 저장: {final_dir}")

    write_report(args.report_json, {
        "task": "train", "stage": "cpt", "status": "ok",
        "n_samples": len(dataset),
        "adapter_dir": final_dir,
        "metrics": {"train_loss": float(stats.training_loss),
                    **{k: v for k, v in (getattr(stats, "metrics", None) or {}).items()}},
    })


if __name__ == "__main__":
    main()
