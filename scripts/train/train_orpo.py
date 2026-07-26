"""(선택) SFT 대체 단계: ORPO (Odds Ratio Preference Optimization)

SFT와 선호 학습을 **한 단계로 합칩니다.** chosen에 대한 일반적인 SFT 손실에
'rejected의 odds를 낮추는' 항을 더한 형태라, reference 모델이 필요 없습니다.
→ VRAM이 덜 들고 파이프라인이 짧아집니다.

**SFT를 대체하므로 train_sft.py와 함께 쓰지 마세요.** 보통 CPT 어댑터에서
바로 시작합니다 (config의 init_from 기본값이 "cpt"인 이유).

반드시 docs/preference_tuning.md를 먼저 읽으세요.

사용법:
    python -m scripts.generate.generate_preference
    python -m scripts.train.train_orpo [--config configs/config.yaml]

출력: outputs/orpo/ (LoRA 어댑터)
"""
import argparse

# unsloth는 transformers/trl보다 먼저 import되어야 함
from scripts.lib.pref_common import (load_config, load_model, load_pref_dataset,
                         save_adapter, trainer_kwargs, warn_if_too_small)

from unsloth import is_bfloat16_supported

from trl import ORPOConfig, ORPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    stage_cfg = cfg["preference"]["orpo"]
    tcfg = stage_cfg["train"]

    model, tokenizer = load_model(cfg, stage_cfg, "orpo")

    dataset = load_pref_dataset(
        cfg["data"]["pref_dataset"],
        required=("prompt", "chosen", "rejected"),
        label="ORPO")
    warn_if_too_small(len(dataset), "ORPO")

    trainer = ORPOTrainer(
        model=model,          # reference 모델 없음 — ORPO의 핵심 장점
        train_dataset=dataset,
        args=ORPOConfig(
            output_dir=stage_cfg["output_dir"],
            beta=tcfg["beta"],          # odds ratio 항의 가중치 (논문의 lambda)
            max_length=tcfg["max_length"],
            max_prompt_length=tcfg["max_prompt_length"],
            num_train_epochs=tcfg["num_epochs"],
            per_device_train_batch_size=tcfg["per_device_batch_size"],
            gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
            learning_rate=float(tcfg["learning_rate"]),
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
        **trainer_kwargs(ORPOTrainer, tokenizer),
    )

    stats = trainer.train()
    print(f"[OK] ORPO 완료 — loss: {stats.training_loss:.4f}")
    print("[i] ORPO는 SFT를 대체합니다. export_model.py의 source_stage를 "
          "\"orpo\"로 바꿔야 이 어댑터가 병합됩니다.")

    save_adapter(model, tokenizer, stage_cfg["output_dir"], "orpo")


if __name__ == "__main__":
    main()
