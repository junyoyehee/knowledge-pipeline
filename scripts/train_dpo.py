"""(선택) SFT 이후 단계: DPO (Direct Preference Optimization)

같은 질문에 대한 '좋은 답(chosen)'과 '나쁜 답(rejected)' 쌍으로 학습해
모델이 나쁜 답 쪽으로 가지 않도록 만듭니다.

**지식을 주입하지 않습니다.** 이미 CPT/SFT로 넣은 지식을 잘못 꺼내 쓰는 것
(그럴듯한 환각, 형식 이탈)을 교정하는 용도입니다.
반드시 docs/preference_tuning.md를 먼저 읽으세요.

사용법:
    python scripts/generate_preference.py     # 데이터 먼저 생성
    python scripts/train_dpo.py [--config configs/config.yaml]

출력: outputs/dpo/ (LoRA 어댑터)
"""
import argparse

# unsloth는 transformers/trl보다 먼저 import되어야 함 (pref_common이 unsloth를 import)
from pref_common import (load_config, load_model, load_pref_dataset,
                         save_adapter, trainer_kwargs, warn_if_too_small)

# 구버전 unsloth는 DPO 최적화를 위해 명시적 패치가 필요했음. 신버전은 불필요.
try:
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
except ImportError:
    pass

from unsloth import is_bfloat16_supported

from trl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    stage_cfg = cfg["preference"]["dpo"]
    tcfg = stage_cfg["train"]

    model, tokenizer = load_model(cfg, stage_cfg, "dpo")

    dataset = load_pref_dataset(
        cfg["data"]["pref_dataset"],
        required=("prompt", "chosen", "rejected"),
        label="DPO")
    warn_if_too_small(len(dataset), "DPO")

    trainer = DPOTrainer(
        model=model,
        # PEFT 모델이므로 ref_model=None이면 어댑터를 끈 상태가 reference가 된다.
        # init_from이 어댑터 단계면 reference = 베이스 모델,
        # 병합모델/base면 reference = 그 모델. (docs/preference_tuning.md 참고)
        ref_model=None,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=stage_cfg["output_dir"],
            beta=tcfg["beta"],
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
        **trainer_kwargs(DPOTrainer, tokenizer),
    )

    stats = trainer.train()
    print(f"[OK] DPO 완료 — loss: {stats.training_loss:.4f}")
    print("[i] 로그의 rewards/accuracies가 0.5 근처에 머물면 학습 신호가 없는 것입니다. "
          "rejected가 충분히 '그럴듯한지' 확인하세요.")

    save_adapter(model, tokenizer, stage_cfg["output_dir"], "dpo")


if __name__ == "__main__":
    main()
