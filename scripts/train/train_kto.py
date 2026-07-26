"""(선택) SFT 이후 단계: KTO (Kahneman-Tversky Optimization)

DPO와 달리 **쌍(pair)이 필요 없습니다.** 응답 하나하나에 "좋음/나쁨" 이진 라벨만
있으면 학습됩니다. 서비스 로그의 엄지척/엄지다운처럼 자연스럽게 쌓이는 신호를
그대로 쓸 수 있어, 실사용 데이터가 있는 환경에서 가장 현실적입니다.

**지식을 주입하지 않습니다.** 반드시 docs/preference_tuning.md를 먼저 읽으세요.

사용법:
    python -m scripts.generate.generate_preference      # 선호 쌍에서 KTO 데이터도 함께 생성됨
    python -m scripts.train.train_kto [--config configs/config.yaml]

출력: outputs/kto/ (LoRA 어댑터)
"""
import argparse

# unsloth는 transformers/trl보다 먼저 import되어야 함
from scripts.lib.pref_common import (load_config, load_model, load_pref_dataset,
                         save_adapter, trainer_kwargs, warn_if_too_small)

from unsloth import is_bfloat16_supported

from trl import KTOConfig, KTOTrainer


def check_balance(dataset, desirable_weight: float, undesirable_weight: float):
    """좋음/나쁨 비율을 확인하고 권장 범위를 벗어나면 경고.

    KTO 논문/TRL 권장: (desirable_weight * n_pos) / (undesirable_weight * n_neg)
    가 대략 1.0 ~ 1.33 사이여야 안정적으로 학습됩니다.
    """
    labels = dataset["label"]
    n_pos = sum(1 for x in labels if x)
    n_neg = len(labels) - n_pos
    print(f"[i] 좋음 {n_pos}개 / 나쁨 {n_neg}개")

    if n_pos == 0 or n_neg == 0:
        raise SystemExit("[!] 좋음 또는 나쁨 샘플이 하나도 없습니다. "
                         "KTO는 양쪽이 모두 필요합니다.")

    ratio = (desirable_weight * n_pos) / (undesirable_weight * n_neg)
    print(f"[i] 유효 비율 = {ratio:.2f} (권장 1.0 ~ 1.33)")
    if not 1.0 <= ratio <= 1.33:
        suggested = (n_neg * 1.15) / n_pos
        print(f"[!] 권장 범위를 벗어났습니다. config의 "
              f"preference.kto.train.desirable_weight를 약 {suggested:.2f}로 "
              f"조정하는 것을 고려하세요 (undesirable_weight=1.0 기준).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    stage_cfg = cfg["preference"]["kto"]
    tcfg = stage_cfg["train"]

    model, tokenizer = load_model(cfg, stage_cfg, "kto")

    dataset = load_pref_dataset(
        cfg["data"]["kto_dataset"],
        required=("prompt", "completion", "label"),
        label="KTO")
    warn_if_too_small(len(dataset), "KTO", minimum=400)  # 쌍이 분해되므로 2배
    check_balance(dataset, tcfg["desirable_weight"], tcfg["undesirable_weight"])

    trainer = KTOTrainer(
        model=model,
        ref_model=None,   # PEFT 모델이므로 어댑터를 끈 상태가 reference
        train_dataset=dataset,
        args=KTOConfig(
            output_dir=stage_cfg["output_dir"],
            beta=tcfg["beta"],
            desirable_weight=tcfg["desirable_weight"],
            undesirable_weight=tcfg["undesirable_weight"],
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
        **trainer_kwargs(KTOTrainer, tokenizer),
    )

    stats = trainer.train()
    print(f"[OK] KTO 완료 — loss: {stats.training_loss:.4f}")

    save_adapter(model, tokenizer, stage_cfg["output_dir"], "kto")


if __name__ == "__main__":
    main()
