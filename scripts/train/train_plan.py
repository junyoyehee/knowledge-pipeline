"""
(선택) 계획수립(planning) 학습 단계 — 목표 → 단계별 계획 능력 주입
- 지식 SFT 어댑터(기본) 위에 '목표를 받아 실행 가능한 단계별 계획을 세우는' 능력을
  전용으로 학습하는 별도 단계
- 응답(계획)에만 loss 계산 (목표/질문 부분은 마스킹)

시작 지점(config의 plan.init_from):
    "sft"(기본) → 지식 SFT 어댑터를 이어받아 계획 능력만 추가
    "base"      → 베이스 모델에서 계획 능력만 학습 (지식과 분리)
    경로/기타   → 그 모델 위에 새 LoRA 부착

사용법:
    python -m scripts.data.prepare_data            # plans_*.jsonl → plan_dataset.jsonl
    python -m scripts.generate.generate_plans          # (선택) LLM으로 자동 생성
    python -m scripts.train.train_plan [--config configs/config.yaml]

출력: outputs/plan/ (LoRA 어댑터)
"""
import argparse
import os

# unsloth는 transformers/trl보다 먼저 import되어야 함 (pref_common이 unsloth를 import)
from scripts.lib.pref_common import load_config, load_stage_model, save_adapter

from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from scripts.lib.common import TEMPLATE_PARTS, add_report_arg, write_report
from scripts.lib.report import make_progress_callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    add_report_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, stage_cfg = cfg["model"], cfg["plan"]
    tcfg = stage_cfg["train"]

    plan_path = cfg["data"].get("plan_dataset")
    if not plan_path or not os.path.exists(plan_path):
        raise SystemExit(
            f"계획수립 데이터셋이 없습니다: {plan_path}\n"
            "plans_*.jsonl을 넣고 prepare_data.py를 돌리거나, "
            "generate_plans.py로 먼저 생성하세요.")

    model, tokenizer = load_stage_model(cfg, stage_cfg, "plan")
    template_name = mcfg["chat_template"]

    # ---------- 데이터셋 ----------
    dataset = load_dataset("json", data_files=plan_path, split="train")
    default_system = stage_cfg.get("system_prompt")

    def to_text(examples):
        texts = []
        for messages in examples["messages"]:
            msgs = messages
            if default_system and msgs[0]["role"] != "system":
                msgs = [{"role": "system", "content": default_system}] + msgs
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True,
                          remove_columns=dataset.column_names)
    print(f"[i] PLAN 학습 샘플 수: {len(dataset)}")

    # ---------- 학습 ----------
    _cb = make_progress_callback(args.progress_json)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=mcfg["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        callbacks=[_cb] if _cb else None,
        args=TrainingArguments(
            output_dir=stage_cfg["output_dir"],
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
    )

    # 응답(계획)만 loss 계산. 목표(질문) 부분은 마스킹.
    if tcfg.get("train_on_responses_only", True):
        parts = TEMPLATE_PARTS.get(template_name)
        if parts:
            trainer = train_on_responses_only(
                trainer, instruction_part=parts[0], response_part=parts[1])
        else:
            print(f"[!] '{template_name}' 템플릿의 구분 토큰을 몰라 "
                  "전체 시퀀스로 학습합니다. common.py의 TEMPLATE_PARTS에 추가하세요.")

    stats = trainer.train()
    print(f"[OK] PLAN 완료 — loss: {stats.training_loss:.4f}")
    print("[i] 병합하려면 export_model.py --stage plan, "
          "테스트는 test_model.py --stage plan 를 쓰세요.")

    save_adapter(model, tokenizer, stage_cfg["output_dir"], "plan")

    write_report(args.report_json, {
        "task": "train", "stage": "plan", "status": "ok",
        "n_samples": len(dataset),
        "adapter_dir": os.path.join(stage_cfg["output_dir"], "final"),
        "metrics": {"train_loss": float(stats.training_loss),
                    **{k: v for k, v in (getattr(stats, "metrics", None) or {}).items()}},
    })


if __name__ == "__main__":
    main()
