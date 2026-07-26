"""
(선택) 추론형(ReAct) 학습 단계 — Thought/Action/Observation 반복 추론 능력 주입
- 지식 SFT 어댑터(기본) 위에 'ReAct 트레이스'를 전용으로 학습하는 별도 단계
- 데이터는 멀티턴 대화로, Observation은 user 턴으로 들어가 있어
  train_on_responses_only가 자동으로 마스킹한다. 즉 모델은 Thought+Action과
  Final Answer만 생성하도록 배우고, Observation은 학습 대상이 아니다.
  (모델이 관찰 결과를 지어내도록 학습하는 것을 방지)

시작 지점(config의 react.init_from):
    "sft"(기본) → 지식 SFT 어댑터를 이어받아 추론 능력만 추가
    "tool"      → 툴 호출 어댑터 위에 추론까지 얹고 싶을 때
    "base"/경로 → 그 모델 위에 새 LoRA 부착

사용법:
    python -m scripts.data.prepare_data            # react_*.jsonl → react_dataset.jsonl
    python -m scripts.generate.generate_react          # (선택) LLM으로 자동 생성
    python -m scripts.train.train_react [--config configs/config.yaml]

출력: outputs/react/ (LoRA 어댑터)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    add_report_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, stage_cfg = cfg["model"], cfg["react"]
    tcfg = stage_cfg["train"]

    react_path = cfg["data"].get("react_dataset")
    if not react_path or not os.path.exists(react_path):
        raise SystemExit(
            f"ReAct 데이터셋이 없습니다: {react_path}\n"
            "react_*.jsonl을 넣고 prepare_data.py를 돌리거나, "
            "generate_react.py로 먼저 생성하세요.")

    model, tokenizer = load_stage_model(cfg, stage_cfg, "react")
    template_name = mcfg["chat_template"]

    # ---------- 데이터셋 ----------
    dataset = load_dataset("json", data_files=react_path, split="train")
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
    print(f"[i] ReAct 학습 샘플 수: {len(dataset)}")

    # ---------- 학습 ----------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=mcfg["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
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

    # 응답(Thought+Action, Final Answer)만 loss 계산.
    # Observation은 user 턴으로 렌더링되어 instruction_part로 마스킹된다.
    if tcfg.get("train_on_responses_only", True):
        parts = TEMPLATE_PARTS.get(template_name)
        if parts:
            trainer = train_on_responses_only(
                trainer, instruction_part=parts[0], response_part=parts[1])
        else:
            print(f"[!] '{template_name}' 템플릿의 구분 토큰을 몰라 "
                  "전체 시퀀스로 학습합니다. common.py의 TEMPLATE_PARTS에 추가하세요.")

    stats = trainer.train()
    print(f"[OK] ReAct 완료 — loss: {stats.training_loss:.4f}")
    print("[i] 병합하려면 export_model.py --stage react, "
          "테스트는 test_model.py --stage react 를 쓰세요.")

    save_adapter(model, tokenizer, stage_cfg["output_dir"], "react")

    write_report(args.report_json, {
        "task": "train", "stage": "react", "status": "ok",
        "n_samples": len(dataset),
        "adapter_dir": os.path.join(stage_cfg["output_dir"], "final"),
        "metrics": {"train_loss": float(stats.training_loss),
                    **{k: v for k, v in (getattr(stats, "metrics", None) or {}).items()}},
    })


if __name__ == "__main__":
    main()
