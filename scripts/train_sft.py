"""
3단계: SFT (Supervised Fine-Tuning) — 주입된 지식을 대화형으로 활용
- CPT로 지식을 축적한 어댑터를 이어받아 QA 데이터로 지시 튜닝
- 응답 부분에만 loss를 계산 (train_on_responses_only)

사용법:
    python scripts/train_sft.py [--config configs/config.yaml]

출력: outputs/sft/ (LoRA 어댑터)
"""
import argparse
import os

# unsloth는 transformers/trl보다 먼저 import해야 최적화가 적용됨
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 채팅 템플릿별 user/assistant 구분 토큰 (train_tool.py와 공용)
from common import load_config, TEMPLATE_PARTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, scfg = cfg["model"], cfg["sft"]
    tcfg, lcfg = scfg["train"], scfg["lora"]

    # ---------- 모델 로드 ----------
    cpt_final = os.path.join(cfg["cpt"]["output_dir"], "final")
    from_cpt = scfg["continue_from_cpt"] and os.path.isdir(cpt_final)

    if from_cpt:
        # CPT 어댑터를 그대로 이어받아 학습 (지식 유지)
        print(f"[i] CPT 어댑터에서 이어서 학습: {cpt_final}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cpt_final,
            max_seq_length=mcfg["max_seq_length"],
            dtype=mcfg["dtype"],
            load_in_4bit=mcfg["load_in_4bit"],
        )
    else:
        print("[i] 베이스 모델에서 새로 SFT 시작 (CPT 어댑터 없음)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=mcfg["name"],
            max_seq_length=mcfg["max_seq_length"],
            dtype=mcfg["dtype"],
            load_in_4bit=mcfg["load_in_4bit"],
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lcfg["r"],
            target_modules=lcfg["target_modules"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=tcfg["seed"],
        )

    # ---------- 채팅 템플릿 ----------
    template_name = mcfg["chat_template"]
    tokenizer = get_chat_template(tokenizer, chat_template=template_name)

    # ---------- 데이터셋 ----------
    dataset = load_dataset("json", data_files=cfg["data"]["sft_dataset"],
                           split="train")

    # 데이터에 system이 없을 때 붙일 기본 system 프롬프트 (config, 없으면 생략)
    default_system = scfg.get("system_prompt")

    def to_text(examples):
        texts = []
        for messages in examples["messages"]:
            if default_system and messages[0]["role"] != "system":
                messages = [{"role": "system", "content": default_system}] + messages
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True,
                          remove_columns=dataset.column_names)
    print(f"[i] SFT 학습 샘플 수: {len(dataset)}")

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
            output_dir=scfg["output_dir"],
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

    # 응답 부분만 loss 계산
    if tcfg.get("train_on_responses_only", True):
        parts = TEMPLATE_PARTS.get(template_name)
        if parts:
            trainer = train_on_responses_only(
                trainer, instruction_part=parts[0], response_part=parts[1])
        else:
            print(f"[!] '{template_name}' 템플릿의 구분 토큰을 몰라 "
                  "전체 시퀀스로 학습합니다. TEMPLATE_PARTS에 추가하세요.")

    stats = trainer.train()
    print(f"[OK] SFT 완료 — loss: {stats.training_loss:.4f}")

    # ---------- 저장 ----------
    final_dir = os.path.join(scfg["output_dir"], "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] SFT 어댑터 저장: {final_dir}")


if __name__ == "__main__":
    main()
