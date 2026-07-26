"""
(선택) 툴 호출 학습 단계 — function calling 능력 주입
- 지식 SFT 어댑터(기본) 위에 툴 호출 데이터로 전용 학습하는 별도 단계
- tools(함수 스키마)를 채팅 템플릿에 함께 넣어 학습하므로, 모델이
  '언제 / 어떤 함수를 / 어떤 인자로' 부르는지를 배운다
- 응답(assistant의 tool_call + 최종 답변)에만 loss 계산, tool 결과는 마스킹

시작 지점(config의 tool.init_from):
    "sft"(기본) → 지식 SFT 어댑터를 이어받아 툴 능력만 추가
    "base"      → 베이스 모델에서 툴 능력만 학습 (지식과 분리)
    경로/기타   → 그 모델 위에 새 LoRA 부착

사용법:
    python scripts/prepare_data.py            # tools_*.jsonl → tool_dataset.jsonl
    python scripts/generate_tool_calls.py     # (선택) LLM으로 자동 생성
    python scripts/train_tool.py [--config configs/config.yaml]

출력: outputs/tool/ (LoRA 어댑터)
"""
import argparse
import json
import os

# unsloth는 transformers/trl보다 먼저 import되어야 함 (pref_common이 unsloth를 import)
from pref_common import load_config, resolve_init_source, save_adapter

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from common import TEMPLATE_PARTS


def load_model(cfg: dict, stage_cfg: dict):
    """툴 학습용 모델·토크나이저 로드 (init_from 해석)."""
    mcfg = cfg["model"]
    init_from = stage_cfg.get("init_from", "sft")
    source, needs_new_lora = resolve_init_source(cfg, init_from)
    print(f"[i] TOOL 시작 지점: {init_from} → {source}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg["dtype"],
        load_in_4bit=mcfg["load_in_4bit"],
    )
    if needs_new_lora:
        lcfg, tcfg = stage_cfg["lora"], stage_cfg["train"]
        print("[i] 새 LoRA 어댑터를 부착합니다 (베이스/병합모델에서 시작)")
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
    else:
        print("[i] 기존 어댑터를 이어서 툴 능력을 추가합니다")

    tokenizer = get_chat_template(tokenizer, chat_template=mcfg["chat_template"])
    return model, tokenizer


def _load_args(arguments):
    """저장 시 JSON 문자열로 두었던 arguments를 dict로 되돌린다.

    채팅 템플릿은 arguments를 객체로 렌더링(| tojson)하므로 dict여야 정확하다.
    """
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _rebuild_message(m: dict) -> dict:
    """datasets가 채운 None 필드를 걷어내고 arguments를 dict로 복원."""
    mm = {"role": m["role"]}
    content = m.get("content")
    if content is not None:
        mm["content"] = content
    tool_calls = m.get("tool_calls")
    if tool_calls:
        mm.setdefault("content", "")   # tool_call만 있는 assistant는 content 비움
        mm["tool_calls"] = [{
            "id": tc.get("id"),
            "type": tc.get("type") or "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": _load_args(tc["function"]["arguments"]),
            },
        } for tc in tool_calls]
    if m.get("tool_call_id") is not None:
        mm["tool_call_id"] = m["tool_call_id"]
    return mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, stage_cfg = cfg["model"], cfg["tool"]
    tcfg = stage_cfg["train"]

    tool_path = cfg["data"].get("tool_dataset")
    if not tool_path or not os.path.exists(tool_path):
        raise SystemExit(
            f"툴 호출 데이터셋이 없습니다: {tool_path}\n"
            "tools_*.jsonl을 넣고 prepare_data.py를 돌리거나, "
            "generate_tool_calls.py로 먼저 생성하세요.")

    model, tokenizer = load_model(cfg, stage_cfg)
    template_name = mcfg["chat_template"]

    # ---------- 데이터셋 ----------
    dataset = load_dataset("json", data_files=tool_path, split="train")
    default_system = stage_cfg.get("system_prompt")

    def to_text(examples):
        texts = []
        for messages, tools_json in zip(examples["messages"], examples["tools"]):
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            msgs = [_rebuild_message(m) for m in messages]
            if default_system and msgs[0]["role"] != "system":
                msgs = [{"role": "system", "content": default_system}] + msgs
            texts.append(tokenizer.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True,
                          remove_columns=dataset.column_names)
    print(f"[i] TOOL 학습 샘플 수: {len(dataset)}")

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

    # 응답(tool_call + 최종 답변)만 loss 계산. tool 결과는 user 블록으로 렌더링되어 마스킹됨.
    if tcfg.get("train_on_responses_only", True):
        parts = TEMPLATE_PARTS.get(template_name)
        if parts:
            trainer = train_on_responses_only(
                trainer, instruction_part=parts[0], response_part=parts[1])
        else:
            print(f"[!] '{template_name}' 템플릿의 구분 토큰을 몰라 "
                  "전체 시퀀스로 학습합니다. common.py의 TEMPLATE_PARTS에 추가하세요.")

    stats = trainer.train()
    print(f"[OK] TOOL 완료 — loss: {stats.training_loss:.4f}")
    print("[i] 병합하려면 export_model.py --stage tool, "
          "테스트는 test_model.py --stage tool 를 쓰세요.")

    save_adapter(model, tokenizer, stage_cfg["output_dir"], "tool")


if __name__ == "__main__":
    main()
