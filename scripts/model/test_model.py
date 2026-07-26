"""
5단계: 학습 결과 확인 — 지식 축적 여부 테스트
- SFT 어댑터(또는 CPT 어댑터)를 로드해 질문을 던져보고 답변 확인
- SFT 데이터셋의 질문 일부를 자동으로 뽑아 테스트하거나, 직접 질문 입력 가능

사용법:
    python -m scripts.model.test_model                          # SFT 데이터에서 3개 자동 테스트
    python -m scripts.model.test_model -q "코어스톤 등급 체계는?"  # 직접 질문
    python -m scripts.model.test_model --stage cpt              # CPT 어댑터로 테스트
    python -m scripts.model.test_model --stage tool             # 툴 호출 테스트 (함수 스키마 제공)
    python -m scripts.model.test_model --stage tool --tools data/raw/tools_catalog.json
    python -m scripts.model.test_model --stage plan             # 계획수립 테스트
    python -m scripts.model.test_model --stage plan -q "축제 준비 계획 세워줘"
    python -m scripts.model.test_model --stage react            # 추론형(ReAct) 테스트
    python -m scripts.model.test_model --stage planact          # 계획-실행 테스트 (함수 스키마 제공)
"""
import argparse
import json
import os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from scripts.lib.common import load_config, stage_adapter


def load_test_tools(args, cfg, dataset_path):
    """tool/planact 스테이지 테스트용 함수 스키마를 로드 (없으면 None).

    --tools가 있으면 그 파일에서, 없으면 해당 데이터셋 첫 줄의 tools에서 읽는다.
    """
    if args.tools:
        with open(args.tools, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tools") if isinstance(data, dict) else data
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            t = json.loads(first).get("tools")
            return json.loads(t) if isinstance(t, str) else t
    print("[!] 함수 스키마를 찾지 못했습니다. --tools로 지정하거나 데이터셋을 먼저 만드세요.")
    return None


def read_first_questions(path, limit=3):
    """데이터셋에서 첫 user 발화들을 뽑아 자동 테스트 질문으로 쓴다."""
    questions = []
    if not os.path.exists(path):
        return questions
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            user_turns = [m["content"] for m in msgs
                          if m["role"] == "user" and m.get("content")]
            if user_turns:
                questions.append(user_turns[0])
            if len(questions) >= limit:
                break
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage",
                        choices=["sft", "cpt", "tool", "plan", "react", "planact",
                                 "dpo", "orpo", "kto"],
                        default="sft")
    parser.add_argument("-q", "--question", action="append", default=None,
                        help="직접 질문 (여러 번 지정 가능)")
    parser.add_argument("--tools", default=None,
                        help="tool 스테이지 테스트 시 쓸 함수 스키마 JSON 파일 "
                             "(생략 시 tool_dataset에서 로드)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg = cfg["model"]
    adapter = stage_adapter(cfg, args.stage)
    if not os.path.isdir(adapter):
        raise SystemExit(f"어댑터가 없습니다: {adapter}")
    print(f"[i] 테스트 대상: {args.stage} → {adapter}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg["dtype"],
        load_in_4bit=mcfg["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=mcfg["chat_template"])
    FastLanguageModel.for_inference(model)  # 2배 빠른 추론 모드

    # 질문 목록 구성 (스테이지별 데이터셋의 첫 user 발화를 자동 추출)
    stage_source = {
        "tool": cfg["data"].get("tool_dataset"),
        "plan": cfg["data"].get("plan_dataset"),
        "react": cfg["data"].get("react_dataset"),
        "planact": cfg["data"].get("planact_dataset"),
    }

    # tool/planact 스테이지는 함수 스키마를 프롬프트에 넣어야 도구 호출이 나온다
    tools = (load_test_tools(args, cfg, stage_source.get(args.stage))
             if args.stage in ("tool", "planact") else None)

    questions = args.question
    if not questions:
        source = stage_source.get(args.stage) or cfg["data"]["sft_dataset"]
        questions = read_first_questions(source) if source else []
        if not questions:
            questions = ["학습한 도메인 지식에 대해 설명해주세요."]

    # 학습 때와 동일한 system 프롬프트를 사용해야 함 (train/serve 불일치 방지)
    stage_key = (args.stage if args.stage in ("tool", "plan", "react", "planact")
                 else "sft")
    system_prompt = cfg.get(stage_key, {}).get("system_prompt")

    for q in questions:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})
        inputs = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.3,
            do_sample=True,
            use_cache=True,
        )
        answer = tokenizer.decode(outputs[0][inputs.shape[1]:],
                                  skip_special_tokens=True)
        print("=" * 70)
        print(f"Q: {q}")
        print(f"A: {answer.strip()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
