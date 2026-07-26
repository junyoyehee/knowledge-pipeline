"""
5단계: 학습 결과 확인 — 지식 축적 여부 테스트
- SFT 어댑터(또는 CPT 어댑터)를 로드해 질문을 던져보고 답변 확인
- SFT 데이터셋의 질문 일부를 자동으로 뽑아 테스트하거나, 직접 질문 입력 가능

사용법:
    python scripts/test_model.py                          # SFT 데이터에서 3개 자동 테스트
    python scripts/test_model.py -q "코어스톤 등급 체계는?"  # 직접 질문
    python scripts/test_model.py --stage cpt              # CPT 어댑터로 테스트
"""
import argparse
import json
import os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=["sft", "cpt"], default="sft")
    parser.add_argument("-q", "--question", action="append", default=None,
                        help="직접 질문 (여러 번 지정 가능)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg = cfg["model"]
    stage_dir = cfg["sft" if args.stage == "sft" else "cpt"]["output_dir"]
    adapter = os.path.join(stage_dir, "final")
    if not os.path.isdir(adapter):
        raise SystemExit(f"어댑터가 없습니다: {adapter}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg["dtype"],
        load_in_4bit=mcfg["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=mcfg["chat_template"])
    FastLanguageModel.for_inference(model)  # 2배 빠른 추론 모드

    # 질문 목록 구성
    questions = args.question
    if not questions:
        questions = []
        sft_path = cfg["data"]["sft_dataset"]
        if os.path.exists(sft_path):
            with open(sft_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # messages에서 첫 user 발화를 질문으로 사용
                    msgs = json.loads(line)["messages"]
                    user_turns = [m["content"] for m in msgs if m["role"] == "user"]
                    if user_turns:
                        questions.append(user_turns[0])
                    if len(questions) >= 3:
                        break
        if not questions:
            questions = ["학습한 도메인 지식에 대해 설명해주세요."]

    # 학습 때와 동일한 system 프롬프트를 사용해야 함 (train/serve 불일치 방지)
    system_prompt = cfg["sft"].get("system_prompt")

    for q in questions:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
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
