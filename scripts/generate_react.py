"""
(선택) 추론형(ReAct) 학습 데이터 자동 생성
- CPT 데이터셋의 각 청크를 LLM에 보내, 그 문서로 답할 수 있는 질문과 함께
  Thought/Action/Observation을 반복하는 추론 트레이스를 생성
- OpenAI 호환 API 사용 (vLLM / Ollama / OpenAI / 사내 게이트웨이)

LLM에는 {question, steps:[{thought,action,observation}], final_answer} 형식으로 받고,
prepare_data.py와 동일한 검증기(normalize_react_sample)를 통과한 것만 저장합니다.

사용법:
    export QA_GEN_BASE_URL="http://localhost:11434/v1"
    export QA_GEN_MODEL="qwen2.5:14b"
    python scripts/generate_react.py [--per-chunk 2] [--overwrite]

출력: data/processed/react_dataset.jsonl 에 append (기본)
    {"messages": [...], "meta": {"origin": "llm:<모델>", "n_steps": N, ...}}
    meta 필드의 의미는 docs/meta_info.md, 형식은 docs/react.md 참고
"""
import argparse
import json
import os

from common import load_config, messages_hash
from llm_client import call_llm, extract_json_array, resolve_env
# prepare_data의 정규화·검증 로직을 그대로 재사용 (unsloth 비의존)
from prepare_data import normalize_react_sample

PROMPT_TEMPLATE = """다음 문서를 근거로, 이 문서를 참고해야 답할 수 있는 질문과
그 질문을 푸는 추론 과정(ReAct)을 {n}세트 만들어주세요.

각 세트는 다음을 포함합니다:
- question: 문서 내용을 근거로 답할 수 있는 질문 (한국어)
- steps: 추론 단계 배열. 각 단계는
    - thought: 지금 무엇을 왜 확인하는지에 대한 생각 (한국어)
    - action: 취하는 행동. "도구이름[입력]" 형태의 한 줄 (예: search[코어스톤 등급])
    - observation: 그 행동으로 얻었을 법한 결과. 반드시 문서 내용에 근거할 것
- final_answer: 관찰들을 종합한 최종 답변

규칙:
- steps는 1~4개. 마지막에 답을 낼 수 있을 만큼만 최소한으로.
- observation은 문서에 실제로 있는 사실만 담을 것 (없는 정보를 지어내지 말 것)
- 각 thought는 바로 다음 action으로 자연스럽게 이어질 것
- 문서를 안 봐도 상식으로 답할 수 있는 질문은 만들지 말 것

반드시 아래 JSON 배열 형식으로만 출력:
[{{"question": "...", "steps": [{{"thought": "...", "action": "search[...]", "observation": "..."}}], "final_answer": "..."}}, ...]

문서:
---
{chunk}
---"""


def load_chunks(path: str) -> list:
    if not os.path.exists(path):
        raise SystemExit(f"CPT 데이터셋이 없습니다: {path}\n"
                         "먼저 prepare_data.py를 실행하세요.")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("meta") or {}
            chunks.append({
                "text": rec["text"],
                "chunk_id": meta.get("id", f"chunk-{lineno:04d}"),
                "source": meta.get("source"),
            })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-chunk", type=int, default=2,
                        help="청크당 생성할 추론 트레이스 수")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="처리할 최대 청크 수 (테스트용)")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 react_dataset.jsonl을 덮어쓰기 (기본은 append)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_url, api_key, model = resolve_env()
    chunks = load_chunks(cfg["data"]["cpt_dataset"])
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    origin = f"llm:{model}"
    out_path = cfg["data"]["react_dataset"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "w" if args.overwrite else "a"

    total = 0
    with open(out_path, mode, encoding="utf-8") as out:
        for i, chunk in enumerate(chunks, 1):
            prompt = PROMPT_TEMPLATE.format(n=args.per_chunk, chunk=chunk["text"])
            try:
                # 추론 경로에 다양성이 필요하므로 온도를 약간 높인다
                response = call_llm(base_url, api_key, model, prompt,
                                    temperature=0.7)
                items = extract_json_array(response)
            except Exception as e:  # noqa: BLE001
                print(f"[!] 청크 {i}/{len(chunks)} 실패: {e}")
                continue

            made = 0
            for j, raw in enumerate(items):
                sample = normalize_react_sample(raw)
                if not sample:
                    continue
                record = {
                    "messages": sample["messages"],
                    "meta": {
                        "id": f"genreact-{chunk['chunk_id']}-{j}",
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "origin": origin,
                        "n_steps": sample["n_steps"],
                        "hash": messages_hash(sample["messages"]),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                made += 1
            total += made
            print(f"[{i}/{len(chunks)}] 트레이스 {made}개 생성 (누적 {total})")

    if not total:
        raise SystemExit("[!] 생성된 트레이스가 없습니다. LLM 응답 형식을 확인하세요.")
    print(f"[OK] 총 {total}개 ReAct 트레이스 → {out_path}")
    print("\n[!] 생성된 트레이스는 사람이 표본 검수하세요. observation이 문서 사실과 "
          "어긋나거나 thought→action 연결이 비논리적이면 잘못된 추론 습관을 가르칩니다.\n"
          "    자세한 내용: docs/react.md")


if __name__ == "__main__":
    main()
