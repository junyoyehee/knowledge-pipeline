"""
(선택) 계획수립(planning) 학습 데이터 자동 생성
- CPT 데이터셋의 각 청크를 LLM에 보내, 그 도메인에서 세울 법한 '목표'와
  '단계별 계획'을 생성 → plan_dataset 보강
- OpenAI 호환 API 사용 (vLLM / Ollama / OpenAI / 사내 게이트웨이)

LLM에는 {goal, steps} 평면 형식으로 받고, prepare_data.py와 동일한 검증기
(normalize_plan_sample)를 통과한 것만 저장합니다.

사용법:
    export QA_GEN_BASE_URL="http://localhost:11434/v1"
    export QA_GEN_MODEL="qwen2.5:14b"
    python -m scripts.generate.generate_plans [--per-chunk 2] [--overwrite]

출력: data/processed/plan_dataset.jsonl 에 append (기본)
    {"messages": [...], "meta": {"origin": "llm:<모델>", "n_steps": N, ...}}
    meta 필드의 의미는 docs/meta_info.md, 형식은 docs/planning.md 참고
"""
import argparse
import json
import os

from scripts.lib.common import load_config, messages_hash
from scripts.lib.llm_client import call_llm, extract_json_array, resolve_env
# prepare_data의 정규화·검증 로직을 그대로 재사용 (unsloth 비의존)
from scripts.data.prepare_data import normalize_plan_sample

PROMPT_TEMPLATE = """다음 문서를 바탕으로, 이 세계관/도메인에서 누군가 세울 법한
'목표(goal)'와 그 목표를 달성하기 위한 '단계별 계획(steps)'을 {n}세트 만들어주세요.

규칙:
- goal: 문서 주제와 관련된 현실적인 목표 (한국어, 한 문장)
- steps: 목표 달성을 위한 순서 있는 단계 배열 (3~6개, 각 단계는 구체적 행동 한 줄)
- 각 단계는 문서 내용에 근거하되, 문서에 없는 사실을 지어내지 말 것
- 단계는 논리적 순서를 따를 것 (앞 단계의 결과가 뒤 단계의 전제가 되도록)
- "문서에 따르면" 같은 표현은 쓰지 말 것

반드시 아래 JSON 배열 형식으로만 출력:
[{{"goal": "목표", "steps": ["1단계 내용", "2단계 내용", "3단계 내용"]}}, ...]

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
                        help="청크당 생성할 계획 세트 수")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="처리할 최대 청크 수 (테스트용)")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 plan_dataset.jsonl을 덮어쓰기 (기본은 append)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_url, api_key, model = resolve_env()
    chunks = load_chunks(cfg["data"]["cpt_dataset"])
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    origin = f"llm:{model}"
    out_path = cfg["data"]["plan_dataset"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "w" if args.overwrite else "a"

    total = 0
    with open(out_path, mode, encoding="utf-8") as out:
        for i, chunk in enumerate(chunks, 1):
            prompt = PROMPT_TEMPLATE.format(n=args.per_chunk, chunk=chunk["text"])
            try:
                # 목표·계획은 다양성이 필요하므로 온도를 약간 높인다
                response = call_llm(base_url, api_key, model, prompt,
                                    temperature=0.7)
                items = extract_json_array(response)
            except Exception as e:  # noqa: BLE001
                print(f"[!] 청크 {i}/{len(chunks)} 실패: {e}")
                continue

            made = 0
            for j, raw in enumerate(items):
                sample = normalize_plan_sample(raw)
                if not sample:
                    continue
                record = {
                    "messages": sample["messages"],
                    "meta": {
                        "id": f"genplan-{chunk['chunk_id']}-{j}",
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
            print(f"[{i}/{len(chunks)}] 계획 {made}개 생성 (누적 {total})")

    if not total:
        raise SystemExit("[!] 생성된 계획이 없습니다. LLM 응답 형식을 확인하세요.")
    print(f"[OK] 총 {total}개 계획 → {out_path}")
    print("\n[!] 생성된 계획은 사람이 표본 검수하세요. 단계가 문서 사실과 어긋나거나 "
          "순서가 비논리적이면 모델에 잘못된 계획 습관을 가르칠 수 있습니다.\n"
          "    자세한 내용: docs/planning.md")


if __name__ == "__main__":
    main()
