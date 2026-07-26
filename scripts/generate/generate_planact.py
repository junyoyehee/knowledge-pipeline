"""
(선택) 계획-실행(plan-and-execute) 학습 데이터 자동 생성
- 함수 스키마 카탈로그 + 원문 청크를 LLM에 주고, 목표 → 계획 → 각 단계를 함수로
  실행 → 관찰 종합 → 최종 답변으로 이어지는 에이전트형 궤적을 생성
- OpenAI 호환 API 사용 (vLLM / Ollama / OpenAI / 사내 게이트웨이)

LLM에는 {goal, plan, steps:[{tool, arguments, observation}], final_answer} 형식으로
받고, prepare_data.py와 동일한 검증기(normalize_planact_sample)를 통과한 것만 저장합니다.

함수 카탈로그 지정 (generate_tool_calls.py와 동일):
    --tools PATH            함수 스키마 배열이 담긴 JSON 파일
    (미지정 시)             data/raw/tools_catalog.json → 없으면 내장 샘플 카탈로그

사용법:
    export QA_GEN_BASE_URL="http://localhost:11434/v1"
    export QA_GEN_MODEL="qwen2.5:14b"
    python -m scripts.generate.generate_planact --per-chunk 1
    python -m scripts.generate.generate_planact --tools data/raw/tools_catalog.json --overwrite

출력: data/processed/planact_dataset.jsonl 에 append (기본)
    {"messages": [...], "tools": "<JSON>", "meta": {"origin": "llm:<모델>", ...}}
    meta 필드의 의미는 docs/meta_info.md, 형식은 docs/planact.md 참고
"""
import argparse
import json
import os

from scripts.lib.common import (load_config, tool_sample_hash, add_report_arg,
                                write_report)
from scripts.lib.llm_client import call_llm, extract_json_array, resolve_env
# 카탈로그 로더는 generate_tool_calls, 정규화는 prepare_data에서 재사용 (unsloth 비의존)
from scripts.generate.generate_tool_calls import load_catalog
from scripts.data.prepare_data import normalize_planact_sample

PROMPT_TEMPLATE = """당신은 '계획 후 도구 실행'(plan-and-execute) 학습 데이터를 만드는 전문가입니다.
아래 [사용 가능한 함수]와 [참고 문서]를 보고, 사용자의 목표를 함수 호출로 해결하는
현실적인 에이전트 궤적을 {n}세트 만들어주세요.

각 세트의 필드:
- goal: 사용자의 목표 (한국어, 한 문장). 여러 번 함수를 써야 풀리는 것이 좋음
- plan: 목표 달성을 위한 단계별 계획 (문자열 배열, 2~4개)
- steps: 계획을 실제로 실행하는 함수 호출 배열. 각 원소는
    - tool: 호출할 함수 이름 (반드시 아래 목록 중 하나)
    - arguments: 함수 인자 (JSON 객체, parameters 스키마를 지킬 것)
    - observation: 그 함수가 반환했을 법한 결과 (참고 문서에 근거한 현실적인 값)
- final_answer: 관찰들을 종합한 사용자 대상 최종 답변

규칙:
- arguments는 해당 함수의 parameters 스키마에 맞을 것
- observation은 참고 문서의 사실과 모순되지 않을 것 (없는 정보를 지어내지 말 것)
- plan은 steps의 실행 순서와 논리적으로 일치할 것
- 함수를 써야만 풀 수 있는 목표일 것

[사용 가능한 함수]
{tools}

[참고 문서]
---
{chunk}
---

반드시 아래 JSON 배열 형식으로만 출력:
[{{"goal": "...", "plan": ["1단계", "2단계"], "steps": [{{"tool": "...", "arguments": {{...}}, "observation": "..."}}], "final_answer": "..."}}, ...]"""


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
    parser.add_argument("--tools", default=None,
                        help="함수 스키마 배열 JSON 파일 (생략 시 tools_catalog.json/샘플)")
    parser.add_argument("--per-chunk", type=int, default=1,
                        help="청크당 생성할 계획-실행 궤적 수")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="처리할 최대 청크 수 (테스트용)")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 planact_dataset.jsonl을 덮어쓰기 (기본은 append)")
    add_report_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_url, api_key, model = resolve_env()
    catalog = load_catalog(args.tools)
    names = {t["function"]["name"] for t in catalog}
    print(f"[i] 함수 {len(names)}개: {', '.join(sorted(names))}")

    chunks = load_chunks(cfg["data"]["cpt_dataset"])
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    tools_str = json.dumps(catalog, ensure_ascii=False, indent=2)
    origin = f"llm:{model}"
    out_path = cfg["data"]["planact_dataset"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "w" if args.overwrite else "a"

    total = 0
    with open(out_path, mode, encoding="utf-8") as out:
        for i, chunk in enumerate(chunks, 1):
            prompt = PROMPT_TEMPLATE.format(n=args.per_chunk, tools=tools_str,
                                            chunk=chunk["text"])
            try:
                response = call_llm(base_url, api_key, model, prompt,
                                    temperature=0.7)
                items = extract_json_array(response)
            except Exception as e:  # noqa: BLE001
                print(f"[!] 청크 {i}/{len(chunks)} 실패: {e}")
                continue

            made = 0
            for j, raw in enumerate(items):
                # 생성분에는 카탈로그를 붙여 검증기로 넘긴다
                if isinstance(raw, dict):
                    raw = {**raw, "tools": catalog}
                sample = normalize_planact_sample(raw)
                if not sample:
                    continue
                record = {
                    "messages": sample["messages"],
                    "tools": json.dumps(sample["tools"], ensure_ascii=False),
                    "meta": {
                        "id": f"genpa-{chunk['chunk_id']}-{j}",
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "origin": origin,
                        "n_tools": len(sample["tools"]),
                        "n_plan": sample["n_plan"],
                        "n_steps": sample["n_steps"],
                        "hash": tool_sample_hash(sample["tools"], sample["messages"]),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                made += 1
            total += made
            print(f"[{i}/{len(chunks)}] 궤적 {made}개 생성 (누적 {total})")

    if not total:
        raise SystemExit("[!] 생성된 궤적이 없습니다. LLM 응답 형식/함수 목록을 확인하세요.")
    print(f"[OK] 총 {total}개 계획-실행 궤적 → {out_path}")
    print("\n[!] 생성된 arguments·observation·계획은 반드시 사람이 표본 검수하세요. "
          "인자가 스키마와 어긋나거나 관찰이 사실과 다르면 잘못된 실행을 가르칩니다.\n"
          "    자세한 내용: docs/planact.md")
    write_report(args.report_json, {"task": "generate", "kind": "planact",
                                    "status": "ok", "generated": total,
                                    "origin": origin})


if __name__ == "__main__":
    main()
