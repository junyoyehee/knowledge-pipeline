"""
(선택) 툴 호출(function calling) 학습 데이터 자동 생성
- 함수 스키마 카탈로그 + 원문 청크를 LLM에 주고, 그 함수를 써야만 답할 수 있는
  현실적인 대화(요청 → tool_call → 결과 → 최종 답변)를 생성
- OpenAI 호환 API 사용 (vLLM / Ollama / OpenAI / 사내 게이트웨이)

LLM에는 조립하기 쉬운 평면 형식으로 받고(파이썬이 정식 messages로 조립),
prepare_data.py와 동일한 검증기(normalize_tool_sample)를 통과한 것만 저장합니다.

함수 카탈로그 지정:
    --tools PATH            함수 스키마 배열이 담긴 JSON 파일
    (미지정 시)             data/raw/tools_catalog.json → 없으면 내장 샘플 카탈로그

사용법:
    export QA_GEN_BASE_URL="http://localhost:11434/v1"
    export QA_GEN_MODEL="qwen2.5:14b"
    python scripts/generate_tool_calls.py --per-chunk 2
    python scripts/generate_tool_calls.py --tools data/raw/tools_catalog.json
    python scripts/generate_tool_calls.py --overwrite      # 기존 파일 덮어쓰기

출력: data/processed/tool_dataset.jsonl 에 append (기본)
    {"messages": [...], "tools": "<JSON>", "meta": {"origin": "llm:<모델>", ...}}
    meta 필드의 의미는 docs/meta_info.md, 형식은 docs/tool_calling.md 참고
"""
import argparse
import json
import os

from common import PROJECT_ROOT, load_config, tool_sample_hash
from llm_client import call_llm, extract_json_array, resolve_env
# prepare_data의 정규화·검증 로직을 그대로 재사용 (unsloth 비의존)
from prepare_data import SAMPLE_TOOLS, _normalize_tools, normalize_tool_sample

PROMPT_TEMPLATE = """당신은 함수 호출(tool calling) 학습 데이터를 만드는 전문가입니다.
아래 [사용 가능한 함수]와 [참고 문서]를 보고, 사용자가 이 함수를 써야만 답할 수 있는
현실적인 대화 예시를 {n}개 만들어주세요.

각 예시의 필드:
- user: 사용자의 자연스러운 한국어 요청
- tool_name: 호출할 함수 이름 (반드시 아래 함수 목록 중 하나)
- arguments: 함수에 넘길 인자 (JSON 객체, 스키마의 파라미터 이름과 타입을 지킬 것)
- tool_result: 그 함수가 반환했을 법한 결과 (참고 문서에 근거한 현실적인 JSON)
- final_answer: tool_result를 바탕으로 사용자에게 준 자연스러운 최종 답변

규칙:
- arguments는 반드시 해당 함수의 parameters 스키마에 맞을 것 (required 인자 포함)
- tool_result는 참고 문서의 사실과 모순되지 않을 것 (문서에 없는 정보는 지어내지 말 것)
- 함수를 써야만 알 수 있는 질문일 것 (상식/일반지식으로 답 가능한 질문 금지)
- final_answer는 tool_result의 내용에 근거할 것

[사용 가능한 함수]
{tools}

[참고 문서]
---
{chunk}
---

반드시 아래 JSON 배열 형식으로만 출력:
[{{"user": "...", "tool_name": "...", "arguments": {{...}}, "tool_result": "...", "final_answer": "..."}}, ...]"""


def load_catalog(path: str | None) -> list:
    """함수 스키마 카탈로그를 로드·정규화한다."""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tools = data.get("tools") if isinstance(data, dict) else data
    else:
        default = os.path.join(PROJECT_ROOT, "data", "raw", "tools_catalog.json")
        if os.path.exists(default):
            with open(default, "r", encoding="utf-8") as f:
                tools = json.load(f)
            print(f"[i] 함수 카탈로그: {default}")
        else:
            print("[i] --tools 미지정, tools_catalog.json도 없어 내장 샘플 카탈로그를 사용합니다.")
            tools = SAMPLE_TOOLS

    norm = _normalize_tools(tools)
    if not norm:
        raise SystemExit(
            "함수 카탈로그 형식이 올바르지 않습니다.\n"
            "함수 스키마 배열(JSON)이어야 합니다. 예: docs/tool_calling.md")
    return norm


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


def assemble(item: dict, catalog: list, names: set) -> dict | None:
    """LLM이 준 평면 항목을 정식 tool 호출 샘플로 조립·검증."""
    if not isinstance(item, dict):
        return None
    user = item.get("user")
    name = item.get("tool_name")
    arguments = item.get("arguments")
    result = item.get("tool_result")
    final = item.get("final_answer")

    if not (isinstance(user, str) and user.strip()
            and name in names
            and isinstance(arguments, (dict, list))
            and result is not None
            and isinstance(final, str) and final.strip()):
        return None

    if isinstance(result, (dict, list)):
        result = json.dumps(result, ensure_ascii=False)

    sample = {
        "tools": catalog,
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": name, "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": str(result)},
            {"role": "assistant", "content": final.strip()},
        ],
    }
    # prepare_data와 동일한 검증기 통과 (arguments/tools 문자열화 포함)
    return normalize_tool_sample(sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tools", default=None,
                        help="함수 스키마 배열 JSON 파일 (생략 시 tools_catalog.json/샘플)")
    parser.add_argument("--per-chunk", type=int, default=2,
                        help="청크당 생성할 툴 호출 예시 수")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="처리할 최대 청크 수 (테스트용)")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 tool_dataset.jsonl을 덮어쓰기 (기본은 append)")
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

    out_path = cfg["data"]["tool_dataset"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "w" if args.overwrite else "a"

    total = 0
    with open(out_path, mode, encoding="utf-8") as out:
        for i, chunk in enumerate(chunks, 1):
            prompt = PROMPT_TEMPLATE.format(n=args.per_chunk, tools=tools_str,
                                            chunk=chunk["text"])
            try:
                # 다양성을 위해 온도를 약간 높게
                response = call_llm(base_url, api_key, model, prompt,
                                    temperature=0.7)
                items = extract_json_array(response)
            except Exception as e:  # noqa: BLE001
                print(f"[!] 청크 {i}/{len(chunks)} 실패: {e}")
                continue

            made = 0
            for j, raw in enumerate(items):
                sample = assemble(raw, catalog, names)
                if not sample:
                    continue
                record = {
                    "messages": sample["messages"],
                    "tools": json.dumps(sample["tools"], ensure_ascii=False),
                    "meta": {
                        "id": f"gentool-{chunk['chunk_id']}-{j}",
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "origin": origin,
                        "n_tools": len(sample["tools"]),
                        "hash": tool_sample_hash(sample["tools"], sample["messages"]),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                made += 1
            total += made
            print(f"[{i}/{len(chunks)}] 툴 호출 {made}개 생성 (누적 {total})")

    if not total:
        raise SystemExit("[!] 생성된 샘플이 없습니다. LLM 응답 형식/함수 목록을 확인하세요.")
    print(f"[OK] 총 {total}개 툴 호출 대화 → {out_path}")
    print("\n[!] 생성된 arguments와 tool_result는 반드시 사람이 표본 검수하세요. "
          "인자·결과가 스키마나 사실과 어긋나면 모델에 잘못된 호출을 가르치게 됩니다.\n"
          "    자세한 내용: docs/tool_calling.md")


if __name__ == "__main__":
    main()
