"""
(선택) 1.5단계: 원문 청크로부터 QA 쌍 자동 생성
- CPT 데이터셋의 각 청크를 LLM에 보내 QA 쌍을 생성 → SFT 데이터셋 보강
- OpenAI 호환 API를 사용 (vLLM, Ollama, OpenAI, 사내 게이트웨이 등 모두 가능)

사용법:
    export QA_GEN_BASE_URL="http://localhost:11434/v1"   # 예: Ollama
    export QA_GEN_API_KEY="dummy"
    export QA_GEN_MODEL="qwen2.5:14b"
    python scripts/generate_qa.py [--config configs/config.yaml] [--per-chunk 3]

출력: data/processed/sft_dataset.jsonl 에 append
"""
import argparse
import json
import os
import re
import urllib.request

from common import load_config

PROMPT_TEMPLATE = """다음 문서 내용을 바탕으로, 문서에 담긴 지식을 확인하는 질문-답변 쌍을 {n}개 만들어주세요.

규칙:
- 답변은 문서 내용만으로 완결되게 작성 (문서를 보지 않은 사람도 이해 가능하게)
- "문서에 따르면" 같은 표현 금지
- 반드시 아래 JSON 배열 형식으로만 출력:
[{{"instruction": "질문", "output": "답변"}}, ...]

문서:
---
{chunk}
---"""


def call_llm(base_url: str, api_key: str, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_qa_pairs(text: str) -> list:
    """LLM 응답에서 JSON 배열 추출."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("instruction") and item.get("output"):
            result.append({"instruction": str(item["instruction"]).strip(),
                           "output": str(item["output"]).strip()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-chunk", type=int, default=3,
                        help="청크당 생성할 QA 쌍 수")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="처리할 최대 청크 수 (테스트용)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_url = os.environ.get("QA_GEN_BASE_URL")
    api_key = os.environ.get("QA_GEN_API_KEY", "dummy")
    model = os.environ.get("QA_GEN_MODEL")
    if not base_url or not model:
        raise SystemExit(
            "QA_GEN_BASE_URL / QA_GEN_MODEL 환경변수를 설정하세요.\n"
            "예) export QA_GEN_BASE_URL=http://localhost:11434/v1\n"
            "    export QA_GEN_MODEL=qwen2.5:14b"
        )

    cpt_path = cfg["data"]["cpt_dataset"]
    if not os.path.exists(cpt_path):
        raise SystemExit("CPT 데이터셋이 없습니다. 먼저 prepare_data.py를 실행하세요.")

    chunks = []
    with open(cpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line)["text"])
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    total = 0
    with open(cfg["data"]["sft_dataset"], "a", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks, 1):
            try:
                response = call_llm(base_url, api_key, model,
                                    PROMPT_TEMPLATE.format(n=args.per_chunk, chunk=chunk))
                pairs = extract_qa_pairs(response)
            except Exception as e:  # noqa: BLE001
                print(f"[!] 청크 {i}/{len(chunks)} 실패: {e}")
                continue
            for p in pairs:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(pairs)
            print(f"[{i}/{len(chunks)}] QA {len(pairs)}개 생성 (누적 {total})")

    print(f"[OK] 총 {total}개 QA 쌍 → {cfg['data']['sft_dataset']}")


if __name__ == "__main__":
    main()
