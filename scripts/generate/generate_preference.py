"""(선택) 선호 학습용 데이터 생성 — DPO / ORPO / KTO 공용

핵심 아이디어는 **그럴듯하지만 틀린 답(rejected)** 을 만드는 것입니다.
횡설수설하거나 비어 있는 rejected는 학습 신호가 없습니다. 모델이 애초에
그런 출력을 내지 않기 때문입니다. 반드시 "자연스럽게 읽히지만 사실이 틀린" 답이어야
합니다. 자세한 배경은 docs/preference_tuning.md 참고.

두 가지 생성 모드:
    from-sft    : 기존 SFT QA의 답을 chosen으로 두고, 그것을 미묘하게 왜곡한
                  오답을 rejected로 생성 (권장 — 정답이 이미 검증되어 있음)
    from-chunks : CPT 청크에서 질문·정답·오답을 한 번에 생성
                  (SFT 데이터가 없거나 커버리지를 넓히고 싶을 때)

추가로 --refusals-per-chunk 를 주면 "자료에 없는 질문"에 대해
거절(chosen) vs 지어낸 답(rejected) 쌍을 만듭니다. 환각 억제에 가장 직접적입니다.

사용법:
    export QA_GEN_BASE_URL="http://localhost:11434/v1"
    export QA_GEN_MODEL="qwen2.5:14b"
    python -m scripts.generate.generate_preference                    # from-sft (기본)
    python -m scripts.generate.generate_preference --mode from-chunks --per-chunk 3
    python -m scripts.generate.generate_preference --refusals-per-chunk 1
    python -m scripts.generate.generate_preference --no-kto           # KTO 파일 생성 생략

출력:
    data/processed/pref_dataset.jsonl  (DPO/ORPO 공용)
        {"prompt": [...], "chosen": [...], "rejected": [...], "meta": {...}}
    data/processed/kto_dataset.jsonl   (KTO 전용, 위 쌍에서 자동 파생)
        {"prompt": [...], "completion": [...], "label": true/false, "meta": {...}}
"""
import argparse
import json
import os

from scripts.lib.common import load_config, messages_hash
from scripts.lib.llm_client import call_llm, extract_json_array, resolve_env

# ---------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------

DISTORT_PROMPT = """아래는 어떤 문서에 근거한 질문과 '정답'입니다.
이 정답을 미묘하게 왜곡한 **오답**을 {n}개 만들어주세요.

오답 작성 규칙 (매우 중요):
- 문체와 자신감은 정답과 똑같이 유지할 것. 어색하거나 얼버무리면 안 됨
- 고유명사, 숫자, 날짜, 등급, 인과관계 중 **하나만** 바꿀 것
- "모르겠습니다", "정보가 없습니다" 같은 회피성 답변은 금지
- 정답과 길이가 비슷할 것
- 읽는 사람이 원문을 모르면 속아 넘어갈 정도로 자연스러울 것

반드시 아래 JSON 배열 형식으로만 출력:
[{{"rejected": "그럴듯한 오답"}}, ...]

질문:
{question}

정답:
{answer}"""


CHUNK_PROMPT = """다음 문서를 바탕으로 질문·정답·오답 세트를 {n}개 만들어주세요.

규칙:
- 정답(chosen)은 문서 내용만으로 완결되게 작성 ("문서에 따르면" 같은 표현 금지)
- 오답(rejected)은 정답의 고유명사·숫자·인과관계 중 하나만 바꾼 것
- 오답도 정답과 동일한 문체·길이·자신감을 유지할 것 (얼버무리면 안 됨)
- 오답이 우연히 문서 내용과 맞아떨어지지 않도록 확인할 것

반드시 아래 JSON 배열 형식으로만 출력:
[{{"question": "질문", "chosen": "정답", "rejected": "그럴듯한 오답"}}, ...]

문서:
---
{chunk}
---"""


REFUSAL_PROMPT = """다음 문서를 읽고, **이 문서로는 답할 수 없는** 질문을 {n}개 만들어주세요.

질문 조건:
- 문서와 같은 주제·세계관에 속해 자연스럽게 물어볼 법한 질문일 것
- 그러나 답이 문서 어디에도 없을 것 (문서에 언급되지 않은 인물·수치·사건 등)
- 문서 내용으로 답할 수 있는 질문이면 안 됨

각 질문에 대해:
- refusal: 자료에 해당 정보가 없다고 밝히되, 아는 범위를 간단히 덧붙이는 정중한 답변
- hallucination: 마치 아는 것처럼 구체적인 수치·이름을 지어내 자신 있게 답한 문장

반드시 아래 JSON 배열 형식으로만 출력:
[{{"question": "질문", "refusal": "모른다고 밝히는 답", "hallucination": "지어낸 답"}}, ...]

문서:
---
{chunk}
---"""


# ---------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------

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


def load_sft_samples(path: str) -> list:
    """SFT 데이터에서 (prompt 메시지들, 정답 문자열)을 뽑는다."""
    if not os.path.exists(path):
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec["messages"]
            if msgs[-1]["role"] != "assistant":
                continue
            samples.append({
                "prompt": msgs[:-1],           # system(선택) + user...
                "answer": msgs[-1]["content"],
                "meta": rec.get("meta") or {},
            })
    return samples


# ---------------------------------------------------------------
# 쌍 생성
# ---------------------------------------------------------------

def make_pair(prompt_msgs: list, chosen: str, rejected: str, meta: dict) -> dict:
    pair = {
        "prompt": prompt_msgs,
        "chosen": [{"role": "assistant", "content": chosen.strip()}],
        "rejected": [{"role": "assistant", "content": rejected.strip()}],
    }
    meta = dict(meta)
    meta["hash"] = messages_hash(prompt_msgs + pair["chosen"] + pair["rejected"])
    pair["meta"] = meta
    return pair


def gen_from_sft(llm, samples: list, per_sample: int) -> list:
    pairs = []
    for i, s in enumerate(samples, 1):
        question = next((m["content"] for m in reversed(s["prompt"])
                         if m["role"] == "user"), None)
        if not question:
            continue
        try:
            resp = llm(DISTORT_PROMPT.format(n=per_sample, question=question,
                                             answer=s["answer"]))
        except Exception as e:  # noqa: BLE001
            print(f"[!] SFT 샘플 {i}/{len(samples)} 실패: {e}")
            continue
        made = 0
        for j, item in enumerate(extract_json_array(resp)):
            rejected = item.get("rejected") if isinstance(item, dict) else None
            if not isinstance(rejected, str) or not rejected.strip():
                continue
            # 오답이 정답과 같으면 학습 신호가 없으므로 버린다
            if rejected.strip() == s["answer"].strip():
                continue
            base_id = s["meta"].get("id", f"sft-{i:04d}")
            pairs.append(make_pair(s["prompt"], s["answer"], rejected, {
                "id": f"pref-{base_id}-{j}",
                "source": s["meta"].get("source"),
                "chunk_id": s["meta"].get("chunk_id"),
                "kind": "distortion",
                "origin": llm.origin,
            }))
            made += 1
        print(f"[{i}/{len(samples)}] 쌍 {made}개 (누적 {len(pairs)})")
    return pairs


def gen_from_chunks(llm, chunks: list, per_chunk: int) -> list:
    pairs = []
    for i, c in enumerate(chunks, 1):
        try:
            resp = llm(CHUNK_PROMPT.format(n=per_chunk, chunk=c["text"]))
        except Exception as e:  # noqa: BLE001
            print(f"[!] 청크 {i}/{len(chunks)} 실패: {e}")
            continue
        made = 0
        for j, item in enumerate(extract_json_array(resp)):
            if not isinstance(item, dict):
                continue
            q, ch, rj = item.get("question"), item.get("chosen"), item.get("rejected")
            if not (q and ch and rj) or str(ch).strip() == str(rj).strip():
                continue
            pairs.append(make_pair([{"role": "user", "content": str(q).strip()}],
                                   str(ch), str(rj), {
                "id": f"pref-{c['chunk_id']}-{j}",
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "kind": "distortion",
                "origin": llm.origin,
            }))
            made += 1
        print(f"[{i}/{len(chunks)}] 쌍 {made}개 (누적 {len(pairs)})")
    return pairs


def gen_refusals(llm, chunks: list, per_chunk: int) -> list:
    pairs = []
    for i, c in enumerate(chunks, 1):
        try:
            resp = llm(REFUSAL_PROMPT.format(n=per_chunk, chunk=c["text"]))
        except Exception as e:  # noqa: BLE001
            print(f"[!] 거절쌍 청크 {i}/{len(chunks)} 실패: {e}")
            continue
        made = 0
        for j, item in enumerate(extract_json_array(resp)):
            if not isinstance(item, dict):
                continue
            q, ref, hal = (item.get("question"), item.get("refusal"),
                           item.get("hallucination"))
            if not (q and ref and hal):
                continue
            pairs.append(make_pair([{"role": "user", "content": str(q).strip()}],
                                   str(ref), str(hal), {
                "id": f"refuse-{c['chunk_id']}-{j}",
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "kind": "refusal",
                "origin": llm.origin,
            }))
            made += 1
        print(f"[거절 {i}/{len(chunks)}] 쌍 {made}개 (누적 {len(pairs)})")
    return pairs


def pairs_to_kto(pairs: list) -> list:
    """선호 쌍을 KTO의 개별 라벨 형식으로 분해."""
    rows = []
    for p in pairs:
        for key, label in (("chosen", True), ("rejected", False)):
            meta = dict(p["meta"])
            meta["id"] = f"{meta.get('id', 'kto')}-{'pos' if label else 'neg'}"
            rows.append({
                "prompt": p["prompt"],
                "completion": p[key],
                "label": label,
                "meta": meta,
            })
    return rows


# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["from-sft", "from-chunks"],
                        default="from-sft")
    parser.add_argument("--per-sample", type=int, default=1,
                        help="from-sft 모드에서 QA 하나당 만들 오답 수")
    parser.add_argument("--per-chunk", type=int, default=3,
                        help="from-chunks 모드에서 청크당 만들 세트 수")
    parser.add_argument("--refusals-per-chunk", type=int, default=0,
                        help="청크당 생성할 '자료에 없는 질문' 거절 쌍 수 (0이면 생략)")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--no-kto", action="store_true",
                        help="KTO 데이터셋 파생 생략")
    parser.add_argument("--append", action="store_true",
                        help="기존 파일에 이어 쓰기 (기본은 덮어쓰기)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_url, api_key, model = resolve_env()

    def llm(prompt: str) -> str:
        # 오답 생성은 다양성이 필요하므로 QA 생성보다 온도를 조금 높인다
        return call_llm(base_url, api_key, model, prompt, temperature=0.7)
    llm.origin = f"llm:{model}"

    chunks = load_chunks(cfg["data"]["cpt_dataset"])
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    pairs = []
    if args.mode == "from-sft":
        samples = load_sft_samples(cfg["data"]["sft_dataset"])
        if not samples:
            raise SystemExit(
                "SFT 데이터셋이 비어 있어 from-sft 모드를 쓸 수 없습니다.\n"
                "--mode from-chunks 를 쓰거나 먼저 SFT 데이터를 준비하세요.")
        print(f"[i] SFT 샘플 {len(samples)}개에서 오답 생성")
        pairs += gen_from_sft(llm, samples, args.per_sample)
    else:
        print(f"[i] 청크 {len(chunks)}개에서 질문·정답·오답 생성")
        pairs += gen_from_chunks(llm, chunks, args.per_chunk)

    if args.refusals_per_chunk > 0:
        print(f"[i] 거절 쌍 생성 (청크당 {args.refusals_per_chunk}개)")
        pairs += gen_refusals(llm, chunks, args.refusals_per_chunk)

    if not pairs:
        raise SystemExit("[!] 생성된 쌍이 없습니다. LLM 응답 형식을 확인하세요.")

    mode = "a" if args.append else "w"
    pref_path = cfg["data"]["pref_dataset"]
    os.makedirs(os.path.dirname(pref_path), exist_ok=True)
    with open(pref_path, mode, encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[OK] 선호 쌍 {len(pairs)}개 → {pref_path}")

    if not args.no_kto:
        kto_rows = pairs_to_kto(pairs)
        kto_path = cfg["data"]["kto_dataset"]
        with open(kto_path, mode, encoding="utf-8") as f:
            for r in kto_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_pos = sum(1 for r in kto_rows if r["label"])
        print(f"[OK] KTO 샘플 {len(kto_rows)}개 (좋음 {n_pos} / 나쁨 "
              f"{len(kto_rows) - n_pos}) → {kto_path}")

    print("\n[!] 생성된 rejected는 반드시 사람이 표본 검수하세요. "
          "오답이 우연히 사실이면 모델에게 거짓을 가르치게 됩니다.\n"
          "    자세한 내용: docs/preference_tuning.md")


if __name__ == "__main__":
    main()
