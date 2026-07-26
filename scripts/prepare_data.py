"""
1단계: 데이터 준비
- data/raw/ 안의 원문 문서(.txt, .md)를 청크로 분할 → CPT용 JSONL 생성
- QA 쌍 JSONL(있다면)을 표준 형식으로 정리 → SFT용 JSONL 생성
- 원문/QA가 하나도 없으면 샘플 데이터를 자동 생성 (파이프라인 동작 확인용)

사용법:
    python scripts/prepare_data.py [--config configs/config.yaml]

입력 형식:
    data/raw/*.txt, *.md          : 도메인 원문 문서
    data/raw/qa_*.jsonl           : (선택) QA 쌍. 아래 형식을 모두 자동 인식
        {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}  (권장)
        {"conversations": [{"from": "human", "value": ...}, ...]}          (ShareGPT)
        {"instruction": "...", "input": "...", "output": "..."}            (Alpaca)
        {"question": "...", "answer": "..."} / {"prompt": ..., "response": ...}

출력 (각 줄에 meta 필드가 함께 붙습니다 — 의미와 활용법은 docs/meta_info.md):
    data/processed/cpt_dataset.jsonl   : {"text": "...", "meta": {...}}
    data/processed/sft_dataset.jsonl   : {"messages": [...], "meta": {...}}
"""
import argparse
import glob
import json
import os

from common import content_hash, load_config, messages_hash, slugify

# ---------------------------------------------------------------
# 샘플 데이터 (원문이 없을 때 파이프라인 검증용으로 생성됨)
# 실제 사용 시에는 data/raw/에 자신의 도메인 문서를 넣으면 됩니다.
# ---------------------------------------------------------------
SAMPLE_DOCUMENT = """\
[가상 게임 '아스테리아 연대기' 세계관 설정집]

## 1. 세계관 개요
아스테리아 연대기는 부유 대륙 '아스테리아'를 배경으로 하는 판타지 세계이다.
아스테리아 대륙은 고대 문명 '엘드라'가 남긴 마력 결정 '코어스톤'의 힘으로
하늘에 떠 있으며, 코어스톤의 마력이 약해지면 대륙 일부가 추락하는
'낙토 현상'이 발생한다.

## 2. 주요 세력
### 2.1 셀레스티아 왕국
대륙 중앙부를 지배하는 최대 왕국. 현 국왕은 '아리안 셀레스티아 7세'이며,
코어스톤 관리 기관인 '성탑 기사단'을 운영한다. 수도는 '루멘하임'.

### 2.2 흑요 상회
대륙 남부 자유도시 연합을 실질 지배하는 상인 길드. 수장은 '베라 무어게이트'.
코어스톤 파편의 밀거래로 부를 축적했으며, 셀레스티아 왕국과 긴장 관계에 있다.

### 2.3 서리엄니 부족연합
북부 빙설 지대의 수렵 부족 연합. 대족장 '카르가 서리엄니'가 이끌며,
코어스톤을 신성한 '하늘의 심장'으로 숭배하여 채굴 자체를 금기시한다.

## 3. 핵심 자원: 코어스톤
코어스톤은 엘드라 문명이 제작한 마력 결정으로, 등급은 총 5단계
(1등급 '티끌'부터 5등급 '심장'까지)로 나뉜다. 5등급 코어스톤은 대륙 전체에
단 세 개만 존재하며, 각각 루멘하임 대성탑, 남부 해구, 북부 빙하 아래에 있다.

## 4. 주요 사건 연표
- 제국력 0년: 엘드라 문명 붕괴, 대륙 부유 시작
- 제국력 312년: 셀레스티아 왕국 건국
- 제국력 847년: 1차 낙토 현상 — 동부 '카엘라 지방' 추락
- 제국력 903년: 흑요 상회 결성
- 제국력 1024년(현재): 코어스톤 마력 감소 징후 재관측, 이야기의 시작점
"""

def _turn(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


SAMPLE_QA = [
    _turn("아스테리아 대륙이 하늘에 떠 있는 이유는 무엇인가요?",
          "아스테리아 대륙은 고대 문명 엘드라가 남긴 마력 결정 '코어스톤'의 힘으로 하늘에 떠 있습니다. 코어스톤의 마력이 약해지면 대륙 일부가 추락하는 '낙토 현상'이 발생합니다."),
    _turn("셀레스티아 왕국의 현재 국왕은 누구인가요?",
          "셀레스티아 왕국의 현 국왕은 아리안 셀레스티아 7세입니다. 왕국은 대륙 중앙부를 지배하는 최대 왕국으로, 수도는 루멘하임이며 코어스톤 관리 기관인 성탑 기사단을 운영합니다."),
    _turn("코어스톤의 등급 체계를 설명해주세요.",
          "코어스톤은 총 5단계 등급으로 나뉩니다. 1등급 '티끌'부터 5등급 '심장'까지 있으며, 최고 등급인 5등급 코어스톤은 대륙 전체에 단 세 개만 존재합니다. 각각 루멘하임 대성탑, 남부 해구, 북부 빙하 아래에 있습니다."),
    _turn("흑요 상회는 어떤 조직인가요?",
          "흑요 상회는 아스테리아 대륙 남부 자유도시 연합을 실질적으로 지배하는 상인 길드입니다. 수장은 베라 무어게이트이며, 코어스톤 파편의 밀거래로 부를 축적했고 셀레스티아 왕국과 긴장 관계에 있습니다."),
    _turn("1차 낙토 현상은 언제 일어났고 어떤 피해가 있었나요?",
          "1차 낙토 현상은 제국력 847년에 발생했으며, 동부 카엘라 지방이 추락하는 피해가 있었습니다."),
    _turn("서리엄니 부족연합이 코어스톤 채굴을 금지하는 이유는?",
          "서리엄니 부족연합은 코어스톤을 신성한 '하늘의 심장'으로 숭배하기 때문에 채굴 자체를 금기시합니다. 이들은 북부 빙설 지대의 수렵 부족 연합으로, 대족장 카르가 서리엄니가 이끌고 있습니다."),
]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """문단 경계를 존중하며 텍스트를 청크로 분할."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
            # 문단 하나가 chunk_size보다 크면 강제 분할
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para
    if current:
        chunks.append(current)

    # 청크 간 overlap 적용 (앞 청크 끝부분을 다음 청크 앞에 붙임)
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(tail + "\n" + chunks[i])
        chunks = overlapped
    return chunks


# ShareGPT 계열의 화자 표기 → OpenAI role
SHAREGPT_ROLES = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
    "system": "system",
}
VALID_ROLES = {"system", "user", "assistant"}


def _clean_messages(messages: list) -> list | None:
    """role/content를 검증·정리. 유효하지 않으면 None."""
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            return None
        cleaned.append({"role": role, "content": content.strip()})

    # system은 맨 앞에만, 마지막은 assistant여야 학습 대상이 존재
    if any(m["role"] == "system" for m in cleaned[1:]):
        return None
    if not cleaned or cleaned[-1]["role"] != "assistant":
        return None
    if not any(m["role"] == "user" for m in cleaned):
        return None
    return cleaned


def normalize_qa(item: dict) -> dict | None:
    """여러 QA 형식을 OpenAI messages 형식으로 통일.

    지원: messages / conversations(ShareGPT) / Alpaca(instruction+input+output)
          / question+answer / prompt+response
    """
    if not isinstance(item, dict):
        return None

    # 1) 이미 OpenAI messages 형식
    if isinstance(item.get("messages"), list):
        messages = _clean_messages(item["messages"])
        return {"messages": messages} if messages else None

    # 2) ShareGPT conversations 형식
    conv = item.get("conversations") or item.get("conversation")
    if isinstance(conv, list):
        converted = []
        for turn in conv:
            if not isinstance(turn, dict):
                return None
            role = SHAREGPT_ROLES.get(str(turn.get("from", "")).lower())
            content = turn.get("value")
            if not role or not isinstance(content, str):
                return None
            converted.append({"role": role, "content": content})
        messages = _clean_messages(converted)
        return {"messages": messages} if messages else None

    # 3) 평면 형식 (Alpaca / question-answer / prompt-response)
    q = item.get("instruction") or item.get("question") or item.get("prompt")
    a = item.get("output") or item.get("answer") or item.get("response")
    if not (isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()):
        return None

    messages = []
    system = item.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})

    # Alpaca의 input 필드는 질문에 딸린 참고 자료 → user 메시지에 합침
    user_content = q.strip()
    extra = item.get("input") or item.get("context")
    if isinstance(extra, str) and extra.strip():
        user_content = f"{user_content}\n\n{extra.strip()}"

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": a.strip()})
    return {"messages": messages}


def build_qa_meta(raw: dict, messages: list, path: str, lineno: int) -> dict:
    """SFT 샘플의 메타정보 생성. 입력에 meta가 이미 있으면 그쪽을 우선한다.

    (필드별 의미와 활용법은 docs/meta_info.md 참고)
    """
    incoming = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    meta = {
        "id": f"{slugify(path)}-{lineno:04d}",
        "source": os.path.basename(path),
        "origin": "human",          # 사람이 작성한 QA. 자동 생성분은 generate_qa.py가 llm:*로 표기
        "hash": messages_hash(messages),
    }
    # 입력 파일이 이미 갖고 있던 메타(출처 청크, 원본 ID 등)를 보존
    meta.update({k: v for k, v in incoming.items() if v is not None})
    # hash는 정규화 후 내용 기준이어야 하므로 항상 다시 계산
    meta["hash"] = messages_hash(messages)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_dir = cfg["data"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg["data"]["cpt_dataset"]), exist_ok=True)

    # ---------- 원문 수집 ----------
    doc_files = sorted(
        glob.glob(os.path.join(raw_dir, "*.txt")) +
        glob.glob(os.path.join(raw_dir, "*.md"))
    )
    qa_files = sorted(glob.glob(os.path.join(raw_dir, "qa_*.jsonl")))

    # 아무것도 없으면 샘플 생성
    if not doc_files and not qa_files:
        print("[i] data/raw/에 문서가 없어 샘플 데이터를 생성합니다.")
        sample_doc = os.path.join(raw_dir, "sample_worldbook.md")
        with open(sample_doc, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DOCUMENT)
        sample_qa = os.path.join(raw_dir, "qa_sample.jsonl")
        with open(sample_qa, "w", encoding="utf-8") as f:
            for item in SAMPLE_QA:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        doc_files, qa_files = [sample_doc], [sample_qa]

    # ---------- CPT 데이터셋 생성 ----------
    # meta.id는 SFT 쪽에서 meta.chunk_id로 참조되므로 안정적으로 유지되어야 한다.
    n_chunks = 0
    with open(cfg["data"]["cpt_dataset"], "w", encoding="utf-8") as out:
        for path in doc_files:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            stem = slugify(path)
            for idx, chunk in enumerate(chunk_text(text, cfg["data"]["chunk_size"],
                                                   cfg["data"]["chunk_overlap"])):
                record = {
                    "text": chunk,
                    "meta": {
                        "id": f"{stem}-{idx:04d}",
                        "source": os.path.basename(path),
                        "chunk_index": idx,
                        "hash": content_hash(chunk),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"[OK] CPT 데이터셋: {n_chunks}개 청크 → {cfg['data']['cpt_dataset']}")

    # ---------- SFT 데이터셋 생성 ----------
    n_qa, n_skipped = 0, 0
    with open(cfg["data"]["sft_dataset"], "w", encoding="utf-8") as out:
        for path in qa_files:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"[!] {os.path.basename(path)}:{lineno} JSON 파싱 실패 — 건너뜀")
                        n_skipped += 1
                        continue
                    item = normalize_qa(raw)
                    if item:
                        item["meta"] = build_qa_meta(raw, item["messages"],
                                                     path, lineno)
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        n_qa += 1
                    else:
                        print(f"[!] {os.path.basename(path)}:{lineno} 인식할 수 없는 형식 — 건너뜀")
                        n_skipped += 1
    if n_qa:
        print(f"[OK] SFT 데이터셋: {n_qa}개 대화 → {cfg['data']['sft_dataset']}"
              + (f" ({n_skipped}개 건너뜀)" if n_skipped else ""))
    else:
        print("[!] QA 파일(qa_*.jsonl)이 없습니다. SFT 단계를 건너뛰거나 "
              "generate_qa.py로 QA를 생성하세요.")


if __name__ == "__main__":
    main()
