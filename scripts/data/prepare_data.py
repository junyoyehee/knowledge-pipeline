"""
1단계: 데이터 준비
- data/raw/ 안의 원문 문서(.txt, .md)를 청크로 분할 → CPT용 JSONL 생성
- QA 쌍 JSONL(있다면)을 표준 형식으로 정리 → SFT용 JSONL 생성
- 원문/QA가 하나도 없으면 샘플 데이터를 자동 생성 (파이프라인 동작 확인용)

사용법:
    python -m scripts.data.prepare_data [--config configs/config.yaml]

입력 형식:
    data/raw/*.txt, *.md          : 도메인 원문 문서
    data/raw/qa_*.jsonl           : (선택) QA 쌍. 아래 형식을 모두 자동 인식
        {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}  (권장)
        {"conversations": [{"from": "human", "value": ...}, ...]}          (ShareGPT)
        {"instruction": "...", "input": "...", "output": "..."}            (Alpaca)
        {"question": "...", "answer": "..."} / {"prompt": ..., "response": ...}
    data/raw/tools_*.jsonl        : (선택) 툴 호출 학습 데이터. docs/tool_calling.md
        {"tools": [<함수 스키마>], "messages": [... tool_calls / role:tool ...]}
    data/raw/react_*.jsonl        : (선택) 추론형(ReAct) 트레이스. docs/react.md
        {"question": "...", "steps": [{"thought","action","observation"}], "final_answer": "..."}
    data/raw/planact_*.jsonl      : (선택) 계획-실행(plan-and-execute) 궤적. docs/planact.md
        {"tools": [...], "goal": "...", "plan": [...],
         "steps": [{"tool","arguments","observation"}], "final_answer": "..."}

출력 (각 줄에 meta 필드가 함께 붙습니다 — 의미와 활용법은 docs/meta_info.md):
    data/processed/cpt_dataset.jsonl   : {"text": "...", "meta": {...}}
    data/processed/sft_dataset.jsonl   : {"messages": [...], "meta": {...}}
    data/processed/tool_dataset.jsonl  : {"messages": [...], "tools": "<JSON>", "meta": {...}}
    data/processed/plan_dataset.jsonl  : {"messages": [...], "meta": {"n_steps": N, ...}}
    data/processed/react_dataset.jsonl : {"messages": [...], "meta": {"n_steps": N, ...}}
    data/processed/planact_dataset.jsonl : {"messages": [...], "tools": "<JSON>", "meta": {...}}

계획수립(planning) 입력 형식 (data/raw/plans_*.jsonl):
    {"goal": "목표", "steps": ["1단계", "2단계", ...]}          (권장)
    {"instruction": "목표", "plan": ["..."], "context": "..."}  (alias 자동 인식)
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}  (이미 대화형)

추론형(ReAct) 입력 형식 (data/raw/react_*.jsonl):
    {"question": "...", "steps": [{"thought": "...", "action": "search[...]", "observation": "..."}],
     "final_answer": "..."}                                    (권장)
    action은 문자열("tool[input]") 또는 {"tool": "...", "input": ...} 모두 인식
"""
import argparse
import glob
import json
import os
import re

from scripts.lib.common import (content_hash, load_config, messages_hash, slugify,
                    tool_sample_hash, add_report_arg, write_report)

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


# ---------------------------------------------------------------
# 샘플 툴 호출 데이터 (tools_*.jsonl이 없을 때 파이프라인 검증용으로 생성됨)
# 실제 사용 시에는 자신의 함수 스키마와 대화를 tools_*.jsonl에 넣으세요.
# 자세한 형식은 docs/tool_calling.md 참고.
# ---------------------------------------------------------------
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_corestone",
            "description": "코어스톤 등급 정보를 조회한다. 등급별 이름·희귀도·존재 위치를 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grade": {"type": "integer",
                              "description": "조회할 코어스톤 등급 (1~5)"},
                },
                "required": ["grade"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_faction_info",
            "description": "아스테리아 세력(왕국·상회·부족연합)의 상세 정보를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "세력 이름. 예: 셀레스티아 왕국, 흑요 상회, 서리엄니 부족연합"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "제국력 연표에서 사건을 조회한다. 기간을 지정하면 그 범위의 사건만 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "era_start": {"type": "integer",
                                  "description": "조회 시작 제국력(연). 생략 시 처음부터"},
                    "era_end": {"type": "integer",
                                "description": "조회 끝 제국력(연). 생략 시 끝까지"},
                },
                "required": [],
            },
        },
    },
]


def _tool_turn(user: str, name: str, arguments: dict,
               tool_result: str, final: str) -> dict:
    """단일 툴 호출 대화(요청 → 호출 → 결과 → 최종 답변) 샘플 생성."""
    return {
        "tools": SAMPLE_TOOLS,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": name, "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
            {"role": "assistant", "content": final},
        ],
    }


SAMPLE_TOOL_SAMPLES = [
    _tool_turn(
        "5등급 코어스톤이 몇 개나 있고 어디에 있는지 알려줘",
        "lookup_corestone", {"grade": 5},
        '{"grade": 5, "name": "심장", "count": 3, '
        '"locations": ["루멘하임 대성탑", "남부 해구", "북부 빙하 아래"]}',
        "5등급 코어스톤 '심장'은 대륙 전체에 단 세 개만 존재합니다. "
        "각각 루멘하임 대성탑, 남부 해구, 북부 빙하 아래에 있습니다."),
    _tool_turn(
        "흑요 상회가 어떤 조직인지 정리해줘",
        "get_faction_info", {"name": "흑요 상회"},
        '{"name": "흑요 상회", "type": "상인 길드", "leader": "베라 무어게이트", '
        '"base": "남부 자유도시 연합", "note": "코어스톤 파편 밀거래, 셀레스티아 왕국과 긴장 관계"}',
        "흑요 상회는 남부 자유도시 연합을 실질 지배하는 상인 길드입니다. "
        "수장은 베라 무어게이트이며, 코어스톤 파편 밀거래로 부를 쌓아 "
        "셀레스티아 왕국과 긴장 관계에 있습니다."),
    _tool_turn(
        "제국력 800년부터 1000년 사이에 무슨 일이 있었어?",
        "list_events", {"era_start": 800, "era_end": 1000},
        '{"events": [{"era": 847, "name": "1차 낙토 현상", "detail": "동부 카엘라 지방 추락"}, '
        '{"era": 903, "name": "흑요 상회 결성"}]}',
        "제국력 800~1000년 사이에는 두 사건이 있었습니다. 847년 1차 낙토 현상으로 "
        "동부 카엘라 지방이 추락했고, 903년에는 흑요 상회가 결성되었습니다."),
]


# ---------------------------------------------------------------
# 샘플 계획수립 데이터 (plans_*.jsonl이 없을 때 파이프라인 검증용으로 생성됨)
# 실제 사용 시에는 자신의 목표·단계를 plans_*.jsonl에 넣으세요. 형식은 docs/planning.md.
# ---------------------------------------------------------------
SAMPLE_PLANS = [
    {"goal": "코어스톤 마력 감소 징후를 조사할 계획을 세워줘",
     "steps": [
         "관측 데이터 수집: 루멘하임 대성탑·남부 해구·북부 빙하의 5등급 코어스톤 마력 수치를 최근 관측치와 비교한다",
         "이해관계자 확인: 성탑 기사단, 흑요 상회, 서리엄니 부족연합의 최근 동향과 보고를 취합한다",
         "원인 가설 수립: 제국력 847년 카엘라 추락 전례와 비교해 마력 감소 패턴의 원인 가설을 정리한다",
         "대응 우선순위 결정: 추락 위험이 큰 지역부터 마력 보강과 주민 대피 순서를 정한다",
         "보고 및 재관측: 셀레스티아 왕국에 보고하고 정기 재관측 일정을 수립한다"]},
    {"goal": "흑요 상회와의 협상을 준비하는 계획을 세워줘",
     "steps": [
         "목표 정의: 코어스톤 파편 밀거래 중단과 정보 공유라는 협상 목표를 명확히 한다",
         "상대 분석: 수장 베라 무어게이트의 이해관계와 남부 자유도시 연합의 요구를 파악한다",
         "지렛대 확보: 왕국이 제시할 교역 이권과 제재 카드를 정리한다",
         "협상안 작성: 최소 합의선과 최대 요구안을 담은 단계별 제안을 준비한다",
         "후속 조치: 합의 이행 점검 방법과 결렬 시 대안을 마련한다"]},
    {"goal": "신입 성탑 기사에게 코어스톤 기초를 가르칠 교육 계획을 세워줘",
     "steps": [
         "개요 교육: 코어스톤의 정의와 대륙 부유 원리, 낙토 현상을 설명한다",
         "등급 체계 학습: 1등급 티끌부터 5등급 심장까지 등급과 희귀도를 익힌다",
         "현장 견학: 루멘하임 대성탑의 5등급 코어스톤 관리 절차를 참관한다",
         "실무 훈련: 마력 수치 측정과 이상 징후 보고 절차를 실습한다",
         "평가: 등급 판별과 비상 대응 시나리오로 이해도를 점검한다"]},
]


# ---------------------------------------------------------------
# 샘플 추론형(ReAct) 데이터 (react_*.jsonl이 없을 때 파이프라인 검증용으로 생성됨)
# 실제 사용 시에는 자신의 추론 트레이스를 react_*.jsonl에 넣으세요. 형식은 docs/react.md.
# ---------------------------------------------------------------
SAMPLE_REACT = [
    {"question": "5등급 코어스톤은 몇 개이고 어디에 있어?",
     "steps": [
         {"thought": "코어스톤 등급별 정보를 찾아야 한다. 등급 조회로 시작하자.",
          "action": "lookup_corestone[5]",
          "observation": "5등급 '심장', 총 3개, 위치: 루멘하임 대성탑·남부 해구·북부 빙하 아래"},
         {"thought": "개수와 위치를 모두 확인했다. 답할 수 있다.",
          "action": "finish[]",
          "observation": "(종료)"}],
     "final_answer": "5등급 코어스톤 '심장'은 총 3개이며, 루멘하임 대성탑·남부 해구·북부 빙하 아래에 있습니다."},
    {"question": "1차 낙토 현상이 흑요 상회 결성보다 먼저야?",
     "steps": [
         {"thought": "두 사건의 연도를 각각 확인해 비교해야 한다. 먼저 낙토 현상 연도를 찾자.",
          "action": "list_events[낙토]",
          "observation": "제국력 847년: 1차 낙토 현상 (동부 카엘라 지방 추락)"},
         {"thought": "이번엔 흑요 상회 결성 연도를 찾자.",
          "action": "list_events[흑요 상회]",
          "observation": "제국력 903년: 흑요 상회 결성"},
         {"thought": "847년 < 903년이므로 낙토 현상이 먼저다.",
          "action": "finish[]",
          "observation": "(종료)"}],
     "final_answer": "네. 1차 낙토 현상은 제국력 847년, 흑요 상회 결성은 903년이므로 낙토 현상이 56년 먼저 일어났습니다."},
]


# ---------------------------------------------------------------
# 샘플 계획-실행(plan-and-execute) 데이터 (planact_*.jsonl이 없을 때 생성됨)
# 목표 → 계획 → 각 단계를 tool_call로 실행 → 관찰 종합 → 최종 답변.
# SAMPLE_TOOLS 카탈로그를 그대로 사용한다. 형식은 docs/planact.md.
# ---------------------------------------------------------------
SAMPLE_PLANACT = [
    {"tools": SAMPLE_TOOLS,
     "goal": "셀레스티아 왕국과 흑요 상회의 관계를 코어스톤 관점에서 정리해줘",
     "plan": [
         "셀레스티아 왕국의 세력 정보를 조회한다",
         "흑요 상회의 세력 정보를 조회한다",
         "두 세력의 코어스톤 관련 이해관계를 비교해 정리한다"],
     "steps": [
         {"tool": "get_faction_info", "arguments": {"name": "셀레스티아 왕국"},
          "observation": '{"name": "셀레스티아 왕국", "leader": "아리안 셀레스티아 7세", "org": "성탑 기사단", "note": "코어스톤 관리 기관 운영"}'},
         {"tool": "get_faction_info", "arguments": {"name": "흑요 상회"},
          "observation": '{"name": "흑요 상회", "leader": "베라 무어게이트", "note": "코어스톤 파편 밀거래, 왕국과 긴장 관계"}'}],
     "final_answer": "셀레스티아 왕국은 성탑 기사단을 통해 코어스톤을 공식 관리하는 반면, 흑요 상회는 코어스톤 파편을 밀거래합니다. 이 밀거래 때문에 두 세력은 긴장 관계에 있습니다."},
    {"tools": SAMPLE_TOOLS,
     "goal": "5등급 코어스톤 현황을 확인하고 가장 시급히 점검할 위치를 제안해줘",
     "plan": [
         "5등급 코어스톤의 개수와 위치를 조회한다",
         "위치 목록을 바탕으로 우선 점검 대상을 제안한다"],
     "steps": [
         {"tool": "lookup_corestone", "arguments": {"grade": 5},
          "observation": '{"grade": 5, "name": "심장", "count": 3, "locations": ["루멘하임 대성탑", "남부 해구", "북부 빙하 아래"]}'}],
     "final_answer": "5등급 '심장'은 루멘하임 대성탑·남부 해구·북부 빙하 아래 세 곳에 있습니다. 수도 방어와 직결된 루멘하임 대성탑을 가장 먼저 점검할 것을 제안합니다."},
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


# ---------------------------------------------------------------
# 툴 호출(function calling) 데이터 정규화
#   - tools: 함수 스키마 리스트를 OpenAI 형식({"type":"function","function":{...}})으로 통일
#   - messages: assistant의 tool_calls, role:"tool" 결과를 검증
#   - arguments/tools는 저장 시 JSON 문자열로 직렬화한다.
#     함수마다 다른 인자 구조가 datasets(Arrow)의 스키마 추론을 깨뜨리기 때문.
# ---------------------------------------------------------------
TOOL_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _normalize_arguments(args) -> str | None:
    """tool_call의 arguments를 JSON 문자열로 통일. 실패하면 None."""
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _normalize_tools(tools: list) -> list | None:
    """함수 스키마 리스트를 OpenAI 형식으로 정규화. 유효하지 않으면 None."""
    if not isinstance(tools, list) or not tools:
        return None
    normalized = []
    for t in tools:
        if not isinstance(t, dict):
            return None
        fn = t["function"] if isinstance(t.get("function"), dict) else \
            {k: v for k, v in t.items() if k != "type"}
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        normalized.append({"type": "function", "function": fn})
    return normalized


def _clean_tool_calls(tool_calls: list) -> list | None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    cleaned = []
    for i, tc in enumerate(tool_calls, 1):
        if not isinstance(tc, dict):
            return None
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        arguments = _normalize_arguments(fn.get("arguments", {}))
        if arguments is None:
            return None
        cleaned.append({
            "id": str(tc.get("id") or f"call_{i}"),
            "type": "function",
            "function": {"name": name.strip(), "arguments": arguments},
        })
    return cleaned


def _clean_tool_messages(messages: list) -> list | None:
    """툴 호출 대화를 검증·정리. 유효하지 않으면 None.

    규칙: system은 맨 앞에만 / user 최소 1개 / 마지막은 assistant /
          tool 결과는 반드시 앞선 tool_call 뒤에 / 툴 호출이 최소 1개 존재.
    """
    if not isinstance(messages, list) or not messages:
        return None
    cleaned = []
    seen_tool_call = False
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            return None
        role = m.get("role")
        if role not in TOOL_VALID_ROLES:
            return None

        if role == "assistant":
            content = m.get("content")
            content = content.strip() if isinstance(content, str) else ""
            tool_calls = m.get("tool_calls")
            if tool_calls is not None:
                tool_calls = _clean_tool_calls(tool_calls)
                if tool_calls is None:
                    return None
                seen_tool_call = True
            if not content and not tool_calls:
                return None  # 내용도 호출도 없는 빈 assistant 메시지
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            cleaned.append(msg)
        elif role == "tool":
            if not seen_tool_call:
                return None  # tool 결과가 호출보다 먼저 나올 수 없음
            content = m.get("content")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            if not isinstance(content, str) or not content.strip():
                return None
            msg = {"role": "tool", "content": content.strip()}
            if m.get("tool_call_id") is not None:
                msg["tool_call_id"] = str(m["tool_call_id"])
            cleaned.append(msg)
        else:  # system / user
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                return None
            cleaned.append({"role": role, "content": content.strip()})

    if any(m["role"] == "system" for m in cleaned[1:]):
        return None
    if not any(m["role"] == "user" for m in cleaned):
        return None
    if cleaned[-1]["role"] != "assistant":
        return None
    if not seen_tool_call:
        return None  # 툴 호출이 하나도 없으면 툴 학습 데이터가 아님 (qa로 넣으세요)
    return cleaned


def normalize_tool_sample(item: dict) -> dict | None:
    """툴 호출 샘플을 검증·정규화 → {"messages": [...], "tools": [...]} 또는 None."""
    if not isinstance(item, dict):
        return None
    tools = _normalize_tools(item.get("tools"))
    if not tools:
        return None
    messages = _clean_tool_messages(item.get("messages"))
    if not messages:
        return None
    return {"messages": messages, "tools": tools}


def build_tool_meta(raw: dict, tools: list, messages: list,
                    path: str, lineno: int) -> dict:
    """툴 호출 샘플의 메타정보 생성. 입력에 meta가 있으면 그쪽을 우선한다."""
    incoming = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    meta = {
        "id": f"{slugify(path)}-{lineno:04d}",
        "source": os.path.basename(path),
        "origin": "human",          # 자동 생성분은 generate_tool_calls.py가 llm:*로 표기
        "n_tools": len(tools),
    }
    meta.update({k: v for k, v in incoming.items() if v is not None})
    meta["hash"] = tool_sample_hash(tools, messages)
    return meta


# ---------------------------------------------------------------
# 계획수립(planning) 데이터 정규화
#   - goal(목표) + steps(단계 배열)를 받아 표준 번호 목록으로 렌더링
#   - 이미 messages 형식이면 그대로 검증만 통과시킨다
# ---------------------------------------------------------------
PLAN_MIN_STEPS = 2  # 단계가 1개뿐이면 '계획'이 아니므로 버린다


def _step_to_text(step) -> str | None:
    """단계 항목(문자열 또는 dict)을 한 줄 텍스트로 변환."""
    if isinstance(step, (int, float)):
        return str(step)
    if isinstance(step, str):
        return step.strip() or None
    if isinstance(step, dict):
        main = (step.get("step") or step.get("title") or step.get("name")
                or step.get("description") or step.get("content"))
        if not isinstance(main, str) or not main.strip():
            return None
        text = main.strip()
        detail = step.get("detail")
        if not detail and main is not step.get("description"):
            detail = step.get("description")
        if isinstance(detail, str) and detail.strip() and detail.strip() != text:
            text = f"{text} — {detail.strip()}"
        return text
    return None


def render_plan(steps: list, intro: str = None) -> str:
    """단계 배열을 표준 번호 목록으로 렌더링 (모델이 배울 계획 형식)."""
    lines = []
    if isinstance(intro, str) and intro.strip():
        lines.append(intro.strip())
        lines.append("")
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def _count_plan_steps(text: str) -> int:
    """이미 렌더링된 계획 텍스트에서 번호 매겨진 단계 수를 센다 (best-effort)."""
    return len(re.findall(r"(?m)^\s*\d+[.)]\s", text))


def normalize_plan_sample(item: dict) -> dict | None:
    """계획 샘플을 검증·정규화 → {"messages": [...], "n_steps": N} 또는 None."""
    if not isinstance(item, dict):
        return None

    # 1) 이미 messages 형식 (SFT용 검증기 재사용)
    if isinstance(item.get("messages"), list):
        messages = _clean_messages(item["messages"])
        if not messages:
            return None
        return {"messages": messages,
                "n_steps": _count_plan_steps(messages[-1]["content"])}

    # 2) goal + steps 구조
    goal = (item.get("goal") or item.get("instruction")
            or item.get("question") or item.get("prompt"))
    steps_raw = item.get("steps") or item.get("plan")
    if not (isinstance(goal, str) and goal.strip()) or not isinstance(steps_raw, list):
        return None
    steps = [t for t in (_step_to_text(s) for s in steps_raw) if t]
    if len(steps) < PLAN_MIN_STEPS:
        return None

    # 목표에 딸린 참고 자료(context/input)는 user 메시지에 합친다
    user_content = goal.strip()
    ctx = item.get("context") or item.get("input")
    if isinstance(ctx, str) and ctx.strip():
        user_content = f"{user_content}\n\n{ctx.strip()}"

    messages = []
    system = item.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant",
                     "content": render_plan(steps, item.get("intro"))})
    return {"messages": messages, "n_steps": len(steps)}


def build_plan_meta(raw: dict, messages: list, n_steps: int,
                    path: str, lineno: int) -> dict:
    """계획 샘플의 메타정보 생성. 입력에 meta가 있으면 그쪽을 우선한다."""
    incoming = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    meta = {
        "id": f"{slugify(path)}-{lineno:04d}",
        "source": os.path.basename(path),
        "origin": "human",          # 자동 생성분은 generate_plans.py가 llm:*로 표기
        "n_steps": n_steps,
    }
    meta.update({k: v for k, v in incoming.items() if v is not None})
    meta["hash"] = messages_hash(messages)
    return meta


# ---------------------------------------------------------------
# 추론형(ReAct) 데이터 정규화
#   - question + steps[{thought, action, observation}] + final_answer 를 받아
#     멀티턴 대화로 변환한다.
#   - Thought+Action은 assistant 턴, Observation은 user 턴으로 넣어
#     train_on_responses_only가 Observation을 자동 마스킹하도록 한다.
#     (모델이 관찰 결과를 지어내도록 학습하는 것을 방지)
# ---------------------------------------------------------------
REACT_MIN_STEPS = 1  # 최소 1회의 thought-action-observation이 있어야 ReAct다
# 트레이스 구조 라벨. 필요하면 이 값만 바꾸면 전체 렌더링이 따라간다.
REACT_LABELS = {"thought": "Thought", "action": "Action",
                "observation": "Observation", "final": "Final Answer"}


def _action_to_text(action) -> str | None:
    """action(문자열 또는 {tool,input})을 'tool[input]' 형태 한 줄로 변환."""
    if isinstance(action, str):
        return action.strip() or None
    if isinstance(action, dict):
        tool = action.get("tool") or action.get("name") or action.get("action")
        if not isinstance(tool, str) or not tool.strip():
            return None
        inp = action.get("input")
        if inp is None:
            inp = action.get("args", action.get("query", ""))
        if isinstance(inp, (dict, list)):
            inp = json.dumps(inp, ensure_ascii=False)
        return f"{tool.strip()}[{str(inp).strip()}]"
    return None


def _clean_react_steps(steps_raw) -> list | None:
    """steps 배열을 (thought, action, observation) 튜플 리스트로 정리."""
    if not isinstance(steps_raw, list):
        return None
    steps = []
    for s in steps_raw:
        if not isinstance(s, dict):
            return None
        thought = s.get("thought") or s.get("reasoning") or s.get("think")
        action = _action_to_text(s.get("action") or s.get("act"))
        obs = s.get("observation")
        if isinstance(obs, (dict, list)):
            obs = json.dumps(obs, ensure_ascii=False)
        if not (isinstance(thought, str) and thought.strip()) or not action:
            return None
        if not (isinstance(obs, str) and obs.strip()):
            return None
        steps.append((thought.strip(), action, obs.strip()))
    return steps


def normalize_react_sample(item: dict) -> dict | None:
    """ReAct 샘플을 검증·정규화 → {"messages": [...], "n_steps": N} 또는 None."""
    if not isinstance(item, dict):
        return None

    # 1) 이미 messages 형식 (SFT용 검증기 재사용, 단계 수는 Action 개수로 추정)
    if isinstance(item.get("messages"), list):
        messages = _clean_messages(item["messages"])
        if not messages:
            return None
        joined = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
        n = len(re.findall(rf"(?m)^{re.escape(REACT_LABELS['action'])}\s*:",
                           joined))
        return {"messages": messages, "n_steps": n}

    # 2) question + steps + final_answer 구조
    question = (item.get("question") or item.get("instruction")
               or item.get("task") or item.get("prompt"))
    final = item.get("final_answer") or item.get("answer") or item.get("final")
    steps = _clean_react_steps(item.get("steps") or item.get("trajectory")
                               or item.get("react"))
    if not (isinstance(question, str) and question.strip()):
        return None
    if not (isinstance(final, str) and final.strip()):
        return None
    if steps is None or len(steps) < REACT_MIN_STEPS:
        return None

    user_content = question.strip()
    ctx = item.get("context") or item.get("input")
    if isinstance(ctx, str) and ctx.strip():
        user_content = f"{user_content}\n\n{ctx.strip()}"

    L = REACT_LABELS
    messages = []
    system = item.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user_content})
    for thought, action, obs in steps:
        # assistant: Thought + Action (여기에 loss)
        messages.append({"role": "assistant",
                         "content": f"{L['thought']}: {thought}\n{L['action']}: {action}"})
        # user: Observation (마스킹됨)
        messages.append({"role": "user", "content": f"{L['observation']}: {obs}"})
    # 마지막 assistant: (선택) 마무리 Thought + Final Answer
    final_thought = item.get("final_thought")
    if isinstance(final_thought, str) and final_thought.strip():
        final_content = (f"{L['thought']}: {final_thought.strip()}\n"
                         f"{L['final']}: {final.strip()}")
    else:
        final_content = f"{L['final']}: {final.strip()}"
    messages.append({"role": "assistant", "content": final_content})

    return {"messages": messages, "n_steps": len(steps)}


def build_react_meta(raw: dict, messages: list, n_steps: int,
                     path: str, lineno: int) -> dict:
    """ReAct 샘플의 메타정보 생성. 입력에 meta가 있으면 그쪽을 우선한다."""
    incoming = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    meta = {
        "id": f"{slugify(path)}-{lineno:04d}",
        "source": os.path.basename(path),
        "origin": "human",          # 자동 생성분은 generate_react.py가 llm:*로 표기
        "n_steps": n_steps,
    }
    meta.update({k: v for k, v in incoming.items() if v is not None})
    meta["hash"] = messages_hash(messages)
    return meta


# ---------------------------------------------------------------
# 계획-실행(plan-and-execute) 데이터 정규화
#   - goal + plan(단계 목록) + steps(각 단계의 tool 실행) + final_answer 를 받아
#     하나의 멀티턴 궤적으로 조립한다:
#       user(goal) → assistant(계획 + 1단계 tool_call) → tool(관찰)
#                  → assistant(다음 tool_call) → tool(관찰) → ... → assistant(최종 답변)
#   - 조립 후 tool 데이터 검증기(normalize_tool_sample)를 그대로 통과시킨다.
#     저장 형식·arguments 문자열화·tool 결과 마스킹이 tool 단계와 완전히 동일해진다.
# ---------------------------------------------------------------
PLANACT_MIN_PLAN = 2  # 계획이 최소 2단계는 되어야 '계획'이라 부를 수 있다


def normalize_planact_sample(item: dict) -> dict | None:
    """계획-실행 샘플 → {"messages": [...], "tools": [...], "n_steps": N, "n_plan": M} 또는 None."""
    if not isinstance(item, dict):
        return None

    # 1) 이미 {messages, tools} 궤적이면 tool 검증기로 그대로 처리
    if isinstance(item.get("messages"), list) and item.get("tools") is not None:
        base = normalize_tool_sample(item)
        if not base:
            return None
        n_steps = sum(1 for m in base["messages"]
                      if m["role"] == "assistant" and m.get("tool_calls"))
        return {**base, "n_steps": n_steps, "n_plan": 0}

    # 2) goal + plan + steps 구조
    tools = _normalize_tools(item.get("tools"))
    if not tools:
        return None
    tool_names = {t["function"]["name"] for t in tools}

    goal = item.get("goal") or item.get("instruction") or item.get("question")
    final = item.get("final_answer") or item.get("answer") or item.get("final")
    plan_raw = item.get("plan") or item.get("plan_steps")
    exec_raw = item.get("steps") or item.get("actions") or item.get("executions")
    if not (isinstance(goal, str) and goal.strip()):
        return None
    if not (isinstance(final, str) and final.strip()):
        return None
    if not isinstance(plan_raw, list) or not isinstance(exec_raw, list) or not exec_raw:
        return None

    plan = [t for t in (_step_to_text(s) for s in plan_raw) if t]
    if len(plan) < PLANACT_MIN_PLAN:
        return None

    user_content = goal.strip()
    ctx = item.get("context") or item.get("input")
    if isinstance(ctx, str) and ctx.strip():
        user_content = f"{user_content}\n\n{ctx.strip()}"

    messages = [{"role": "user", "content": user_content}]
    plan_text = render_plan(plan, item.get("intro"))
    for idx, st in enumerate(exec_raw):
        if not isinstance(st, dict):
            return None
        tool = st.get("tool") or st.get("name")
        if tool not in tool_names:      # 카탈로그에 없는 함수는 거부
            return None
        arguments = st.get("arguments")
        if arguments is None:
            arguments = st.get("input", {})
        obs = st.get("observation")
        if isinstance(obs, (dict, list)):
            obs = json.dumps(obs, ensure_ascii=False)
        if not (isinstance(obs, str) and obs.strip()):
            return None
        # 첫 assistant 턴에만 계획을 담고, 이후 턴은 tool_call만
        content = plan_text if idx == 0 else ""
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [{"id": f"call_{idx + 1}", "type": "function",
                                         "function": {"name": tool, "arguments": arguments}}]})
        messages.append({"role": "tool", "tool_call_id": f"call_{idx + 1}",
                         "content": obs.strip()})
    messages.append({"role": "assistant", "content": final.strip()})

    # tool 데이터 검증기로 최종 검증·정규화 (arguments 문자열화 등)
    base = normalize_tool_sample({"tools": tools, "messages": messages})
    if not base:
        return None
    return {**base, "n_steps": len(exec_raw), "n_plan": len(plan)}


def build_planact_meta(raw: dict, tools: list, messages: list,
                       n_steps: int, n_plan: int, path: str, lineno: int) -> dict:
    """계획-실행 샘플의 메타정보 생성. 입력에 meta가 있으면 그쪽을 우선한다."""
    incoming = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    meta = {
        "id": f"{slugify(path)}-{lineno:04d}",
        "source": os.path.basename(path),
        "origin": "human",          # 자동 생성분은 generate_planact.py가 llm:*로 표기
        "n_tools": len(tools),
        "n_plan": n_plan,
        "n_steps": n_steps,
    }
    meta.update({k: v for k, v in incoming.items() if v is not None})
    meta["hash"] = tool_sample_hash(tools, messages)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    add_report_arg(parser)
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
    tool_files = sorted(glob.glob(os.path.join(raw_dir, "tools_*.jsonl")))
    plan_files = sorted(glob.glob(os.path.join(raw_dir, "plans_*.jsonl")))
    react_files = sorted(glob.glob(os.path.join(raw_dir, "react_*.jsonl")))
    planact_files = sorted(glob.glob(os.path.join(raw_dir, "planact_*.jsonl")))

    # 아무것도 없으면 샘플 생성 (원문 + QA + 툴 호출 + 계획수립 + ReAct + 계획-실행)
    if not (doc_files or qa_files or tool_files or plan_files or react_files
            or planact_files):
        print("[i] data/raw/에 문서가 없어 샘플 데이터를 생성합니다.")
        sample_doc = os.path.join(raw_dir, "sample_worldbook.md")
        with open(sample_doc, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DOCUMENT)
        sample_qa = os.path.join(raw_dir, "qa_sample.jsonl")
        with open(sample_qa, "w", encoding="utf-8") as f:
            for item in SAMPLE_QA:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        sample_tools = os.path.join(raw_dir, "tools_sample.jsonl")
        with open(sample_tools, "w", encoding="utf-8") as f:
            for item in SAMPLE_TOOL_SAMPLES:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        sample_plans = os.path.join(raw_dir, "plans_sample.jsonl")
        with open(sample_plans, "w", encoding="utf-8") as f:
            for item in SAMPLE_PLANS:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        sample_react = os.path.join(raw_dir, "react_sample.jsonl")
        with open(sample_react, "w", encoding="utf-8") as f:
            for item in SAMPLE_REACT:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        sample_planact = os.path.join(raw_dir, "planact_sample.jsonl")
        with open(sample_planact, "w", encoding="utf-8") as f:
            for item in SAMPLE_PLANACT:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        doc_files, qa_files = [sample_doc], [sample_qa]
        tool_files, plan_files = [sample_tools], [sample_plans]
        react_files, planact_files = [sample_react], [sample_planact]

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

    # ---------- 툴 호출 데이터셋 생성 (선택) ----------
    tool_out = cfg["data"].get("tool_dataset")
    if tool_files and tool_out:
        os.makedirs(os.path.dirname(tool_out), exist_ok=True)
        n_tool, n_tool_skipped = 0, 0
        with open(tool_out, "w", encoding="utf-8") as out:
            for path in tool_files:
                with open(path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "JSON 파싱 실패 — 건너뜀")
                            n_tool_skipped += 1
                            continue
                        item = normalize_tool_sample(raw)
                        if not item:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "인식할 수 없는 툴 호출 형식 — 건너뜀")
                            n_tool_skipped += 1
                            continue
                        record = {
                            "messages": item["messages"],
                            # tools는 함수마다 구조가 달라 JSON 문자열로 저장한다
                            "tools": json.dumps(item["tools"], ensure_ascii=False),
                            "meta": build_tool_meta(raw, item["tools"],
                                                    item["messages"], path, lineno),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_tool += 1
        if n_tool:
            print(f"[OK] 툴 호출 데이터셋: {n_tool}개 대화 → {tool_out}"
                  + (f" ({n_tool_skipped}개 건너뜀)" if n_tool_skipped else ""))
        else:
            print("[!] 유효한 툴 호출 샘플이 없습니다. docs/tool_calling.md의 형식을 "
                  "확인하세요.")
    elif tool_out:
        print("[i] 툴 호출 파일(tools_*.jsonl)이 없어 tool 단계 데이터는 만들지 "
              "않았습니다. generate_tool_calls.py로 생성하거나 tool 단계를 건너뛰세요.")

    # ---------- 계획수립 데이터셋 생성 (선택) ----------
    plan_out = cfg["data"].get("plan_dataset")
    if plan_files and plan_out:
        os.makedirs(os.path.dirname(plan_out), exist_ok=True)
        n_plan, n_plan_skipped = 0, 0
        with open(plan_out, "w", encoding="utf-8") as out:
            for path in plan_files:
                with open(path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "JSON 파싱 실패 — 건너뜀")
                            n_plan_skipped += 1
                            continue
                        item = normalize_plan_sample(raw)
                        if not item:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "인식할 수 없는 계획 형식 — 건너뜀")
                            n_plan_skipped += 1
                            continue
                        record = {
                            "messages": item["messages"],
                            "meta": build_plan_meta(raw, item["messages"],
                                                    item["n_steps"], path, lineno),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_plan += 1
        if n_plan:
            print(f"[OK] 계획수립 데이터셋: {n_plan}개 계획 → {plan_out}"
                  + (f" ({n_plan_skipped}개 건너뜀)" if n_plan_skipped else ""))
        else:
            print("[!] 유효한 계획 샘플이 없습니다. docs/planning.md의 형식을 확인하세요.")
    elif plan_out:
        print("[i] 계획 파일(plans_*.jsonl)이 없어 plan 단계 데이터는 만들지 "
              "않았습니다. generate_plans.py로 생성하거나 plan 단계를 건너뛰세요.")

    # ---------- 추론형(ReAct) 데이터셋 생성 (선택) ----------
    react_out = cfg["data"].get("react_dataset")
    if react_files and react_out:
        os.makedirs(os.path.dirname(react_out), exist_ok=True)
        n_react, n_react_skipped = 0, 0
        with open(react_out, "w", encoding="utf-8") as out:
            for path in react_files:
                with open(path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "JSON 파싱 실패 — 건너뜀")
                            n_react_skipped += 1
                            continue
                        item = normalize_react_sample(raw)
                        if not item:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "인식할 수 없는 ReAct 형식 — 건너뜀")
                            n_react_skipped += 1
                            continue
                        record = {
                            "messages": item["messages"],
                            "meta": build_react_meta(raw, item["messages"],
                                                     item["n_steps"], path, lineno),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_react += 1
        if n_react:
            print(f"[OK] ReAct 데이터셋: {n_react}개 트레이스 → {react_out}"
                  + (f" ({n_react_skipped}개 건너뜀)" if n_react_skipped else ""))
        else:
            print("[!] 유효한 ReAct 샘플이 없습니다. docs/react.md의 형식을 확인하세요.")
    elif react_out:
        print("[i] ReAct 파일(react_*.jsonl)이 없어 react 단계 데이터는 만들지 "
              "않았습니다. generate_react.py로 생성하거나 react 단계를 건너뛰세요.")

    # ---------- 계획-실행(plan-and-execute) 데이터셋 생성 (선택) ----------
    planact_out = cfg["data"].get("planact_dataset")
    if planact_files and planact_out:
        os.makedirs(os.path.dirname(planact_out), exist_ok=True)
        n_pa, n_pa_skipped = 0, 0
        with open(planact_out, "w", encoding="utf-8") as out:
            for path in planact_files:
                with open(path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "JSON 파싱 실패 — 건너뜀")
                            n_pa_skipped += 1
                            continue
                        item = normalize_planact_sample(raw)
                        if not item:
                            print(f"[!] {os.path.basename(path)}:{lineno} "
                                  "인식할 수 없는 계획-실행 형식 — 건너뜀")
                            n_pa_skipped += 1
                            continue
                        record = {
                            "messages": item["messages"],
                            # tools는 함수마다 구조가 달라 JSON 문자열로 저장 (tool 단계와 동일)
                            "tools": json.dumps(item["tools"], ensure_ascii=False),
                            "meta": build_planact_meta(raw, item["tools"],
                                                       item["messages"], item["n_steps"],
                                                       item["n_plan"], path, lineno),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_pa += 1
        if n_pa:
            print(f"[OK] 계획-실행 데이터셋: {n_pa}개 궤적 → {planact_out}"
                  + (f" ({n_pa_skipped}개 건너뜀)" if n_pa_skipped else ""))
        else:
            print("[!] 유효한 계획-실행 샘플이 없습니다. docs/planact.md의 형식을 확인하세요.")
    elif planact_out:
        print("[i] 계획-실행 파일(planact_*.jsonl)이 없어 planact 단계 데이터는 만들지 "
              "않았습니다. generate_planact.py로 생성하거나 planact 단계를 건너뛰세요.")

    # ---------- 구조화 리포트 (--report-json) ----------
    report_datasets = []
    for kind, key in (("cpt", "cpt_dataset"), ("sft", "sft_dataset"),
                      ("tool", "tool_dataset"), ("plan", "plan_dataset"),
                      ("react", "react_dataset"), ("planact", "planact_dataset")):
        path = cfg["data"].get(key)
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            report_datasets.append({"kind": kind, "count": n,
                                    "file": os.path.basename(path)})
    write_report(args.report_json, {"task": "prepare", "status": "ok",
                                    "datasets": report_datasets})


if __name__ == "__main__":
    main()
