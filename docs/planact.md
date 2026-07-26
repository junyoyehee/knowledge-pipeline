# 계획-실행(plan-and-execute) 학습 가이드

모델에게 **목표를 받아 (1) 계획을 세우고 → (2) 각 단계를 실제 함수 호출로 실행하며 →
(3) 관찰을 종합해 최종 답변까지** 내는 에이전트형 능력을 가르치는 별도 전용 단계입니다.

```
CPT (지식) ─► SFT (활용) ─► [planact] 계획-실행 학습 ─► 병합 ─► 테스트
                            목표 → 계획 → tool_call 실행 → 관찰 → 최종 답변
```

> **툴 호출 단계의 확장입니다.** 데이터 형식·저장 방식·학습 로직이 tool 단계와 동일하며,
> 차이는 **첫 assistant 턴에 '계획'이 실린다**는 점뿐입니다. 그래서 `train_planact.py`는
> `train_tool.py`의 학습 함수를 그대로 재사용합니다. 기본 모델 Qwen2.5는 tool 호출
> 템플릿을 지원합니다(→ [tool_calling.md](tool_calling.md) 참고).

---

## 1. 데이터 형식

`data/raw/planact_*.jsonl` 에 한 줄에 하나씩 넣습니다. **권장 형식은 `tools` +
`goal` + `plan` + `steps` + `final_answer`** 입니다.

```json
{
  "tools": [
    {"type": "function", "function": {"name": "get_faction_info",
      "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}
  ],
  "goal": "셀레스티아 왕국과 흑요 상회의 관계를 정리해줘",
  "plan": ["셀레스티아 왕국 정보 조회", "흑요 상회 정보 조회", "두 세력 관계 비교"],
  "steps": [
    {"tool": "get_faction_info", "arguments": {"name": "셀레스티아 왕국"}, "observation": "{...}"},
    {"tool": "get_faction_info", "arguments": {"name": "흑요 상회"}, "observation": "{...}"}
  ],
  "final_answer": "셀레스티아 왕국은 코어스톤을 공식 관리하고, 흑요 상회는 파편을 밀거래해 둘은 긴장 관계입니다."
}
```

위 한 줄은 아래 멀티턴 궤적으로 변환되어 학습됩니다:

```
[user]      목표
[assistant] 1. 셀레스티아 왕국 정보 조회        ← 계획(첫 assistant 턴 content)
            2. 흑요 상회 정보 조회
            3. 두 세력 관계 비교
            + tool_call: get_faction_info(셀레스티아 왕국)
[tool]      관찰 결과                            ← 마스킹(학습 안 함)
[assistant] + tool_call: get_faction_info(흑요 상회)
[tool]      관찰 결과                            ← 마스킹
[assistant] Final: 최종 답변
```

- **계획은 첫 assistant 턴의 content**에 번호 목록으로 실리고, 곧바로 1단계 `tool_call`이 이어집니다.
- 이후 각 단계는 `tool_call → tool(관찰)`로 진행되고, 마지막 assistant 턴이 최종 답변입니다.
- `plan`(계획 설명)과 `steps`(실제 실행)의 개수가 꼭 1:1일 필요는 없습니다.

### 자동 인식되는 입력들

| 필드 | 인식되는 키 |
|---|---|
| 함수 스키마 | `tools` (평면 `{name, parameters}`도 자동 래핑) |
| 목표 | `goal` / `instruction` / `question` |
| 계획 | `plan` / `plan_steps` (문자열 또는 `{step, detail}` 배열) |
| 실행 단계 | `steps` / `actions` / `executions` (각 `{tool, arguments, observation}`) |
| 최종 답변 | `final_answer` / `answer` / `final` |
| 이미 궤적 | `{"tools": [...], "messages": [...]}` (tool 형식 그대로 검증) |

### 검증 규칙 (위반 줄은 건너뜀)

- `tools`가 비어 있지 않고 각 함수에 `name`이 있을 것
- `goal`·`final_answer`가 비어 있지 않을 것
- `plan`이 **2단계 이상** (`PLANACT_MIN_PLAN`)
- `steps`가 1개 이상이며, 각 `tool`은 `tools` 카탈로그에 있는 함수일 것 / `observation` 필수
- 조립 후 tool 데이터 검증기(마지막 assistant, tool 결과는 호출 뒤, tool_call ≥ 1)를 통과할 것

### 저장 형식 (`data/processed/planact_dataset.jsonl`)

tool 데이터셋과 **완전히 동일한 스키마**입니다(그래서 학습 코드를 공유):

```json
{"messages": [...], "tools": "<함수 스키마 JSON 문자열>", "meta": {"n_plan": M, "n_steps": N, "n_tools": K, ...}}
```

`arguments`와 `tools`는 `datasets`(Arrow) 스키마 안정화를 위해 JSON 문자열로 저장되고,
`train_planact.py`가 학습 직전 dict/list로 복원합니다. `meta`에는 `n_plan`(계획 단계 수)와
`n_steps`(실행 단계 수)가 붙습니다. → [meta_info.md](meta_info.md)

---

## 2. 데이터 만들기

### (A) 직접 작성

`data/raw/planact_운영.jsonl`에 넣고 `prepare_data.py`를 돌립니다.

```bash
python scripts/prepare_data.py
# [OK] 계획-실행 데이터셋: N개 궤적 → data/processed/planact_dataset.jsonl
```

### (B) LLM으로 자동 생성

함수 카탈로그(툴 호출 단계와 동일한 `--tools` 규칙)와 원문 청크로 궤적을 생성합니다.

```bash
export QA_GEN_BASE_URL="http://localhost:11434/v1"
export QA_GEN_MODEL="qwen2.5:14b"
python scripts/generate_planact.py --tools data/raw/tools_catalog.json --per-chunk 1
```

LLM에게 `{goal, plan, steps, final_answer}` 형식으로 받은 뒤 **`prepare_data.py`와
동일한 검증기**를 통과한 것만 저장합니다. 기본 append, `--overwrite`로 새로 쓰기.

> ⚠️ 생성된 arguments·observation·계획은 반드시 사람이 표본 검수하세요. 인자가 스키마와
> 어긋나거나 관찰이 사실과 다르면 잘못된 실행을 가르칩니다.

---

## 3. 학습

```bash
python scripts/train_planact.py
```

- 시작 지점은 `config.yaml`의 `planact.init_from`으로 정합니다.
  - `"sft"`(기본): 지식 SFT 위에 계획+실행을 함께 학습 (자립적)
  - `"tool"`: 이미 학습한 툴 호출 어댑터 위에 계획 능력을 얹고 싶을 때
  - `"base"` / 병합모델 경로: 그 위에서 시작
- `train_on_responses_only`가 켜져 있어 **계획+tool_call과 최종 답변에만** loss가 걸리고,
  **tool 결과(관찰)는 마스킹**됩니다.

출력: `outputs/planact/final` (LoRA 어댑터)

---

## 4. 테스트 · 병합

```bash
python scripts/test_model.py --stage planact          # 함수 스키마를 자동 제공
python scripts/test_model.py --stage planact -q "코어스톤 위기 대응을 준비해줘"

python scripts/export_model.py --stage planact
```

> **테스트 유의점:** 실제 도구/환경이 없으므로 모델은 관찰(tool 결과)까지 스스로 지어내며
> 궤적을 이어갑니다. 이는 형식(계획→호출→답변)이 학습됐는지 확인하는 스모크 테스트이며,
> 실제 품질은 각 tool_call을 진짜로 실행해 결과를 되먹이는 에이전트 런타임에서 평가하세요.

전체 파이프라인:

```bash
bash run_pipeline.sh --with-planact
```

---

## 5. 다른 단계와의 관계

| 단계 | 무엇을 배우나 | 도구 실행 | 계획 |
|---|---|---|---|
| tool | 단발성 함수 호출 | ✅ (구조화 tool_call) | ✕ |
| plan | 목표 → 단계별 계획 | ✕ | ✅ |
| react | 생각↔행동↔관찰 반복 | 텍스트 action | 암묵적 |
| **planact** | **계획 세우고 → 함수로 실행 → 종합** | ✅ (구조화 tool_call) | ✅ (명시적) |

- **plan vs planact**: plan은 계획만, planact는 계획을 실제 함수 호출로 **실행**까지 합니다.
- **react vs planact**: react는 사전 계획 없이 매 스텝 판단하고 action이 텍스트인 반면,
  planact는 **앞단에 전체 계획**을 세우고 **구조화된 tool_call**로 실행합니다.
- **tool 위에 얹기**: `planact.init_from: "tool"`로 툴 호출 어댑터 위에 계획 능력을 추가할 수 있습니다.

---

## 6. 팁 & 흔한 실수

- **계획과 실행의 정합성.** plan에 적은 순서와 steps의 tool 호출 순서가 어긋나면 모델이
  "말 따로 행동 따로"를 배웁니다. 계획이 실제 실행을 반영하게 하세요.
- **관찰(tool 결과)은 사실 기반.** 생성분은 observation이 원문/스키마와 맞는지 검수하세요.
- **arguments 스키마 일치**가 핵심입니다(→ tool 단계와 동일). 카탈로그에 없는 함수를 부르는
  단계는 자동으로 탈락합니다.
- **train/eval 분할**은 `meta.chunk_id`(생성분) 또는 `hash`로 그룹을 지어 누수를 막으세요.
- 계획 없이 바로 호출만 하길 원하면 tool 단계를, 실행 없이 계획만 원하면 plan 단계를 쓰세요.
