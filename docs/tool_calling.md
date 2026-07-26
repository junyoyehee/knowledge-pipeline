# 툴 호출(function calling) 학습 가이드

모델에게 **"언제 / 어떤 함수를 / 어떤 인자로"** 호출해야 하는지를 가르치는
별도 전용 단계입니다. 지식 주입(CPT)이나 지시 튜닝(SFT)과는 목적이 다릅니다.

```
CPT (지식 축적) ──► SFT (지식 활용) ──► [tool] 툴 호출 학습 ──► 병합 ──► 테스트
                     지식 SFT 어댑터 위에 function calling 능력만 추가
```

> **전제:** 베이스 모델과 `model.chat_template`이 tool 호출 템플릿을 지원해야 합니다.
> 기본값 `unsloth/Qwen2.5-7B-Instruct` + `qwen-2.5` 템플릿은 지원합니다.
> Llama-3.1 계열도 가능합니다. 지원하지 않는 모델/템플릿에서는 `tools`가
> 프롬프트에 렌더링되지 않아 학습 신호가 생기지 않습니다.

---

## 1. 데이터 형식

`data/raw/tools_*.jsonl` 에 한 줄에 하나씩 넣습니다. 각 줄은 `tools`(함수 스키마)와
`messages`(대화)로 구성됩니다.

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 조회한다.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "도시 이름"}},
          "required": ["city"]
        }
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "서울 날씨 어때?"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": {"city": "서울"}}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 25, \"sky\": \"맑음\"}"},
    {"role": "assistant", "content": "서울은 현재 25도이고 맑습니다."}
  ]
}
```

### 검증 규칙 (`prepare_data.py`가 자동 적용, 위반 줄은 건너뜀)

- `tools`: 비어 있지 않은 함수 스키마 배열. 각 함수는 `function.name` 필수.
  평면 형식(`{"name": ..., "parameters": ...}`)도 자동으로 OpenAI 형식으로 감쌉니다.
- `messages`:
  - `system`은 맨 앞에만, `user`는 최소 1개, **마지막은 반드시 `assistant`**
  - `role: "tool"`(함수 결과)은 반드시 앞선 `tool_calls` **뒤에** 올 것
  - **대화에 `tool_calls`가 최소 하나** 있어야 함 (없으면 툴 학습 데이터가 아니므로
    `qa_*.jsonl`로 넣으세요)
- `arguments`는 dict로 적어도 되고 JSON 문자열로 적어도 됩니다(내부적으로 통일).

### 저장 형식 (`data/processed/tool_dataset.jsonl`)

```json
{"messages": [...], "tools": "<함수 스키마 JSON 문자열>", "meta": {...}}
```

`tools`와 tool_call의 `arguments`는 **JSON 문자열**로 저장됩니다. 함수마다 인자
구조가 달라 `datasets`(Arrow)의 스키마 추론이 깨지는 것을 막기 위함입니다.
`train_tool.py`가 학습 직전에 다시 dict/list로 복원해 채팅 템플릿에 넣습니다.

`meta` 필드 의미는 [meta_info.md](meta_info.md)를 참고하세요. 툴 데이터에는
`n_tools`(제공한 함수 수)가 추가로 붙습니다.

---

## 2. 데이터 만들기

### (A) 직접 작성

`data/raw/tools_사내API.jsonl` 처럼 넣고 `prepare_data.py`를 돌립니다.

```bash
python -m scripts.data.prepare_data
# [OK] 툴 호출 데이터셋: N개 대화 → data/processed/tool_dataset.jsonl
```

### (B) LLM으로 자동 생성

함수 스키마 카탈로그와 원문 청크를 LLM에 주고, 그 함수를 써야만 답할 수 있는
대화를 생성합니다. OpenAI 호환 API(vLLM/Ollama/사내 게이트웨이)면 됩니다.

```bash
export QA_GEN_BASE_URL="http://localhost:11434/v1"
export QA_GEN_MODEL="qwen2.5:14b"

# 함수 카탈로그: 함수 스키마 배열이 담긴 JSON 파일
#   생략하면 data/raw/tools_catalog.json → 그것도 없으면 내장 샘플 카탈로그 사용
python -m scripts.generate.generate_tool_calls --tools data/raw/tools_catalog.json --per-chunk 2
```

`tools_catalog.json` 예시(함수 스키마만 담은 배열):

```json
[
  {"type": "function", "function": {"name": "lookup_corestone",
    "description": "코어스톤 등급 정보를 조회한다.",
    "parameters": {"type": "object",
      "properties": {"grade": {"type": "integer"}}, "required": ["grade"]}}}
]
```

생성기는 LLM에게 조립하기 쉬운 평면 형식
(`user / tool_name / arguments / tool_result / final_answer`)으로 받은 뒤,
파이썬이 정식 `messages`로 조립하고 **`prepare_data.py`와 동일한 검증기**를
통과한 것만 저장합니다. 기본은 `tool_dataset.jsonl`에 append이며,
`--overwrite`로 새로 쓸 수 있습니다.

> ⚠️ 생성된 `arguments`와 `tool_result`는 반드시 사람이 표본 검수하세요.
> 인자가 스키마와 어긋나거나 결과가 사실과 다르면 모델에게 잘못된 호출을 가르칩니다.

---

## 3. 학습

```bash
python -m scripts.train.train_tool
```

- 시작 지점은 `config.yaml`의 `tool.init_from`으로 정합니다.
  - `"sft"`(기본): 지식 SFT 어댑터를 이어받아 툴 능력만 추가
  - `"base"`: 베이스 모델에서 툴 능력만 학습(지식과 분리하고 싶을 때)
  - `"cpt"` / 병합모델 경로: 그 위에서 시작
- `tools`를 채팅 템플릿에 함께 넣어 학습하므로, 모델이 함수 목록을 보고
  적절한 함수·인자를 고르는 법을 배웁니다.
- `train_on_responses_only`가 켜져 있어 **assistant의 tool_call과 최종 답변에만**
  loss가 걸립니다. `role: tool` 결과는 (Qwen/ChatML 계열에서 user 블록으로
  렌더링되어) 자동으로 마스킹됩니다.

출력: `outputs/tool/final` (LoRA 어댑터)

---

## 4. 테스트 · 병합

```bash
# 테스트: 함수 스키마를 자동으로 제공해 실제로 tool_call이 나오는지 확인
python -m scripts.model.test_model --stage tool
python -m scripts.model.test_model --stage tool -q "3등급 코어스톤 정보 알려줘"

# 병합 (16bit / GGUF)
python -m scripts.model.export_model --stage tool
```

`test_model.py --stage tool`은 `--tools`가 없으면 `tool_dataset.jsonl`의
첫 줄에서 함수 스키마를 읽어 프롬프트에 넣습니다. 스키마를 안 주면 모델이
tool_call을 낼 이유가 없으니, 툴 테스트에서는 반드시 스키마가 제공되어야 합니다.

전체 파이프라인에 붙이려면:

```bash
bash run_pipeline.sh --with-tool     # SFT 뒤에 tool 학습 + tool 단계로 병합/테스트
```

---

## 5. 팁 & 흔한 실수

- **부정 예시(함수를 부르지 말아야 할 때)** 도 섞으면 과호출을 줄일 수 있습니다.
  단, 현재 `train_tool.py`는 `tool_calls`가 있는 샘플만 받습니다. 순수한
  "함수 없이 답" 예시는 일반 `qa_*.jsonl`(SFT)로 넣으세요.
- **멀티 호출/멀티턴**도 지원됩니다. 여러 `tool_calls`, 여러 `tool` 결과,
  중간 assistant 발화를 자유롭게 이어 붙일 수 있습니다(마지막만 assistant면 됨).
- **train/eval 분할**은 `meta.chunk_id`(생성분) 또는 `hash`로 그룹을 지어
  누수를 막으세요. 같은 청크에서 나온 예시가 train/eval에 갈리면 점수가 부풀려집니다.
- **인자 스키마 일치**가 핵심입니다. `arguments`의 키/타입이 `parameters`와
  어긋난 데이터를 학습하면 그대로 잘못된 호출을 배웁니다. 생성분은 꼭 검수하세요.
- 학습 후 tool_call이 안 나오면: (1) 모델/템플릿의 tool 지원 여부, (2) 테스트 때
  `tools` 제공 여부, (3) 데이터의 `tool_calls` 존재 여부를 순서대로 확인하세요.
