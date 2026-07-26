# 추론형(ReAct) 학습 가이드

모델에게 **Thought(생각) → Action(행동) → Observation(관찰)** 을 반복하며 문제를
풀고, 마지막에 **Final Answer** 로 마무리하는 추론 방식을 가르치는 별도 전용 단계입니다.

```
CPT (지식 축적) ──► SFT (지식 활용) ──► [react] 추론형 학습 ──► 병합 ──► 테스트
                     지식 SFT 어댑터 위에 'ReAct 추론 루프' 능력만 추가
```

> **핵심 설계 — 관찰(Observation) 마스킹.** ReAct 트레이스를 멀티턴 대화로 변환할 때
> Thought+Action은 **assistant 턴**, Observation은 **user 턴**으로 넣습니다.
> `train_on_responses_only`가 user 턴을 마스킹하므로, 모델은 **생각·행동·최종 답변만**
> 학습하고 **관찰 결과는 학습하지 않습니다.** 관찰은 실제 환경/도구가 주는 것이지
> 모델이 지어내야 할 것이 아니기 때문입니다.

---

## 1. 데이터 형식

`data/raw/react_*.jsonl` 에 한 줄에 하나씩 넣습니다. **권장 형식은 `question` +
`steps` + `final_answer`** 이며, `prepare_data.py`가 표준 트레이스로 렌더링합니다.

```json
{"question": "5등급 코어스톤은 몇 개이고 어디에 있어?",
 "steps": [
   {"thought": "등급별 정보를 조회하자.", "action": "lookup_corestone[5]",
    "observation": "5등급 '심장', 총 3개, 위치: 루멘하임 대성탑·남부 해구·북부 빙하 아래"},
   {"thought": "개수와 위치를 확인했다.", "action": "finish[]", "observation": "(종료)"}],
 "final_answer": "5등급 코어스톤 '심장'은 총 3개이며, 루멘하임 대성탑·남부 해구·북부 빙하 아래에 있습니다."}
```

위 한 줄은 아래 멀티턴 대화로 변환되어 학습됩니다:

```
[user]      5등급 코어스톤은 몇 개이고 어디에 있어?
[assistant] Thought: 등급별 정보를 조회하자.
            Action: lookup_corestone[5]
[user]      Observation: 5등급 '심장', 총 3개, 위치: ...        ← 마스킹(학습 안 함)
[assistant] Thought: 개수와 위치를 확인했다.
            Action: finish[]
[user]      Observation: (종료)                                  ← 마스킹
[assistant] Final Answer: 5등급 코어스톤 '심장'은 총 3개이며 ...
```

### 자동 인식되는 입력들

| 필드 | 인식되는 키 |
|---|---|
| 질문 | `question` / `instruction` / `task` / `prompt` |
| 단계 배열 | `steps` / `trajectory` / `react` |
| 생각 | `thought` / `reasoning` / `think` |
| 행동 | `action` (문자열 `"tool[input]"` 또는 `{"tool": "...", "input": ...}`) |
| 관찰 | `observation` (문자열 또는 객체 → JSON 문자열로 변환) |
| 최종 답변 | `final_answer` / `answer` / `final` |
| (선택) 마무리 생각 | `final_thought` → Final Answer 앞에 붙음 |

- `context`/`input`은 질문에 딸린 참고 자료로 user 메시지에 합쳐집니다.
- `system`을 넣으면 맨 앞 system 메시지로 보존됩니다.
- 이미 대화형(`messages`) 트레이스면 그대로 검증만 통과합니다(단계 수는 `Action:` 개수로 추정).

### 검증 규칙 (위반 줄은 건너뜀)

- `question`과 `final_answer`가 비어 있지 않을 것
- `steps`가 **1개 이상** (`REACT_MIN_STEPS`)
- 각 단계는 `thought` + `action` + `observation` 을 모두 가질 것 (하나라도 비면 그 줄 탈락)

### 저장 형식 (`data/processed/react_dataset.jsonl`)

```json
{"messages": [{"role": "user", ...}, {"role": "assistant", ...}, {"role": "user", ...}, ...],
 "meta": {"id": "...", "source": "...", "origin": "human", "n_steps": 2, "hash": "..."}}
```

`meta`에는 SFT와 같은 필드에 더해 `n_steps`(추론 단계 수)가 붙습니다.
자세한 내용은 [meta_info.md](meta_info.md) 참고.

### 라벨 커스터마이즈

`Thought/Action/Observation/Final Answer` 라벨은 `scripts/prepare_data.py`의
`REACT_LABELS` 상수에서 한 곳으로 관리됩니다. 한국어 라벨(`생각/행동/관찰/최종 답변`)
등으로 바꾸고 싶으면 이 값만 수정하면 렌더링 전체가 따라갑니다.

---

## 2. 데이터 만들기

### (A) 직접 작성

`data/raw/react_운영.jsonl` 처럼 넣고 `prepare_data.py`를 돌립니다.

```bash
python scripts/prepare_data.py
# [OK] ReAct 데이터셋: N개 트레이스 → data/processed/react_dataset.jsonl
```

### (B) LLM으로 자동 생성

원문 청크를 근거로 질문과 추론 트레이스를 생성합니다.

```bash
export QA_GEN_BASE_URL="http://localhost:11434/v1"
export QA_GEN_MODEL="qwen2.5:14b"
python scripts/generate_react.py --per-chunk 2
```

LLM에게 `{question, steps, final_answer}` 형식으로 받은 뒤 **`prepare_data.py`와
동일한 검증기**를 통과한 것만 저장합니다. 기본은 append이며 `--overwrite`로 새로 쓸 수 있습니다.

> ⚠️ 생성된 트레이스는 사람이 표본 검수하세요. observation이 문서 사실과 어긋나거나
> thought→action 연결이 비논리적이면 잘못된 추론 습관을 가르칩니다.

---

## 3. 학습

```bash
python scripts/train_react.py
```

- 시작 지점은 `config.yaml`의 `react.init_from`으로 정합니다.
  - `"sft"`(기본): 지식 SFT 어댑터를 이어받아 추론 능력만 추가
  - `"tool"`: 툴 호출 어댑터 위에 추론까지 얹고 싶을 때
  - `"base"` / 병합모델 경로: 그 위에서 시작
- `train_on_responses_only`가 켜져 있어 **Thought+Action과 Final Answer(assistant 턴)에만**
  loss가 걸리고, **Observation(user 턴)은 마스킹**됩니다.

출력: `outputs/react/final` (LoRA 어댑터)

---

## 4. 테스트 · 병합

```bash
python scripts/test_model.py --stage react
python scripts/test_model.py --stage react -q "코어스톤과 낙토 현상의 관계를 조사해줘"

python scripts/export_model.py --stage react
```

> **테스트 시 유의점:** 학습된 모델은 질문을 받으면 `Thought:`/`Action:`을 생성합니다.
> 그러나 `test_model.py`에는 실제 도구/환경이 없으므로, 모델은 Observation까지 스스로
> **지어내며** 트레이스를 이어갑니다. 이는 형식이 학습됐는지 확인하는 스모크 테스트일
> 뿐이며, 실제 추론 품질은 **각 Action을 실제로 실행해 Observation을 되먹이는 실행
> 루프**(에이전트 런타임)에서 평가해야 합니다.

전체 파이프라인에 붙이려면:

```bash
bash run_pipeline.sh --with-react     # SFT 뒤에 react 학습 + react 단계로 병합/테스트
```

---

## 5. tool / plan 단계와의 관계

| 단계 | 무엇을 배우나 | 형식 |
|---|---|---|
| **tool** | 언제/어떤 함수를 어떤 인자로 부를지 | OpenAI `tool_calls` (구조화) |
| **plan** | 목표를 단계별 계획으로 분해 | 번호 목록 (실행 없음) |
| **react** | 생각↔행동↔관찰을 반복하며 문제 해결 | Thought/Action/Observation 트레이스 |

- **plan vs react**: plan은 실행 없이 계획만, react는 관찰을 받아가며 다단계로 추론합니다.
- **tool vs react**: react의 Action은 텍스트(`search[...]`)라 특정 모델의 tool 템플릿에
  의존하지 않습니다. 구조화된 네이티브 함수 호출을 원하면 tool 단계를 쓰세요. 둘을 함께
  쓰고 싶으면 `react.init_from: "tool"`로 툴 어댑터 위에 추론을 얹을 수 있습니다.

---

## 6. 팁 & 흔한 실수

- **Observation을 학습시키지 마세요.** 이 파이프라인은 Observation을 user 턴으로 넣어
  자동 마스킹합니다. 직접 대화형(`messages`)으로 데이터를 만들 때도 관찰은 반드시
  user 턴에 두세요. assistant 턴에 관찰을 넣으면 모델이 관찰을 지어내도록 학습됩니다.
- **thought → action 인과관계**가 분명해야 합니다. "무엇을 왜 확인하는지"를 생각에
  담고, 그 다음 행동이 자연스럽게 이어지게 하세요.
- **관찰은 사실에 근거**해야 합니다. 생성분은 특히 observation이 원문과 맞는지 검수하세요.
- **train/eval 분할**은 `meta.chunk_id`(생성분) 또는 `hash`로 그룹을 지어 누수를 막으세요.
- 트레이스가 무한히 늘어지거나 Final Answer가 안 나오면: (1) 학습 데이터의 단계 수를
  1~4개로 제한하고, (2) 마지막 단계에서 확실히 `Final Answer`로 끝나는 예시를 늘리고,
  (3) 추론 실행 루프에서 최대 스텝 수 상한을 두세요.
