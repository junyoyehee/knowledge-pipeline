# 지식축적형 학습 파이프라인 (unsloth)

새로운 도메인 지식을 LLM에 주입하는 2단계 학습 파이프라인입니다.

```
원문 문서 ──► [1] 데이터 준비 ──► [2] CPT (지식 주입) ──► [3] SFT (지시 튜닝) ──► [4] 병합 ──► [5] 테스트
(txt/md)      청크 분할            원문 이어서 사전학습      대화 데이터로 튜닝      16bit/GGUF
QA jsonl      messages로 정규화     embed/lm_head 학습        응답만 loss 계산
```

**왜 2단계인가?** SFT만으로는 모델이 "답변 형식"만 배우고 지식 자체는 잘 흡수하지
못합니다. CPT로 원문을 이어서 사전학습해 지식을 파라미터에 축적한 뒤, SFT로 그
지식을 대화 형태로 꺼내 쓰도록 만드는 것이 unsloth 공식 권장 방식입니다.
핵심은 CPT 단계에서 `embed_tokens`/`lm_head`까지 학습하고, 이 둘의 학습률을
본체보다 낮게(`embedding_learning_rate`) 주는 것입니다.

## 디렉터리 구조

```
knowledge-pipeline/
├── configs/config.yaml       # 모든 설정 (모델, 하이퍼파라미터, 경로)
├── docs/
│   ├── meta_info.md          # 데이터셋 meta 필드 의미와 활용법
│   ├── preference_tuning.md  # DPO/ORPO/KTO 가이드 — 쓰기 전 필독
│   ├── tool_calling.md       # 툴 호출(function calling) 학습 가이드
│   ├── planning.md           # 계획수립(planning) 학습 가이드
│   ├── react.md              # 추론형(ReAct) 학습 가이드
│   └── planact.md            # 계획-실행(plan-and-execute) 학습 가이드
├── requirements.txt
├── run_pipeline.sh           # 전체 파이프라인 원클릭 실행
├── data/
│   ├── raw/                  # ← 원문(.txt/.md), qa_/tools_/plans_/react_/planact_*.jsonl
│   └── processed/            # 가공된 학습 데이터 (자동 생성)
├── scripts/
│   ├── common.py             # config 로더, 메타정보 유틸
│   ├── llm_client.py         # OpenAI 호환 API 클라이언트 (데이터 생성 공용)
│   ├── prepare_data.py       # [1] 원문 → CPT/SFT/툴/계획 데이터셋 (없으면 샘플 생성)
│   ├── generate_qa.py        # [1.5] (선택) LLM으로 원문에서 QA 자동 생성
│   ├── train_cpt.py          # [2] Continued Pretraining
│   ├── train_sft.py          # [3] Supervised Fine-Tuning
│   ├── generate_tool_calls.py# (선택) LLM으로 툴 호출 데이터 자동 생성
│   ├── train_tool.py         # (선택) 툴 호출(function calling) 전용 학습
│   ├── generate_plans.py     # (선택) LLM으로 계획수립 데이터 자동 생성
│   ├── train_plan.py         # (선택) 계획수립(planning) 전용 학습
│   ├── generate_react.py     # (선택) LLM으로 ReAct 트레이스 자동 생성
│   ├── train_react.py        # (선택) 추론형(ReAct) 전용 학습
│   ├── generate_planact.py   # (선택) LLM으로 계획-실행 궤적 자동 생성
│   ├── train_planact.py      # (선택) 계획-실행(plan-and-execute) 전용 학습
│   ├── export_model.py       # [4] LoRA 병합 (16bit / GGUF)
│   ├── test_model.py         # [5] 학습 결과 확인
│   ├── generate_preference.py# (선택) 선호 학습 데이터 생성
│   ├── pref_common.py        # (선택) 선호 학습 트레이너 공용 로직
│   ├── train_dpo.py          # (선택) DPO — SFT 뒤에 추가
│   ├── train_orpo.py         # (선택) ORPO — SFT를 대체
│   └── train_kto.py          # (선택) KTO — 이진 라벨 기반
└── outputs/                  # 학습 결과물 (자동 생성)
```

## 설치

```bash
pip install -r requirements.txt
# 또는 최소한으로:
pip install unsloth
```

CUDA GPU가 필요합니다. 24GB 이하 GPU(RTX 3090/4090 등) 기준으로 4bit QLoRA
설정이 기본값입니다.

## 사용법

### 1. 데이터 넣기

`data/raw/`에 도메인 원문 문서(`.txt`, `.md`)를 넣습니다.
QA 쌍이 있으면 `qa_이름.jsonl`로 넣습니다 (한 줄에 하나). **OpenAI messages 형식이
표준**이며, system 프롬프트와 멀티턴 대화를 그대로 담을 수 있습니다:

```json
{"messages": [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}
{"messages": [{"role": "system", "content": "너는 아스테리아 세계관 전문가다."}, {"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}
```

기존에 갖고 있는 데이터가 다른 형식이어도 `prepare_data.py`가 자동으로
messages 형식으로 변환합니다:

| 입력 형식 | 예시 |
|---|---|
| ShareGPT | `{"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}` |
| Alpaca | `{"instruction": ..., "input": ..., "output": ...}` (`input`은 user 메시지에 합쳐짐) |
| QA | `{"question": ..., "answer": ...}` / `{"prompt": ..., "response": ...}` |

변환 규칙: 마지막 메시지는 반드시 `assistant`여야 하고(학습 대상), `system`은 맨 앞에만
올 수 있습니다. 위반하는 줄은 경고를 출력하고 건너뜁니다.

가공된 데이터셋의 각 줄에는 출처·작성주체·중복 판별용 `meta` 필드가 자동으로
붙습니다(학습에는 사용되지 않음). 필드별 의미와 활용법은
[docs/meta_info.md](docs/meta_info.md)를 참고하세요.

**아무 데이터도 없으면 샘플 데이터(가상 게임 세계관)가 자동 생성**되어
파이프라인 동작을 바로 확인할 수 있습니다.

### 2. 전체 실행

```bash
bash run_pipeline.sh              # 전체: 데이터→CPT→SFT→병합→테스트
bash run_pipeline.sh --skip-export  # 병합 생략 (디스크 절약)
```

### 3. 단계별 실행

```bash
python scripts/prepare_data.py    # 데이터 가공
python scripts/train_cpt.py       # 지식 주입
python scripts/train_sft.py       # 지시 튜닝
python scripts/export_model.py    # 병합
python scripts/test_model.py -q "코어스톤 등급 체계를 설명해줘"
```

### (선택) QA 자동 생성

원문만 있고 QA가 없다면, OpenAI 호환 API(vLLM/Ollama/사내 게이트웨이)로
청크별 QA를 자동 생성할 수 있습니다:

```bash
export QA_GEN_BASE_URL=http://localhost:11434/v1
export QA_GEN_MODEL=qwen2.5:14b
python scripts/generate_qa.py --per-chunk 3
```

### (선택) 툴 호출 학습 — function calling

모델에게 "언제 / 어떤 함수를 / 어떤 인자로" 호출할지 가르치는 **별도 전용 단계**입니다.
`data/raw/tools_*.jsonl`에 함수 스키마(`tools`)와 대화(`messages`)를 넣습니다:

```json
{"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "서울 날씨 어때?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "서울"}}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 25}"}, {"role": "assistant", "content": "서울은 25도입니다."}]}
```

```bash
# (선택) LLM으로 툴 호출 데이터 자동 생성
export QA_GEN_BASE_URL=http://localhost:11434/v1
export QA_GEN_MODEL=qwen2.5:14b
python scripts/generate_tool_calls.py --per-chunk 2

python scripts/prepare_data.py          # tools_*.jsonl → tool_dataset.jsonl
python scripts/train_tool.py            # 지식 SFT 위에 툴 능력 추가
python scripts/test_model.py --stage tool
python scripts/export_model.py --stage tool

# 전체 파이프라인에 붙이려면:
bash run_pipeline.sh --with-tool
```

> 기본 모델 Qwen2.5는 툴 호출 템플릿을 지원합니다. 데이터 형식·검증 규칙·자동
> 생성·주의사항은 **[docs/tool_calling.md](docs/tool_calling.md)** 를 참고하세요.

### (선택) 계획수립 학습 — planning

목표(goal)를 받아 **단계별 계획(steps)** 을 세우는 능력을 가르치는 **별도 전용 단계**입니다.
`data/raw/plans_*.jsonl`에 목표와 단계를 넣으면 표준 번호 목록으로 변환되어 학습됩니다:

```json
{"goal": "코어스톤 위기를 조사할 계획을 세워줘", "steps": ["관측 데이터 수집: ...", "원인 가설 수립: ...", "대응 우선순위 결정: ..."]}
```

```bash
# (선택) LLM으로 계획 데이터 자동 생성
export QA_GEN_BASE_URL=http://localhost:11434/v1
export QA_GEN_MODEL=qwen2.5:14b
python scripts/generate_plans.py --per-chunk 2

python scripts/prepare_data.py          # plans_*.jsonl → plan_dataset.jsonl
python scripts/train_plan.py            # 지식 SFT 위에 계획 능력 추가
python scripts/test_model.py --stage plan
python scripts/export_model.py --stage plan

# 전체 파이프라인에 붙이려면:
bash run_pipeline.sh --with-plan
```

> 데이터 형식·검증 규칙·자동 생성·주의사항은 **[docs/planning.md](docs/planning.md)** 를
> 참고하세요. 계획을 세운 뒤 실제 도구 실행까지 하려면 툴 호출 학습과 함께 쓰면 됩니다.

### (선택) 추론형 학습 — ReAct

**Thought(생각) → Action(행동) → Observation(관찰)** 을 반복하며 문제를 풀도록
가르치는 **별도 전용 단계**입니다. `data/raw/react_*.jsonl`에 질문·추론 단계·최종
답변을 넣으면 멀티턴 트레이스로 변환됩니다(**Observation은 자동 마스킹**되어 학습 대상에서 제외):

```json
{"question": "5등급 코어스톤은 몇 개이고 어디에 있어?", "steps": [{"thought": "등급 정보를 조회하자.", "action": "lookup_corestone[5]", "observation": "5등급 '심장', 3개, 루멘하임·남부 해구·북부 빙하"}], "final_answer": "심장은 3개이며 루멘하임 대성탑·남부 해구·북부 빙하에 있습니다."}
```

```bash
# (선택) LLM으로 ReAct 트레이스 자동 생성
export QA_GEN_BASE_URL=http://localhost:11434/v1
export QA_GEN_MODEL=qwen2.5:14b
python scripts/generate_react.py --per-chunk 2

python scripts/prepare_data.py          # react_*.jsonl → react_dataset.jsonl
python scripts/train_react.py           # 지식 SFT 위에 추론 능력 추가
python scripts/test_model.py --stage react
python scripts/export_model.py --stage react

# 전체 파이프라인에 붙이려면:
bash run_pipeline.sh --with-react
```

> 데이터 형식·관찰 마스킹 원리·tool/plan 단계와의 차이·주의사항은
> **[docs/react.md](docs/react.md)** 를 참고하세요.

### (선택) 계획-실행 학습 — plan-and-execute

목표를 받아 **계획을 세우고 → 각 단계를 실제 tool_call로 실행하며 → 관찰을 종합해
최종 답변**까지 내는 에이전트형 궤적을 학습하는 **별도 전용 단계**입니다(툴 호출 형식의
확장 — 첫 assistant 턴에 계획이 실림). `data/raw/planact_*.jsonl`:

```json
{"tools": [{"type": "function", "function": {"name": "get_faction_info", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}], "goal": "두 세력 관계 정리", "plan": ["A 조회", "B 조회", "비교"], "steps": [{"tool": "get_faction_info", "arguments": {"name": "셀레스티아 왕국"}, "observation": "{...}"}], "final_answer": "..."}
```

```bash
# (선택) LLM으로 계획-실행 궤적 자동 생성
export QA_GEN_BASE_URL=http://localhost:11434/v1
export QA_GEN_MODEL=qwen2.5:14b
python scripts/generate_planact.py --per-chunk 1

python scripts/prepare_data.py          # planact_*.jsonl → planact_dataset.jsonl
python scripts/train_planact.py         # 지식 SFT 위에 계획+실행 능력 추가
python scripts/test_model.py --stage planact
python scripts/export_model.py --stage planact

# 전체 파이프라인에 붙이려면:
bash run_pipeline.sh --with-planact
```

> tool 단계와 데이터 형식·학습 로직을 공유합니다. 계획/실행 정합성·관찰 마스킹·다른
> 단계와의 차이는 **[docs/planact.md](docs/planact.md)** 를 참고하세요.

### (선택) 선호 학습 — DPO / ORPO / KTO

SFT 이후 **환각 억제**나 형식 교정이 필요할 때 추가하는 단계입니다.

```bash
python scripts/generate_preference.py   # 선호 데이터 생성 (LLM API 필요)
python scripts/train_dpo.py             # 또는 train_orpo.py / train_kto.py
python scripts/test_model.py --stage dpo
python scripts/export_model.py --stage dpo
```

> ⚠️ **이 단계는 지식을 주입하지 않습니다.** 데이터가 부족하거나 rejected 품질이
> 나쁘면 모델이 오히려 나빠집니다. 실행 전
> **[docs/preference_tuning.md](docs/preference_tuning.md)를 반드시 읽으세요.**
> 언제 쓰면 안 되는지, rejected를 어떻게 만들어야 하는지, 학습 후 무엇을
> 확인해야 하는지가 정리되어 있습니다.

## 주요 설정 (configs/config.yaml)

| 항목 | 설명 |
|---|---|
| `model.name` | 베이스 모델. unsloth 지원 모델 아무거나 (Qwen2.5, Llama3.1, EXAONE 등) |
| `model.chat_template` | SFT용 채팅 템플릿. 모델에 맞게 변경 (`qwen-2.5`, `llama-3.1`, `chatml`) |
| `model.max_seq_length` | VRAM 부족 시 2048로 축소 |
| `cpt.lora.r` | CPT rank. 지식량이 많으면 128~256, 적으면 64 |
| `cpt.train.num_epochs` | 지식 주입 반복 횟수. 데이터가 적으면 3~10 |
| `cpt.train.embedding_learning_rate` | embed/lm_head 학습률 — 본체의 1/5~1/10 유지 |
| `sft.continue_from_cpt` | CPT 어댑터를 이어받을지 여부 |
| `sft.system_prompt` | 데이터에 system이 없을 때 붙일 기본 system 프롬프트. `null`이면 미사용. 학습·추론에 동일 적용됨 |
| `tool.init_from` | 툴 호출 학습 시작 지점 (`sft`(권장)/`cpt`/`base`/경로). [docs/tool_calling.md](docs/tool_calling.md) |
| `plan.init_from` | 계획수립 학습 시작 지점 (`sft`(권장)/`cpt`/`base`/경로). [docs/planning.md](docs/planning.md) |
| `react.init_from` | 추론형(ReAct) 학습 시작 지점 (`sft`(권장)/`tool`/`base`/경로). [docs/react.md](docs/react.md) |
| `planact.init_from` | 계획-실행 학습 시작 지점 (`sft`(권장)/`tool`/`base`/경로). [docs/planact.md](docs/planact.md) |
| `export.source_stage` | 병합할 단계 (`sft`/`tool`/`plan`/`react`/`planact`/`dpo`/`orpo`/`kto`). 전용 단계를 썼다면 반드시 변경 |
| `export.save_gguf` | Ollama/llama.cpp용 GGUF 저장 여부 |
| `preference.*` | 선호 학습 설정. [docs/preference_tuning.md](docs/preference_tuning.md) 참고 |

## VRAM 부족(OOM) 시 체크리스트

1. `model.max_seq_length`: 4096 → 2048
2. `per_device_batch_size`: 2 → 1 (`gradient_accumulation_steps`를 2배로)
3. `cpt.lora.r`: 128 → 64
4. 더 작은 베이스 모델 사용 (7B → 3B)

## 팁

- **지식 축적 품질은 CPT 데이터 반복 노출에 비례**합니다. 같은 지식을 다른
  표현으로 여러 번 담은 문서(요약본, 상세본, 목록형)를 함께 넣으면 좋습니다.
- SFT용 QA는 청크당 3개 이상, 같은 지식을 다른 각도로 묻는 질문이 효과적입니다.
- 일반 능력 손실(catastrophic forgetting)이 우려되면 CPT 데이터에 일반 코퍼스를
  10~30% 섞으세요.
- 학습 후 `test_model.py --stage cpt`로 CPT만의 효과도 따로 확인할 수 있습니다.
