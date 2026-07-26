# 계획수립(planning) 학습 가이드

모델에게 **목표(goal)를 받아 실행 가능한 단계별 계획(steps)을 세우는** 능력을
가르치는 별도 전용 단계입니다. 지식 주입(CPT)·지시 튜닝(SFT)과는 목적이 다릅니다.

```
CPT (지식 축적) ──► SFT (지식 활용) ──► [plan] 계획수립 학습 ──► 병합 ──► 테스트
                     지식 SFT 어댑터 위에 '목표 → 단계별 계획' 능력만 추가
```

> 이 단계는 **계획만** 생성합니다(도구 실행 없음). 계획을 세운 뒤 실제로 도구를
> 호출해 실행까지 하려면 [tool_calling.md](tool_calling.md)의 툴 호출 학습과 함께
> 쓰세요.

---

## 1. 데이터 형식

`data/raw/plans_*.jsonl` 에 한 줄에 하나씩 넣습니다. **권장 형식은 `goal` + `steps`** 이며,
`prepare_data.py`가 표준 번호 목록으로 렌더링해 `messages`로 변환합니다.

```json
{"goal": "코어스톤 마력 감소 징후를 조사할 계획을 세워줘",
 "steps": ["관측 데이터 수집: ...", "이해관계자 확인: ...", "원인 가설 수립: ...", "대응 우선순위 결정: ...", "보고 및 재관측: ..."]}
```

위 한 줄은 아래처럼 변환되어 학습됩니다(assistant가 배우는 계획 형식):

```
1. 관측 데이터 수집: ...
2. 이해관계자 확인: ...
3. 원인 가설 수립: ...
4. 대응 우선순위 결정: ...
5. 보고 및 재관측: ...
```

### 자동 인식되는 입력들

| 형식 | 예시 |
|---|---|
| 권장 | `{"goal": ..., "steps": ["...", "..."]}` |
| alias | `{"instruction": 목표, "plan": [...], "context": 참고자료}` (`question`/`prompt`도 goal로 인식) |
| dict 단계 | `{"goal": ..., "steps": [{"step": "제목", "detail": "상세"}]}` → `제목 — 상세`로 렌더링 |
| 이미 대화형 | `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` (그대로 검증만) |

- `context`/`input`은 목표에 딸린 참고 자료로 user 메시지에 합쳐집니다.
- `system`을 넣으면 맨 앞 system 메시지로 보존됩니다.
- `intro`를 넣으면 번호 목록 앞에 한 줄 안내로 붙습니다(선택).

### 검증 규칙 (위반 줄은 건너뜀)

- `goal`이 비어 있지 않을 것
- `steps`가 **2개 이상** (`PLAN_MIN_STEPS`) — 단계 1개는 계획으로 보지 않습니다
- 각 단계는 비어 있지 않은 문자열/딕셔너리
- 대화형 입력은 마지막이 `assistant`여야 함

### 저장 형식 (`data/processed/plan_dataset.jsonl`)

```json
{"messages": [{"role": "user", "content": 목표}, {"role": "assistant", "content": "1. ...\n2. ..."}],
 "meta": {"id": "...", "source": "...", "origin": "human", "n_steps": 5, "hash": "..."}}
```

`meta`에는 SFT와 같은 필드에 더해 `n_steps`(단계 수)가 붙습니다. 자세한 내용은
[meta_info.md](meta_info.md) 참고.

---

## 2. 데이터 만들기

### (A) 직접 작성

`data/raw/plans_운영계획.jsonl` 처럼 넣고 `prepare_data.py`를 돌립니다.

```bash
python scripts/prepare_data.py
# [OK] 계획수립 데이터셋: N개 계획 → data/processed/plan_dataset.jsonl
```

### (B) LLM으로 자동 생성

원문 청크를 LLM에 주고, 그 도메인에서 세울 법한 목표와 단계별 계획을 생성합니다.

```bash
export QA_GEN_BASE_URL="http://localhost:11434/v1"
export QA_GEN_MODEL="qwen2.5:14b"
python scripts/generate_plans.py --per-chunk 2
```

LLM에게 `{goal, steps}` 평면 형식으로 받은 뒤 **`prepare_data.py`와 동일한 검증기**를
통과한 것만 저장합니다. 기본은 append이며 `--overwrite`로 새로 쓸 수 있습니다.

> ⚠️ 생성된 계획은 사람이 표본 검수하세요. 단계가 문서 사실과 어긋나거나 순서가
> 비논리적이면 모델에 잘못된 계획 습관을 가르칩니다.

---

## 3. 학습

```bash
python scripts/train_plan.py
```

- 시작 지점은 `config.yaml`의 `plan.init_from`으로 정합니다.
  - `"sft"`(기본): 지식 SFT 어댑터를 이어받아 계획 능력만 추가
  - `"base"`: 베이스 모델에서 계획 능력만 학습(지식과 분리)
  - `"cpt"` / 병합모델 경로: 그 위에서 시작
- `train_on_responses_only`가 켜져 있어 **계획(assistant 응답)에만** loss가 걸리고
  목표(질문) 부분은 마스킹됩니다.

출력: `outputs/plan/final` (LoRA 어댑터)

---

## 4. 테스트 · 병합

```bash
python scripts/test_model.py --stage plan
python scripts/test_model.py --stage plan -q "축제 준비 계획을 세워줘"

python scripts/export_model.py --stage plan
```

전체 파이프라인에 붙이려면:

```bash
bash run_pipeline.sh --with-plan     # SFT 뒤에 plan 학습 + plan 단계로 병합/테스트
bash run_pipeline.sh --with-tool --with-plan   # 둘 다 학습 (병합/테스트 기준은 plan)
```

---

## 5. 팁 & 흔한 실수

- **단계 형식의 일관성**이 핵심입니다. 데이터마다 "1. / 1) / - / 문단형"이 섞이면
  모델이 형식을 안정적으로 배우지 못합니다. `goal`+`steps` 형식을 쓰면 항상 동일한
  번호 목록으로 렌더링되어 일관성이 확보됩니다.
- **단계는 행동 단위**로 쓰세요. "잘 준비한다" 같은 추상적 단계보다 "X를 조사한다 →
  결과로 Y를 결정한다"처럼 앞 단계 결과가 뒤 단계의 전제가 되게 쓰면 품질이 좋습니다.
- **도메인 지식과 결합**하려면 CPT/SFT로 지식을 먼저 넣고 `init_from: sft`로 계획을
  얹으세요. 그래야 "무엇을(지식) + 어떤 순서로(계획)"가 함께 나옵니다.
- **train/eval 분할**은 `meta.chunk_id`(생성분) 또는 `hash`로 그룹을 지어 누수를
  막으세요. 같은 청크에서 나온 계획이 train/eval에 갈리면 점수가 부풀려집니다.
- 계획이 너무 짧거나 형식이 무너지면: (1) `num_epochs`를 늘리고, (2) 데이터의 단계
  수·표현을 다양화하고, (3) `system_prompt`로 역할을 고정해 보세요.
