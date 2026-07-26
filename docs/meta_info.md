# 데이터셋 메타정보 (meta) 설계

`data/processed/`에 생성되는 데이터셋의 각 줄에는 `meta` 객체가 붙습니다.

```json
{"text": "...", "meta": {"id": "sample_worldbook-0000", "source": "sample_worldbook.md", "chunk_index": 0, "hash": "244c9367bbad"}}
{"messages": [...], "meta": {"id": "qa_sample-0001", "source": "qa_sample.jsonl", "origin": "human", "hash": "8ffa4df8689d"}}
{"messages": [...], "tools": "<JSON>", "meta": {"id": "tools_sample-0001", "source": "tools_sample.jsonl", "origin": "human", "n_tools": 3, "hash": "aae945524d40"}}
```

> 툴 호출 데이터셋(`tool_dataset.jsonl`)은 SFT와 같은 메타 필드에 더해 `n_tools`
> (샘플에 제공된 함수 수)가 붙습니다. `hash`는 함수 스키마·tool_calls·결과까지 포함해
> 계산합니다. 형식은 [tool_calling.md](tool_calling.md) 참고.
>
> 계획수립 데이터셋(`plan_dataset.jsonl`)은 SFT와 같은 메타 필드에 더해 `n_steps`
> (계획 단계 수)가 붙습니다. 형식은 [planning.md](planning.md) 참고.
>
> 추론형 데이터셋(`react_dataset.jsonl`)은 SFT와 같은 메타 필드에 더해 `n_steps`
> (Thought/Action/Observation 단계 수)가 붙습니다. 형식은 [react.md](react.md) 참고.

## 전제: 메타는 학습에 들어가지 않습니다

`meta`는 **모델이 보는 데이터가 아닙니다.** 두 학습 스크립트가 텍스트 컬럼만 남기고
나머지를 명시적으로 제거합니다:

- [`train_cpt.py`](../scripts/train_cpt.py) — `add_eos` 매핑에서 `text` 외 컬럼 제거
- [`train_sft.py`](../scripts/train_sft.py) — `remove_columns=dataset.column_names`로 전량 제거 후 `text`만 생성

따라서 메타를 추가해도 학습 결과·VRAM·속도에 영향이 없습니다. 메타의 목적은
**학습이 아니라 데이터 운영**(추적·중복 제거·분할·필터링)입니다.

---

## 필드 정의

### 공통 필드

| 필드 | 타입 | 생성 주체 | 의미 |
|---|---|---|---|
| `id` | string | 자동 | 이 샘플의 사람이 읽을 수 있는 식별자 |
| `source` | string | 자동 | 이 샘플이 유래한 `data/raw/`의 파일명 |
| `hash` | string | 자동 | 내용 기반 지문 (SHA-1 앞 12자) |

### CPT 전용 (`cpt_dataset.jsonl`)

| 필드 | 타입 | 의미 |
|---|---|---|
| `chunk_index` | int | 해당 원문 문서 내에서의 청크 순번 (0부터) |

### SFT 전용 (`sft_dataset.jsonl`)

| 필드 | 타입 | 의미 |
|---|---|---|
| `origin` | string | 이 QA를 누가 만들었는가. `human` 또는 `llm:<모델명>` |
| `chunk_id` | string | (선택) 이 QA가 근거한 CPT 청크의 `meta.id` |

---

## 각 필드의 의미와 활용

### `id` — 식별자

형식은 `<파일명 stem>-<번호 4자리>`입니다.

- CPT: `sample_worldbook-0000` (번호 = 문서 내 청크 순번)
- SFT (사람 작성): `qa_sample-0001` (번호 = 원본 jsonl의 **줄 번호**)
- SFT (자동 생성): `gen-<청크 id>-<n>` 예) `gen-sample_worldbook-0000-2`

**활용:** 학습 후 이상한 답변이 나왔을 때 로그·평가 결과에 `id`를 남겨두면
원본 줄로 바로 되짚을 수 있습니다.

> **주의 — `id`는 줄 번호 기반이라 원본 파일을 편집하면 밀립니다.** 중간에 한 줄을
> 삽입하면 그 아래 모든 `id`가 바뀝니다. 편집을 넘어 안정적으로 같은 샘플을 가리켜야
> 한다면 `id`가 아니라 `hash`를 쓰세요. `id`는 "지금 이 파일의 이 줄"을 가리키는
> 위치 포인터, `hash`는 "이 내용"을 가리키는 내용 포인터입니다.

### `source` — 출처 파일

`data/raw/` 기준 파일명(경로 없이)입니다.

**활용 1 — 문서 단위 평가.** 특정 문서의 지식만 골라 평가 세트를 만들 수 있습니다.

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("data/processed/sft_dataset.jsonl", encoding="utf-8")]
target = [r for r in rows if r["meta"]["source"] == "제품매뉴얼.md"]
print(len(target))
PY
```

**활용 2 — 문서 회수.** 원문에 오류가 발견되면 그 문서에서 파생된 CPT 청크와
SFT QA를 `source`로 한 번에 찾아 제거할 수 있습니다.

### `hash` — 내용 지문

대화 전체(`role:content`를 이어붙인 문자열)를 공백 정규화한 뒤 SHA-1으로 해시한
앞 12자입니다. CPT는 청크 텍스트 기준입니다.

공백 정규화 때문에 **들여쓰기나 줄바꿈만 다른 중복도 같은 해시**가 됩니다.
역으로 답변 한 글자만 달라도 다른 해시가 됩니다.

**활용 1 — 중복 제거.** 여러 소스에서 데이터를 모을 때 필수입니다.

```bash
python - <<'PY'
import json
seen, out = set(), []
for l in open("data/processed/sft_dataset.jsonl", encoding="utf-8"):
    r = json.loads(l)
    h = r["meta"]["hash"]
    if h not in seen:
        seen.add(h); out.append(r)
print(f"중복 {sum(1 for _ in open('data/processed/sft_dataset.jsonl', encoding='utf-8')) - len(out)}건 제거")
PY
```

`generate_qa.py`를 여러 번 돌리면 같은 청크에서 비슷한 QA가 반복 생성되는데,
완전히 동일한 것들은 이걸로 걸러집니다.

**활용 2 — 재현성 확인.** 파이프라인을 다시 돌렸을 때 해시 집합이 같으면 데이터가
동일하게 재생성된 것입니다. 학습 결과가 달라졌을 때 "데이터가 바뀐 건지 학습이
바뀐 건지"를 가르는 기준이 됩니다.

### `origin` — 작성 주체 (SFT 전용)

- `human` — `data/raw/qa_*.jsonl`에 사람이 넣은 QA ([`prepare_data.py`](../scripts/prepare_data.py)가 부여)
- `llm:<모델명>` — [`generate_qa.py`](../scripts/generate_qa.py)가 자동 생성. 예) `llm:qwen2.5:14b`

**이 필드가 없으면 생기는 문제:** `generate_qa.py`는 `sft_dataset.jsonl`에 **append**
합니다. 한 번 섞이고 나면 어느 게 자동 생성분인지 구분할 수 없어, 나중에 생성 품질이
나빴다고 판단해도 그것만 걷어낼 수 없습니다.

**활용 1 — 생성분만 회수.** 더 좋은 모델로 다시 생성하고 싶을 때:

```bash
python - <<'PY'
import json
keep = [l for l in open("data/processed/sft_dataset.jsonl", encoding="utf-8")
        if json.loads(l)["meta"]["origin"] == "human"]
open("data/processed/sft_dataset.jsonl", "w", encoding="utf-8").writelines(keep)
PY
python scripts/generate_qa.py --per-chunk 3   # 새 모델로 재생성
```

**활용 2 — 모델별 품질 비교.** 여러 모델로 생성한 뒤 `origin`별로 나눠 학습·평가하면
어느 생성 모델이 나은지 판단할 수 있습니다.

**활용 3 — 신뢰도 가중.** 사람 작성분을 여러 epoch 노출시키고 생성분은 한 번만
쓰는 식의 커리큘럼을 짤 때 분리 기준이 됩니다.

### `chunk_id` — 근거 청크 (SFT 전용)

`generate_qa.py`가 생성한 QA에만 붙으며, 그 QA를 만들 때 LLM에게 준 CPT 청크의
`meta.id`를 가리킵니다. 사람이 작성한 QA에는 없습니다(근거 청크가 특정되지 않으므로).
직접 넣고 싶으면 `qa_*.jsonl`에 `"meta": {"chunk_id": "..."}`를 적으면 보존됩니다.

**활용 1 — train/eval 분할 시 누수 방지 (가장 중요).** 같은 청크에서 뽑은 QA 3개가
train과 eval로 갈리면, eval 질문의 답이 이미 train에 다른 표현으로 들어 있어
평가 점수가 실제보다 높게 나옵니다. **청크 단위로 묶어서 분할해야 합니다.**

```bash
python - <<'PY'
import json, random
rows = [json.loads(l) for l in open("data/processed/sft_dataset.jsonl", encoding="utf-8")]
groups = {}
for r in rows:
    # chunk_id가 없으면(사람 작성분) 자기 자신을 그룹으로
    key = r["meta"].get("chunk_id") or r["meta"]["id"]
    groups.setdefault(key, []).append(r)

keys = sorted(groups)
random.Random(42).shuffle(keys)
n_eval = max(1, len(keys) // 10)
eval_keys = set(keys[:n_eval])

for name, sel in (("train", lambda k: k not in eval_keys),
                  ("eval",  lambda k: k in eval_keys)):
    with open(f"data/processed/sft_{name}.jsonl", "w", encoding="utf-8") as f:
        for k in keys:
            if sel(k):
                for r in groups[k]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"청크 {len(keys)}개 → eval {len(eval_keys)}개 그룹")
PY
```

**활용 2 — 지식 커버리지 점검.** 어느 청크에서 QA가 하나도 안 나왔는지 확인해
지식 공백을 찾습니다. CPT로는 주입됐지만 SFT로 꺼내 쓰는 법을 안 배운 영역입니다.

> **주의:** 사람이 작성한 QA에는 `chunk_id`가 없으므로, 아래 결과는 실제보다
> 커버리지를 낮게 잡습니다(사람 QA가 다루고 있는 청크도 "누락"으로 집계됨).
> 정확히 보려면 `qa_*.jsonl`에 `meta.chunk_id`를 직접 적어주거나, 이 결과를
> "자동 생성이 닿지 않은 청크" 목록으로 해석하세요.

```bash
python - <<'PY'
import json
chunks = {json.loads(l)["meta"]["id"] for l in open("data/processed/cpt_dataset.jsonl", encoding="utf-8")}
covered = {json.loads(l)["meta"].get("chunk_id") for l in open("data/processed/sft_dataset.jsonl", encoding="utf-8")}
missing = sorted(chunks - covered)
print(f"QA가 없는 청크 {len(missing)}개: {missing[:10]}")
PY
```

**활용 3 — 환각 추적.** 모델이 틀린 사실을 말했을 때, 해당 QA의 `chunk_id`로
원문 청크를 찾아 "원문이 틀렸는지 / 생성 LLM이 왜곡했는지"를 판별합니다.

---

## 직접 넣는 메타 (커스텀 필드)

`data/raw/qa_*.jsonl`에 `meta`를 직접 적으면 그대로 보존되며, **자동 생성값보다
우선합니다**(`hash`는 정규화 후 내용 기준이라 항상 재계산됩니다).

```json
{"messages": [...], "meta": {"origin": "human", "reviewer": "kim", "difficulty": "hard"}}
```

정의되지 않은 필드(`reviewer`, `difficulty` 등)도 그대로 통과하므로 팀 사정에 맞게
확장할 수 있습니다.

---

## 의도적으로 넣지 않은 필드

| 필드 | 제외 이유 |
|---|---|
| `created_at` | 재실행할 때마다 값이 바뀌어 데이터 diff가 무의미해집니다. 생성 시각이 필요하면 파일 mtime이나 git 커밋 시각을 쓰세요 |
| `weight`, `quality_score` | 현재 이 값을 읽어 샘플링·필터링하는 코드가 없습니다. 죽은 필드가 되므로 실제로 가중 학습을 구현할 때 함께 추가하는 편이 낫습니다 |
| `token_count` | 토크나이저에 종속적이라 `model.name`을 바꾸면 값이 전부 무효가 됩니다. 필요한 시점에 계산하세요 |

---

## 요약

| 알고 싶은 것 | 쓸 필드 |
|---|---|
| 이 샘플 어디서 왔지? | `source`, `chunk_id` |
| 이거 사람이 쓴 거야 생성한 거야? | `origin` |
| 중복 아닌가? | `hash` |
| eval 세트를 어떻게 갈라야 안전하지? | `chunk_id`로 그룹 분할 |
| 학습 로그의 이 샘플이 원본 어디야? | `id` |
| 커버 안 된 지식이 있나? | CPT `id` vs SFT `chunk_id` 차집합 |
