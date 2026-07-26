# 선호 학습 (DPO / ORPO / KTO) 활용 가이드

이 문서는 **읽지 않고 쓰면 모델을 망가뜨리는** 단계에 대한 것입니다.
스크립트를 실행하기 전에 최소한 "언제 쓰면 안 되는가"까지는 읽어주세요.

---

## 0. 가장 먼저 — 이 단계는 지식을 주입하지 않습니다

선호 학습은 모델이 **이미 만들 수 있는 여러 출력 중 무엇을 고를지**를 조정합니다.
모르는 사실을 알게 만들지 못합니다.

| 증상 | 원인 | 처방 |
|---|---|---|
| 사실을 **아예 모른다** ("그런 건 없습니다") | CPT 부족 | CPT 데이터·epoch 증가. **선호 학습은 무의미** |
| 사실을 **틀리게, 자신 있게 말한다** | 환각 | ✅ 선호 학습이 듣는 지점 |
| 지식은 맞는데 **형식·말투가 엉망** | SFT 부족 | SFT 데이터 개선 (선호 학습도 보조 가능) |
| 자료 밖 질문에 **지어낸다** | 거절 학습 부재 | ✅ 거절 쌍으로 선호 학습 |

**따라서 순서는 반드시 이렇습니다:**

```
1. 실제 도메인 문서로 CPT → SFT 완주
2. test_model.py로 실패 양상 확인
3. "틀리게 말한다"가 주된 실패일 때만 선호 학습 추가
```

2번을 건너뛰고 3번부터 하는 것이 이 단계에서 가장 흔한 실수입니다.

---

## 1. 세 방법 중 무엇을 쓸 것인가

| | DPO | ORPO | KTO |
|---|---|---|---|
| 위치 | SFT **뒤에** 추가 | SFT를 **대체** | SFT **뒤에** 추가 |
| 데이터 | 쌍 (chosen/rejected) | 쌍 (chosen/rejected) | **개별** 좋음/나쁨 라벨 |
| reference 모델 | 필요 | **불필요** | 필요 |
| VRAM | 높음 | 낮음 | 높음 |
| 파이프라인 | CPT→SFT→DPO | CPT→ORPO | CPT→SFT→KTO |

### 선택 기준

- **처음 시도한다면 → DPO.** 자료가 가장 많고 동작을 이해하기 쉽습니다. 검증된 SFT 모델을 남겨둔 채 위에 얹으므로, 실패해도 SFT로 되돌아가면 됩니다.
- **VRAM이 빠듯하거나 단계를 줄이고 싶다면 → ORPO.** reference 모델이 없어 메모리가 덜 들고 학습도 한 번으로 끝납니다. 다만 SFT를 대체하므로 **실패하면 SFT 결과물도 없습니다.** SFT가 이미 잘 동작한다면 굳이 갈아탈 이유는 없습니다.
- **서비스 로그에 엄지척/엄지다운이 쌓인다면 → KTO.** 쌍을 만드는 건 사람 손이 많이 가지만, 개별 응답의 좋음/나쁨은 사용자가 자연스럽게 남깁니다. **실사용 피드백이 있는 환경에서는 이게 가장 현실적입니다.**

세 개를 다 돌려서 비교하는 건 권하지 않습니다. 평가 세트가 부실한 상태에서
셋을 비교하면 노이즈를 보고 고르게 됩니다.

---

## 2. rejected 데이터가 전부입니다

선호 학습의 성패는 **rejected 품질 하나**로 갈립니다.

### 나쁜 rejected (효과 없음)

```json
{"chosen": "셀레스티아 왕국의 국왕은 아리안 셀레스티아 7세입니다.",
 "rejected": "잘 모르겠습니다."}
```

모델은 애초에 이런 답을 내지 않습니다. 학습 신호가 0에 가깝고, 최악의 경우
"모른다고 말하지 마라"를 배워 **환각이 오히려 늘어납니다.**

### 좋은 rejected (효과 있음)

```json
{"chosen": "셀레스티아 왕국의 국왕은 아리안 셀레스티아 7세입니다.",
 "rejected": "셀레스티아 왕국의 국왕은 아리안 셀레스티아 5세입니다."}
```

문체·길이·자신감이 동일하고 **사실 하나만 다릅니다.** 이것이 모델이 실제로 저지르는
실수의 형태이며, 그래서 교정 신호가 됩니다.

### 원칙

1. rejected는 **모델이 실제로 낼 법한 답**이어야 합니다
2. chosen과 **형식·길이·톤이 같아야** 합니다. 아니면 모델은 사실이 아니라 길이나 말투를 학습합니다
3. 차이는 **한 가지 축**(이름/숫자/날짜/인과)만
4. **rejected가 우연히 사실이면 안 됩니다** — 모델에게 거짓을 가르치는 셈입니다

[`generate_preference.py`](../scripts/generate/generate_preference.py)의 프롬프트는 이 원칙을
반영했지만, **LLM 생성물은 반드시 사람이 표본 검수해야 합니다.** 최소 30~50건은
직접 읽어보세요.

---

## 3. 데이터 만들기

```bash
export QA_GEN_BASE_URL="http://localhost:11434/v1"
export QA_GEN_MODEL="qwen2.5:14b"

# 권장: 검증된 SFT 정답을 chosen으로 두고 오답만 생성
python -m scripts.generate.generate_preference --mode from-sft --per-sample 1

# SFT 데이터가 없거나 커버리지를 넓히고 싶을 때
python -m scripts.generate.generate_preference --mode from-chunks --per-chunk 3

# 환각 억제에 가장 직접적: "자료에 없는 질문" 거절 쌍
python -m scripts.generate.generate_preference --refusals-per-chunk 1
```

`--mode from-sft`를 권하는 이유는 **chosen이 이미 검증된 값**이기 때문입니다.
`from-chunks`는 정답까지 LLM이 만들므로 chosen 자체가 틀릴 위험이 있습니다.

출력은 두 파일입니다 (KTO 파일은 쌍에서 자동 분해, `--no-kto`로 생략 가능):

```json
// pref_dataset.jsonl — DPO/ORPO 공용
{"prompt": [{"role":"user","content":"..."}],
 "chosen": [{"role":"assistant","content":"..."}],
 "rejected": [{"role":"assistant","content":"..."}],
 "meta": {"kind": "distortion", "chunk_id": "...", "origin": "llm:..."}}

// kto_dataset.jsonl — KTO 전용
{"prompt": [...], "completion": [...], "label": true, "meta": {...}}
```

`meta.kind`는 `distortion`(사실 왜곡) 또는 `refusal`(거절 학습)입니다.
학습 후 어느 쪽이 효과가 있었는지 나눠 분석할 때 씁니다.
나머지 meta 필드는 [meta_info.md](meta_info.md)와 동일한 규칙을 따릅니다.

### 권장 데이터 규모

| 샘플 수 | 판단 |
|---|---|
| ~200 미만 | **하지 마세요.** 효과가 없거나 모델이 나빠집니다 |
| 500 ~ 2,000 | 실용적인 최소선. 환각 억제 효과를 관측할 수 있는 구간 |
| 5,000+ | 충분. 다만 품질이 낮으면 양은 의미 없습니다 |

트레이너들은 이 기준으로 실행 시 경고를 출력합니다 (KTO는 쌍이 분해되므로 2배 기준).

---

## 4. 학습하기

```bash
# DPO — SFT 뒤에 추가
python -m scripts.train.train_dpo

# ORPO — SFT 대신 (CPT 어댑터에서 시작)
python -m scripts.train.train_orpo

# KTO — SFT 뒤에 추가
python -m scripts.train.train_kto
```

단계별 결과 확인과 병합:

```bash
python -m scripts.model.test_model --stage dpo
python -m scripts.model.export_model --stage dpo
```

**ORPO를 썼다면 `export.source_stage`를 `orpo`로 바꾸는 것을 잊지 마세요.**
기본값은 `sft`라 그냥 두면 ORPO 결과가 아니라 SFT 어댑터가 병합됩니다.

---

## 5. reference 모델 — 알고 넘어가야 할 함정

DPO와 KTO는 "reference 모델에서 얼마나 벗어났는가"로 학습을 제어합니다.
QLoRA 환경에서는 별도 모델을 올리지 않고 **LoRA 어댑터를 끈 상태**를 reference로
씁니다. 그래서 `init_from` 설정에 따라 reference의 정체가 달라집니다.

| `init_from` | 동작 | reference | 평가 |
|---|---|---|---|
| `"sft"` (기본) | SFT 어댑터를 이어서 학습 | **베이스 모델** | 간편·VRAM 절약. 다만 KL 기준점이 SFT가 아니라 베이스라 이론적으로는 정석이 아님 |
| `"outputs/final_model"` 등 경로 | 병합 모델 위에 **새 LoRA** | **SFT 모델** | 정석. 대신 SFT를 먼저 병합·저장해야 하고 디스크가 더 필요 |

기본값 `"sft"`는 실무에서 널리 쓰이고 대체로 잘 동작합니다. 하지만 학습이 불안정하거나
결과가 기대에 못 미치면 정석 경로를 시도해볼 값어치가 있습니다:

```bash
python -m scripts.model.export_model --stage sft        # SFT를 16bit로 병합
# config에서 preference.dpo.init_from을 "outputs/final_model"로 변경
python -m scripts.train.train_dpo
```

ORPO는 reference 모델 자체가 없으므로 이 논점이 없습니다.

---

## 6. 하이퍼파라미터 주의사항

### learning_rate — 가장 중요

`5.0e-6`이 기본값입니다. **SFT의 `2.0e-4`보다 40배 낮습니다.** 이건 오타가 아닙니다.
선호 학습은 학습률에 극도로 민감해서, SFT 수준의 학습률을 쓰면 몇 스텝 만에
모델이 붕괴합니다 (반복 출력, 문장 미완성).

### beta — KL 제약 강도

- 낮으면(0.05) 선호를 공격적으로 반영하지만 일반 능력이 손상되기 쉬움
- 높으면(0.5) 안전하지만 효과가 미미함
- **0.1에서 시작**하고, 효과가 없으면 낮추기 전에 **rejected 품질부터 의심하세요**

### num_epochs

DPO/KTO는 **1 epoch면 충분한 경우가 많습니다.** 여러 epoch는 과최적화로 이어집니다.
ORPO는 SFT 역할을 겸하므로 2 정도가 적당합니다.

### KTO의 가중치 균형

좋음/나쁨 샘플 수가 불균형하면 학습이 한쪽으로 쏠립니다. 권장 유효 비율은

```
(desirable_weight × 좋음 개수) / (undesirable_weight × 나쁨 개수) ≈ 1.0 ~ 1.33
```

[`train_kto.py`](../scripts/train/train_kto.py)가 실제 비율을 계산해 벗어나면 권장값을
제안합니다. `generate_preference.py`로 만든 데이터는 쌍에서 분해되어 1:1이므로
기본값 그대로 두면 됩니다.

---

## 7. 학습이 잘 되고 있는지 판단하기

### 로그에서 볼 것

- **`rewards/accuracies`** — chosen에 rejected보다 높은 점수를 준 비율
  - **0.5 근처에서 안 움직임** → 학습 신호가 없음. rejected가 너무 쉽거나 너무 어려움
  - **1.0에 금방 도달** → rejected가 너무 쉬움 (횡설수설 수준). 데이터를 다시 만드세요
  - **0.6 → 0.8로 완만히 상승** → 정상
- **`rewards/margins`** — chosen과 rejected의 점수 차이. 완만히 커지는 게 정상
- **loss가 급격히 0으로** → 과최적화. epoch를 줄이거나 beta를 높이세요

ORPO는 `rewards/*` 대신 `log_odds_ratio`, `log_odds_chosen`을 봅니다.

### 반드시 확인할 부작용

선호 학습의 대표적 실패는 **성능이 떨어지는 게 아니라 이상해지는 것**입니다.

```bash
# 학습한 도메인 질문 — 정확도가 올랐는가
python -m scripts.model.test_model --stage dpo

# 도메인과 무관한 일반 질문 — 능력이 망가지지 않았는가 (필수!)
python -m scripts.model.test_model --stage dpo -q "파이썬으로 리스트를 정렬하는 법은?"
python -m scripts.model.test_model --stage dpo -q "안녕하세요, 오늘 기분이 어때요?"
```

두 번째가 핵심입니다. 도메인 정확도만 보고 배포하면, 일반 대화가 망가진 걸
운영 중에 발견하게 됩니다.

**흔한 부작용:**

| 증상 | 원인 | 대응 |
|---|---|---|
| 답변이 계속 길어짐 | chosen이 rejected보다 일관되게 길었음 | 길이를 맞춰 데이터 재생성 |
| 아무 질문에나 "자료에 없습니다" | 거절 쌍 비중 과다 | 거절 쌍을 전체의 10~20% 이하로 |
| 같은 문장 반복 | 학습률 과다 / 과최적화 | lr 낮추기, epoch 줄이기 |
| 일반 능력 저하 | beta 과소 | beta 높이기 (0.1→0.3) |

---

## 8. 되돌리기

선호 학습은 **별도 디렉터리**(`outputs/dpo` 등)에 저장되며 SFT 어댑터를 덮어쓰지
않습니다. 결과가 나쁘면 그냥 SFT를 쓰면 됩니다:

```bash
python -m scripts.model.export_model --stage sft
```

**단, ORPO는 예외입니다.** SFT를 대체하므로 `init_from: "cpt"`로 CPT에서 시작하며,
결과가 나쁘면 되돌아갈 SFT 모델이 없습니다. ORPO를 시도하기 전에
`train_sft.py`를 먼저 돌려 SFT 어댑터를 확보해두는 편이 안전합니다.

---

## 9. 요약 체크리스트

시작 전:

- [ ] 실제 도메인 문서로 CPT→SFT를 완주했다
- [ ] `test_model.py`로 실패 양상을 확인했고, 그것이 "모름"이 아니라 "틀리게 말함"이다
- [ ] 평가용 질문 세트를 따로 준비했다 (도메인 + 일반 능력 양쪽)

데이터:

- [ ] rejected가 chosen과 형식·길이·톤이 같다
- [ ] rejected를 30건 이상 직접 읽어봤다
- [ ] rejected 중 우연히 사실인 것이 없다
- [ ] 최소 200건, 가능하면 500건 이상이다
- [ ] 거절 쌍 비중이 20%를 넘지 않는다

학습 후:

- [ ] `rewards/accuracies`가 0.5에 머물거나 1.0으로 튀지 않았다
- [ ] 도메인 질문 정확도가 올랐다
- [ ] **도메인과 무관한 일반 질문이 멀쩡하다**
- [ ] 답변 길이가 비정상적으로 늘지 않았다
- [ ] (ORPO를 썼다면) `export.source_stage`를 바꿨다
