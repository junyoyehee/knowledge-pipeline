# 프로젝트 구조 · 폴더별 역할

이 문서는 리포지토리의 폴더 구조와 **각 폴더/파일의 정확한 역할**을 정리합니다.
파이프라인 개념은 [README.md](../README.md), 각 학습 단계는 [docs/](.)의 개별 가이드를,
API는 [api/README.md](../api/README.md)를 참고하세요.

## 전체 트리

```
knowledge-pipeline/
├── README.md                 # 프로젝트 개요·사용법
├── requirements.txt          # 학습 스택 의존성 (unsloth, torch, trl, peft ...)
├── run_pipeline.sh           # 전체 파이프라인 원클릭 실행 스크립트
├── .gitignore
│
├── configs/
│   └── config.yaml           # 단일 설정 소스 (모델·하이퍼파라미터·경로·단계별 설정)
│
├── data/
│   ├── raw/                  # 입력: 원문(.md/.txt) + qa_/tools_/plans_/react_/planact_*.jsonl (+샘플)
│   └── processed/            # 출력: 가공된 데이터셋 6종 (자동 생성, git 미포함)
│
├── docs/                     # 설계·가이드 문서
│   ├── project_structure.md  # (이 문서) 폴더 구조·역할
│   ├── meta_info.md          # 데이터셋 meta 필드 의미·활용
│   ├── preference_tuning.md  # DPO/ORPO/KTO 가이드
│   ├── tool_calling.md       # 툴 호출 학습 가이드
│   ├── planning.md           # 계획수립 학습 가이드
│   ├── react.md              # 추론형(ReAct) 학습 가이드
│   ├── planact.md            # 계획-실행 학습 가이드
│   └── api_design.md         # HTTP API 설계 문서
│
├── scripts/                  # 파이프라인 스크립트 (파이썬 패키지 — `python -m` 으로 실행)
│   ├── lib/                  # 공용 유틸 (여러 스크립트가 공유)
│   │   ├── common.py         #   config 로더·경로 처리·메타/해시 유틸·TEMPLATE_PARTS
│   │   ├── llm_client.py     #   OpenAI 호환 API 클라이언트 (생성 스크립트 공용)
│   │   └── pref_common.py    #   학습 공용 로직 (init_from 해석·모델 로드·데이터셋 로드)
│   ├── data/
│   │   └── prepare_data.py   # [1] 원문/jsonl → 데이터셋 6종 정규화 (없으면 샘플 생성)
│   ├── generate/             # (선택) LLM으로 학습 데이터 자동 생성
│   │   ├── generate_qa.py            #   원문 청크 → QA
│   │   ├── generate_tool_calls.py    #   툴 호출 대화
│   │   ├── generate_plans.py         #   목표→단계별 계획
│   │   ├── generate_react.py         #   ReAct 트레이스
│   │   ├── generate_planact.py       #   계획-실행 궤적
│   │   └── generate_preference.py    #   선호(DPO/ORPO/KTO) 쌍
│   ├── train/                # 학습 단계 (각 단계 = LoRA 어댑터 산출)
│   │   ├── train_cpt.py      #   [2] Continued Pretraining (지식 주입)
│   │   ├── train_sft.py      #   [3] Supervised Fine-Tuning (지식 활용)
│   │   ├── train_tool.py     #   (선택) 툴 호출 — train_tool_style() 공용 학습 로직 보유
│   │   ├── train_plan.py     #   (선택) 계획수립
│   │   ├── train_react.py    #   (선택) 추론형(ReAct)
│   │   ├── train_planact.py  #   (선택) 계획-실행 (train_tool_style 재사용)
│   │   ├── train_dpo.py      #   (선택) DPO — SFT 뒤에 추가
│   │   ├── train_orpo.py     #   (선택) ORPO — SFT 대체
│   │   └── train_kto.py      #   (선택) KTO — 이진 라벨
│   └── model/
│       ├── export_model.py   # [4] LoRA 병합 → 16bit / GGUF
│       └── test_model.py     # [5] 학습 결과 간이 추론 테스트
│
├── api/                      # (선택) 파이프라인 HTTP API (FastAPI) — 상세: api/README.md
│   ├── main.py, auth.py, config.py, schemas.py
│   ├── store.py, storage.py, secrets.py, config_builder.py
│   ├── runner.py, worker.py  # 잡 → `python -m scripts.*` 서브프로세스 실행
│   ├── routers/              # projects · pipeline · jobs 엔드포인트
│   └── tests/test_e2e.py
│
└── outputs/                  # 학습 산출물 (어댑터·병합모델, 자동 생성, git 미포함)
```

---

## 폴더별 역할 (상세)

### `configs/`
- **역할:** 파이프라인 전체의 **단일 설정 소스**. 모델 이름·채팅 템플릿, 데이터 경로,
  각 단계(cpt/sft/tool/plan/react/planact/preference)의 LoRA·학습 하이퍼파라미터,
  export 설정을 담습니다.
- 모든 스크립트는 `--config`로 이 파일을 읽습니다. API는 요청마다 이 파일을 템플릿으로
  복제·병합해 잡별 config를 만듭니다.

### `data/`
- **`raw/` (입력):** 사용자가 넣는 원문과 jsonl. 파일명 접두로 종류를 구분합니다.
  - `*.md`, `*.txt` — 도메인 원문
  - `qa_*.jsonl` — QA(SFT), `tools_*.jsonl` — 툴 호출, `plans_*.jsonl` — 계획,
    `react_*.jsonl` — ReAct, `planact_*.jsonl` — 계획-실행
  - 아무것도 없으면 `prepare_data`가 **샘플(아스테리아 세계관)**을 자동 생성합니다.
- **`processed/` (출력):** `prepare_data`가 만든 학습용 데이터셋
  (`cpt/sft/tool/plan/react/planact_dataset.jsonl`). **재생성 가능하므로 git에 포함하지 않습니다.**

### `docs/`
- **역할:** 설계·가이드 문서. 코드가 아니라 "왜/어떻게"를 설명합니다.
- 데이터 형식·검증 규칙은 각 단계 가이드(tool_calling/planning/react/planact),
  운영 메타는 meta_info, 서비스화는 api_design 참고.

### `scripts/` — 파이썬 **패키지**
- **역할:** 파이프라인 실행 단위. 4.x부터 **역할별 하위 패키지**로 구성되며,
  리포지토리 루트에서 **모듈 형식으로 실행**합니다.
  ```bash
  python -m scripts.data.prepare_data
  python -m scripts.train.train_sft --config configs/config.yaml
  python -m scripts.model.test_model --stage sft
  ```
- **`lib/` (공용):** 다른 스크립트들이 import하는 공유 코드. 자체 실행 진입점이 아닙니다.
  - `common.py` — 설정 로드, 경로 절대화, `stage_adapter`, 해시/메타 유틸, `TEMPLATE_PARTS`
  - `llm_client.py` — 생성 스크립트가 쓰는 OpenAI 호환 API 호출부
  - `pref_common.py` — 학습 스크립트 공용(`load_stage_model`, `resolve_init_source`, 데이터셋 로드 등)
- **`data/`:** 원문/jsonl을 검증·정규화해 `data/processed/`에 데이터셋을 씁니다.
- **`generate/`:** (선택) LLM API로 학습 데이터를 합성합니다. 모두 `lib/`의 정규화·검증을
  재사용하므로 사람 작성분과 동일한 품질 기준을 통과한 것만 저장됩니다.
- **`train/`:** 각 단계 학습. 입력은 `data/processed/`의 데이터셋, 출력은 `outputs/<stage>/final` 어댑터.
- **`model/`:** 학습된 어댑터를 병합(export)하거나 간이 추론(test)합니다.

> **의존 방향:** `generate/`·`data/`·`train/`·`model/` → `lib/` (한 방향). `lib/`는
> 다른 하위 패키지를 import하지 않습니다. 예외적으로 `train_planact`는 공용 학습 로직
> 재사용을 위해 `train_tool`을, `generate_planact`는 카탈로그 로더 재사용을 위해
> `generate_tool_calls`를 import합니다.

### `api/`
- **역할:** (선택) 위 스크립트들을 감싼 **HTTP API**. 요청별 config를 만들어
  `python -m scripts.*` 를 서브프로세스로 실행하는 비동기 Job 게이트웨이입니다.
- 스크립트 자체는 API에 의존하지 않습니다(단방향: api → scripts). 설계는
  [api_design.md](api_design.md), 실행·구조는 [api/README.md](../api/README.md).

### `outputs/` (자동 생성)
- **역할:** 학습 산출물. `outputs/<stage>/final`(LoRA 어댑터), `outputs/final_model`(병합 16bit),
  `outputs/final_model_gguf`(GGUF). **재생성 가능하므로 git 미포함.**

---

## 실행 방식 요약

| 방법 | 명령 | 비고 |
|---|---|---|
| 개별 단계 | `python -m scripts.<group>.<name> [--config ...]` | 리포 루트에서 실행 |
| 전체 파이프라인 | `bash run_pipeline.sh [--with-tool/plan/react/planact]` | 준비→CPT→SFT→(전용)→병합→테스트 |
| HTTP API | `uvicorn api.main:app` | 비동기 Job, `/v1/docs` |

> **왜 `python -m` 인가?** `scripts/`가 패키지가 되면서 하위 폴더 스크립트가 공용
> 모듈을 `from scripts.lib.common import ...` 처럼 절대 경로로 안전하게 import합니다.
> 따라서 `python scripts/x.py`가 아니라 루트에서 `python -m scripts.<group>.<name>` 으로 실행합니다.
