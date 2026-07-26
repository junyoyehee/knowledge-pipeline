# scripts/ — 파이프라인 스크립트 패키지

역할별 하위 패키지로 구성된 파이썬 패키지입니다. **리포지토리 루트에서 `python -m`
으로 실행**하세요. 전체 폴더 역할은 [../docs/project_structure.md](../docs/project_structure.md).

| 하위 폴더 | 역할 | 실행 진입점? |
|---|---|---|
| `lib/` | 공용 유틸 (`common`·`llm_client`·`pref_common`) | ✗ (import 전용) |
| `data/` | `prepare_data` — 원문/jsonl → 데이터셋 6종 | ✓ |
| `generate/` | (선택) LLM 기반 데이터 생성 `generate_*` | ✓ |
| `train/` | 학습 `train_*` (cpt/sft/tool/plan/react/planact/dpo/orpo/kto) | ✓ |
| `model/` | `export_model`(병합) · `test_model`(추론 테스트) | ✓ |

## 실행 예

```bash
python -m scripts.data.prepare_data
python -m scripts.generate.generate_planact --per-chunk 1
python -m scripts.train.train_sft --config configs/config.yaml
python -m scripts.model.export_model --stage planact
python -m scripts.model.test_model --stage planact
```

## 의존 방향

`data`·`generate`·`train`·`model` → `lib` (단방향). `lib`는 다른 하위 패키지를
import하지 않습니다. 예외: `train.train_planact`→`train.train_tool`(공용 학습 로직 재사용),
`generate.generate_planact`→`generate.generate_tool_calls`(카탈로그 로더 재사용).
