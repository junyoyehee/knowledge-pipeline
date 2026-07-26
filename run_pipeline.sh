#!/usr/bin/env bash
# =========================================================
# 지식축적형 학습 파이프라인 전체 실행
#   1) 데이터 준비 → 2) CPT → 3) SFT → (선택) 툴 호출 → 4) 병합 → 5) 테스트
# 사용법: bash run_pipeline.sh [--skip-export] [--with-tool]
#   --with-tool : SFT 뒤에 툴 호출 전용 학습을 추가하고, 병합/테스트도 tool 단계 기준
# =========================================================
set -euo pipefail
cd "$(dirname "$0")"

SKIP_EXPORT=false
WITH_TOOL=false
for arg in "$@"; do
  [ "$arg" = "--skip-export" ] && SKIP_EXPORT=true
  [ "$arg" = "--with-tool" ] && WITH_TOOL=true
done

echo "===== [1/6] 데이터 준비 ====="
python scripts/prepare_data.py

echo "===== [2/6] CPT (지식 주입) ====="
python scripts/train_cpt.py

echo "===== [3/6] SFT (지시 튜닝) ====="
python scripts/train_sft.py

if [ "$WITH_TOOL" = true ]; then
  echo "===== [4/6] 툴 호출 학습 ====="
  python scripts/train_tool.py
  STAGE_ARGS=(--stage tool)
else
  echo "===== [4/6] 툴 호출 학습 건너뜀 (--with-tool 로 활성화) ====="
  STAGE_ARGS=()
fi

if [ "$SKIP_EXPORT" = false ]; then
  echo "===== [5/6] 모델 병합 ====="
  python scripts/export_model.py "${STAGE_ARGS[@]}"
else
  echo "===== [5/6] 모델 병합 건너뜀 ====="
fi

echo "===== [6/6] 결과 테스트 ====="
python scripts/test_model.py "${STAGE_ARGS[@]}"

echo "===== 파이프라인 완료 ====="
