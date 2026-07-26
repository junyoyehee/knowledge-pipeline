#!/usr/bin/env bash
# =========================================================
# 지식축적형 학습 파이프라인 전체 실행
#   1) 데이터 준비 → 2) CPT → 3) SFT → 4) 병합 → 5) 테스트
# 사용법: bash run_pipeline.sh [--skip-export]
# =========================================================
set -euo pipefail
cd "$(dirname "$0")"

SKIP_EXPORT=false
for arg in "$@"; do
  [ "$arg" = "--skip-export" ] && SKIP_EXPORT=true
done

echo "===== [1/5] 데이터 준비 ====="
python scripts/prepare_data.py

echo "===== [2/5] CPT (지식 주입) ====="
python scripts/train_cpt.py

echo "===== [3/5] SFT (지시 튜닝) ====="
python scripts/train_sft.py

if [ "$SKIP_EXPORT" = false ]; then
  echo "===== [4/5] 모델 병합 ====="
  python scripts/export_model.py
else
  echo "===== [4/5] 모델 병합 건너뜀 ====="
fi

echo "===== [5/5] 결과 테스트 ====="
python scripts/test_model.py

echo "===== 파이프라인 완료 ====="
