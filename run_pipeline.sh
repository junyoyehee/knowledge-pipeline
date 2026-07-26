#!/usr/bin/env bash
# =========================================================
# 지식축적형 학습 파이프라인 전체 실행
#   1) 데이터 준비 → 2) CPT → 3) SFT → (선택) 툴/계획 → 4) 병합 → 5) 테스트
# 사용법: bash run_pipeline.sh [--skip-export] [--with-tool] [--with-plan] [--with-react] [--with-planact]
#   --with-tool    : SFT 뒤에 툴 호출 전용 학습을 추가
#   --with-plan    : SFT 뒤에 계획수립 전용 학습을 추가
#   --with-react   : SFT 뒤에 추론형(ReAct) 전용 학습을 추가
#   --with-planact : SFT 뒤에 계획-실행(plan-and-execute) 전용 학습을 추가
#   여러 플래그를 함께 주면 모두 학습하며, 병합/테스트 기준은 planact > react > plan > tool 순.
#   (다른 단계를 병합/테스트하려면 export_model.py/test_model.py에 --stage로 직접 지정)
# =========================================================
set -euo pipefail
cd "$(dirname "$0")"

SKIP_EXPORT=false
WITH_TOOL=false
WITH_PLAN=false
WITH_REACT=false
WITH_PLANACT=false
for arg in "$@"; do
  [ "$arg" = "--skip-export" ] && SKIP_EXPORT=true
  [ "$arg" = "--with-tool" ] && WITH_TOOL=true
  [ "$arg" = "--with-plan" ] && WITH_PLAN=true
  [ "$arg" = "--with-react" ] && WITH_REACT=true
  [ "$arg" = "--with-planact" ] && WITH_PLANACT=true
done

echo "===== [1/6] 데이터 준비 ====="
python -m scripts.data.prepare_data

echo "===== [2/6] CPT (지식 주입) ====="
python -m scripts.train.train_cpt

echo "===== [3/6] SFT (지시 튜닝) ====="
python -m scripts.train.train_sft

# 병합/테스트에 쓸 단계 (지정된 전용 단계가 우선, 없으면 SFT)
STAGE_ARGS=()
if [ "$WITH_TOOL" = false ] && [ "$WITH_PLAN" = false ] \
   && [ "$WITH_REACT" = false ] && [ "$WITH_PLANACT" = false ]; then
  echo "===== [4/6] 전용 단계 건너뜀 (--with-tool/--with-plan/--with-react/--with-planact 로 활성화) ====="
else
  echo "===== [4/6] 전용 단계 학습 ====="
  if [ "$WITH_TOOL" = true ]; then
    echo "----- 툴 호출 학습 -----"
    python -m scripts.train.train_tool
    STAGE_ARGS=(--stage tool)
  fi
  if [ "$WITH_PLAN" = true ]; then
    echo "----- 계획수립 학습 -----"
    python -m scripts.train.train_plan
    STAGE_ARGS=(--stage plan)   # plan을 병합/테스트 기준으로 (tool보다 우선)
  fi
  if [ "$WITH_REACT" = true ]; then
    echo "----- 추론형(ReAct) 학습 -----"
    python -m scripts.train.train_react
    STAGE_ARGS=(--stage react)  # react를 병합/테스트 기준으로 (plan/tool보다 우선)
  fi
  if [ "$WITH_PLANACT" = true ]; then
    echo "----- 계획-실행(plan-and-execute) 학습 -----"
    python -m scripts.train.train_planact
    STAGE_ARGS=(--stage planact)  # planact를 병합/테스트 기준으로 (최우선)
  fi
fi

if [ "$SKIP_EXPORT" = false ]; then
  echo "===== [5/6] 모델 병합 ====="
  python -m scripts.model.export_model "${STAGE_ARGS[@]}"
else
  echo "===== [5/6] 모델 병합 건너뜀 ====="
fi

echo "===== [6/6] 결과 테스트 ====="
python -m scripts.model.test_model "${STAGE_ARGS[@]}"

echo "===== 파이프라인 완료 ====="
