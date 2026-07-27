"""
(선택) 계획-실행(plan-and-execute) 학습 단계 — 에이전트형 궤적 학습
- 목표를 받아 (1) 계획을 세우고 (2) 각 단계를 실제 tool_call로 실행하며
  (3) 관찰을 종합해 최종 답변까지 내는 능력을 전용으로 학습한다.
- 데이터 형식이 툴 호출 단계와 동일하므로(계획은 첫 assistant 턴의 content에 실림)
  train_tool.py의 학습 로직(train_tool_style)을 그대로 재사용한다.
- 응답(계획 + tool_call + 최종 답변)에만 loss, tool 결과는 마스킹.

시작 지점(config의 planact.init_from):
    "sft"(기본) → 지식 SFT 위에 계획+실행을 함께 학습 (자립적)
    "tool"      → 이미 학습한 툴 호출 어댑터 위에 계획 능력을 얹고 싶을 때
    "base"/경로 → 그 모델 위에 새 LoRA 부착

사용법:
    python -m scripts.data.prepare_data            # planact_*.jsonl → planact_dataset.jsonl
    python -m scripts.generate.generate_planact        # (선택) LLM으로 자동 생성
    python -m scripts.train.train_planact [--config configs/config.yaml]

출력: outputs/planact/ (LoRA 어댑터)
"""
import argparse
import os

# unsloth는 transformers/trl보다 먼저 import되어야 함.
# train_tool이 pref_common→unsloth를 먼저 import하므로 여기서 가져오면 순서가 지켜진다.
from scripts.train.train_tool import train_tool_style
from scripts.lib.pref_common import load_config
from scripts.lib.common import add_report_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    add_report_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    planact_path = cfg["data"].get("planact_dataset")
    if not planact_path or not os.path.exists(planact_path):
        raise SystemExit(
            f"계획-실행 데이터셋이 없습니다: {planact_path}\n"
            "planact_*.jsonl을 넣고 prepare_data.py를 돌리거나, "
            "generate_planact.py로 먼저 생성하세요.")

    train_tool_style(cfg, cfg["planact"], planact_path, "planact",
                     report_path=args.report_json,
                     progress_path=args.progress_json)


if __name__ == "__main__":
    main()
