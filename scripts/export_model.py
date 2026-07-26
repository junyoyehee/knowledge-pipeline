"""
4단계: 모델 병합/내보내기
- SFT 어댑터를 베이스 모델에 병합하여 16bit 모델로 저장
- 선택적으로 GGUF(q4_k_m)로도 저장 (llama.cpp / Ollama 배포용)

사용법:
    python scripts/export_model.py [--config configs/config.yaml]
    python scripts/export_model.py --stage dpo    # 선호 학습 어댑터 병합
"""
import argparse
import os

from unsloth import FastLanguageModel

from common import load_config, stage_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default=None,
                        help="병합할 단계 (sft/dpo/orpo/kto). "
                             "생략하면 config의 export.source_stage 사용")
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, ecfg = cfg["model"], cfg["export"]
    stage = args.stage or ecfg.get("source_stage", "sft")
    adapter = stage_adapter(cfg, stage)
    if not os.path.isdir(adapter):
        raise SystemExit(f"'{stage}' 단계의 어댑터가 없습니다: {adapter}\n"
                         f"해당 단계를 먼저 학습하거나 --stage를 바꾸세요.")
    print(f"[i] 병합 대상: {stage} → {adapter}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg["dtype"],
        load_in_4bit=mcfg["load_in_4bit"],
    )

    # 16bit 병합 저장 (vLLM/transformers 서빙용)
    model.save_pretrained_merged(
        ecfg["merged_dir"], tokenizer, save_method="merged_16bit")
    print(f"[OK] 병합 모델 저장: {ecfg['merged_dir']}")

    if ecfg.get("save_gguf"):
        gguf_dir = ecfg["merged_dir"] + "_gguf"
        model.save_pretrained_gguf(
            gguf_dir, tokenizer,
            quantization_method=ecfg.get("gguf_quantization", "q4_k_m"))
        print(f"[OK] GGUF 저장: {gguf_dir}")


if __name__ == "__main__":
    main()
