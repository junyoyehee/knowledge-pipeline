"""
4단계: 모델 병합/내보내기
- SFT 어댑터를 베이스 모델에 병합하여 16bit 모델로 저장
- 선택적으로 GGUF(q4_k_m)로도 저장 (llama.cpp / Ollama 배포용)

사용법:
    python scripts/export_model.py [--config configs/config.yaml]
"""
import argparse
import os

from unsloth import FastLanguageModel

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, ecfg = cfg["model"], cfg["export"]
    sft_final = os.path.join(cfg["sft"]["output_dir"], "final")
    if not os.path.isdir(sft_final):
        raise SystemExit(f"SFT 어댑터가 없습니다: {sft_final}\n"
                         "train_sft.py를 먼저 실행하세요.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_final,
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
