"""선호 학습(DPO/ORPO/KTO) 트레이너 공용 로직.

- 시작 지점(init_from) 해석 및 모델 로드
- 데이터셋 로드 및 meta 컬럼 제거
- TRL 버전에 따라 달라지는 인자명 흡수

주의: 이 모듈은 unsloth를 import하므로, 이를 사용하는 스크립트는
반드시 transformers/trl보다 먼저 이 모듈(또는 unsloth)을 import해야 합니다.
"""
import inspect
import os

from unsloth import FastLanguageModel

from common import load_config, stage_adapter  # noqa: F401  (재수출)


def resolve_init_source(cfg: dict, init_from: str) -> tuple:
    """init_from 설정 → (모델 경로, 새 LoRA를 붙여야 하는가).

    "cpt"/"sft"/"dpo"/... : 해당 단계의 어댑터를 이어서 학습 → 새 LoRA 불필요
    "base"                : config의 베이스 모델에서 시작 → 새 LoRA 필요
    그 외 문자열          : 병합된 모델 디렉터리 경로로 취급 → 새 LoRA 필요
    """
    if init_from == "base":
        return cfg["model"]["name"], True

    known_stages = ["cpt", "sft"] + list(cfg.get("preference", {}).keys())
    if init_from in known_stages:
        path = stage_adapter(cfg, init_from)
        if not os.path.isdir(path):
            raise SystemExit(
                f"'{init_from}' 단계의 어댑터가 없습니다: {path}\n"
                f"해당 단계를 먼저 학습하거나 config의 init_from을 바꾸세요.")
        return path, False

    # 경로로 간주
    path = init_from if os.path.isabs(init_from) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), init_from)
    if not os.path.isdir(path):
        raise SystemExit(f"init_from 경로를 찾을 수 없습니다: {path}")
    return path, True


def load_model(cfg: dict, stage_cfg: dict, stage_name: str):
    """선호 학습용 모델·토크나이저 로드."""
    mcfg = cfg["model"]
    init_from = stage_cfg.get("init_from", "sft")
    source, needs_new_lora = resolve_init_source(cfg, init_from)

    print(f"[i] {stage_name.upper()} 시작 지점: {init_from} → {source}")
    if not needs_new_lora:
        print("[i] 기존 어댑터를 이어서 학습합니다 "
              "(reference 모델은 베이스 모델이 됩니다 — docs/preference_tuning.md 참고)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=mcfg["max_seq_length"],
        dtype=mcfg["dtype"],
        load_in_4bit=mcfg["load_in_4bit"],
    )

    if needs_new_lora:
        lcfg = stage_cfg["lora"]
        tcfg = stage_cfg["train"]
        print("[i] 새 LoRA 어댑터를 부착합니다 "
              "(reference 모델은 시작 지점 모델이 됩니다)")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lcfg["r"],
            target_modules=lcfg["target_modules"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=tcfg["seed"],
        )

    tokenizer = _apply_chat_template(tokenizer, mcfg["chat_template"])
    return model, tokenizer


def _apply_chat_template(tokenizer, template_name: str):
    from unsloth.chat_templates import get_chat_template
    return get_chat_template(tokenizer, chat_template=template_name)


def load_pref_dataset(path: str, required: tuple, label: str):
    """JSONL 로드 후 학습에 불필요한 컬럼(meta 등) 제거."""
    from datasets import load_dataset

    if not os.path.exists(path):
        raise SystemExit(
            f"{label} 데이터셋이 없습니다: {path}\n"
            "scripts/generate_preference.py로 먼저 생성하세요.")

    ds = load_dataset("json", data_files=path, split="train")
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise SystemExit(f"{label} 데이터셋에 필수 컬럼이 없습니다: {missing}\n"
                         f"현재 컬럼: {ds.column_names}")

    drop = [c for c in ds.column_names if c not in required]
    if drop:
        ds = ds.remove_columns(drop)
    print(f"[i] {label} 학습 샘플 수: {len(ds)}")
    return ds


def warn_if_too_small(n: int, label: str, minimum: int = 200):
    """선호 학습은 데이터가 적으면 효과가 없거나 모델을 망가뜨린다."""
    if n < minimum:
        print(f"\n{'=' * 66}\n"
              f"[경고] {label} 샘플이 {n}개뿐입니다 (권장 최소 {minimum}개).\n"
              f"이 규모의 선호 학습은 효과가 없거나 모델 품질을 떨어뜨립니다.\n"
              f"먼저 데이터를 늘리는 것을 강력히 권합니다.\n"
              f"자세한 내용: docs/preference_tuning.md\n"
              f"{'=' * 66}\n")


def trainer_kwargs(trainer_cls, tokenizer) -> dict:
    """TRL 버전에 따라 tokenizer / processing_class 중 맞는 인자명을 고른다."""
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def save_adapter(model, tokenizer, output_dir: str, stage_name: str):
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] {stage_name.upper()} 어댑터 저장: {final_dir}")
