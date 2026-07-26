"""공용 유틸리티 — config 로드, 경로 처리, 데이터 메타정보 생성."""
import hashlib
import json
import os
import re

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = None) -> dict:
    """config.yaml을 로드하고 상대 경로를 프로젝트 루트 기준 절대 경로로 변환."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 경로 필드를 절대 경로로 변환
    for key in ("raw_dir", "cpt_dataset", "sft_dataset", "tool_dataset",
                "plan_dataset", "react_dataset", "pref_dataset", "kto_dataset"):
        if cfg["data"].get(key):
            cfg["data"][key] = _abs(cfg["data"][key])
    cfg["cpt"]["output_dir"] = _abs(cfg["cpt"]["output_dir"])
    cfg["sft"]["output_dir"] = _abs(cfg["sft"]["output_dir"])
    for optional_stage in ("tool", "plan", "react"):
        if cfg.get(optional_stage):
            cfg[optional_stage]["output_dir"] = _abs(cfg[optional_stage]["output_dir"])
    cfg["export"]["merged_dir"] = _abs(cfg["export"]["merged_dir"])
    for stage in cfg.get("preference", {}).values():
        stage["output_dir"] = _abs(stage["output_dir"])
    return cfg


def stage_adapter(cfg: dict, stage: str) -> str:
    """단계 이름 → 해당 단계의 최종 어댑터 디렉터리 경로."""
    if stage in ("cpt", "sft", "tool", "plan", "react"):
        if stage not in cfg:
            raise ValueError(f"config에 '{stage}' 설정이 없습니다.")
        return os.path.join(cfg[stage]["output_dir"], "final")
    pref = cfg.get("preference", {})
    if stage in pref:
        return os.path.join(pref[stage]["output_dir"], "final")
    raise ValueError(f"알 수 없는 단계: {stage}")


# 채팅 템플릿별 user/assistant 구분 토큰 (응답만 학습할 때 사용)
# train_sft.py / train_tool.py / train_plan.py / train_react.py가 공용으로 사용한다.
# ReAct의 Observation은 user 블록으로 넣으므로 instruction_part로 함께 마스킹된다. tool 결과는
# Qwen/ChatML 계열에서 user 블록으로 렌더링되므로 instruction_part로 함께 마스킹된다.
TEMPLATE_PARTS = {
    "qwen-2.5":  ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "chatml":    ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama-3.1": ("<|start_header_id|>user<|end_header_id|>\n\n",
                  "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-3":   ("<|start_header_id|>user<|end_header_id|>\n\n",
                  "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}


def _abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# ---------------------------------------------------------------
# 메타정보 유틸 (자세한 설계 근거는 docs/meta_info.md 참고)
# ---------------------------------------------------------------

def slugify(name: str) -> str:
    """파일명 등을 ID에 쓸 수 있는 형태로 정규화."""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", stem).strip("-")
    return stem or "unnamed"


def content_hash(text: str) -> str:
    """내용 기반 지문 — 중복 제거·재현 확인용.

    공백을 정규화한 뒤 해시하므로 들여쓰기·줄바꿈만 다른 중복도 잡힌다.
    """
    normalized = " ".join(text.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def messages_hash(messages: list) -> str:
    """대화 전체(role 포함)의 지문. 답변만 다른 같은 질문도 구분된다."""
    joined = "\n".join(f"{m['role']}:{m.get('content') or ''}" for m in messages)
    return content_hash(joined)


def tool_sample_hash(tools: list, messages: list) -> str:
    """툴 호출 샘플의 지문 — 함수 스키마·tool_calls·결과까지 포함해 계산한다.

    tool 호출 메시지는 content가 비어 있고 tool_calls/arguments로 내용이 결정되므로
    messages_hash만으로는 서로 다른 호출을 구분하지 못한다. 전체를 정규화 직렬화한다.
    """
    payload = json.dumps({"tools": tools, "messages": messages},
                         ensure_ascii=False, sort_keys=True)
    return content_hash(payload)
