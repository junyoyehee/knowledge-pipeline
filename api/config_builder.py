"""요청별 config.yaml 생성 (docs/api_design.md §8).

최종 config = 템플릿(configs/config.yaml)
              ⊕ 프로젝트 설정(model.name, chat_template)
              ⊕ 서버 주입 경로(프로젝트 스토리지 기준 절대경로)
              ⊕ 요청 overrides(화이트리스트 필드만)

경로 필드는 서버가 강제로 주입하며, 클라이언트 override로는 절대 바꿀 수 없다
(경로 탈출 방지).
"""
import copy

import yaml

from . import storage
from .config import TEMPLATE_CONFIG

# override로 건드릴 수 있는 최상위 섹션
_ALLOWED_TOP = {"model", "data", "cpt", "sft", "tool", "plan", "react",
                "planact", "preference", "export"}
# override로 절대 못 바꾸는 경로 (프로젝트/서버 관리 영역)
_FORBIDDEN_PATHS = {"model.name", "model.chat_template"}


def _blocked(path: str) -> bool:
    if path in _FORBIDDEN_PATHS:
        return True
    leaf = path.split(".")[-1]
    return (leaf.endswith("_dir") or leaf.endswith("_dataset")
            or leaf in {"raw_dir", "output_dir", "merged_dir"})


def _allowed(path: str) -> bool:
    return path.split(".")[0] in _ALLOWED_TOP and not _blocked(path)


def _load_template() -> dict:
    with open(TEMPLATE_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _inject_paths(cfg: dict, pid: str) -> None:
    """모든 경로 필드를 프로젝트 스토리지 기준 절대경로로 강제 주입."""
    raw = storage.raw_dir(pid)
    proc = storage.processed_dir(pid)
    out = storage.outputs_dir(pid)

    cfg["data"]["raw_dir"] = str(raw)
    dataset_files = {
        "cpt_dataset": "cpt_dataset.jsonl", "sft_dataset": "sft_dataset.jsonl",
        "tool_dataset": "tool_dataset.jsonl", "plan_dataset": "plan_dataset.jsonl",
        "react_dataset": "react_dataset.jsonl",
        "planact_dataset": "planact_dataset.jsonl",
        "pref_dataset": "pref_dataset.jsonl", "kto_dataset": "kto_dataset.jsonl",
    }
    for key, fname in dataset_files.items():
        if key in cfg["data"]:
            cfg["data"][key] = str(proc / fname)

    for stage in ("cpt", "sft", "tool", "plan", "react", "planact"):
        if stage in cfg:
            cfg[stage]["output_dir"] = str(out / stage)
    for name, stage_cfg in cfg.get("preference", {}).items():
        stage_cfg["output_dir"] = str(out / name)
    cfg["export"]["merged_dir"] = str(out / "final_model")


def _apply_overrides(cfg: dict, overrides: dict, prefix: str = "") -> list[str]:
    """화이트리스트에 맞는 override만 깊은 병합. 거부된 경로 목록을 반환."""
    rejected: list[str] = []
    for key, val in (overrides or {}).items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            node = cfg.get(key)
            if not isinstance(node, dict):
                # 경로가 허용될 때만 새 dict 생성
                if not _allowed(path):
                    rejected.append(path)
                    continue
                node = cfg[key] = {}
            rejected += _apply_overrides(node, val, path + ".")
        else:
            if _allowed(path):
                cfg[key] = val
            else:
                rejected.append(path)
    return rejected


def build_config(pid: str, project: dict, overrides: dict | None) -> tuple[dict, list[str]]:
    """(최종 config dict, 거부된 override 경로) 반환."""
    cfg = _load_template()
    cfg["model"]["name"] = project["base_model"]
    cfg["model"]["chat_template"] = project["chat_template"]
    _inject_paths(cfg, pid)
    rejected = _apply_overrides(cfg, copy.deepcopy(overrides or {}))
    return cfg, rejected


def write_config(cfg: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
