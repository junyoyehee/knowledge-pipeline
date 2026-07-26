"""공용 유틸리티 — config 로드, 경로 처리."""
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = None) -> dict:
    """config.yaml을 로드하고 상대 경로를 프로젝트 루트 기준 절대 경로로 변환."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 경로 필드를 절대 경로로 변환
    for key in ("raw_dir", "cpt_dataset", "sft_dataset"):
        cfg["data"][key] = _abs(cfg["data"][key])
    cfg["cpt"]["output_dir"] = _abs(cfg["cpt"]["output_dir"])
    cfg["sft"]["output_dir"] = _abs(cfg["sft"]["output_dir"])
    cfg["export"]["merged_dir"] = _abs(cfg["export"]["merged_dir"])
    return cfg


def _abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)
