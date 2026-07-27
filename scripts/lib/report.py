"""학습 실시간 진행률 리포트 (#9).

- write_progress: 진행률 dict를 원자적으로 파일에 기록(러너가 부분 파일을 읽지 않도록 tmp→replace)
- progress_from_state: TrainerState/logs → 진행률 dict (순수 함수, transformers 비의존 → 테스트 가능)
- make_progress_callback: transformers TrainerCallback 생성 (transformers는 지연 import)

원격 GPU 워커(#8①)에서는 워커가 로컬 progress 파일을 폴링해 공유 DB에 반영하므로
실시간 진행률이 그대로 API로 노출된다. SSH 러너(#8②)는 최종 리포트만 회수된다.
"""
import json
import os


def write_progress(path: str, data: dict) -> None:
    """진행률 JSON을 원자적으로 기록. path가 없으면 무시."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)   # 원자적 교체


def progress_from_state(step, total, logs: dict | None) -> dict:
    """(global_step, max_steps, logs) → 진행률 dict. None 값은 제외."""
    d: dict = {"step": int(step) if step else None,
               "total_steps": int(total) if total else None}
    if isinstance(logs, dict):
        for src, dst in (("loss", "loss"), ("eval_loss", "eval_loss"),
                         ("learning_rate", "lr"), ("epoch", "epoch")):
            v = logs.get(src)
            if v is not None:
                d[dst] = v
    if d.get("step") and d.get("total_steps"):
        d["pct"] = round(100.0 * d["step"] / d["total_steps"], 1)
    return {k: v for k, v in d.items() if v is not None}


def make_progress_callback(path: str):
    """path에 진행률을 주기적으로 기록하는 TrainerCallback. path 없으면 None.

    transformers는 함수 안에서 지연 import하므로, 이 모듈 자체는 GPU/transformers
    없이도 import된다(순수 헬퍼 테스트 용이).
    """
    if not path:
        return None
    from transformers import TrainerCallback

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            write_progress(path, progress_from_state(
                getattr(state, "global_step", None),
                getattr(state, "max_steps", None), logs or {}))

        def on_train_end(self, args, state, control, **kwargs):
            data = progress_from_state(getattr(state, "global_step", None),
                                       getattr(state, "max_steps", None), {})
            data["done"] = True
            write_progress(path, data)

    return _ProgressCallback()
