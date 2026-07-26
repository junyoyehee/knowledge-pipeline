"""파이프라인 실행 라우터 — prepare / generate / train / export / infer.

모두 비동기 Job을 생성하고 202로 반환한다.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .. import storage, store
from ..auth import require_api_key
from ..schemas import (ExportRequest, GenerateRequest, InferRequest,
                       PrepareRequest, TrainRequest)
from ._common import err, get_project_or_404

router = APIRouter(dependencies=[Depends(require_api_key)])


def _accepted(job: dict) -> JSONResponse:
    return JSONResponse(status_code=202, content=job)


@router.post("/projects/{pid}/datasets:prepare")
def prepare(pid: str, body: PrepareRequest):
    get_project_or_404(pid)
    job = store.create_job(pid, "prepare", {"overrides": body.overrides},
                           callback_url=body.callback_url,
                           idempotency_key=body.idempotency_key)
    return _accepted(job)


@router.post("/projects/{pid}/datasets:generate")
def generate(pid: str, body: GenerateRequest):
    get_project_or_404(pid)
    params = {"kind": body.kind, **body.params}
    if body.llm:
        params["llm"] = body.llm.model_dump(exclude_none=True)
    job = store.create_job(pid, "generate", params,
                           callback_url=body.callback_url,
                           idempotency_key=body.idempotency_key)
    return _accepted(job)


@router.post("/projects/{pid}/train")
def train(pid: str, body: TrainRequest):
    get_project_or_404(pid)
    _require_prereqs(pid, body.stage, body.overrides)
    job = store.create_job(pid, "train", {"overrides": body.overrides},
                           stage=body.stage, callback_url=body.callback_url,
                           idempotency_key=body.idempotency_key)
    return _accepted(job)


@router.post("/projects/{pid}/export")
def export(pid: str, body: ExportRequest):
    get_project_or_404(pid)
    overrides = {"export": {}}
    if body.save_gguf is not None:
        overrides["export"]["save_gguf"] = body.save_gguf
    if body.gguf_quantization:
        overrides["export"]["gguf_quantization"] = body.gguf_quantization
    job = store.create_job(pid, "export", {"overrides": overrides},
                           stage=body.stage, callback_url=body.callback_url,
                           idempotency_key=body.idempotency_key)
    return _accepted(job)


@router.post("/projects/{pid}/infer")
def infer(pid: str, body: InferRequest):
    get_project_or_404(pid)
    adapter = storage.outputs_dir(pid) / body.stage / "final"
    if not adapter.is_dir():
        raise err(409, "conflict",
                  f"'{body.stage}' 어댑터가 없습니다. 먼저 학습하세요.")
    params = {"questions": body.questions, "max_new_tokens": body.max_new_tokens}
    job = store.create_job(pid, "infer", params, stage=body.stage,
                           callback_url=body.callback_url)
    return _accepted(job)


def _require_prereqs(pid: str, stage: str, overrides: dict) -> None:
    """선행 어댑터가 필요한 단계는 존재 여부를 검증한다 (없으면 409)."""
    out = storage.outputs_dir(pid)

    def has(s: str) -> bool:
        return (out / s / "final").is_dir()

    scfg = (overrides or {}).get(stage, {}) if isinstance(overrides, dict) else {}

    if stage == "sft":
        # continue_from_cpt 기본 True — cpt 어댑터가 있으면 이어받고, 없으면 베이스에서 시작
        return
    if stage in ("tool", "plan", "react", "planact"):
        init = scfg.get("init_from", "sft")
        if init in ("cpt", "sft", "tool", "plan", "react", "planact") and not has(init):
            raise err(409, "conflict",
                      f"init_from='{init}' 어댑터가 없습니다. 먼저 {init} 단계를 학습하세요.")
    # dpo/orpo/kto의 init_from은 config 기본값에 의존 — 스크립트가 자체 검증
