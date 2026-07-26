"""프로젝트 · 시크릿 · 파일 · 데이터셋 · 어댑터/모델 라우터."""
import shutil

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from .. import config, secrets, storage, store
from ..auth import require_api_key
from ..schemas import ProjectCreate, SecretsUpdate
from ._common import err, get_project_or_404

router = APIRouter(dependencies=[Depends(require_api_key)])


# ---------- 프로젝트 ----------

@router.post("/projects", status_code=201)
def create_project(body: ProjectCreate):
    p = store.create_project(body.name, body.base_model, body.chat_template)
    storage.init_project_dirs(p["id"])
    return p


@router.get("/projects")
def list_projects():
    return {"projects": store.list_projects()}


@router.get("/projects/{pid}")
def get_project(pid: str):
    project = get_project_or_404(pid)
    return {**project,
            "datasets": _list_datasets(pid),
            "adapters": _list_adapters(pid),
            "secrets": secrets.secret_names(pid)}


@router.delete("/projects/{pid}", status_code=204)
def delete_project(pid: str, confirm: bool = False):
    get_project_or_404(pid)
    if not confirm:
        raise err(400, "invalid_argument", "삭제하려면 ?confirm=true 가 필요합니다.")
    store.delete_project(pid)
    shutil.rmtree(storage.project_dir(pid), ignore_errors=True)


# ---------- 시크릿 ----------

@router.put("/projects/{pid}/secrets", status_code=204)
def put_secrets(pid: str, body: SecretsUpdate):
    get_project_or_404(pid)
    secrets.save_secrets(pid, body.values)


# ---------- 파일 ----------

@router.post("/projects/{pid}/raw-files", status_code=201)
async def upload_raw(pid: str, file: UploadFile = File(...),
                     kind: str = Form("doc")):
    get_project_or_404(pid)
    if kind not in storage.KIND_PREFIX:
        raise err(400, "invalid_argument",
                  f"kind는 {list(storage.KIND_PREFIX)} 중 하나여야 합니다.")
    name = storage.raw_filename(kind, file.filename or "upload")
    dest = storage.raw_dir(pid) / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close(); dest.unlink(missing_ok=True)
                raise err(400, "invalid_argument", "업로드 크기 상한 초과")
            out.write(chunk)

    lines = None
    if name.endswith(".jsonl"):
        with open(dest, "r", encoding="utf-8") as fh:
            lines = sum(1 for line in fh if line.strip())
    return {"name": name, "kind": kind, "bytes": size, "lines": lines}


@router.get("/projects/{pid}/raw-files")
def list_raw(pid: str):
    get_project_or_404(pid)
    d = storage.raw_dir(pid)
    files = [{"name": f.name, "bytes": f.stat().st_size}
             for f in sorted(d.glob("*")) if f.is_file()]
    return {"files": files}


@router.delete("/projects/{pid}/raw-files/{name}", status_code=204)
def delete_raw(pid: str, name: str):
    get_project_or_404(pid)
    target = storage.raw_dir(pid) / name
    # 경로 탈출 방지
    if target.parent != storage.raw_dir(pid) or not target.exists():
        raise err(404, "not_found", f"파일 없음: {name}")
    target.unlink()


# ---------- 데이터셋 ----------

def _list_datasets(pid: str) -> list[dict]:
    out = []
    for f in sorted(storage.processed_dir(pid).glob("*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
        out.append({"kind": f.stem.replace("_dataset", ""), "file": f.name,
                    "count": n})
    return out


@router.get("/projects/{pid}/datasets")
def list_datasets(pid: str):
    get_project_or_404(pid)
    return {"datasets": _list_datasets(pid)}


@router.get("/projects/{pid}/datasets/{kind}")
def get_dataset(pid: str, kind: str, sample: int = 3, download: bool = False):
    get_project_or_404(pid)
    path = storage.processed_dir(pid) / f"{kind}_dataset.jsonl"
    if not path.exists():
        raise err(404, "not_found", f"데이터셋 없음: {kind}")
    if download:
        return FileResponse(path, filename=path.name,
                            media_type="application/x-ndjson")
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= sample:
                break
    return {"kind": kind, "sample": rows}


# ---------- 어댑터 / 모델 ----------

def _list_adapters(pid: str) -> list[dict]:
    out = []
    base = storage.outputs_dir(pid)
    for stage in ("cpt", "sft", "tool", "plan", "react", "planact",
                  "dpo", "orpo", "kto"):
        final = base / stage / "final"
        if final.is_dir():
            out.append({"stage": stage, "path": str(final)})
    return out


@router.get("/projects/{pid}/adapters")
def list_adapters(pid: str):
    get_project_or_404(pid)
    return {"adapters": _list_adapters(pid)}


@router.get("/projects/{pid}/models")
def list_models(pid: str):
    get_project_or_404(pid)
    merged = storage.outputs_dir(pid) / "final_model"
    models = []
    if merged.is_dir():
        models.append({"id": "final_model", "path": str(merged)})
    gguf = storage.outputs_dir(pid) / "final_model_gguf"
    if gguf.is_dir():
        models.append({"id": "final_model_gguf", "path": str(gguf)})
    return {"models": models}
