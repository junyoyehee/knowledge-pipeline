"""요청/응답 Pydantic 스키마."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Stage = Literal["cpt", "sft", "tool", "plan", "react", "planact",
                "dpo", "orpo", "kto"]
GenKind = Literal["qa", "tool", "plan", "react", "planact", "preference"]


class ProjectCreate(BaseModel):
    name: str
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    chat_template: str = "qwen-2.5"


class SecretsUpdate(BaseModel):
    # 값은 저장만 되고 응답엔 노출되지 않음. None이면 삭제.
    values: dict[str, Optional[str]]


class PrepareRequest(BaseModel):
    overrides: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    callback_url: Optional[str] = None


class LLMConfig(BaseModel):
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


class GenerateRequest(BaseModel):
    kind: GenKind
    params: dict[str, Any] = Field(default_factory=dict)
    llm: Optional[LLMConfig] = None
    callback_url: Optional[str] = None
    idempotency_key: Optional[str] = None


class TrainRequest(BaseModel):
    stage: Stage
    overrides: dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None
    idempotency_key: Optional[str] = None


class ExportRequest(BaseModel):
    stage: Optional[Stage] = None          # 생략 시 config의 export.source_stage
    save_gguf: Optional[bool] = None
    gguf_quantization: Optional[str] = None
    callback_url: Optional[str] = None
    idempotency_key: Optional[str] = None


class InferRequest(BaseModel):
    stage: Stage = "sft"
    questions: list[str] = Field(default_factory=list)
    max_new_tokens: Optional[int] = None
    callback_url: Optional[str] = None
