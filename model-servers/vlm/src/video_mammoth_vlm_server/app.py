from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import time
from typing import Literal
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from pydantic import Field, JsonValue

from .config import load_settings
from .runtime import ChatMessage, VLMRuntime, WireModel
from .structured import StructuredOutputError


def create_app(config_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(config_path)
    runtime = VLMRuntime(settings)
    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        try:
            runtime.load()
            yield
        finally:
            runtime.close()

    app = FastAPI(title="Video Mammoth VLM Server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, JsonValue]:
        return {
            "status": "busy" if inference_lock.locked() else "ready",
            "model": runtime.model_id,
            "capabilities": {"json_schema": True},
            "settings": settings.model_dump(mode="json"),
        }

    @app.post("/v1/chat/completions")
    def chat(payload: ChatPayload) -> dict[str, JsonValue]:
        if payload.model != runtime.model_id:
            raise HTTPException(status_code=404, detail="Unknown VLM model.")

        if not inference_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="The VLM is busy. Retry after the current request.",
            )

        try:
            result = runtime.generate(
                payload.messages,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                json_schema=(
                    payload.response_format.json_schema.schema_
                    if payload.response_format is not None
                    else None
                ),
            )
        except ValueError as cause:
            raise HTTPException(status_code=400, detail=str(cause)) from cause
        except StructuredOutputError as cause:
            raise HTTPException(status_code=502, detail=str(cause)) from cause
        except torch.cuda.OutOfMemoryError as cause:
            raise HTTPException(
                status_code=503,
                detail="GPU memory exhausted. Use fewer or smaller images.",
            ) from cause
        finally:
            inference_lock.release()

        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time()),
            "model": runtime.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
        }

    return app


class SchemaDefinition(WireModel):
    name: str = Field(min_length=1)
    strict: Literal[True] = True
    schema_: dict[str, JsonValue] = Field(alias="schema")


class ResponseFormat(WireModel):
    type: Literal["json_schema"]
    json_schema: SchemaDefinition


class ChatPayload(WireModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    response_format: ResponseFormat | None = None
    stream: Literal[False] = False
    n: Literal[1] = 1
