from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .config import load_settings
from .runtime import EmbeddingRuntime


def create_app(config_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(config_path)
    runtime = EmbeddingRuntime(settings)
    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        try:
            runtime.load()
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Video Mammoth Embedding Server",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, JsonValue]:
        return {
            "status": "ready",
            "model": runtime.model_id,
            "settings": settings.model_dump(mode="json"),
        }

    @app.post("/v1/embeddings")
    def embeddings(payload: EmbeddingPayload) -> dict[str, JsonValue]:
        if payload.model != runtime.model_id:
            raise HTTPException(
                status_code=404,
                detail="Unknown embedding model.",
            )

        try:
            with inference_lock:
                vectors = runtime.embed(
                    payload.input,
                    instruction=payload.instruction,
                    input_type=payload.input_type,
                )

        except ValueError as cause:
            raise HTTPException(
                status_code=400,
                detail=str(cause),
            ) from cause

        except torch.cuda.OutOfMemoryError as cause:
            raise HTTPException(
                status_code=503,
                detail=("GPU memory exhausted. Reduce batch_size or max_pixels."),
            ) from cause

        return {
            "object": "list",
            "model": runtime.model_id,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector,
                }
                for index, vector in enumerate(vectors)
            ],
        }

    return app


class EmbeddingPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    model: str = Field(min_length=1, pattern=r"^\S+$")
    input: list[dict[str, JsonValue]] = Field(
        min_length=1,
    )
    instruction: str | None = None
    input_type: Literal[
        "search_query",
        "search_document",
    ] = "search_document"
    encoding_format: Literal["float"] = "float"
