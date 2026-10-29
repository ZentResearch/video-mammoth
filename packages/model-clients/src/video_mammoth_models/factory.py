import os
from collections.abc import AsyncGenerator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import EndpointProfile, ModelProfiles
from .contracts import EmbeddingClient, VLMClient
from .embeddings import OpenAICompatibleEmbeddings
from .errors import ErrorKind, ModelClientError
from .vlm import OpenAICompatibleVLM


@dataclass(frozen=True, slots=True)
class ModelClients:
    vlm: VLMClient
    embeddings: EmbeddingClient


def _api_key(
    profile: EndpointProfile,
    environ: Mapping[str, str],
) -> str:
    name = profile.api_key_env

    if name is None:
        # Explicit placeholder prevents fallback to OPENAI_API_KEY.
        return "not-required"

    value = environ.get(name, "").strip()

    if not value or any(character.isspace() for character in value):
        raise ModelClientError(
            ErrorKind.AUTHENTICATION,
            f"Set a valid API key in environment variable {name}.",
        )

    return value


def _sdk(
    profile: EndpointProfile,
    api_key: str,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=profile.base_url,
        timeout=profile.timeout_seconds,
        max_retries=0,
    )


@asynccontextmanager
async def create_model_clients(
    profiles: ModelProfiles,
    *,
    environ: Mapping[str, str] | None = None,
) -> AsyncGenerator[ModelClients]:
    environment = os.environ if environ is None else environ

    vlm_key = _api_key(profiles.vlm, environment)
    embedding_key = _api_key(profiles.embeddings, environment)

    async with AsyncExitStack() as stack:
        vlm_sdk = await stack.enter_async_context(_sdk(profiles.vlm, vlm_key))
        embedding_sdk = await stack.enter_async_context(
            _sdk(profiles.embeddings, embedding_key)
        )

        yield ModelClients(
            vlm=OpenAICompatibleVLM(
                profiles.vlm,
                vlm_sdk,
            ),
            embeddings=OpenAICompatibleEmbeddings(
                profiles.embeddings,
                embedding_sdk,
            ),
        )
