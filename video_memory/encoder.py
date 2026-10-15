"""One GPU model, with interchangeable Sentence Transformers checkpoints."""

import sys
from collections.abc import Sequence
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

MODEL_ALIASES = {
    "qwen": "Qwen/Qwen3-VL-Embedding-2B",
    "wemm": "tencent/WeMM-Embedding-2B",
}


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "qwen",
        *,
        device: str = "cuda",
        batch_size: int = 1,
        max_pixels: int = 1_048_576,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = MODEL_ALIASES.get(model_name, model_name)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch cannot access CUDA. Run on a GPU-enabled host or explicitly select --device cpu."
            )
        self.batch_size = batch_size
        self.max_pixels = max_pixels
        self.is_wemm = "wemm" in self.model_id.lower()
        self.trust_remote_code = trust_remote_code or self.is_wemm
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(
            f"Loading {self.model_id} on {self.device}...", file=sys.stderr, flush=True
        )
        with redirect_stdout(sys.stderr):
            self.model = SentenceTransformer(
                self.model_id,
                device=str(self.device),
                revision=revision,
                trust_remote_code=self.trust_remote_code,
                model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
            )
        self.model.eval()
        self.dimension = self.model.get_embedding_dimension()
        self.revision = getattr(self.model[0].config, "_commit_hash", None) or revision

    @torch.inference_mode()
    def encode(
        self, inputs: Sequence[str | dict], instruction: str, *, is_query: bool = False
    ) -> torch.Tensor:
        if self.is_wemm:
            encode = self.model.encode_query if is_query else self.model.encode_document
        else:
            encode = self.model.encode
        with redirect_stdout(sys.stderr):
            vectors = encode(
                list(inputs),
                prompt=instruction,
                batch_size=self.batch_size,
                processing_kwargs={
                    "image": {
                        "size": {
                            "shortest_edge": 65_536 if self.is_wemm else 4_096,
                            "longest_edge": self.max_pixels,
                        }
                    }
                },
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(inputs)
            or not torch.isfinite(vectors).all()
        ):
            raise RuntimeError("The model returned invalid embeddings.")
        return F.normalize(vectors.float(), dim=-1)

    def embed_frames(self, paths: list[str], prompts: list[dict]) -> torch.Tensor:
        if not paths:
            return torch.empty((0, len(prompts), self.dimension), dtype=torch.float32)
        inputs = [{"image": path} for path in paths]
        channels = [
            self.encode(inputs, prompt["instruction"]).cpu() for prompt in prompts
        ]
        return torch.stack(channels, dim=1)
