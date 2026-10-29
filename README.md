# Video Mammoth

A research project for investigating questions about a video using sampled images, supplied subtitles, instruction-guided embeddings, and a vision-language model.

The modular implementation now lives at the repository root. The legacy application has been removed.

## What it does

1. Inspect a video, sample frames, and read its optional SRT subtitles.
2. Organize saved evidence into fixed windows of at most 30 seconds. Clips reference complete overlapping subtitle cues without splitting their text or timestamps.
3. Embed every frame under each configured visual perspective and each unique referenced subtitle cue once.
4. Save float32 vectors in chunked PyTorch files, then search them with one query embedding per search.
5. Investigate a question: build an outline, plan searches, inspect original evidence, keep notes, search again, and review a proposed answer against its cited sources.

Preparation, indexing, retrieval, model servers, and the investigation coordinator are implemented. Component checks and offline investigation checks have passed. A complete live investigation through the actual models remains to be verified. Indexing performance tuning is deferred.

## Project layout

```text
packages/
  model-clients/       Typed requests and compatible HTTP model clients
  core/                Media, indexing, storage, retrieval, and investigation
model-servers/
  embedding/           Resident local embedding model and HTTP endpoint
  vlm/                 Resident local VLM and HTTP endpoint
config/
  models.toml          Client URLs, model IDs, capabilities, and timeouts
  servers.toml         Local model loading, inference, and processing settings
  embedding_prompts.toml
apps/
  api/                 Reserved for the application API
  cli/                 Reserved for the CLI
  web/                 Reserved for a web UI
tests/                 Reserved for maintained tests
data/                  Local source data; ignored by Git
memory/                Prepared evidence and embeddings; ignored by Git
```

The application API, CLI, web UI, and maintained test suite are not implemented yet. A Python caller currently connects the core functions.

## Setup

Use Python 3.13 or newer and UV. The local inference servers require a CUDA GPU; frame preparation also requires CUDA video decoding and `ffprobe` on the executable path.

Run commands from the repository root:

```bash
uv sync --all-packages --locked
```

Model weights are loaded by the separate server processes. Opening a model client creates HTTP connections; it does not load model weights.

## Start the local model servers

Run each command in its own terminal at the repository root.

Embedding server:

```bash
uv run --package video-mammoth-embedding-server uvicorn video_mammoth_embedding_server.app:create_app --factory --host 127.0.0.1 --port 8001
```

VLM server:

```bash
uv run --package video-mammoth-vlm-server uvicorn video_mammoth_vlm_server.app:create_app --factory --host 127.0.0.1 --port 8002
```

The current configuration selects `Qwen/Qwen3-VL-Embedding-2B` and `Qwen/Qwen3.5-2B`. Each server loads its model at startup and reuses it across requests. `/health` reports the model and loaded settings.

Only the embedding server is needed for indexing. Investigation uses both servers. Restart a server after changing its settings.

The server configuration defaults to `config/servers.toml`. Set `VIDEO_MAMMOTH_SERVER_CONFIG` to use another file. Local client profiles require no API keys; external profiles can name an environment variable through `api_key_env`.

## Core entry points

| Phase | Calls | Result |
| --- | --- | --- |
| Prepare | `probe_video()` → `prepare_video()` | Saved PNG images and a `PreparedEvidence` manifest containing frames, cues, and clip references. |
| Index | `load_embedding_prompts()` → `embed_prepared()` | A saved embedding collection with channel instructions, model metadata, and vector chunks. |
| Reopen | `load_prepared()` → `open_embeddings()` | Validated source catalogs and lazy access to saved vectors. |
| Search | `search_index()` | Ranked candidate clips with matching source IDs, timestamps, and scores. |
| Investigate | `investigate()` | A completed or incomplete outcome, with notebook events, evidence references, answer review, and saved result. |

Use `create_model_clients()` as an async context manager to provide the VLM and embedding clients to those operations. The caller supplies paths, question, budgets, and the appropriate token counter.

Local indexing state starts empty: previous indexes, prepared frames, cached outlines, and run/notebook output have been removed. Prepare and index a video to create a new collection under a caller-chosen directory such as `memory/<video>/prepared`. Source videos and subtitles remain local input files and are ignored by Git.

## Documentation

- [Architecture and system walkthrough](SYSTEM_WALKTHROUGH.md): the standalone reference for agents and contributors, including the package map, complete flow, data types, configuration, persistence, verification status, current limits, and a fictional example.

## Local artifacts and Git

Source media, prepared frames, vector collections, generated runs, model weights, environments, caches, and credentials are excluded by `.gitignore`. Keep the UV lockfile, source code, configuration templates, and documentation in Git. Images and JSON files used as application source are not globally ignored.

Saved preparation manifests record source-file paths and identities. Keep the original files unchanged, and move a prepared directory together with its frames and embedding collection. Notebook recovery does not currently implement full investigation resume.
