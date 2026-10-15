"""The CLI delegates to the same functions that an agent can call later."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .indexing import DEFAULT_PROMPTS, index_video, load_prompts
from .media import export_clip
from .retrieval import search
from .storage import get_clip, list_indexes, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index and search videos using SRT transcripts and prompted frame embeddings."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser(
        "index", help="Build or resume an ordered video memory."
    )
    index.add_argument("video", type=Path)
    index.add_argument("--srt", type=Path, required=True)
    index.add_argument("--memory", type=Path, default=Path("memory"))
    index.add_argument(
        "--output",
        type=Path,
        help="Explicit index directory instead of an automatically unique directory.",
    )
    index.add_argument(
        "--model",
        default="qwen",
        help="qwen, wemm, or a Sentence Transformers model ID/path.",
    )
    index.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    index.add_argument("--clip-seconds", type=float, default=30)
    index.add_argument("--boundary-slack", type=float, default=5)
    index.add_argument("--fps", type=float, default=1)
    index.add_argument(
        "--subtitle-offset",
        type=float,
        default=0,
        help="Seconds to add to all SRT cue timestamps.",
    )
    index.add_argument("--batch-size", type=int, default=1)
    index.add_argument("--max-pixels", type=int, default=1_048_576)
    index.add_argument("--device", default="cuda")
    index.add_argument("--decode-device", choices=("cuda", "cpu"), default="cuda")
    index.add_argument(
        "--copy-video",
        action="store_true",
        help="Copy the video into the memory for a self-contained index.",
    )
    index.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code; enabled automatically for the wemm alias.",
    )
    index.add_argument(
        "--model-revision", help="Pin a Hugging Face checkpoint revision."
    )

    query = commands.add_parser(
        "query", aliases=["search"], help="Rank unique candidate clips."
    )
    query.add_argument("query")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument(
        "--channels", nargs="+", help="For example: transcript ocr objects"
    )
    query.add_argument("--fusion", choices=("max", "rrf"), default="max")
    query.add_argument(
        "--frame-top-n",
        type=int,
        default=1,
        help="Average the N strongest frames per visual channel; default keeps a single peak.",
    )
    query.add_argument(
        "--query-instruction",
        help="Override the saved query instructions for this search.",
    )
    query.add_argument("--device", default="cuda")
    query.add_argument(
        "--output",
        type=Path,
        help="Also save the complete machine-readable result as JSON.",
    )

    show = commands.add_parser(
        "show", help="Read a clip's original subtitle cues and all sampled frame paths."
    )
    show.add_argument("clip_id")
    show.add_argument("--index", type=Path, required=True)

    export = commands.add_parser(
        "export", help="Render an MP4 clip and its clip-relative SRT on demand."
    )
    export.add_argument("clip_id")
    export.add_argument("--index", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    listing = commands.add_parser(
        "list", help="List indexed videos without loading a model."
    )
    listing.add_argument("--memory", type=Path, default=Path("memory"))
    for command in (index, query, show, export, listing):
        command.add_argument(
            "--json",
            action="store_true",
            help="Write JSON to stdout; progress stays on stderr.",
        )
    return parser


def display_candidates(result: dict) -> None:
    print(
        f"{len(result['candidates'])} candidates from {result['total_clips']} clips; fusion={result['fusion']}"
    )
    for rank, candidate in enumerate(result["candidates"], 1):
        print(
            f"{rank}. Clip {candidate['clip_id']}  {candidate['start']}–{candidate['end']}  score={candidate['score']:.4f}  best={candidate['best_channel']}"
        )
        print(
            "   "
            + "; ".join(
                f"{name}={score:.4f}"
                for name, score in candidate["channel_scores"].items()
            )
        )
        if candidate["frame_matches"]:
            best = candidate["frame_matches"][0]
            print(f"   Best frame: {best['timestamp']}  {best['path']}")
        if candidate["transcript"]:
            print("   Transcript: " + " ".join(candidate["transcript"].split())[:180])


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "index":
            directory = index_video(
                args.video,
                args.srt,
                memory_directory=args.memory,
                output=args.output,
                model_name=args.model,
                prompts=load_prompts(args.prompts),
                clip_seconds=args.clip_seconds,
                boundary_slack=args.boundary_slack,
                fps=args.fps,
                subtitle_offset=args.subtitle_offset,
                batch_size=args.batch_size,
                max_pixels=args.max_pixels,
                device=args.device,
                decode_device=args.decode_device,
                copy_video=args.copy_video,
                trust_remote_code=args.trust_remote_code,
                model_revision=args.model_revision,
            )
            result = {"index": str(directory), "status": "complete"}
            if not args.json:
                print(f"Saved video memory: {directory}")
        elif args.command in {"query", "search"}:
            result = search(
                args.index,
                args.query,
                top_k=args.top_k,
                channels=args.channels,
                fusion=args.fusion,
                frame_top_n=args.frame_top_n,
                device=args.device,
                query_instruction=args.query_instruction,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, result)
            if not args.json:
                display_candidates(result)
        elif args.command == "show":
            result = get_clip(args.index, args.clip_id)
            if not args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "export":
            result = export_clip(args.index, args.clip_id, args.output)
            if not args.json:
                print(f"Exported {result['start']}–{result['end']}: {result['video']}")
        else:
            result = list_indexes(args.memory.expanduser().resolve())
            if not args.json:
                for item in result:
                    print(
                        f"{item['video_id']}  {item['status']}  {item['duration']}  {item['completed_clips']}/{item['clips']} clips  {item['index']}"
                    )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
