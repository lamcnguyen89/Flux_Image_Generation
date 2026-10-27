#!/usr/bin/env python3
"""Generate SDXL images from catalog settings, or validate with --dry-run."""

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation, request_snapshot, run_generation
from image_generator.workflows.sdxl import SAMPLERS


BASE_MODEL_ID = "model_a4278bb0-94f4-4960-ada0-39ca89cec194"


def seed_value(value):
    if value.lower() == "random":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("seed must be an integer or 'random'") from None


def lora_value(value):
    identifier, separator, strength = value.partition(":")
    if not identifier.strip():
        raise argparse.ArgumentTypeError("LoRA ID must not be empty")
    link = {"id": identifier, "enabled": True}
    if separator:
        try:
            link["strength"] = float(strength)
        except ValueError:
            raise argparse.ArgumentTypeError("LoRA must be ID or ID:STRENGTH") from None
    return link


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "Models/Model_Families")
    parser.add_argument("--model-id", default=BASE_MODEL_ID, help="permanent catalog model ID; defaults to SDXL base")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--guidance", dest="guidance_scale", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=seed_value, default=argparse.SUPPRESS, help="integer or random")
    parser.add_argument("--num-images", dest="num_images_per_prompt", type=int, default=1)
    parser.add_argument("--clip-skip", dest="clip_skip", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sampler", choices=["model_default", *SAMPLERS])
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--lora", type=lora_value, action="append", metavar="ID[:STRENGTH]", help="repeat for multiple adapters; replaces saved links")
    adapters.add_argument("--no-loras", action="store_true", help="disable all saved adapters for this request")
    parser.add_argument("--include-trigger-words", action="store_true", help="insert model and enabled adapter trigger words")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Images")
    parser.add_argument("--device", default="cuda:0", help="cuda, cuda:<index>, or explicit cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-dir", type=Path, help="exact Hugging Face hub cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="never download pipeline components")
    parser.add_argument("--dry-run", action="store_true", help="print validated request JSON without importing GPU libraries or generating outputs; missing catalog manifests are initialized")
    args = parser.parse_args(argv)
    overrides = {key: getattr(args, key) for key in (
        "prompt", "negative_prompt", "width", "height", "num_inference_steps",
        "guidance_scale", "seed", "num_images_per_prompt", "clip_skip",
    ) if hasattr(args, key)}
    if args.sampler is not None:
        overrides["sampling"] = {"sampler": args.sampler}
    if args.lora is not None:
        overrides["loras"] = args.lora
    elif args.no_loras:
        overrides["loras"] = []
    try:
        request = prepare_generation(
            discover_catalog(args.root), args.model_id, overrides=overrides,
            include_trigger_words=args.include_trigger_words,
        )
        if args.dry_run:
            print(json.dumps(request_snapshot(request), indent=2, ensure_ascii=False, allow_nan=False))
            return 0

        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        cache_root = Path(os.getenv("CACHE_LOCATION") or PROJECT_ROOT / "Models").expanduser()
        if not cache_root.is_absolute():
            cache_root = PROJECT_ROOT / cache_root
        results = run_generation(
            request, args.output_dir, device=args.device, dtype=args.dtype,
            cache_dir=args.cache_dir if args.cache_dir is not None else cache_root / "hub",
            local_files_only=args.local_files_only,
            token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or None,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(f"Image: {result.image_path}\nMetadata: {result.metadata_path}\nSeed: {result.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
