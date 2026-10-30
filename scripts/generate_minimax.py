#!/usr/bin/env python3
"""Generate MiniMax H3 videos from catalog manifests, or validate with --dry-run."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_generator.cli import run_family


if __name__ == "__main__":
    raise SystemExit(run_family("minimax"))
