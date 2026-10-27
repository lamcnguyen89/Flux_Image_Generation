#!/usr/bin/env python3
"""Run WAN 2.2 image-to-video generation from the model catalog."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_generator.cli import run_family


def main(argv=None):
    return run_family("wan", argv)


if __name__ == "__main__":
    raise SystemExit(main())
