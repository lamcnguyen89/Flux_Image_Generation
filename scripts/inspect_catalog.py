#!/usr/bin/env python3
"""Inspect catalog data without installing or importing GPU dependencies."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_generator.catalog import CatalogValidationError, discover_catalog


def asset_document(asset):
    return {
        "id": asset.id,
        "kind": asset.kind,
        "manifest_path": str(asset.manifest_path),
        "user_path": str(asset.user_path),
        "defaults": asset.defaults,
        "user_overrides": asset.user_overrides,
        "asset_paths": {
            key: str(value) if value is not None else None
            for key, value in asset.asset_paths.items()
        },
        "effective_settings": asset.resolve_settings(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "Models/Model_Families",
        help="model family directory (defaults to this project's Models/Model_Families)",
    )
    parser.add_argument("--id", dest="asset_id", help="inspect one model or LoRA by permanent ID")
    parser.add_argument("--json", action="store_true", help="print full catalog data and resolved settings as JSON")
    args = parser.parse_args(argv)
    try:
        catalog = discover_catalog(args.root)
    except CatalogValidationError as exc:
        if args.json:
            print(json.dumps({"errors": [
                {"path": str(issue.path), "field": issue.field, "message": issue.message}
                for issue in exc.issues
            ]}, indent=2))
        else:
            print(str(exc), file=sys.stderr)
        return 1

    if args.asset_id:
        try:
            asset = catalog.get_asset(args.asset_id)
        except KeyError:
            if args.json:
                print(json.dumps({"errors": [{"field": "id", "message": f"unknown asset ID {args.asset_id!r}"}]}))
            else:
                print(f"Unknown asset ID: {args.asset_id}", file=sys.stderr)
            return 1
        print(json.dumps(asset_document(asset), indent=2, allow_nan=False))
    elif args.json:
        print(json.dumps({
            "root": str(catalog.root),
            "models": [asset_document(asset) for asset in catalog.models.values()],
            "loras": [asset_document(asset) for asset in catalog.loras.values()],
        }, indent=2, allow_nan=False))
    else:
        print(f"Catalog: {catalog.root}")
        for label, assets in (("Models", catalog.models), ("LoRAs", catalog.loras)):
            print(f"\n{label} ({len(assets)}):")
            for asset in assets.values():
                print(f"  {asset.display_name} [{asset.family}]\n    {asset.id}")
        print("\nUse --id <ID> for paths and effective settings, or --json for the full catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
