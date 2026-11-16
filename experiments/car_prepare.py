"""Prepare a content-disjoint HPatches pair manifest; does not run inference."""

import argparse
import json
from pathlib import Path

from .car_data import make_hpatches_manifest
from .common import digest_file


def validate_partitions(pairs, image_hashes=None):
    owners = {}
    ids = set()
    for pair in pairs:
        if pair["id"] in ids:
            raise ValueError("Duplicate pair ID")
        ids.add(pair["id"])
        for image in (pair["image0"], pair["image1"]):
            image = str(Path(image).resolve())
            if image in owners and owners[image] != pair["partition"]:
                raise ValueError(f"Image leakage across partitions: {image}")
            owners[image] = pair["partition"]
    if image_hashes is not None:
        content_owners = {}
        for image, partition in owners.items():
            digest = image_hashes[image]
            if digest in content_owners and content_owners[digest] != partition:
                raise ValueError(f"Identical image content across partitions: {image}")
            content_owners[digest] = partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/hpatches-sequences-release")
    )
    parser.add_argument("--output", type=Path, default=Path("data/hpatches_pairs.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Manifest already exists; use it or choose a new --output")
    if args.verbose:
        print(
            f"Scanning full-image sequences in {args.data}; seed={args.seed}",
            flush=True,
        )
    manifest = make_hpatches_manifest(args.data, args.seed)
    paths = {
        str(Path(p[k]).resolve())
        for p in manifest["pairs"]
        for k in ("image0", "image1")
    }
    if args.verbose:
        print(
            f"Checking image content across partitions ({len(paths)} images)",
            flush=True,
        )
    validate_partitions(manifest["pairs"], {p: digest_file(Path(p)) for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    if args.verbose:
        for partition, sequences in manifest["sequences"].items():
            count = sum(p["partition"] == partition for p in manifest["pairs"])
            print(f"{partition}: {len(sequences)} sequences, {count} pairs", flush=True)
    print(f"Manifest: {args.output}", flush=True)


if __name__ == "__main__":
    main()
