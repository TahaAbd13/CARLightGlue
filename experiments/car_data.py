"""Reproducible, sequence-disjoint controller data and frozen feature caching."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from lightglue import SuperPoint
from lightglue.utils import load_image

from .common import digest_file, digest_tensors, discover_sequences


def group_identical_content(sequences, groups):
    """Keep sequences sharing even one identical image in a single partition."""
    parent = {path.name: path.name for path in sequences}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    seen = {}
    for path in sequences:
        for image in sorted(path.glob("*.ppm")):
            digest = digest_file(image)
            if digest in seen:
                parent[root(path.name)] = root(seen[digest])
            seen[digest] = path.name
    components = {}
    for name in parent:
        components.setdefault(root(name), []).append(name)
    priority = {name: index for index, name in enumerate(groups)}
    original = {name: split for split, names in groups.items() for name in names}
    assigned = {split: [] for split in groups}
    membership = {}
    for representative, names in components.items():
        split = min((original[name] for name in names), key=priority.get)
        assigned[split].extend(names)
        membership.update({name: representative for name in names})
    duplicates = [sorted(names) for names in components.values() if len(names) > 1]
    return assigned, membership, duplicates


def make_hpatches_manifest(root, seed=17):
    sequences = discover_sequences(Path(root))
    rng = np.random.default_rng(seed)
    groups = {name: [] for name in ("train", "validation", "calibration", "test")}
    for prefix in ("i_", "v_"):
        names = [p.name for p in sequences if p.name.startswith(prefix)]
        rng.shuffle(names)
        a, b = int(len(names) * 0.55), int(len(names) * 0.15)
        for split, subset in zip(
            groups,
            (names[:a], names[a : a + b], names[a + b : a + 2 * b], names[a + 2 * b :]),
        ):
            groups[split].extend(subset)
    groups, content_groups, duplicates = group_identical_content(sequences, groups)
    by_name = {p.name: p for p in sequences}
    pairs = []
    for split, names in groups.items():
        names.sort()
        for index, name in enumerate(names):
            # Same-partition distractors only. Labels are explicitly weak/assumed.
            candidates = [
                names[(index + len(names) // 2 + offset) % len(names)]
                for offset in range(len(names))
            ]
            other = next(
                (
                    candidate
                    for candidate in candidates
                    if content_groups[candidate] != content_groups[name]
                ),
                None,
            )
            if other is None:
                raise ValueError("Each partition needs at least two distinct scenes")
            for target in range(2, 7):
                common = {
                    "partition": split,
                    "sequence": name,
                    "image0": str((by_name[name] / "1.ppm").resolve()),
                }
                pairs.append(
                    {
                        **common,
                        "id": f"{name}/1-{target}",
                        "kind": "illumination"
                        if name.startswith("i_")
                        else "viewpoint",
                        "image1": str((by_name[name] / f"{target}.ppm").resolve()),
                        "homography": np.loadtxt(
                            by_name[name] / f"H_1_{target}"
                        ).tolist(),
                    }
                )
                pairs.append(
                    {
                        **common,
                        "id": f"negative/{name}-{other}-{target}",
                        "kind": "negative",
                        "negative_label": "assumed_cross_sequence",
                        "image1": str((by_name[other] / f"{target}.ppm").resolve()),
                    }
                )
    return {
        "schema": 1,
        "seed": seed,
        "sequences": groups,
        "identical_content_groups": duplicates,
        "pairs": pairs,
        "negative_label_warning": "Cross-sequence distractors are assumed "
        "unrelated, not independently verified. Report negative metrics "
        "conditional on these labels.",
    }


class FeatureStore:
    def __init__(self, directory, device="cpu", max_keypoints=512, resize=640):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.extractor = (
            SuperPoint(max_num_keypoints=max_keypoints)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        self.settings = {
            "max_keypoints": max_keypoints,
            "resize": resize,
            "extractor_hash": digest_tensors(self.extractor.state_dict()),
            "torch": str(torch.__version__),
            "origin": "center",
        }
        self.memory = OrderedDict()
        self.image_hashes = {}

    @torch.inference_mode()
    def get(self, image_path):
        path = Path(image_path).resolve()
        image_hash = self.image_hashes.setdefault(str(path), None)
        if image_hash is None:
            image_hash = self.image_hashes[str(path)] = digest_file(path)
        key = hashlib.sha256(
            (image_hash + json.dumps(self.settings, sort_keys=True)).encode()
        ).hexdigest()
        if key in self.memory:
            self.memory.move_to_end(key)
            cpu = self.memory[key]
        else:
            cache_path = self.directory / f"{key}.pt"
            if cache_path.exists():
                cpu = torch.load(cache_path, map_location="cpu", weights_only=True)
            else:
                image = load_image(path).to(self.device)
                feature = self.extractor.extract(
                    image, resize=self.settings["resize"] or None
                )
                cpu = {k: v.cpu() for k, v in feature.items()}
                torch.save(cpu, cache_path)
            self.memory[key] = cpu
            if len(self.memory) > 32:
                self.memory.popitem(last=False)
        return {k: v.to(self.device) for k, v in cpu.items()}

    def pair(self, record):
        return {f"image{i}": self.get(record[f"image{i}"]) for i in range(2)}


def load_megadepth(root):
    """Read the official Glue Factory MegaDepth-1500 views/pairs archive."""
    root = Path(root)
    if (root / "megadepth1500").exists():
        root = root / "megadepth1500"
    views = {}
    for line in (root / "views.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, *values = line.split()
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(values[:9], float).reshape(3, 3)
        transform[:3, 3] = np.asarray(values[9:12], float)
        model = values[12]
        params = np.asarray(values[15:], float)
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE":
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError(f"Unsupported camera {model}; undistort before evaluation")
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        views[name] = (transform, intrinsics)
    pairs = []
    for index, line in enumerate((root / "pairs.txt").read_text().splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        name0, name1 = line.split()
        t0, k0 = views[name0]
        t1, k1 = views[name1]
        pairs.append(
            {
                "id": f"md1500/{index}",
                "kind": "megadepth",
                "partition": "test",
                "image0": str((root / "images" / name0).resolve()),
                "image1": str((root / "images" / name1).resolve()),
                "K0": k0.tolist(),
                "K1": k1.tolist(),
                "T_0to1": (t1 @ np.linalg.inv(t0)).tolist(),
            }
        )
    return pairs
