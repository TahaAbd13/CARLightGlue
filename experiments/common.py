"""Shared dataset validation, hashing, and synchronized timing helpers."""

import hashlib
from time import perf_counter

import torch


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_tensors(tensors):
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def discover_sequences(root):
    if (root / "hpatches-sequences-release").is_dir():
        root = root / "hpatches-sequences-release"
    sequences = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith(("i_", "v_"))
    )
    if not sequences:
        raise ValueError(
            f"No i_*/v_* sequences in {root}; "
            "use HPatches full images, not cropped patches"
        )
    for sequence in sequences:
        required = [sequence / f"{i}.ppm" for i in range(1, 7)]
        required += [sequence / f"H_1_{i}" for i in range(2, 7)]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise ValueError(f"Incomplete full-image sequence: {missing}")
    return sequences


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(call, device):
    sync(device)
    start = perf_counter()
    result = call()
    sync(device)
    return result, (perf_counter() - start) * 1000


def same_prediction(a, b):
    for key in ("matches0", "matches1", "matching_scores0", "matching_scores1"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)
