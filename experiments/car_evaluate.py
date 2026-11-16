"""Online accuracy and synchronized latency, including all controller overhead."""

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue
from lightglue.car_match import CAR, CARConfig
from lightglue.utils import read_image

from .car_data import FeatureStore, load_megadepth
from .car_metrics import PairEvaluator
from .car_report import generate_report
from .common import digest_file, digest_tensors, same_prediction, timed
from .visualization import visualize


def make_models(args):
    full = (
        LightGlue(
            features="superpoint", depth_confidence=-1, width_confidence=-1, flash=False
        )
        .eval()
        .requires_grad_(False)
        .to(args.device)
    )
    adaptive = (
        LightGlue(features="superpoint", flash=False)
        .eval()
        .requires_grad_(False)
        .to(args.device)
    )
    backbone_hash = digest_tensors(full.state_dict())
    common = dict(
        accept_threshold=args.accept_threshold,
        reject_usable_threshold=args.reject_usable_threshold,
        reject_recoverable_threshold=args.reject_recoverable_threshold,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        pruning=args.prune_keypoints,
        pruning_threshold=args.pruning_threshold,
        pruning_min_keypoints=args.pruning_min_keypoints,
        pruning_start_depth=args.pruning_start_depth,
    )
    geometry = "fundamental" if args.megadepth else "homography"
    configurations = {
        "full_9": full,
        "adaptive": adaptive,
        "CAR": CAR(full, config=CARConfig(**common, geometry=geometry)),
    }
    criterion = {
        "min_correct": 20,
        "min_precision": 0.8,
        "hpatches_pixels": 3,
        "megadepth_squared_epi": 1e-4,
    }
    return configurations, criterion, backbone_hash


@torch.inference_mode()
def evaluate(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty evaluation output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(
            f"Initializing frozen SuperPoint and LightGlue on {args.device}", flush=True
        )
    torch.set_num_threads(4)
    torch.manual_seed(0)
    store = FeatureStore(args.cache, args.device, args.max_keypoints, args.resize)
    models, criterion, backbone_hash = make_models(args)
    if args.megadepth:
        pairs = load_megadepth(args.megadepth)
        negative_warning = None
    else:
        manifest = json.loads(args.manifest.read_text())
        pairs = [
            p
            for p in manifest["pairs"]
            if args.partition == "all" or p["partition"] == args.partition
        ]
        negative_warning = manifest.get("negative_label_warning")
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No evaluation pairs")
    metadata = {
        "pairs": len(pairs),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda")
        else "CPU",
        "features": store.settings,
        "backbone_hash": backbone_hash,
        "criterion": criterion,
        "partition": "all" if args.megadepth else args.partition,
        "dataset": "megadepth1500" if args.megadepth else "hpatches",
        "negative_label_warning": negative_warning,
        "timing_repeats": args.timing_repeats,
        "warmup": args.warmup,
        "configurations": {
            name: asdict(model.config)
            for name, model in models.items()
            if isinstance(model, CAR)
        },
        "profiling": {
            "enabled": args.profile,
            "repeats": args.profile_repeats,
            "method": "Separate CAR pass; synchronized wall-clock stage boundaries",
            "normal_latency_includes_profiling": False,
        },
        "status": "running",
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "pair_manifest_sha256": hashlib.sha256(
            json.dumps(pairs, sort_keys=True).encode()
        ).hexdigest(),
        "source_hashes": {
            str(path): digest_file(path)
            for path in [
                *Path("experiments").glob("car_*.py"),
                Path("lightglue/car_match.py"),
                Path("lightglue/lightglue.py"),
                Path("experiments/hpatches_metrics.py"),
                Path("experiments/common.py"),
                Path("experiments/visualization.py"),
            ]
        },
    }
    if args.verbose:
        print(
            f"Device: {metadata['device_name']}; pairs: {len(pairs)}; "
            f"models: {', '.join(models)}",
            flush=True,
        )
        print(
            f"Output: {args.output}; warmup={args.warmup}; "
            f"timing repeats={args.timing_repeats}; profiling={args.profile}",
            flush=True,
        )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2))
    # Oracle diagnostics are outside latency and never supplied to an online model.
    tracer = CAR(models["full_9"], config=CARConfig())
    visuals = defaultdict(int)
    with (args.output / "pairs.jsonl").open("w") as stream, (
        args.output / "layers.jsonl"
    ).open("w") as layer_stream, (args.output / "profiles.jsonl").open(
        "w"
    ) as profile_stream:
        for index, record in enumerate(pairs):
            if args.verbose:
                print(
                    f"[{index + 1}/{len(pairs)}] {record['id']}: loading features",
                    flush=True,
                )
            data = store.pair(record)
            if args.verbose:
                print(
                    "  Checking full-depth trace and ground-truth metrics", flush=True
                )
            evaluator = PairEvaluator(
                record, data, criterion["min_correct"], criterion["min_precision"]
            )
            trace = tracer(data, collect=True)
            oracle = [evaluator.measure(h["prediction"]) for h in trace["history"]]
            same_prediction(models["full_9"](data), trace)
            for h, metrics in zip(trace["history"], oracle):
                layer_stream.write(
                    json.dumps(
                        {
                            "pair": record["id"],
                            "kind": record["kind"],
                            "depth": h["depth"],
                            **metrics,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            predictions, elapsed = {}, defaultdict(list)
            calls = {
                name: (lambda model=model: model(data))
                for name, model in models.items()
            }
            if args.verbose:
                print(f"  Timing {list(models)} (profiling disabled)", flush=True)
            for _ in range(args.warmup):
                for call in calls.values():
                    call()
            for repeat in range(args.timing_repeats):
                names = list(calls)
                offset = (index + repeat) % len(names)
                for name in names[offset:] + names[:offset]:
                    prediction, milliseconds = timed(calls[name], store.device)
                    predictions[name] = prediction
                    elapsed[name].append(milliseconds)
            if args.profile:
                for name in (n for n in models if isinstance(models[n], CAR)):
                    if args.verbose:
                        print(
                            f"  Profiling {name} separately "
                            f"({args.profile_repeats} passes)",
                            flush=True,
                        )
                    for repeat in range(args.profile_repeats):
                        profiled = models[name](data, profile=True)
                        same_prediction(predictions[name], profiled)
                        if profiled["status"] != predictions[name]["status"]:
                            raise AssertionError(
                                "Profiling changed the controller decision"
                            )
                        profile_stream.write(
                            json.dumps(
                                {
                                    "pair": record["id"],
                                    "kind": record["kind"],
                                    "configuration": name,
                                    "repeat": repeat,
                                    "stop": int(profiled["stop"]),
                                    **profiled["profile"],
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                profile_stream.flush()
            full_ms = float(np.median(elapsed["full_9"]))
            for name, prediction in predictions.items():
                metrics = evaluator.measure(
                    prediction, estimate_pose=args.megadepth is not None
                )
                depth = int(prediction["stop"])
                status = prediction.get("status", "RETURN")
                # Audit baselines with the same minimum-match gate as CAR. Empty
                # proposals are not false acceptances of an unrelated pair.
                accepted = status == "ACCEPT" or (
                    status == "RETURN"
                    and metrics["num_matches"] >= criterion["min_correct"]
                )
                # Adaptive pruning can change outputs relative to a fixed-depth trace.
                # Judge every returned proposal using its own measured matches.
                current_usable = (
                    metrics["usable"]
                    if status in ("ACCEPT", "RETURN")
                    else oracle[depth - 1]["usable"]
                )
                budget = args.max_depth if isinstance(models[name], CAR) else 9
                ever_usable = any(m["usable"] for m in oracle[:budget])
                future_usable = any(m["usable"] for m in oracle[depth:budget])
                latency = float(np.median(elapsed[name]))
                error = metrics.get("pose_error")
                if error is not None and not np.isfinite(error):
                    metrics[
                        "pose_error"
                    ] = None  # interpreted as failure/infinity by report
                row = {
                    "pair": record["id"],
                    "kind": record["kind"],
                    "configuration": name,
                    "status": status,
                    "accepted": accepted,
                    "depth": depth,
                    "executed_layers": prediction.get("executed_layers", depth),
                    "active_keypoints": prediction.get("active_keypoints"),
                    "latency_ms": latency,
                    "full_latency_ms": full_ms,
                    "latency_savings_fraction": 1 - latency / full_ms,
                    "current_usable": bool(current_usable),
                    "ever_usable": ever_usable,
                    "future_usable": future_usable,
                    "false_accept": bool(accepted and not current_usable),
                    "false_reject": bool(status == "REJECT" and ever_usable),
                    "false_reject_recoverable": bool(
                        status == "REJECT" and future_usable
                    ),
                    "usable_score": prediction.get("usable_score"),
                    "recovery_score": prediction.get("recovery_score"),
                    "decisions": [
                        {
                            k: h[k]
                            for k in (
                                "depth",
                                "action",
                                "usable_score",
                                "recovery_score",
                                "active_keypoints",
                                "retained_keypoints",
                            )
                            if k in h
                        }
                        for h in prediction.get("history", [])
                    ],
                    **metrics,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if args.verbose:
                    print(
                        f"  {name}: {status}, depth={depth}, "
                        f"matches={metrics['num_matches']}, {latency:.3f} ms, "
                        f"active keypoints={prediction.get('active_keypoints', 'n/a')}",
                        flush=True,
                    )
                if (
                    visuals[(record["kind"], name)] < args.visualize_per_split
                    and evaluator.errors is not None
                ):
                    path = (
                        args.output
                        / "correspondences"
                        / f"{record['id'].replace('/', '_')}_{name}.png"
                    )
                    images = [read_image(record[f"image{i}"]) for i in range(2)]
                    matches = prediction["matches"][0].cpu().numpy()
                    visualize(
                        path,
                        *images,
                        *evaluator.points,
                        matches,
                        evaluator.errors,
                        f"{name}: {status}, depth {depth}, "
                        f"{latency:.2f} ms (normal latency)",
                    )
                    visuals[(record["kind"], name)] += 1
            stream.flush()
            layer_stream.flush()
            if args.verbose or (index + 1) % 20 == 0:
                print(f"Evaluated {index + 1}/{len(pairs)} pairs", flush=True)
    metadata.update(status="complete", image_hashes=store.image_hashes)
    generate_report(args.output, metadata, verbose=args.verbose)
    manifest_path.write_text(json.dumps(metadata, indent=2))
    print(f"Report: {args.output / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/hpatches_pairs.json")
    )
    parser.add_argument("--megadepth", type=Path)
    parser.add_argument(
        "--partition",
        choices=("all", "test", "validation", "calibration"),
        default="test",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/car_features"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--accept-threshold", type=float, default=0.95)
    parser.add_argument("--reject-usable-threshold", type=float, default=0.1)
    parser.add_argument("--reject-recoverable-threshold", type=float, default=0.05)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--visualize-per-split", type=int, default=1)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each pair, timing phase, and model result",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Add separate CAR stage-timing passes and plots",
    )
    parser.add_argument(
        "--prune-keypoints",
        action="store_true",
        help="Prune low-matchability CAR keypoints after CONTINUE checkpoints",
    )
    parser.add_argument("--pruning-threshold", type=float, default=0.1)
    parser.add_argument("--pruning-min-keypoints", type=int, default=64)
    parser.add_argument("--pruning-start-depth", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=1)
    args = parser.parse_args()
    if min(args.timing_repeats, args.profile_repeats, args.max_keypoints) < 1:
        parser.error("Timing repetitions and keypoint count must be positive")
    if min(args.warmup, args.limit, args.resize, args.visualize_per_split) < 0:
        parser.error(
            "Warmup, limits, resize, and visualization count cannot be negative"
        )
    if not 1 <= args.min_depth <= args.max_depth <= 9:
        parser.error("Require 1 <= min-depth <= max-depth <= 9")
    if not 0 <= args.pruning_threshold <= 1 or args.pruning_min_keypoints < 1:
        parser.error("Pruning threshold must be in [0, 1] and minimum points positive")
    if not 1 <= args.pruning_start_depth <= 9:
        parser.error("Pruning start depth must be in [1, 9]")
    evaluate(args)


if __name__ == "__main__":
    main()
