"""Rebuild tables/plots: python -m experiments.car_report RESULT_DIRECTORY."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from .car_metrics import pose_auc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def aggregate(rows):
    summaries = []
    for kind in ("all", "illumination", "viewpoint", "negative", "megadepth"):
        selected = [r for r in rows if kind == "all" or r["kind"] == kind]
        for name in sorted({r["configuration"] for r in selected}):
            group = [r for r in selected if r["configuration"] == name]
            count = len(group)
            accepted = sum(r["accepted"] for r in group)
            reject = sum(r["status"] == "REJECT" for r in group)
            usable = sum(r["ever_usable"] for r in group)
            false_accept = sum(r["false_accept"] for r in group)
            false_reject = sum(r["false_reject"] for r in group)

            def mean(key):
                return float(np.mean([r[key] for r in group]))

            result = {
                "kind": kind,
                "configuration": name,
                "pairs": count,
                "mean_layers": mean("executed_layers"),
                "layer_savings": 1 - mean("executed_layers") / 9,
                "mean_latency_ms": mean("latency_ms"),
                "latency_savings": 1 - mean("latency_ms") / mean("full_latency_ms"),
                "accepted": accepted,
                "rejected": reject,
                "unresolved": sum(r["status"] == "UNRESOLVED" for r in group),
                "false_accept_count": false_accept,
                "false_accept_per_accepted": false_accept / accepted
                if accepted
                else None,
                "false_accept_per_pair": false_accept / count,
                "false_reject_count": false_reject,
                "oracle_usable_pairs": usable,
                "false_reject_per_usable": false_reject / usable if usable else None,
                "false_reject_per_rejected": false_reject / reject if reject else None,
                "recoverable_rejections": sum(
                    r["false_reject_recoverable"] for r in group
                ),
                "accepted_usable_per_usable": sum(
                    r["accepted"] and r["current_usable"] and r["ever_usable"]
                    for r in group
                )
                / usable
                if usable
                else None,
                "negative_early_rejection_rate": reject / count
                if kind == "negative"
                else None,
                "mean_matches": mean("num_matches"),
            }
            hp = [r for r in group if r["kind"] in ("illumination", "viewpoint")]
            for threshold in (1, 3, 5):
                if hp:
                    correct = sum(r[f"correct_{threshold}px"] for r in hp)
                    predicted = sum(r["num_matches"] for r in hp)
                    possible = sum(r[f"matchable_{threshold}px"] for r in hp)
                    result.update(
                        {
                            f"mma_{threshold}px": float(
                                np.mean([r[f"precision_{threshold}px"] for r in hp])
                            ),
                            f"precision_{threshold}px": correct / predicted
                            if predicted
                            else 0,
                            f"recall_{threshold}px": correct / possible
                            if possible
                            else 0,
                            f"correct_{threshold}px_per_pair": correct / len(hp),
                        }
                    )
            md = [r for r in group if r["kind"] == "megadepth"]
            if md:
                for threshold in (5, 10, 20):
                    result[f"pose_auc_{threshold}"] = pose_auc(
                        [
                            r["pose_error"]
                            if r["pose_error"] is not None
                            else float("inf")
                            for r in md
                        ],
                        threshold,
                    )
                for threshold in (1e-4, 5e-4, 1e-3):
                    key = f"epi_precision_{threshold}"
                    result[key] = float(np.mean([r[key] for r in md]))
            summaries.append(result)
    return summaries


def profile_summary(records):
    """Average repetitions within each pair, then give every pair equal weight."""
    rows = []
    kinds = ["all"] + sorted({r["kind"] for r in records})
    for kind in kinds:
        selected = [r for r in records if kind == "all" or r["kind"] == kind]
        by_pair = defaultdict(list)
        for record in selected:
            by_pair[record["pair"]].append(record)
        if not by_pair:
            continue
        total = float(
            np.mean(
                [np.mean([r["total_ms"] for r in group]) for group in by_pair.values()]
            )
        )
        stages = sorted({stage for r in selected for stage in r["stages_ms"]})
        for stage in stages:
            mean = float(
                np.mean(
                    [
                        np.mean([r["stages_ms"].get(stage, 0) for r in group])
                        for group in by_pair.values()
                    ]
                )
            )
            rows.append(
                {
                    "kind": kind,
                    "stage": stage,
                    "pairs": len(by_pair),
                    "mean_ms": mean,
                    "fraction_of_profiled_total": mean / total if total else 0,
                    "profiled_total_ms": total,
                }
            )
    return rows


def plot_profiles(output, records, verbose=False):
    summaries = profile_summary(records)
    (output / "profile_summary.json").write_text(json.dumps(summaries, indent=2))
    with (output / "profile_summary.csv").open("w", newline="") as stream:
        fields = [
            "kind",
            "stage",
            "pairs",
            "mean_ms",
            "fraction_of_profiled_total",
            "profiled_total_ms",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    # Keep historical coverage measurements readable; new CAR runs omit them.
    order = [
        "preparation",
        "transformer",
        "assignment",
        "confidence",
        "margin",
        "coverage",
        "geometry",
        "decision",
        "pruning",
        "other",
    ]
    colors = dict(zip(order, plt.get_cmap("tab10").colors))
    for kind in sorted({r["kind"] for r in summaries}):
        selected = [r for r in summaries if r["kind"] == kind]
        by_stage = {r["stage"]: r["mean_ms"] for r in selected}
        stages = [stage for stage in order if stage in by_stage]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        values = [by_stage[stage] for stage in stages]
        axes[0].barh(stages, values, color=[colors[stage] for stage in stages])
        for i, value in enumerate(values):
            axes[0].text(value, i, f" {value:.3f}", va="center", fontsize=8)
        axes[0].set(
            xlabel="Mean ms per pair (separate profiled passes)",
            title=f"CAR stage costs; total {selected[0]['profiled_total_ms']:.2f} ms",
        )
        group = [r for r in records if kind == "all" or r["kind"] == kind]
        depths = sorted(
            {e["depth"] for r in group for e in r["events"] if e["depth"] > 0}
        )
        bottom = np.zeros(len(depths))
        reached = []
        for depth in depths:
            reached.append(
                len(
                    {
                        r["pair"]
                        for r in group
                        if any(e["depth"] == depth for e in r["events"])
                    }
                )
            )
        for stage in order:
            if stage in ("other", "preparation") or stage not in by_stage:
                continue
            values = []
            for depth in depths:
                by_pair = defaultdict(list)
                for record in group:
                    if not any(e["depth"] == depth for e in record["events"]):
                        continue
                    by_pair[record["pair"]].append(
                        sum(
                            e["milliseconds"]
                            for e in record["events"]
                            if e["depth"] == depth and e["stage"] == stage
                        )
                    )
                values.append(float(np.mean([np.mean(v) for v in by_pair.values()])))
            axes[1].bar(depths, values, bottom=bottom, label=stage, color=colors[stage])
            bottom += values
        axes[1].set(
            xticks=depths,
            xticklabels=[f"{d}\nn={n}" for d, n in zip(depths, reached)],
            xlabel="Depth and number of pairs reaching it",
            ylabel="Mean ms among pairs reaching depth",
            title="Stage time by executed depth (different pair subsets)",
        )
        axes[1].legend(fontsize=7)
        fig.suptitle(
            f"{kind}: instrumented CAR timing; synchronization changes latency"
        )
        fig.tight_layout()
        path = output / f"time_profile_{kind}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        if verbose:
            print(f"Wrote {path}", flush=True)


def generate_report(output, metadata, verbose=False):
    output = Path(output)
    rows = [
        json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()
    ]
    layers = [
        json.loads(line) for line in (output / "layers.jsonl").read_text().splitlines()
    ]
    summary = aggregate(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    keys = sorted({key for r in summary for key in r})
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# CAR evaluation",
        "",
        f"Dataset: {metadata['dataset']}; partition: {metadata['partition']}; "
        f"pairs: {metadata['pairs']}; device: {metadata['device']}.",
        "",
        "All matcher times include head evaluation, feature extraction for "
        "CAR, GPU synchronization required by decisions, "
        "optional keypoint pruning, and scheduled geometry. "
        "SuperPoint extraction, disk I/O, "
        "feature transfers, downstream pose estimation and GT evaluation are excluded.",
        "",
        "ACCEPT returns correspondences. REJECT and UNRESOLVED withhold all "
        "matches; both contribute zero to pair MMA and pose AUC. RETURN is "
        "the original baselines' unconditional output, not a calibrated "
        "acceptance decision. For decision metrics only, a baseline proposal "
        "counts as accepted when its match count meets the same minimum-count "
        "gate as CAR (20 by default); empty returns are not false accepts. "
        "Accuracy metrics still use all original returned matches.",
        "",
        "False acceptance = accepted/returned output failing the defined "
        "usability criterion. FA/accepted divides by accepted pairs. False "
        "rejection = REJECT on a pair usable at any depth within budget; "
        "FR/usable divides by oracle-usable pairs. Recoverable rejection "
        "separately requires a usable strictly later layer. Unresolved pairs "
        "are NOT counted as rejections; their loss is visible in usable-pair "
        "retention and unconditional accuracy.",
        "",
        "Negative rejection rates are "
        "conditional on the supplied labels. Geometry is a fitted homography "
        "for HPatches and a fundamental matrix for MegaDepth, never a GT "
        "transform.",
        "",
        "HPatches MMA is macro pair precision; pooled precision = total "
        "correct/total predicted; recall = total correct/total geometrically "
        "matchable source detections. Errors use original-image pixels. "
        "Negative pairs are excluded from HPatches accuracy averages, but "
        "included in runtime and decision metrics for the 'all' group.",
        "",
        "MegaDepth correspondence precision uses squared symmetric epipolar "
        "distance in normalized camera coordinates at 1e-4/5e-4/1e-3. Pose "
        "AUC integrates recall up to 5/10/20 degrees using max(rotation, "
        "sign-invariant translation-direction error); all failed/withheld "
        "estimates remain in its denominator. OpenCV essential-matrix RANSAC "
        "uses a fixed 1-pixel threshold. These settings are not identical to "
        "Glue Factory's default PoseLib benchmark.",
        "",
        "CAR uses fixed heuristic thresholds without controller training. "
        "HPatches and MegaDepth use different geometric models. "
        "Select thresholds on validation data before evaluating test data.",
        "",
    ]
    profile_path = output / "profiles.jsonl"
    profiles = (
        [json.loads(line) for line in profile_path.read_text().splitlines()]
        if profile_path.exists()
        else []
    )
    lines += [
        "Component profiling is measured in separate CAR passes with synchronized "
        "stage boundaries. It includes CPU work and transfers, and perturbs execution. "
        "Normal latency excludes these passes. Stage summaries average repetitions "
        "within each pair, then average pairs; skipped stages contribute zero. "
        "The residual 'other' includes Python bookkeeping and profiling overhead.",
        "",
        (
            "Component measurements: profile_summary.csv/json and time_profile_*.png."
            if profiles
            else "No component timing was recorded; no breakdown can be inferred."
        ),
        "",
    ]
    car_config = metadata.get("configurations", {}).get("CAR", {})
    if car_config.get("pruning"):
        lines += [
            "CAR keypoint pruning is enabled: matchability threshold "
            f"{car_config['pruning_threshold']}, minimum retained per image "
            f"{car_config['pruning_min_keypoints']}, starting at checkpoint depth "
            f"{car_config['pruning_start_depth']}. Pruning follows CONTINUE only. "
            "Match fractions retain original keypoint denominators; similarity "
            "margins use active points. Full-depth oracle traces remain unpruned. "
            "For withheld outputs, current_usable refers to the unpruned reference, "
            "not the pruned candidate. Active/retained counts are in pairs.jsonl.",
            "",
        ]
    if metadata.get("historical_note"):
        lines += ["Historical run: " + metadata["historical_note"], ""]
    if metadata.get("negative_label_warning"):
        lines += [
            "Negative-label limitation: " + metadata["negative_label_warning"],
            "",
        ]
    lines += [
        "| Split | Model | Layers | ms | Time saved | Accepted | Rejected | "
        "Unresolved | FA/accepted | FR/usable | MMA@3 / epi precision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:

        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"

        quality = r.get("mma_3px", r.get("epi_precision_0.0001"))
        lines.append(
            f"| {r['kind']} | {r['configuration']} | {r['mean_layers']:.2f} | "
            f"{r['mean_latency_ms']:.2f} | {r['latency_savings']:.1%} | "
            f"{r['accepted']} | {r['rejected']} | {r['unresolved']} | "
            f"{fmt(r['false_accept_per_accepted'])} | "
            f"{fmt(r['false_reject_per_usable'])} | {fmt(quality)} |"
        )
    lines += [
        "",
        "Pose AUC, all three HPatches thresholds and usable-pair retention "
        "are in summary.csv/json. Raw decisions and heuristic scores are in "
        "pairs.jsonl. Negative time savings mean CAR was slower than "
        "the full-depth baseline.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    for kind in sorted({r["kind"] for r in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        group = [r for r in rows if r["kind"] == kind]
        for name in sorted({r["configuration"] for r in group}):
            chosen = [r for r in group if r["configuration"] == name]
            by_depth = defaultdict(list)
            for r in chosen:
                by_depth[r["depth"]].append(r)
            depths = sorted(by_depth)
            quality_key = (
                "epi_precision_0.0001" if kind == "megadepth" else "precision_3px"
            )
            quality = [
                np.mean([r.get(quality_key, 0) for r in by_depth[d]]) for d in depths
            ]
            axes[0].plot(depths, quality, "o-", label=name)
            axes[1].plot(
                depths,
                [np.mean([r["latency_ms"] for r in by_depth[d]]) for d in depths],
                "o-",
                label=name,
            )
            axes[2].plot(depths, [len(by_depth[d]) for d in depths], "o-", label=name)
        if kind != "negative":
            fixed = [r for r in layers if r["kind"] == kind]
            x = sorted({r["depth"] for r in fixed})
            axes[0].plot(
                x,
                [
                    np.mean([r[quality_key] for r in fixed if r["depth"] == d])
                    for d in x
                ],
                "k--",
                label="fixed-depth oracle trace",
            )
        for ax, title in zip(
            axes,
            (
                "Quality (all outcomes at this depth)",
                "Online matcher latency (ms)",
                "Pair count at stopping depth",
            ),
        ):
            ax.set_title(title)
            ax.set_xlabel("Executed/decision depth")
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(
            f"{kind}: stopping-depth groups contain different pairs; dashed "
            f"curve uses every pair"
        )
        fig.tight_layout()
        fig.savefig(output / f"quality_runtime_{kind}.png", dpi=140)
        plt.close(fig)
    # Unconditional aggregate accuracy avoids comparing different stopping subsets.
    kinds = sorted({r["kind"] for r in rows} - {"negative"}) or ["negative"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7 * len(kinds), 5), squeeze=False)
    for kind, ax in zip(kinds, axes[0]):
        for item in summary:
            if item["kind"] != kind:
                continue
            quality = item.get(
                "mma_3px",
                item.get("pose_auc_10", item.get("negative_early_rejection_rate")),
            )
            ax.scatter(item["mean_latency_ms"], quality)
            ax.annotate(
                item["configuration"],
                (item["mean_latency_ms"], quality),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set(
            xlabel="Mean online latency (ms)",
            ylabel=(
                "Pose AUC@10"
                if kind == "megadepth"
                else "Early rejection fraction"
                if kind == "negative"
                else "Unconditional MMA@3"
            ),
            title=kind,
        )
        ax.grid(alpha=0.25)
        ax.margins(x=0.3)
    fig.tight_layout()
    fig.savefig(output / "accuracy_latency.png", dpi=140)
    plt.close(fig)
    if profiles:
        plot_profiles(output, profiles, verbose)
    if verbose:
        print(f"Wrote summaries and result plots to {output}", flush=True)
    md = [r for r in rows if r["kind"] == "megadepth"]
    if md:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in sorted({r["configuration"] for r in md}):
            errors = np.array(
                [
                    r["pose_error"] if r["pose_error"] is not None else np.inf
                    for r in md
                    if r["configuration"] == name
                ]
            )
            thresholds = np.linspace(0, 20, 201)
            ax.plot(thresholds, [(errors <= t).mean() for t in thresholds], label=name)
        ax.set(
            xlabel="Maximum rotation/translation error (degrees)",
            ylabel="Fraction of all test pairs",
            xlim=(0, 20),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "pose_recall.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    generate_report(
        args.output,
        json.loads((args.output / "manifest.json").read_text()),
        verbose=args.verbose,
    )
