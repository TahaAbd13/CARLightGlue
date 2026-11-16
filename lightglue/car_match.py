"""Confidence-adaptive control of the original, independently trained layers.

This module never consumes ground truth. The controller uses fixed rules
and frozen pretrained weights.
REJECT means unlikely to become usable within the configured budget, not no overlap.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import torch
from torch import nn

from .lightglue import filter_matches, normalize_keypoints
from .profiling import StageProfiler


class Action(str, Enum):
    ACCEPT = "ACCEPT"
    CONTINUE = "CONTINUE"
    REJECT = "REJECT"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class CARConfig:
    checkpoints: tuple = (1, 3, 5, 7, 9)
    min_depth: int = 1
    max_depth: int = 9
    accept_threshold: float = 0.95
    reject_usable_threshold: float = 0.1
    reject_recoverable_threshold: float = 0.05
    reject_min_depth: int = 5
    rejection_patience: int = 2
    min_matches: int = 20
    geometry: str = "none"  # none, homography, fundamental
    geometry_layers: tuple = (3, 7, 9)
    geometry_threshold: float = 3.0  # original-image pixels
    geometry_max_matches: int = 256
    geometry_max_iterations: int = 500
    pruning: bool = False
    pruning_threshold: float = 0.1  # pretrained keypoint matchability
    pruning_min_keypoints: int = 64  # retained independently in each image
    pruning_start_depth: int = 3

    def validate(self, layers):
        if not 0 <= self.pruning_threshold <= 1:
            raise ValueError("Pruning threshold must be finite and in [0, 1]")
        if self.pruning_min_keypoints < 1:
            raise ValueError("Pruning minimum keypoints must be positive")
        if not 1 <= self.pruning_start_depth <= layers:
            raise ValueError("Invalid pruning start depth")
        if not 1 <= self.min_depth <= self.max_depth <= layers:
            raise ValueError("Require 1 <= min_depth <= max_depth <= model depth")
        if not self.checkpoints or any(not 1 <= i <= layers for i in self.checkpoints):
            raise ValueError("Invalid controller checkpoints")
        if tuple(sorted(set(self.checkpoints))) != tuple(self.checkpoints):
            raise ValueError("Checkpoints must be sorted and unique")
        if self.geometry not in ("none", "homography", "fundamental"):
            raise ValueError("Unknown geometric model")
        if self.rejection_patience < 1 or self.min_matches < 1:
            raise ValueError("Patience and minimum matches must be positive")
        if self.geometry_max_matches < 8 or self.geometry_max_iterations < 1:
            raise ValueError("Invalid geometry budget")
        if self.geometry_threshold <= 0 or not np.isfinite(self.geometry_threshold):
            raise ValueError("Geometry threshold must be finite and positive")
        if not 1 <= self.reject_min_depth <= layers:
            raise ValueError("Invalid minimum rejection depth")
        if any(not 1 <= i <= layers for i in self.geometry_layers):
            raise ValueError("Invalid geometry layers")
        if self.geometry != "none" and not set(self.geometry_layers).issubset(
            set(self.checkpoints) | {self.max_depth}
        ):
            raise ValueError("Geometry layers must be decision checkpoints")
        for value in (
            self.accept_threshold,
            self.reject_usable_threshold,
            self.reject_recoverable_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Decision-score thresholds must be in [0, 1]")


FEATURE_NAMES = (
    "log_matches",
    "match_fraction",
    "score_mean",
    "similarity_margin",
    "delta_match_fraction",
    "geometry_enabled",
    "geometry_available",
    "geometry_inlier_fraction",
)


class DecisionPolicy:
    """State belongs to a single pair and resets on every matcher call."""

    def __init__(self, config):
        self.config = config
        self.low_streak = 0

    def decide(self, depth, num_matches, usable, recoverable):
        conf = self.config
        finite = np.isfinite([usable, recoverable]).all()
        if finite and depth >= conf.min_depth and num_matches >= conf.min_matches:
            if usable >= conf.accept_threshold:
                return Action.ACCEPT
        # Maximum depth is not evidence of unrelated images.
        if depth >= conf.max_depth:
            return Action.UNRESOLVED
        low = (
            finite
            and depth >= max(conf.min_depth, conf.reject_min_depth)
            and usable <= conf.reject_usable_threshold
            and recoverable <= conf.reject_recoverable_threshold
        )
        self.low_streak = self.low_streak + 1 if low else 0
        if self.low_streak >= conf.rejection_patience:
            return Action.REJECT
        return Action.CONTINUE


def car_scores(features, min_matches=20):
    """Heuristic decision scores, NOT calibrated probabilities."""
    f = dict(zip(FEATURE_NAMES, features.detach().cpu().tolist()))
    adequate = (
        round(float(np.expm1(f["log_matches"]))) >= min_matches
        and f["score_mean"] >= 0.3
        and f["similarity_margin"] >= 0.1
    )
    if f["geometry_enabled"]:
        adequate &= f["geometry_available"] > 0 and f["geometry_inlier_fraction"] >= 0.6
    usable = 0.99 if adequate else 0.01 if f["match_fraction"] < 0.02 else 0.5
    recoverable = (
        0.01 if f["match_fraction"] < 0.02 and f["delta_match_fraction"] <= 0 else 0.8
    )
    return usable, recoverable


def empty_prediction(data, depth):
    desc = data["image0"]["descriptors"]
    b, m = data["image0"]["keypoints"].shape[:2]
    n = data["image1"]["keypoints"].shape[1]
    return {
        "matches0": desc.new_full((b, m), -1, dtype=torch.long),
        "matches1": desc.new_full((b, n), -1, dtype=torch.long),
        "matching_scores0": desc.new_zeros((b, m)),
        "matching_scores1": desc.new_zeros((b, n)),
        "matches": [desc.new_empty((0, 2), dtype=torch.long) for _ in range(b)],
        "scores": [desc.new_empty(0) for _ in range(b)],
        "stop": depth,
        "prune0": desc.new_full((b, m), depth),
        "prune1": desc.new_full((b, n), depth),
    }


def prediction_from_assignment(assignment, threshold, depth):
    m0, m1, s0, s1 = filter_matches(assignment, threshold)
    indices = torch.where(m0[0] >= 0)[0]
    return {
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": s0,
        "matching_scores1": s1,
        "matches": [torch.stack((indices, m0[0, indices]), -1)],
        "scores": [s0[0, indices]],
        "stop": depth,
        "prune0": torch.full_like(s0, depth),
        "prune1": torch.full_like(s1, depth),
    }


def pruning_indices(matchability, threshold, minimum):
    """Keep high-matchability points, with a top-score floor and original order."""
    scores = matchability.flatten()
    # Nonfinite evidence is not a reason to remove a point.
    scores = torch.where(torch.isfinite(scores), scores, torch.ones_like(scores))
    keep = scores >= threshold
    floor = min(minimum, scores.numel())
    if floor:
        # Stable sorting makes ties deterministic; retained indices remain ordered.
        best = torch.argsort(scores, descending=True, stable=True)[:floor]
        keep[best] = True
    return torch.where(keep)[0]


def restore_prediction(prediction, indices, data, depth):
    """Scatter compact predictions into the original keypoint index space."""
    restored = empty_prediction(data, depth)
    for side in (0, 1):
        local = prediction[f"matches{side}"][0]
        valid = local >= 0
        mapped = torch.full_like(local, -1)
        mapped[valid] = indices[1 - side][local[valid]]
        restored[f"matches{side}"][0, indices[side]] = mapped
        restored[f"matching_scores{side}"][0, indices[side]] = prediction[
            f"matching_scores{side}"
        ][0]
    matches = prediction["matches"][0]
    restored["matches"] = [
        torch.stack((indices[0][matches[:, 0]], indices[1][matches[:, 1]]), dim=-1)
    ]
    restored["scores"] = prediction["scores"]
    return restored


def geometric_quality(data, prediction, config):
    """Bounded robust fit using predictions only; no GT or camera pose is read."""
    matches = prediction["matches"][0]
    needed = 4 if config.geometry == "homography" else 8
    if len(matches) < needed:
        return (1.0, 0.0, 0.0)
    # Fit on a deterministic spatially unselected subset, ordered by source index.
    idx = torch.linspace(
        0,
        len(matches) - 1,
        min(len(matches), config.geometry_max_matches),
        device=matches.device,
    ).long()
    selected = matches[idx]
    p0 = data["image0"]["keypoints"][0, selected[:, 0]].float().cpu().numpy()
    p1 = data["image1"]["keypoints"][0, selected[:, 1]].float().cpu().numpy()
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return (1.0, 0.0, 0.0)
    cv2.setRNGSeed(0)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    try:
        if config.geometry == "homography":
            model, mask = cv2.findHomography(
                p0,
                p1,
                method,
                config.geometry_threshold,
                maxIters=config.geometry_max_iterations,
                confidence=0.99,
            )
        else:
            model, mask = cv2.findFundamentalMat(
                p0,
                p1,
                method,
                config.geometry_threshold,
                0.99,
                config.geometry_max_iterations,
            )
        if model is None or mask is None or not np.isfinite(model).all():
            return (1.0, 0.0, 0.0)
        support = float(np.asarray(mask, bool).sum())
        return (1.0, support / len(selected), support / config.geometry_max_matches)
    except cv2.error:
        return (1.0, 0.0, 0.0)


def extract_quality(
    data, prediction, similarity, previous, config, geometry, depth=0, profiler=None
):
    """Only compute evidence consumed by CAR's fixed rules."""
    section = profiler.section if profiler else lambda *args: nullcontext()
    scores = prediction["scores"][0].float()
    matches = prediction["matches"][0]
    k0, k1 = [data[k]["keypoints"][0].float() for k in ("image0", "image1")]
    zero = k0.new_zeros(())
    count = len(matches)
    fraction = count / max(min(len(k0), len(k1)), 1)
    with section("confidence", depth):
        mean = scores.mean() if count else zero
    with section("margin", depth):
        if similarity is not None and min(similarity.shape[-2:]) >= 2:
            margins = []
            for dimension in (1, 2):
                top = similarity.float().topk(2, dim=dimension).values
                margins.append(
                    (top.select(dimension, 0) - top.select(dimension, 1)).flatten()
                )
            margin = torch.cat(margins).clamp(max=20).mean()
        else:
            margin = zero
    delta = zero if previous is None else fraction - previous[1]
    values = [
        np.log1p(count),
        fraction,
        mean,
        margin,
        delta,
        float(config.geometry != "none"),
        geometry[0],
        geometry[1],
    ]
    features = torch.stack(
        [v if torch.is_tensor(v) else zero.new_tensor(v) for v in values]
    )
    return torch.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)


class CAR(nn.Module):
    """A frozen LightGlue backbone with fixed confidence-adaptive rules.

    Online control is pairwise (B=1). Optional matchability pruning.
    Disabled control delegates directly to standard full-depth LightGlue.
    """

    def __init__(self, matcher, config=None):
        super().__init__()
        if matcher.conf.depth_confidence > 0 or matcher.conf.width_confidence > 0:
            raise ValueError(
                "Construct the CAR backbone with depth/width_confidence=-1"
            )
        if matcher.static_lengths is not None:
            raise ValueError("CAR currently requires an uncompiled backbone")
        self.matcher = matcher.eval().requires_grad_(False)
        self.config = config or CARConfig(max_depth=matcher.conf.n_layers)
        self.config.validate(matcher.conf.n_layers)

    def train(self, mode=True):
        super().train(mode)
        self.matcher.eval()
        self.matcher.requires_grad_(False)
        return self

    @torch.no_grad()
    def forward(self, data, enabled=True, collect=False, profile=False):
        if not enabled:
            return self.matcher(data)
        profiler = (
            StageProfiler(data["image0"]["descriptors"].device) if profile else None
        )
        output = self._controlled(data, collect, profiler)
        if profiler:
            output["profile"] = profiler.finish()
        return output

    def _controlled(self, data, collect, profiler=None):
        conf, matcher = self.config, self.matcher
        section = profiler.section if profiler else lambda *args: nullcontext()
        # Explicitly restrict the input contract; unknown fields are never features.
        items = [data[f"image{i}"] for i in range(2)]
        if any(f["keypoints"].shape[0] != 1 for f in items):
            raise ValueError("Online CAR supports one image pair per call")
        for f in items:
            if "image_size" not in f:
                raise ValueError("CAR requires image_size for keypoint normalization")
        empty = any(f["keypoints"].shape[1] == 0 for f in items)
        # Diagnostic collection always follows the unpruned full-depth path.
        do_prune = conf.pruning and not collect
        indices = [
            torch.arange(f["keypoints"].shape[1], device=f["keypoints"].device)
            for f in items
        ]
        prune_depths = [torch.zeros_like(index)[None] for index in indices]
        checkpoints = set(conf.checkpoints) | {conf.max_depth}
        history, previous = [], None
        geometry = (0.0, 0.0, 0.0)
        policy = DecisionPolicy(conf)
        with torch.autocast(enabled=matcher.conf.mp, device_type="cuda"):
            with section("preparation", 0):
                if not empty:
                    positions = [
                        normalize_keypoints(f["keypoints"], f["image_size"]).clone()
                        for f in items
                    ]
                    if matcher.conf.add_scale_ori:
                        positions = [
                            torch.cat(
                                [p] + [f[k].unsqueeze(-1) for k in ("scales", "oris")],
                                -1,
                            )
                            for p, f in zip(positions, items)
                        ]
                    descriptors = [
                        f["descriptors"].detach().contiguous() for f in items
                    ]
                    if torch.is_autocast_enabled():
                        descriptors = [d.half() for d in descriptors]
                    descriptors = [matcher.input_proj(d) for d in descriptors]
                    encodings = [matcher.posenc(p) for p in positions]
            for depth in range(1, conf.max_depth + 1):
                if do_prune:
                    for side in (0, 1):
                        prune_depths[side][0, indices[side]] = depth
                assignment = similarity = None
                with section("transformer", depth):
                    if not empty:
                        descriptors = matcher.transformers[depth - 1](
                            *descriptors, *encodings
                        )
                if depth not in checkpoints and not collect:
                    continue
                with section("assignment", depth):
                    if empty:
                        prediction = empty_prediction(data, depth)
                    else:
                        assignment, similarity = matcher.log_assignment[depth - 1](
                            *descriptors
                        )
                        prediction = prediction_from_assignment(
                            assignment, matcher.conf.filter_threshold, depth
                        )
                        if do_prune:
                            prediction = restore_prediction(
                                prediction, indices, data, depth
                            )
                if (
                    conf.geometry != "none"
                    and depth in conf.geometry_layers
                    and depth in checkpoints
                ):
                    with section("geometry", depth):
                        geometry = geometric_quality(data, prediction, conf)
                features = extract_quality(
                    data,
                    prediction,
                    similarity,
                    previous,
                    conf,
                    geometry,
                    depth,
                    profiler,
                )
                with section("decision", depth):
                    if collect:
                        usable, recoverable = float("nan"), float("nan")
                        action = Action.CONTINUE
                    else:
                        usable, recoverable = car_scores(features, conf.min_matches)
                        action = policy.decide(
                            depth, len(prediction["matches"][0]), usable, recoverable
                        )
                history.append(
                    {
                        "depth": depth,
                        "features": features,
                        "prediction": prediction if collect else None,
                        "usable_score": usable,
                        "recovery_score": recoverable,
                        "action": action.value,
                        "active_keypoints": [len(index) for index in indices],
                    }
                )
                if depth in checkpoints:
                    previous = features
                if (collect) and depth == conf.max_depth:
                    return {**prediction, "history": history, "status": "TRACE"}
                if action != Action.CONTINUE:
                    output = (
                        prediction
                        if action == Action.ACCEPT
                        else empty_prediction(data, depth)
                    )
                    if do_prune:
                        output.update(prune0=prune_depths[0], prune1=prune_depths[1])
                    return {
                        **output,
                        "status": action.value,
                        "history": history,
                        "executed_layers": 0 if empty else depth,
                        "usable_score": usable,
                        "recovery_score": recoverable,
                        "active_keypoints": [len(index) for index in indices],
                    }
                if (
                    do_prune
                    and not empty
                    and depth >= conf.pruning_start_depth
                    and depth < conf.max_depth
                ):
                    with section("pruning", depth):
                        descriptors, encodings = list(descriptors), list(encodings)
                        # Assignment's unmatched entries are log(1 - matchability).
                        matchabilities = (
                            1 - assignment[0, :-1, -1].exp(),
                            1 - assignment[0, -1, :-1].exp(),
                        )
                        for side in (0, 1):
                            keep = pruning_indices(
                                matchabilities[side],
                                conf.pruning_threshold,
                                conf.pruning_min_keypoints,
                            )
                            descriptors[side] = descriptors[side].index_select(1, keep)
                            encodings[side] = encodings[side].index_select(-2, keep)
                            indices[side] = indices[side].index_select(0, keep)
                    history[-1]["retained_keypoints"] = [len(i) for i in indices]
        raise RuntimeError("No terminal decision")
