"""GT is confined to offline evaluation, never controller inference."""

import cv2
import numpy as np

from .hpatches_metrics import correspondence_geometry, pair_metrics


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def normalized_points(points, intrinsics):
    return (
        np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(intrinsics).T
    )[:, :2]


def epipolar_errors(p0, p1, k0, k1, transform):
    x0 = np.column_stack((normalized_points(p0, k0), np.ones(len(p0))))
    x1 = np.column_stack((normalized_points(p1, k1), np.ones(len(p1))))
    essential = skew(transform[:3, 3]) @ transform[:3, :3]
    e0, e1 = x0 @ essential.T, x1 @ essential
    residual = np.sum(x1 * e0, axis=-1) ** 2
    # Squared symmetric epipolar distance in normalized camera coordinates.
    return residual * (
        1 / np.maximum((e0[:, :2] ** 2).sum(-1), 1e-15)
        + 1 / np.maximum((e1[:, :2] ** 2).sum(-1), 1e-15)
    )


def pose_error(p0, p1, k0, k1, ground_truth, threshold=1.0):
    if len(p0) < 5:
        return float("inf")
    x0, x1 = normalized_points(p0, k0), normalized_points(p1, k1)
    focal = np.mean([k0[0, 0], k0[1, 1], k1[0, 0], k1[1, 1]])
    cv2.setRNGSeed(0)
    try:
        essential, mask = cv2.findEssentialMat(
            x0,
            x1,
            np.eye(3),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=threshold / focal,
        )
        if essential is None or mask is None:
            return float("inf")
        best_count, best_error = -1, float("inf")
        for candidate in np.split(essential, len(essential) // 3):
            count, rotation, translation, _ = cv2.recoverPose(
                candidate, x0, x1, np.eye(3), mask=mask.copy()
            )
            if count > best_count:
                rcos = (np.trace(rotation.T @ ground_truth[:3, :3]) - 1) / 2
                truth = ground_truth[:3, 3]
                norm = np.linalg.norm(truth) * np.linalg.norm(translation)
                if norm <= 1e-12:
                    return float("inf")
                tcos = abs(float(translation[:, 0] @ truth) / norm)
                best_error = max(
                    np.degrees(np.arccos(np.clip(rcos, -1, 1))),
                    np.degrees(np.arccos(np.clip(tcos, 0, 1))),
                )
                best_count = count
        return best_error if best_count >= 5 else float("inf")
    except cv2.error:
        return float("inf")


def pose_auc(errors, threshold):
    errors = np.sort(np.asarray(errors, float))
    if not len(errors):
        return 0.0
    recall = np.arange(1, len(errors) + 1) / len(errors)
    last = np.searchsorted(errors, threshold, side="right")
    x = np.r_[0.0, errors[:last], threshold]
    y = np.r_[0.0, recall[:last], recall[last - 1] if last else 0.0]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x) / threshold)


class PairEvaluator:
    def __init__(self, record, data, min_correct=20, min_precision=0.8):
        self.record = record
        self.points = [
            data[f"image{i}"]["keypoints"][0].cpu().numpy() for i in range(2)
        ]
        self.min_correct, self.min_precision = min_correct, min_precision
        self.errors = self.possible = None
        if "homography" in record:
            size = data["image1"]["image_size"][0].cpu().numpy()
            self.errors, self.possible = correspondence_geometry(
                *self.points, record["homography"], size
            )

    def measure(self, prediction, estimate_pose=False):
        matches = prediction["matches"][0].detach().cpu().numpy()
        count = len(matches)
        if self.errors is not None:
            metrics = pair_metrics(matches, self.errors, self.possible)
            correct, precision = metrics["correct_3px"], metrics["precision_3px"]
        elif self.record["kind"] == "negative":
            metrics = {"num_matches": count}
            correct = precision = 0
        else:
            k0, k1, transform = [
                np.asarray(self.record[k], float) for k in ("K0", "K1", "T_0to1")
            ]
            p0, p1 = [p[matches[:, i]] for i, p in enumerate(self.points)]
            distances = epipolar_errors(p0, p1, k0, k1, transform)
            metrics = {"num_matches": count}
            for threshold in (1e-4, 5e-4, 1e-3):
                metrics[f"epi_precision_{threshold}"] = (
                    float((distances <= threshold).mean()) if count else 0.0
                )
            correct = int((distances <= 1e-4).sum())
            precision = correct / count if count else 0.0
            if estimate_pose:
                metrics["pose_error"] = pose_error(p0, p1, k0, k1, transform)
        metrics.update(
            {
                "usable": bool(
                    correct >= self.min_correct and precision >= self.min_precision
                ),
                "criterion_correct": int(correct),
                "criterion_precision": float(precision),
            }
        )
        return metrics
