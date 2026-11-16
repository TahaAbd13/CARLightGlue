"""Directional homography metrics in original-image pixel coordinates."""

import numpy as np

THRESHOLDS = (1, 3, 5)


def correspondence_geometry(keypoints0, keypoints1, homography, size1):
    """Return pairwise forward errors and threshold-specific recall denominators.

    A source keypoint is matchable at t if its homographic projection is finite,
    lies in image 1, and has at least one detected target keypoint within t pixels.
    This is directional detection-conditioned recall, not repeatability or a
    count of all possible keypoint pairs. Correct one-to-one predictions are a
    subset of these source points, so recall cannot exceed one.
    """
    k0, k1 = np.asarray(keypoints0, float), np.asarray(keypoints1, float)
    h = np.asarray(homography, float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
        raise ValueError("Expected a finite, nonsingular 3x3 homography")
    p = np.column_stack((k0, np.ones(len(k0)))) @ h.T
    projected = np.full((len(k0), 2), np.nan)
    finite = np.abs(p[:, 2]) > 1e-12
    projected[finite] = p[finite, :2] / p[finite, 2:3]
    w, height = size1
    visible = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < w)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    errors = np.full((len(k0), len(k1)), np.inf)
    # Chunk the temporary NxMx2 array to bound evaluation memory.
    for start in range(0, len(k0), 256):
        stop = start + 256
        delta = projected[start:stop, None] - k1[None]
        errors[start:stop] = np.linalg.norm(delta, axis=-1)
    errors[~visible] = np.inf
    errors[~np.isfinite(errors)] = np.inf
    possible = {t: int((errors <= t).any(axis=1).sum()) for t in THRESHOLDS}
    return errors, possible


def pair_metrics(matches, errors, possible):
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    count = len(matches)
    if count:
        if (matches < 0).any() or (matches >= np.array(errors.shape)).any():
            raise ValueError("Match indices outside original keypoint arrays")
        if any(len(np.unique(matches[:, i])) != count for i in (0, 1)):
            raise ValueError("Expected one-to-one correspondences")
    distances = errors[matches[:, 0], matches[:, 1]]
    result = {"num_matches": count}
    for threshold in THRESHOLDS:
        correct = int((distances <= threshold).sum())
        result.update(
            {
                f"correct_{threshold}px": correct,
                f"matchable_{threshold}px": possible[threshold],
                f"precision_{threshold}px": correct / count if count else 0.0,
                f"recall_{threshold}px": correct / possible[threshold]
                if possible[threshold]
                else 0.0,
            }
        )
    return result
