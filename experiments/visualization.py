"""HPatches correspondence visualization using ground-truth pixel errors."""

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def visualize(path, image0, image1, k0, k1, matches, errors, title, limit=300):
    """Deterministic evenly spaced subset; green <=3px, red >3px."""
    h = max(image0.shape[0], image1.shape[0])
    w0 = image0.shape[1]
    canvas = np.ones((h, w0 + image1.shape[1], 3), dtype=np.uint8) * 255
    canvas[: image0.shape[0], :w0] = image0
    canvas[: image1.shape[0], w0:] = image1
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)
    selection = np.linspace(0, len(matches) - 1, min(limit, len(matches)), dtype=int)
    shown = matches[selection]
    if len(shown):
        p0, p1 = k0[shown[:, 0]], k1[shown[:, 1]] + [w0, 0]
        correct = errors[shown[:, 0], shown[:, 1]] <= 3
        lines = np.stack([p0, p1], axis=1)
        ax.add_collection(
            LineCollection(
                lines,
                colors=np.where(correct, "lime", "red"),
                linewidths=0.6,
                alpha=0.65,
            )
        )
    ax.set_title(
        f"{title}\nGreen: <=3px; red: >3px / outside overlap. "
        f"Showing {len(shown)}/{len(matches)} matches"
    )
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
