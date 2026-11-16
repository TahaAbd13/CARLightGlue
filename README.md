# CARLightGlue

**Confidence-Adaptive Refinement Controller (CAR)** for
[LightGlue](https://github.com/cvg/LightGlue). CAR uses fixed matching and geometry
rules to stop refinement early, with optional keypoint pruning to reduce later
computation. All pretrained weights remain frozen; no training is required.

## How it works

At checkpoints 1, 3, 5, 7, and 9, CAR checks match count, mean confidence,
similarity margin, match-count progress, and estimated geometric consistency.
It accepts the current matches, continues refinement, or returns no matches
when evidence remains insufficient. Geometry uses homographies on HPatches
and fundamental matrices on MegaDepth. Ground truth is used only for evaluation.

Optional pruning removes low-matchability keypoints after a CONTINUE decision,
while preserving original output indices. Defaults: threshold **0.1**, at least
**64 points per image**, starting at layer **3**.

Evaluations compare **full nine-layer LightGlue**, **original adaptive LightGlue**,
and **CAR** using identical SuperPoint features. Standard LightGlue remains
available through its original Python API.

## Installation

Requires Python 3.10+ and PyTorch 2.x. Use a CUDA-compatible PyTorch build for GPU execution.

```bash
git clone https://github.com/TahaAbd13/CARLightGlue.git
cd CARLightGlue
python -m pip install -v -e .
```

The package is `carlightglue`; Python imports remain `lightglue`. Pretrained
weights download on first use. Datasets, caches, and full local results are
excluded from Git. Run the following commands from the repository root.

## Evaluate

### HPatches

Download the [full-image sequences](https://github.com/hpatches/hpatches-dataset#full-image-sequences)
—not descriptor patches—and extract them into `data/hpatches-sequences-release`.

```bash
# Create the pair manifest once
python -u -m experiments.car_prepare \
  --data data/hpatches-sequences-release \
  --output data/hpatches_pairs.json --seed 17 --verbose

# Compare all three methods
python -u -m experiments.car_evaluate \
  --manifest data/hpatches_pairs.json --partition test \
  --output results/HPatches/car_01 \
  --device cuda --max-keypoints 512 --resize 640 \
  --warmup 1 --timing-repeats 3 \
  --profile --profile-repeats 3 --verbose
```

### MegaDepth-1500

```bash
# Skip this if the dataset is already downloaded
python -u -m experiments.download_megadepth \
  --destination data --workers 16 --verbose

python -u -m experiments.car_evaluate \
  --megadepth data/megadepth1500 \
  --output results/Megadepth/car_01 \
  --device cuda --max-keypoints 512 --resize 640 \
  --warmup 1 --timing-repeats 3 \
  --profile --profile-repeats 3 --verbose
```

| Option | Purpose |
|---|---|
| `--device cpu` | Run without CUDA |
| `--prune-keypoints` | Enable CAR keypoint pruning; disabled by default |
| `--pruning-threshold 0.1` | Minimum matchability, subject to the retention floor |
| `--pruning-min-keypoints 64` | Minimum retained points per image |
| `--pruning-start-depth 3` | First eligible pruning checkpoint |
| `--limit 6` | Evaluate only the first six pairs for a quick check |
| `--profile` | Collect separate component-timing measurements and plots |
| `--verbose` | Print detailed progress |

Use a **new or empty output directory** for each run. Open its `REPORT.md` for
results; `summary.csv`, `pairs.jsonl`, and `manifest.json` provide measurements,
decisions, and settings. All options are available with:

```bash
python -m experiments.car_evaluate --help
```

HPatches reports illumination/viewpoint precision, recall, MMA, and correct-match
counts at 1/3/5 pixels. Precision pools match counts; MMA averages pair precision
(empty outputs count as zero). Recall is relative to geometrically matchable
source detections. MegaDepth reports epipolar precision and pose AUC at 5/10/20°.
Failed or withheld poses remain in the denominator.

## Saved MegaDepth-1500 results

Previously recorded **pruned run**, 1,500 pairs; RTX 3060 Laptop GPU,
SuperPoint with 512 keypoints and resize 640, PyTorch 2.9.1+cu128.
One warmup and three timing repetitions; flash attention and mixed precision off.

| Method | Mean layers | Latency (ms) ↓ | AUC@5° (%) ↑ | AUC@10° (%) ↑ | AUC@20° (%) ↑ |
|---|---:|---:|---:|---:|---:|
| Full LightGlue | 9.00 | 14.25 | 27.59 | 45.05 | 61.86 |
| Adaptive LightGlue | 5.79 | 10.45 | 26.98 | 45.17 | 62.58 |
| CAR + pruning | 3.51 | 9.46 | 25.92 | 42.83 | 59.71 |

CAR used approximately **9.5% less matcher time than adaptive LightGlue**, with
lower pose accuracy. It withheld 26 unresolved pairs and rejected none early.
This is a measured accuracy–latency trade-off, not a guarantee of improvement.

Latency averages per-pair medians and excludes feature extraction, image I/O,
feature transfers, and downstream pose estimation. Pose evaluation uses OpenCV
essential-matrix RANSAC at 1 pixel; results are not directly interchangeable with
other benchmark protocols.

![Accuracy versus matcher latency](assets/results/car_pruned_01/accuracy_latency.png)

<details>
<summary>Component timing and archived measurements</summary>

![CAR component timing](assets/results/car_pruned_01/time_profile_megadepth.png)

Profiling uses separate synchronized passes. The 10.02 ms instrumented total
must not replace the 9.46 ms normal latency; later-depth bars include only pairs
reaching that depth.

- Pruned run: [report](assets/results/car_pruned_01/REPORT.md), [summary](assets/results/car_pruned_01/summary.csv), [settings](assets/results/car_pruned_01/manifest.json), [pose-recall plot](assets/results/car_pruned_01/pose_recall.png).
- Unpruned run: [report](assets/results/car_1500_01/REPORT.md), [summary](assets/results/car_1500_01/summary.csv), [settings](assets/results/car_1500_01/manifest.json).

Baseline timings differ between runs; their raw latency difference does not
isolate the effect of pruning.

</details>

## Development

```bash
# Synthetic tests; no dataset download or benchmark
python -m unittest discover -s tests -v
```

Controller: [`lightglue/car_match.py`](lightglue/car_match.py).
Evaluation and reporting: [`experiments/`](experiments/).

## Attribution and license

Based on **LightGlue: Local Feature Matching at Light Speed**, Philipp
Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys, ICCV 2023. Please cite
the [original paper](https://arxiv.org/abs/2306.13643) when using this work.
CAR is an experimental addition, not part of the original publication.

LightGlue code and weights use [Apache-2.0](LICENSE). SuperPoint has a
[separate restrictive license](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE),
including its weights and inference code. ALIKED uses BSD-3-Clause.
