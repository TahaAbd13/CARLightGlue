# CAR evaluation

Dataset: megadepth1500; partition: all; pairs: 1500; device: cuda.

All matcher times include head evaluation, feature extraction for CAR, GPU synchronization required by decisions, optional keypoint pruning, and scheduled geometry. SuperPoint extraction, disk I/O, feature transfers, downstream pose estimation and GT evaluation are excluded.

ACCEPT returns correspondences. REJECT and UNRESOLVED withhold all matches; both contribute zero to pair MMA and pose AUC. RETURN is the original baselines' unconditional output, not a calibrated acceptance decision. For decision metrics only, a baseline proposal counts as accepted when its match count meets the same minimum-count gate as CAR (20 by default); empty returns are not false accepts. Accuracy metrics still use all original returned matches.

False acceptance = accepted/returned output failing the defined usability criterion. FA/accepted divides by accepted pairs. False rejection = REJECT on a pair usable at any depth within budget; FR/usable divides by oracle-usable pairs. Recoverable rejection separately requires a usable strictly later layer. Unresolved pairs are NOT counted as rejections; their loss is visible in usable-pair retention and unconditional accuracy.

Negative rejection rates are conditional on the supplied labels. Geometry is a fitted homography for HPatches and a fundamental matrix for MegaDepth, never a GT transform.

HPatches MMA is macro pair precision; pooled precision = total correct/total predicted; recall = total correct/total geometrically matchable source detections. Errors use original-image pixels. Negative pairs are excluded from HPatches accuracy averages, but included in runtime and decision metrics for the 'all' group.

MegaDepth correspondence precision uses squared symmetric epipolar distance in normalized camera coordinates at 1e-4/5e-4/1e-3. Pose AUC integrates recall up to 5/10/20 degrees using max(rotation, sign-invariant translation-direction error); all failed/withheld estimates remain in its denominator. OpenCV essential-matrix RANSAC uses a fixed 1-pixel threshold. These settings are not identical to Glue Factory's default PoseLib benchmark.

CAR uses fixed heuristic thresholds without controller training. HPatches and MegaDepth use different geometric models. Select thresholds on validation data before evaluating test data.

Component profiling is measured in separate CAR passes with synchronized stage boundaries. It includes CPU work and transfers, and perturbs execution. Normal latency excludes these passes. Stage summaries average repetitions within each pair, then average pairs; skipped stages contribute zero. The residual 'other' includes Python bookkeeping and profiling overhead.

Component measurements: profile_summary.csv/json and time_profile_*.png.

CAR keypoint pruning is enabled: matchability threshold 0.1, minimum retained per image 64, starting at checkpoint depth 3. Pruning follows CONTINUE only. Match fractions retain original keypoint denominators; similarity margins use active points. Full-depth oracle traces remain unpruned. For withheld outputs, current_usable refers to the unpruned reference, not the pruned candidate. Active/retained counts are in pairs.jsonl.

| Split | Model | Layers | ms | Time saved | Accepted | Rejected | Unresolved | FA/accepted | FR/usable | MMA@3 / epi precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | CAR | 3.51 | 9.46 | 33.6% | 1474 | 0 | 26 | 0.038 | 0.000 | 0.925 |
| all | adaptive | 5.79 | 10.45 | 26.7% | 1492 | 0 | 0 | 0.021 | 0.000 | 0.970 |
| all | full_9 | 9.00 | 14.25 | 0.0% | 1493 | 0 | 0 | 0.019 | 0.000 | 0.973 |
| megadepth | CAR | 3.51 | 9.46 | 33.6% | 1474 | 0 | 26 | 0.038 | 0.000 | 0.925 |
| megadepth | adaptive | 5.79 | 10.45 | 26.7% | 1492 | 0 | 0 | 0.021 | 0.000 | 0.970 |
| megadepth | full_9 | 9.00 | 14.25 | 0.0% | 1493 | 0 | 0 | 0.019 | 0.000 | 0.973 |

Pose AUC, all three HPatches thresholds and usable-pair retention are in summary.csv/json. Raw decisions and heuristic scores are in pairs.jsonl. Negative time savings mean CAR was slower than the full-depth baseline.
