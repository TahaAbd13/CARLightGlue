import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.car_data import group_identical_content, load_megadepth
from experiments.car_metrics import (
    epipolar_errors,
    pose_auc,
    pose_error,
)
from experiments.car_prepare import validate_partitions
from lightglue import LightGlue
from lightglue.car_match import (
    CAR,
    FEATURE_NAMES,
    Action,
    CARConfig,
    DecisionPolicy,
    car_scores,
    extract_quality,
    geometric_quality,
    pruning_indices,
)


class CARTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)
        self.backbone = LightGlue(
            features=None,
            input_dim=16,
            descriptor_dim=16,
            num_heads=4,
            flash=False,
            filter_threshold=0,
            depth_confidence=-1,
            width_confidence=-1,
        ).eval()
        self.data = {
            f"image{i}": {
                "keypoints": torch.rand(1, n, 2) * 64,
                "descriptors": torch.rand(1, n, 16),
                "image_size": torch.tensor([[64.0, 64.0]]),
            }
            for i, n in enumerate((12, 10))
        }

    def test_disabled_and_trace_equal_standard_nine_layers(self):
        car = CAR(self.backbone)
        expected = self.backbone(self.data)
        for output in (car(self.data, enabled=False), car(self.data, collect=True)):
            for key in (
                "matches0",
                "matches1",
                "matching_scores0",
                "matching_scores1",
                "prune0",
                "prune1",
            ):
                torch.testing.assert_close(expected[key], output[key], rtol=0, atol=0)
            self.assertEqual(output["stop"], 9)

    def test_distinct_layers_and_heads_used_once_and_frozen(self):
        called = []
        handles = [
            module.register_forward_hook(
                lambda m, a, o, name=f"{kind}{i}": called.append(name)
            )
            for kind, modules in (
                ("layer", self.backbone.transformers),
                ("head", self.backbone.log_assignment),
            )
            for i, module in enumerate(modules)
        ]
        before = {k: v.clone() for k, v in self.backbone.state_dict().items()}
        car = CAR(self.backbone).train()
        trace = car(self.data, collect=True)
        self.assertFalse(self.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in self.backbone.parameters()))
        self.assertEqual(
            called, [name for i in range(9) for name in (f"layer{i}", f"head{i}")]
        )
        self.assertTrue(
            all(torch.isfinite(h["features"]).all() for h in trace["history"])
        )
        for name, value in self.backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for handle in handles:
            handle.remove()

    def test_accept_uses_corresponding_head_and_original_indices(self):
        config = CARConfig(min_matches=1, min_depth=3)
        car = CAR(self.backbone, config)
        expected = car(self.data, collect=True)["history"][2]["prediction"]
        with patch("lightglue.car_match.car_scores", return_value=(1.0, 0.9)):
            result = car(self.data)
        self.assertEqual(result["status"], "ACCEPT")
        self.assertEqual(result["executed_layers"], 3)
        self.assertEqual(
            [h["action"] for h in result["history"]], ["CONTINUE", "ACCEPT"]
        )
        torch.testing.assert_close(expected["matches0"], result["matches0"])
        matches = result["matches"][0]
        torch.testing.assert_close(result["matches0"][0, matches[:, 0]], matches[:, 1])

    def test_reject_withholds_all_correspondences(self):
        config = CARConfig(min_matches=1, reject_min_depth=1, rejection_patience=2)
        with patch("lightglue.car_match.car_scores", return_value=(0.0, 0.0)):
            result = CAR(self.backbone, config)(self.data)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["stop"], 3)
        self.assertEqual(len(result["matches"][0]), 0)
        self.assertTrue((result["matches0"] == -1).all())
        self.assertTrue((result["matching_scores0"] == 0).all())

    def test_recoverable_continues_and_maximum_is_unresolved(self):
        with patch("lightglue.car_match.car_scores", return_value=(0.0, 0.9)):
            result = CAR(self.backbone)(self.data)
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertEqual(result["stop"], 9)
        self.assertEqual(len(result["matches"][0]), 0)

    def test_geometry_is_sparse_and_optional(self):
        config = CARConfig(geometry="homography")
        with patch(
            "lightglue.car_match.geometric_quality", return_value=(1, 0.8, 0.5)
        ) as geo, patch("lightglue.car_match.car_scores", return_value=(0.0, 0.9)):
            CAR(self.backbone, config)(self.data)
            self.assertEqual(geo.call_count, 3)
        with patch("lightglue.car_match.geometric_quality", side_effect=AssertionError):
            CAR(self.backbone)(self.data)

    def test_gt_fields_are_not_consumed(self):
        car = CAR(self.backbone, CARConfig(min_matches=1))
        expected = car(self.data)
        result = car(
            {
                **self.data,
                "homography": object(),
                "gt_matches": object(),
                "pose": object(),
            }
        )
        torch.testing.assert_close(
            result["matching_scores0"], expected["matching_scores0"]
        )

    def test_scale_orientation_and_input_projection_match_original(self):
        backbone = LightGlue(
            features=None,
            input_dim=8,
            descriptor_dim=16,
            num_heads=4,
            add_scale_ori=True,
            flash=False,
            depth_confidence=-1,
            width_confidence=-1,
        ).eval()
        for item in self.data.values():
            count = item["keypoints"].shape[1]
            item.update(
                descriptors=torch.rand(1, count, 8),
                scales=torch.rand(1, count),
                oris=torch.rand(1, count),
            )
        expected = backbone(self.data)
        trace = CAR(backbone)(self.data, collect=True)
        torch.testing.assert_close(
            expected["matching_scores0"], trace["matching_scores0"], atol=0, rtol=0
        )
        torch.testing.assert_close(expected["matches0"], trace["matches0"])

    def test_empty_input_and_noncheckpoint_budget(self):
        self.data["image0"]["keypoints"] = torch.empty(1, 0, 2)
        self.data["image0"]["descriptors"] = torch.empty(1, 0, 16)
        car = CAR(self.backbone, CARConfig(max_depth=2))
        result = car(self.data)
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertEqual(result["stop"], 2)
        self.assertEqual(result["executed_layers"], 0)
        self.assertTrue(
            all(torch.isfinite(h["features"]).all() for h in result["history"])
        )

    def test_geometric_quality_translation(self):
        pts = torch.rand(1, 30, 2) * 100
        data = {"image0": {"keypoints": pts}, "image1": {"keypoints": pts + 10}}
        indices = torch.arange(30)
        prediction = {"matches": [torch.stack((indices, indices), -1)]}
        available, fraction, _ = geometric_quality(
            data, prediction, CARConfig(geometry="homography")
        )
        self.assertEqual(available, 1)
        self.assertGreater(fraction, 0.95)

    def test_profile_preserves_output_and_accounts_for_stages(self):
        car = CAR(self.backbone)
        normal = car(self.data)
        profiled = car(self.data, profile=True)
        self.assertNotIn("profile", normal)
        self.assertEqual(normal["status"], profiled["status"])
        self.assertEqual(normal["stop"], profiled["stop"])
        torch.testing.assert_close(normal["matches0"], profiled["matches0"])
        profile = profiled["profile"]
        self.assertAlmostEqual(sum(profile["stages_ms"].values()), profile["total_ms"])
        self.assertTrue(all(e["milliseconds"] >= 0 for e in profile["events"]))
        self.assertIn("margin", profile["stages_ms"])
        self.assertNotIn("geometry", profile["stages_ms"])
        self.assertNotIn("coverage", profile["stages_ms"])
        depths = [e["depth"] for e in profile["events"] if e["stage"] == "transformer"]
        self.assertEqual(depths, list(range(1, normal["stop"] + 1)))

    def test_normal_inference_never_constructs_profiler(self):
        with patch("lightglue.car_match.StageProfiler", side_effect=AssertionError):
            CAR(self.backbone)(self.data)

    def test_pruning_compacts_descriptors_and_preserves_original_indices(self):
        sizes = []
        saved_encodings = {}

        def before_layer(module, args):
            sizes.append([args[0].shape[1], args[1].shape[1]])
            if len(sizes) == 1:
                saved_encodings["first"] = [args[2].clone(), args[3].clone()]
            if len(sizes) == 2:
                for side, keep in enumerate(([1, 3, 7], [2, 5])):
                    expected = saved_encodings["first"][side].index_select(
                        -2, torch.tensor(keep)
                    )
                    torch.testing.assert_close(args[side + 2], expected)

        handles = [
            m.register_forward_pre_hook(before_layer)
            for m in self.backbone.transformers
        ]

        def head(d0, d1):
            m, n = d0.shape[1], d1.shape[1]
            scores = d0.new_full((1, m + 1, n + 1), -10)
            diagonal = torch.arange(min(m, n))
            scores[0, diagonal, diagonal] = np.log(0.9)
            ma0, ma1 = d0.new_full((m,), 0.01), d0.new_full((n,), 0.01)
            if m == 12:
                ma0[[1, 3, 7]] = 0.9
                ma1[[2, 5]] = 0.9
            scores[0, :-1, -1] = (1 - ma0).log()
            scores[0, -1, :-1] = (1 - ma1).log()
            return scores, torch.eye(m, n)[None]

        config = CARConfig(
            pruning=True,
            pruning_threshold=0.5,
            pruning_min_keypoints=1,
            pruning_start_depth=1,
            min_depth=3,
            max_depth=3,
            min_matches=1,
            geometry="homography",
            geometry_layers=(3,),
        )
        before = {k: v.clone() for k, v in self.backbone.state_dict().items()}
        with patch.object(
            self.backbone.log_assignment[0], "forward", side_effect=head
        ), patch.object(
            self.backbone.log_assignment[2], "forward", side_effect=head
        ), patch(
            "lightglue.car_match.car_scores", return_value=(1.0, 0.8)
        ), patch(
            "lightglue.car_match.geometric_quality", return_value=(1, 0.8, 0)
        ) as geo:
            output = CAR(self.backbone, config)(self.data, profile=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(sizes, [[12, 10], [3, 2], [3, 2]])
        self.assertEqual(output["status"], "ACCEPT")
        expected = torch.tensor([[1, 2], [3, 5]])
        torch.testing.assert_close(output["matches"][0], expected)
        torch.testing.assert_close(geo.call_args.args[1]["matches"][0], expected)
        self.assertEqual(output["matches0"].shape, (1, 12))
        self.assertEqual(output["matches1"].shape, (1, 10))
        self.assertEqual(output["matches0"][0, 1].item(), 2)
        self.assertEqual(output["matches1"][0, 5].item(), 3)
        self.assertEqual(output["matches0"][0, 0].item(), -1)
        self.assertEqual(output["matching_scores0"][0, 0].item(), 0)
        self.assertEqual(output["prune0"][0, 0].item(), 1)
        self.assertEqual(output["prune0"][0, 1].item(), 3)
        self.assertEqual(output["history"][0]["retained_keypoints"], [3, 2])
        # Keep the original denominator (min(12,10)), not min(3,2).
        self.assertAlmostEqual(output["history"][-1]["features"][1].item(), 0.2)
        self.assertIn("pruning", output["profile"]["stages_ms"])
        self.assertEqual(
            [
                e["depth"]
                for e in output["profile"]["events"]
                if e["stage"] == "pruning"
            ],
            [1],
        )
        for key, value in self.backbone.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertEqual(self.data["image0"]["descriptors"].shape[1], 12)

    def test_repeated_pruning_and_unresolved_output(self):
        sizes = []
        handles = [
            m.register_forward_pre_hook(
                lambda module, args: sizes.append([args[0].shape[1], args[1].shape[1]])
            )
            for m in self.backbone.transformers
        ]

        def wrapped_head(original):
            def forward(d0, d1):
                assignment, similarity = original(d0, d1)
                assignment = assignment.clone()
                for side, n in enumerate((d0.shape[1], d1.shape[1])):
                    ma = d0.new_full((n,), 0.01)
                    ma[::2] = 0.9
                    if side == 0:
                        assignment[0, :-1, -1] = (1 - ma).log()
                    else:
                        assignment[0, -1, :-1] = (1 - ma).log()
                return assignment, similarity

            return forward

        with ExitStack() as stack:
            for index in (0, 2):
                module = self.backbone.log_assignment[index]
                stack.enter_context(
                    patch.object(
                        module, "forward", side_effect=wrapped_head(module.forward)
                    )
                )
            stack.enter_context(
                patch("lightglue.car_match.car_scores", return_value=(0.0, 0.8))
            )
            result = CAR(
                self.backbone,
                CARConfig(
                    pruning=True,
                    pruning_threshold=0.5,
                    pruning_min_keypoints=1,
                    pruning_start_depth=1,
                    max_depth=5,
                ),
            )(self.data, profile=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(sizes, [[12, 10], [6, 5], [6, 5], [3, 3], [3, 3]])
        self.assertEqual(result["active_keypoints"], [3, 3])
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertTrue((result["matches0"] == -1).all())
        self.assertTrue((result["matching_scores0"] == 0).all())
        self.assertEqual(result["matches0"].shape, (1, 12))
        self.assertEqual(result["matches1"].shape, (1, 10))
        torch.testing.assert_close(
            torch.where(result["prune0"][0] == 5)[0], torch.tensor([0, 4, 8])
        )
        self.assertEqual(
            [
                e["depth"]
                for e in result["profile"]["events"]
                if e["stage"] == "pruning"
            ],
            [1, 3],
        )

    def test_pruning_noop_and_diagnostic_trace(self):
        normal = CAR(self.backbone)(self.data)
        car = CAR(
            self.backbone,
            CARConfig(
                pruning=True,
                pruning_threshold=0,
                pruning_start_depth=1,
                pruning_min_keypoints=1,
            ),
        )
        pruned = car(self.data)
        self.assertEqual(normal["status"], pruned["status"])
        for key in ("matches0", "matches1", "matching_scores0", "matching_scores1"):
            torch.testing.assert_close(pruned[key], normal[key], atol=0, rtol=0)
        with patch("lightglue.car_match.pruning_indices", side_effect=AssertionError):
            trace = car(self.data, collect=True)
            standard = self.backbone(self.data)
            torch.testing.assert_close(trace["matches0"], standard["matches0"])
            car(self.data, enabled=False)

    def test_pruning_empty_input_and_single_retained_point(self):
        self.data["image0"]["keypoints"] = torch.empty(1, 0, 2)
        self.data["image0"]["descriptors"] = torch.empty(1, 0, 16)
        output = CAR(self.backbone, CARConfig(pruning=True))(self.data)
        self.assertEqual(output["executed_layers"], 0)
        self.assertEqual(output["matches0"].shape, (1, 0))
        self.assertEqual(output["matches1"].shape, (1, 10))
        features = extract_quality(
            {
                "image0": {"keypoints": torch.ones(1, 12, 2)},
                "image1": {"keypoints": torch.ones(1, 10, 2)},
            },
            {"scores": [torch.tensor([0.8])], "matches": [torch.tensor([[3, 4]])]},
            torch.ones(1, 1, 1),
            None,
            CARConfig(),
            (0, 0, 0),
        )
        self.assertEqual(features[FEATURE_NAMES.index("similarity_margin")].item(), 0)


class PolicyAndMetricsTests(unittest.TestCase):
    def test_pruning_threshold_floor_ties_and_nonfinite(self):
        scores = torch.tensor([0.01, 0.9, 0.02, 0.8])
        torch.testing.assert_close(
            pruning_indices(scores, 0.5, 1), torch.tensor([1, 3])
        )
        torch.testing.assert_close(
            pruning_indices(scores, 1.0, 2), torch.tensor([1, 3])
        )
        torch.testing.assert_close(pruning_indices(scores, 0.5, 10), torch.arange(4))
        torch.testing.assert_close(
            pruning_indices(torch.zeros(4), 0.5, 2), torch.tensor([0, 1])
        )
        self.assertEqual(
            pruning_indices(torch.tensor([float("nan")]), 0.5, 1).numel(), 1
        )
        for kwargs in (
            {"pruning_threshold": float("nan")},
            {"pruning_threshold": 1.1},
            {"pruning_min_keypoints": 0},
            {"pruning_start_depth": 0},
        ):
            with self.assertRaises(ValueError):
                CARConfig(**kwargs).validate(9)

    def test_concentrated_matches_do_not_require_coverage(self):
        # All matched points lie in one grid cell. Other evidence is sufficient.
        data = {
            f"image{i}": {
                "keypoints": torch.ones(1, 20, 2),
                "image_size": torch.tensor([[100.0, 100.0]]),
            }
            for i in range(2)
        }
        indices = torch.arange(20)
        prediction = {
            "matches": [torch.stack((indices, indices), -1)],
            "scores": [torch.full((20,), 0.8)],
        }
        features = extract_quality(
            data,
            prediction,
            torch.eye(20)[None],
            None,
            CARConfig(geometry="homography"),
            (1.0, 0.8, 0.0),
        )
        self.assertEqual(len(FEATURE_NAMES), 8)
        self.assertEqual(len(features), 8)
        self.assertFalse(any("coverage" in name for name in FEATURE_NAMES))
        usable, recovery = car_scores(features)
        self.assertEqual(usable, 0.99)
        self.assertEqual(
            DecisionPolicy(CARConfig()).decide(3, 20, usable, recovery), Action.ACCEPT
        )

    def test_car_rules_mean_confidence_and_geometry(self):
        f = dict.fromkeys(FEATURE_NAMES, 0.0)
        f.update(
            log_matches=np.log1p(20),
            match_fraction=0.2,
            score_mean=0.5,
            similarity_margin=0.2,
        )

        def scores():
            return car_scores(torch.tensor([f[k] for k in FEATURE_NAMES]))

        self.assertEqual(scores()[0], 0.99)
        f["score_mean"] = 0.2
        self.assertEqual(scores()[0], 0.5)
        f.update(score_mean=0.5, geometry_enabled=1)
        self.assertEqual(scores()[0], 0.5)
        f.update(geometry_available=1, geometry_inlier_fraction=0.8)
        self.assertEqual(scores()[0], 0.99)
        f.update(match_fraction=0.01, log_matches=np.log1p(1), delta_match_fraction=0)
        self.assertEqual(scores(), (0.01, 0.01))

    def test_profile_plots_from_synthetic_records(self):
        from experiments.car_report import plot_profiles

        records = [
            {
                "pair": "synthetic",
                "kind": "viewpoint",
                "total_ms": 10,
                "stages_ms": {
                    "transformer": 6,
                    "margin": 2,
                    "pruning": 0.5,
                    "other": 1.5,
                },
                "events": [
                    {"stage": "transformer", "depth": 1, "milliseconds": 6},
                    {"stage": "margin", "depth": 1, "milliseconds": 2},
                    {"stage": "pruning", "depth": 1, "milliseconds": 0.5},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plot_profiles(root, records)
            for name in (
                "profile_summary.csv",
                "profile_summary.json",
                "time_profile_all.png",
                "time_profile_viewpoint.png",
            ):
                self.assertGreater((root / name).stat().st_size, 0)

    def test_profile_summary_weights_pairs_not_repetitions(self):
        from experiments.car_report import profile_summary

        records = [
            {
                "pair": "a",
                "kind": "viewpoint",
                "total_ms": 10,
                "stages_ms": {"margin": 2, "other": 8},
            },
            {
                "pair": "a",
                "kind": "viewpoint",
                "total_ms": 10,
                "stages_ms": {"margin": 4, "other": 6},
            },
            {
                "pair": "b",
                "kind": "viewpoint",
                "total_ms": 20,
                "stages_ms": {"other": 20},
            },
        ]
        rows = [r for r in profile_summary(records) if r["kind"] == "all"]
        margin = next(r for r in rows if r["stage"] == "margin")
        self.assertEqual(margin["mean_ms"], 1.5)
        self.assertEqual(margin["pairs"], 2)
        self.assertEqual(margin["profiled_total_ms"], 15)
        self.assertAlmostEqual(sum(r["fraction_of_profiled_total"] for r in rows), 1)

    def test_geometry_schedule_must_match_decision_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "decision checkpoints"):
            CARConfig(checkpoints=(2, 4, 9), geometry="homography").validate(9)
        CARConfig(
            checkpoints=(2, 4, 9), geometry="homography", geometry_layers=(4, 9)
        ).validate(9)

    def test_epipolar_and_pose_geometry_on_known_camera_pair(self):
        rng = np.random.default_rng(5)
        xyz = rng.uniform(-1, 1, (100, 3))
        xyz[:, 2] += 5
        transform = np.eye(4)
        transform[0, 3] = 0.7
        other = xyz + transform[:3, 3]
        k = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
        p0 = (xyz @ k.T)[:, :2] / xyz[:, 2:]
        p1 = (other @ k.T)[:, :2] / other[:, 2:]
        self.assertLess(epipolar_errors(p0, p1, k, k, transform).max(), 1e-12)
        corrupted = p1.copy()
        corrupted[:, 1] += 30
        self.assertGreater(epipolar_errors(p0, corrupted, k, k, transform).min(), 1e-3)
        self.assertLess(pose_error(p0, p1, k, k, transform), 0.01)

    def test_megadepth_world_to_camera_composition(self):
        rotation = "1 0 0 0 1 0 0 0 1"
        camera = "PINHOLE 640 480 500 501 320 240"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "views.txt").write_text(
                f"a.jpg {rotation} 1 0 0 {camera}\n"
                f"b.jpg {rotation} 3 0 0 {camera}\n"
            )
            (root / "pairs.txt").write_text("a.jpg b.jpg\n")
            pair = load_megadepth(root)[0]
            self.assertEqual(pair["T_0to1"][0][3], 2)
            self.assertEqual(pair["K1"][1][1], 501)

    def test_conservative_patience_and_terminal_order(self):
        policy = DecisionPolicy(CARConfig(min_depth=3, reject_min_depth=3))
        self.assertEqual(policy.decide(1, 100, 1, 1), Action.CONTINUE)
        self.assertEqual(policy.decide(3, 100, 0, 0), Action.CONTINUE)
        self.assertEqual(policy.decide(5, 100, 0, 0.9), Action.CONTINUE)
        self.assertEqual(policy.decide(7, 100, 0, 0), Action.CONTINUE)
        self.assertEqual(policy.decide(9, 100, 0, 0), Action.UNRESOLVED)
        self.assertEqual(policy.decide(9, 100, 1, 0), Action.ACCEPT)
        self.assertEqual(policy.decide(9, 0, 1, 0), Action.UNRESOLVED)
        self.assertEqual(policy.decide(9, 100, float("nan"), 0), Action.UNRESOLVED)

    def test_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_partitions(
                [
                    {"id": "a", "partition": "train", "image0": "a", "image1": "b"},
                    {"id": "b", "partition": "test", "image0": "b", "image1": "c"},
                ]
            )

    def test_pose_auc_includes_failed_pairs(self):
        self.assertEqual(pose_auc([float("inf")], 5), 0)
        self.assertEqual(pose_auc([0], 5), 1)
        self.assertAlmostEqual(pose_auc([0, float("inf")], 5), 0.5)

    def test_identical_contents_cannot_cross_partitions(self):
        pairs = [
            {"id": "a", "partition": "train", "image0": "/a", "image1": "/b"},
            {"id": "b", "partition": "test", "image0": "/c", "image1": "/d"},
        ]
        with self.assertRaisesRegex(ValueError, "Identical image content"):
            validate_partitions(
                pairs, {"/a": "same", "/b": "b", "/c": "same", "/d": "d"}
            )

    def test_transitive_duplicate_sequences_stay_together(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / name for name in "abcd"]
            for path, contents in zip(paths, ((b"x",), (b"x", b"y"), (b"y",), (b"z",))):
                path.mkdir()
                for index, data in enumerate(contents):
                    (path / f"{index}.ppm").write_bytes(data)
            groups, membership, duplicates = group_identical_content(
                paths, {"train": ["a"], "validation": ["b"], "test": ["c", "d"]}
            )
            self.assertEqual(sorted(groups["train"]), ["a", "b", "c"])
            self.assertEqual(groups["test"], ["d"])
            self.assertEqual(membership["a"], membership["c"])
            self.assertEqual(duplicates, [["a", "b", "c"]])


if __name__ == "__main__":
    unittest.main()
