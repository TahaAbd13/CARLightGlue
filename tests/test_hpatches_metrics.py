import unittest

import numpy as np

from experiments.hpatches_metrics import correspondence_geometry, pair_metrics


class MetricTest(unittest.TestCase):
    def test_translation_thresholds_visibility_and_denominators(self):
        k0 = np.array([[0, 0], [10, 10], [50, 50], [95, 95]])
        k1 = np.array([[5, 0], [17, 10], [70, 50]])
        h = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]])
        errors, possible = correspondence_geometry(k0, k1, h, (100, 100))
        self.assertEqual(possible, {1: 1, 3: 2, 5: 2})
        self.assertTrue(np.isinf(errors[3]).all())
        metrics = pair_metrics([[0, 0], [1, 1], [2, 2]], errors, possible)
        self.assertEqual(metrics["precision_1px"], 1 / 3)
        self.assertEqual(metrics["precision_3px"], 2 / 3)
        self.assertEqual(metrics["recall_3px"], 1.0)
        self.assertEqual(pair_metrics([], errors, possible)["precision_3px"], 0)
        with self.assertRaises(ValueError):
            pair_metrics([[0, 0], [0, 1]], errors, possible)

    def test_empty_detections(self):
        errors, possible = correspondence_geometry(
            np.empty((0, 2)), np.empty((0, 2)), np.eye(3), (10, 10)
        )
        self.assertEqual(possible, {1: 0, 3: 0, 5: 0})
        self.assertEqual(pair_metrics([], errors, possible)["recall_1px"], 0)


if __name__ == "__main__":
    unittest.main()
