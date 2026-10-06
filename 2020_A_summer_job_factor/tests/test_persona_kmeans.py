"""K-Means permutation, silhouette, and pack-pipeline I/O."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.clustering.persona_kmeans import (
    CONTEST_K,
    elbow_k_from_inertia,
    fit_kmeans,
    permute_labels_to_prototypes,
    run_persona_pipeline,
)


class SyntheticBlobTests(unittest.TestCase):
    def test_three_blobs_recover_k(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(loc=(2.0, 0.0, 0.0), scale=0.15, size=(20, 3))
        b = rng.normal(loc=(-1.0, 2.0, 0.0), scale=0.15, size=(20, 3))
        c = rng.normal(loc=(0.0, 0.0, -2.0), scale=0.15, size=(20, 3))
        x = np.vstack([a, b, c])
        model = fit_kmeans(x, 3, random_state=0)
        sil = float(silhouette_score(x, model.labels_))
        self.assertGreater(sil, 0.70)
        self.assertEqual(len(np.unique(model.labels_)), 3)

    def test_permutation_maps_to_prototypes(self) -> None:
        centroids = np.array(
            [
                [0.0, 0.0, -1.4],
                [1.3, 0.1, 0.0],
                [-0.5, 1.2, 0.1],
            ]
        )
        labels = np.array([0, 0, 1, 1, 2, 2])
        new_labels, new_c, perm, sse = permute_labels_to_prototypes(centroids, labels)
        self.assertEqual(set(perm), {0, 1, 2})
        self.assertEqual(len(np.unique(new_labels)), 3)
        self.assertGreater(new_c[0, 0], new_c[1, 0])
        self.assertGreater(new_c[1, 1], new_c[0, 1])
        self.assertLess(new_c[2, 2], new_c[0, 2])
        self.assertTrue(np.all(sse >= 0.0))

    def test_elbow_curvature_on_l_shape(self) -> None:
        inertias = np.array([100.0, 40.0, 22.0, 18.0, 16.0, 15.0])
        k_hat = elbow_k_from_inertia(inertias, k_min=1)
        self.assertGreaterEqual(k_hat, 2)
        self.assertLessEqual(k_hat, 4)


class PackPipelineTests(unittest.TestCase):
    def test_pipeline_k3_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_persona_pipeline(
                labels_path=root / "labels.csv",
                centroids_path=root / "centroids.csv",
                ksel_path=root / "ksel.csv",
                persona_md_path=root / "personas.md",
                figure_3d_path=root / "fig3d.png",
                figure_k_path=root / "figk.png",
            )
            self.assertEqual(result.n_obs, 50)
            self.assertEqual(result.k, CONTEST_K)
            self.assertEqual(result.labels.shape, (50,))
            self.assertEqual(set(result.labels.tolist()), {0, 1, 2})
            self.assertEqual(result.centroids.shape, (3, 3))
            self.assertEqual(len(result.personas), 3)
            self.assertEqual(sum(p.n_members for p in result.personas), 50)
            lab = pd.read_csv(root / "labels.csv")
            self.assertEqual(len(lab), 50)
            self.assertTrue((root / "fig3d.png").is_file())
            self.assertGreater((root / "fig3d.png").stat().st_size, 1000)
            md = (root / "personas.md").read_text(encoding="utf-8")
            self.assertIn("The Pragmatist", md)
            self.assertIn("not the designed cash", md)


if __name__ == "__main__":
    unittest.main()
