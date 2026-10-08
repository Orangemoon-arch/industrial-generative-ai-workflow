from __future__ import annotations

import unittest

import numpy as np

from build_ngw_planetary_demo import extrude_radial_annulus, sampled_external_radius
from extract_spur_gear_parameters import compare_to_reference, extract, infer_tooth_count


class SpurGearParameterExtractionTests(unittest.TestCase):
    def test_fft_detects_27_periods(self) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 8192, endpoint=False)
        envelope = 10.0 + 0.8 * np.cos(27.0 * theta)

        result = infer_tooth_count(envelope)

        self.assertEqual(result["estimated_tooth_count"], 27)
        self.assertGreater(result["confidence_ratio_to_second"], 10.0)

    def test_extracts_known_involute_planet_parameters(self) -> None:
        theta, radius, _ = sampled_external_radius(
            teeth=27,
            module=1.5,
            pressure_angle_deg=20.0,
            tooth_thinning_mm=0.15,
            samples_per_tooth=32,
        )
        mesh = extrude_radial_annulus(theta, radius, 2.65, 8.0, "test_planet")

        result, _ = extract(mesh)

        self.assertEqual(result["estimated_tooth_count"], 27)
        self.assertTrue(result["tooth_count_signal_confident"])
        self.assertAlmostEqual(result["outside_diameter_mesh_units"], 43.5, places=3)
        self.assertAlmostEqual(result["root_diameter_mesh_units"], 36.75, places=3)
        self.assertAlmostEqual(result["bore_diameter_mesh_units"], 5.3, places=3)
        self.assertAlmostEqual(result["thickness_mesh_units"], 8.0, places=6)
        self.assertAlmostEqual(result["module_from_outside_diameter_mesh_units"], 1.5, places=3)

    def test_reference_comparison_aligns_outside_diameter(self) -> None:
        reference = {
            "estimated_tooth_count": 27,
            "outside_diameter_mesh_units": 43.5,
            "root_diameter_mesh_units": 36.75,
            "bore_diameter_mesh_units": 5.3,
            "thickness_mesh_units": 8.0,
            "module_from_outside_diameter_mesh_units": 1.5,
            "pitch_diameter_mesh_units": 40.5,
        }
        candidate = {key: value / 20.0 for key, value in reference.items() if key != "estimated_tooth_count"}
        candidate["estimated_tooth_count"] = 27

        comparison = compare_to_reference(candidate, reference)

        self.assertTrue(comparison["tooth_count_matches"])
        self.assertAlmostEqual(comparison["scale_candidate_to_reference_by_outside_diameter"], 20.0)
        for error in comparison["relative_error_after_outside_diameter_alignment"].values():
            self.assertAlmostEqual(error, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
