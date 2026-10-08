from __future__ import annotations

import unittest

from rebuild_external_spur_gear import build_mesh, topology


class RebuildExternalSpurGearTests(unittest.TestCase):
    def test_builds_confirmed_27_tooth_planet(self) -> None:
        mesh, dimensions = build_mesh(
            teeth=27,
            module_mm=1.5,
            pressure_angle_degrees=20.0,
            tooth_thinning_mm=0.15,
            bore_diameter_mm=5.3,
            face_width_mm=8.0,
            samples_per_tooth=64,
        )
        result = topology(mesh)

        self.assertAlmostEqual(dimensions["outside_diameter_mm"], 43.5)
        self.assertAlmostEqual(dimensions["root_diameter_mm"], 36.75)
        self.assertEqual(result["components"], 1)
        self.assertTrue(result["watertight"])
        self.assertTrue(result["winding_consistent"])
        self.assertEqual(result["boundary_edges"], 0)
        self.assertEqual(result["nonmanifold_edges"], 0)
        self.assertAlmostEqual(result["extents"][2], 8.0)

    def test_rejects_bore_larger_than_root(self) -> None:
        with self.assertRaises(ValueError):
            build_mesh(
                teeth=27,
                module_mm=1.5,
                pressure_angle_degrees=20.0,
                tooth_thinning_mm=0.15,
                bore_diameter_mm=40.0,
                face_width_mm=8.0,
                samples_per_tooth=32,
            )


if __name__ == "__main__":
    unittest.main()
