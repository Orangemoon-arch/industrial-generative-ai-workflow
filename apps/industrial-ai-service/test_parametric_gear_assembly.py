"""CPU geometry tests; run with the Hunyuan environment that owns trimesh."""

from __future__ import annotations

import unittest

import build_parametric_gear_assembly as gear_assembly


class ParametricGearAssemblyTest(unittest.TestCase):
    def test_three_parts_pass_mesh_gate_and_clearances_are_positive(self):
        parameters = gear_assembly.default_parameters()
        meshes = {
            "gear": gear_assembly.make_gear(parameters),
            "shaft": gear_assembly.make_shaft(parameters),
            "cap": gear_assembly.make_cap(parameters),
        }
        for name, mesh in meshes.items():
            metrics = gear_assembly.validate_mesh(name, mesh)
            self.assertEqual(metrics["components"], 1)
            self.assertTrue(metrics["watertight"])
            self.assertEqual(metrics["boundary_edges"], 0)
            self.assertEqual(metrics["nonmanifold_edges"], 0)
        self.assertEqual(meshes["gear"].extents.tolist(), [60.0, 60.0, 10.0])
        self.assertEqual(meshes["shaft"].extents.tolist(), [24.0, 24.0, 18.0])
        self.assertEqual(meshes["cap"].extents.tolist(), [18.0, 18.0, 6.0])
        self.assertGreater(parameters["gear_bore_diametral_clearance_mm"], 0)
        self.assertGreater(parameters["cap_socket_diametral_clearance_mm"], 0)

    def test_scaled_geometry_preserves_validated_absolute_clearances(self):
        parameters = gear_assembly.scaled_parameters(2.5)
        self.assertEqual(parameters["gear_outer_diameter_mm"], 150.0)
        self.assertEqual(parameters["shaft_diameter_mm"], 30.0)
        self.assertEqual(parameters["retaining_peg_diameter_mm"], 20.0)
        self.assertEqual(parameters["gear_bore_diametral_clearance_mm"], 0.50)
        self.assertEqual(parameters["cap_socket_diametral_clearance_mm"], 0.30)
        meshes = {
            "gear": gear_assembly.make_gear(parameters),
            "shaft": gear_assembly.make_shaft(parameters),
            "cap": gear_assembly.make_cap(parameters),
        }
        for name, mesh in meshes.items():
            gear_assembly.validate_mesh(name, mesh)
        self.assertEqual(meshes["gear"].extents.tolist(), [150.0, 150.0, 25.0])
        self.assertEqual(meshes["shaft"].extents.tolist(), [60.0, 60.0, 45.0])
        self.assertEqual(meshes["cap"].extents.tolist(), [45.0, 45.0, 15.0])

    def test_fit_clearances_can_be_tightened_without_scaling_nominal_geometry(self):
        parameters = gear_assembly.scaled_parameters(
            1.0,
            gear_bore_clearance_mm=0.35,
            cap_socket_clearance_mm=0.20,
        )
        self.assertEqual(parameters["gear_bore_diametral_clearance_mm"], 0.35)
        self.assertEqual(parameters["cap_socket_diametral_clearance_mm"], 0.20)
        meshes = {
            "gear": gear_assembly.make_gear(parameters),
            "shaft": gear_assembly.make_shaft(parameters),
            "cap": gear_assembly.make_cap(parameters),
        }
        for name, mesh in meshes.items():
            gear_assembly.validate_mesh(name, mesh)
        self.assertEqual(meshes["gear"].extents.tolist(), [60.0, 60.0, 10.0])
        self.assertEqual(meshes["shaft"].extents.tolist(), [24.0, 24.0, 18.0])
        self.assertEqual(meshes["cap"].extents.tolist(), [18.0, 18.0, 6.0])

    def test_fit_clearance_override_rejects_unsafe_values(self):
        for value in (0.0, 1.01, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    gear_assembly.scaled_parameters(1.0, gear_bore_clearance_mm=value)

    def test_gear_and_socket_up_cap_can_share_one_print_plate(self):
        parameters = gear_assembly.scaled_parameters(
            1.0,
            gear_bore_clearance_mm=0.30,
            cap_socket_clearance_mm=0.10,
        )
        gear = gear_assembly.make_gear(parameters)
        cap = gear_assembly.make_cap(parameters)
        cap.apply_transform(
            gear_assembly.trimesh.transformations.rotation_matrix(
                gear_assembly.math.pi, [1, 0, 0]
            )
        )
        cap.apply_translation([0.0, 0.0, float(parameters["cap_height_mm"])])
        combined = gear_assembly.make_gear_cap_print_layout(parameters, gear, cap)
        components = combined.split(only_watertight=False)
        self.assertEqual(len(components), 2)
        self.assertTrue(all(component.is_watertight for component in components))
        self.assertAlmostEqual(float(combined.bounds[0, 2]), 0.0, places=8)
        self.assertLessEqual(float(combined.extents[0]), 81.0)
        self.assertEqual(float(combined.extents[1]), 60.0)


if __name__ == "__main__":
    unittest.main()
