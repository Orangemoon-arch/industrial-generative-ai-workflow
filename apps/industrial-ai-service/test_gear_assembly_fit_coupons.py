"""CPU geometry tests for the V1 gear assembly fit coupons."""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

import build_gear_assembly_fit_coupons as coupons
import build_parametric_gear_assembly as gear_assembly


class GearAssemblyFitCouponsTest(unittest.TestCase):
    def test_combined_plate_has_eight_watertight_components(self):
        parameters = gear_assembly.default_parameters()
        meshes = [
            coupons.make_d_hole_ring(parameters, 0.30, 20.0),
            coupons.make_d_hole_ring(parameters, 0.50, 22.0),
            coupons.make_d_hole_ring(parameters, 0.70, 24.0),
            coupons.make_d_plug(parameters),
            coupons.print_oriented_round_socket(parameters, 0.10, 14.0),
            coupons.print_oriented_round_socket(parameters, 0.30, 16.0),
            coupons.print_oriented_round_socket(parameters, 0.50, 18.0),
            coupons.make_round_plug(parameters),
        ]
        placements = [
            (-39.0, 16.0, 0.0),
            (-13.0, 16.0, 0.0),
            (13.0, 16.0, 0.0),
            (39.0, 16.0, 0.0),
            (-39.0, -16.0, 0.0),
            (-13.0, -16.0, 0.0),
            (13.0, -16.0, 0.0),
            (39.0, -16.0, 0.0),
        ]
        placed = []
        for mesh, translation in zip(meshes, placements):
            part = mesh.copy()
            part.apply_translation(translation)
            placed.append(part)

        combined = trimesh.util.concatenate(placed)
        combined.remove_unreferenced_vertices()
        components = combined.split(only_watertight=False)
        edge_counts = np.bincount(
            combined.edges_unique_inverse,
            minlength=len(combined.edges_unique),
        )

        self.assertEqual(len(components), 8)
        self.assertTrue(all(component.is_watertight for component in components))
        self.assertTrue(combined.is_winding_consistent)
        self.assertEqual(int((edge_counts == 1).sum()), 0)
        self.assertEqual(int((edge_counts > 2).sum()), 0)
        np.testing.assert_allclose(combined.extents, [98.0, 53.0, 11.0], atol=1e-9)
        self.assertGreaterEqual(float(combined.bounds[0, 2]), -1e-12)


if __name__ == "__main__":
    unittest.main()
