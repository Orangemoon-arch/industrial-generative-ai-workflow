from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import trimesh

    from mesh_repair_tool import (
        assess_repairability,
        bidirectional_surface_distance_metrics,
        detached_artifact_cleanup_candidate,
        export_reload_gate,
        metrics,
        repair_degenerate_faces_by_edge_flip,
        safe_gate,
        safe_repair,
        standard_repair,
        surface_quality_metrics,
        surface_remesh_candidate,
        voxel_reconstruction_candidate,
    )
except ModuleNotFoundError:
    trimesh = None

try:
    import pymeshlab
except ModuleNotFoundError:
    pymeshlab = None


@unittest.skipIf(
    trimesh is None,
    "网格测试需要 Hunyuan 环境中的 trimesh",
)
class GenericSafeMeshRepairTests(unittest.TestCase):
    def test_surface_quality_is_diagnostic_only(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        quality = surface_quality_metrics(source)

        self.assertGreater(quality["edge_length"]["p50"], 0.0)
        self.assertGreaterEqual(quality["triangle_aspect_ratio"]["p99"], 1.0)
        self.assertGreaterEqual(
            quality["adjacent_normal_angle_degrees"]["max"], 0.0
        )
        # The diagnostic must not mutate the input mesh.
        self.assertTrue(source.is_watertight)

    def test_repairability_routes_simple_and_severe_topology(self) -> None:
        healthy = metrics(trimesh.creation.icosphere(subdivisions=2, radius=1.0))
        healthy_route = assess_repairability(healthy)
        self.assertEqual(healthy_route["route"], "BASE_GEOMETRY_CHECK_PASSED")

        severe = dict(healthy)
        severe["boundary_edges"] = 1200
        severe_route = assess_repairability(severe)
        self.assertEqual(severe_route["route"], "RECONSTRUCTION_RECOMMENDED")

    def test_repairability_routes_dominant_body_with_tiny_fragment(self) -> None:
        values = metrics(trimesh.creation.box())
        values["components"] = 2
        values["component_area_shares"] = [0.9999, 0.0001]

        route = assess_repairability(values)

        self.assertEqual(route["route"], "DETACHED_ARTIFACT_CLEANUP_CANDIDATE")

    def test_repairability_routes_surface_quality_to_visual_review(self) -> None:
        values = metrics(trimesh.creation.icosphere(subdivisions=2, radius=1.0))
        values["surface_quality"] = dict(values["surface_quality"])
        values["surface_quality"]["triangle_aspect_ratio"] = dict(
            values["surface_quality"]["triangle_aspect_ratio"]
        )
        values["surface_quality"]["triangle_aspect_ratio"]["p99"] = 25.0

        route = assess_repairability(values)

        self.assertEqual(route["route"], "SURFACE_QUALITY_VISUAL_REVIEW_REQUIRED")

    def test_bidirectional_surface_distance_for_identical_mesh_is_zero(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        distance = bidirectional_surface_distance_metrics(
            source, source.copy(), samples=2000, seed=1234
        )

        self.assertLess(distance["p99_to_diagonal_ratio"], 1e-10)
        self.assertLess(distance["max_to_diagonal_ratio"], 1e-10)

    def test_repairs_exact_duplicate_vertices(self) -> None:
        source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        vertices = np.vstack([source.vertices, source.vertices[0]])
        faces = np.asarray(source.faces).copy()
        first_reference = np.argwhere(faces == 0)[0]
        faces[tuple(first_reference)] = len(vertices) - 1
        damaged = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        repaired, operations = safe_repair(damaged)
        accepted = {item["name"] for item in operations if item["accepted"]}
        result = metrics(repaired)

        self.assertIn("merge_exact_duplicate_vertices", accepted)
        self.assertEqual(result["duplicate_vertices_exact"], 0)
        self.assertTrue(result["watertight"])
        self.assertEqual(result["boundary_edges"], 0)
        self.assertEqual(result["nonmanifold_edges"], 0)

    def test_repairs_duplicate_faces(self) -> None:
        source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        faces = np.vstack([source.faces, source.faces[0]])
        damaged = trimesh.Trimesh(
            vertices=source.vertices,
            faces=faces,
            process=False,
        )

        repaired, operations = safe_repair(damaged)
        accepted = {item["name"] for item in operations if item["accepted"]}
        result = metrics(repaired)

        self.assertIn("remove_duplicate_faces", accepted)
        self.assertEqual(result["duplicate_faces"], 0)
        self.assertTrue(result["watertight"])
        self.assertEqual(result["nonmanifold_edges"], 0)

    def test_repairs_inconsistent_winding(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=1)
        faces = np.asarray(source.faces).copy()
        faces[0] = faces[0][::-1]
        damaged = trimesh.Trimesh(vertices=source.vertices, faces=faces, process=False)
        self.assertFalse(damaged.is_winding_consistent)

        repaired, operations = safe_repair(damaged)
        accepted = {item["name"] for item in operations if item["accepted"]}

        self.assertIn("fix_winding_and_normals", accepted)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertTrue(repaired.is_watertight)

    def test_gate_rejects_new_boundary_and_lost_watertightness(self) -> None:
        source = trimesh.creation.box()
        damaged = source.copy()
        damaged.update_faces(np.arange(len(damaged.faces)) != 0)
        before = metrics(source)
        after = metrics(damaged)

        passed, failures = safe_gate(before, after)

        self.assertFalse(passed)
        self.assertIn("破坏了输入网格的水密性", failures)
        self.assertIn("边界边数量增加", failures)

    def test_export_reload_gate_allows_float_rounding_only(self) -> None:
        source = metrics(trimesh.creation.box())
        reloaded = dict(source)
        reloaded["extents"] = [value * (1.0 + 1e-8) for value in source["extents"]]
        reloaded["area"] = source["area"] * (1.0 + 1e-8)
        reloaded["volume"] = source["volume"] * (1.0 + 1e-8)

        passed, failures = export_reload_gate(source, reloaded)

        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_standard_repairs_one_small_missing_triangle(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        damaged = source.copy()
        damaged.update_faces(np.arange(len(damaged.faces)) != 0)
        self.assertFalse(damaged.is_watertight)

        repaired, operations = standard_repair(damaged)
        hole_operation = operations[-1]

        self.assertTrue(hole_operation["accepted"])
        self.assertEqual(hole_operation["faces_added"], 1)
        self.assertTrue(repaired.is_watertight)
        self.assertEqual(metrics(repaired)["boundary_edges"], 0)

    def test_standard_rejects_large_missing_triangle(self) -> None:
        source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        damaged = source.copy()
        damaged.update_faces(np.arange(len(damaged.faces)) != 0)

        repaired, operations = standard_repair(damaged)
        hole_operation = operations[-1]

        self.assertFalse(hole_operation["accepted"])
        self.assertEqual(len(repaired.faces), len(damaged.faces))
        self.assertFalse(repaired.is_watertight)

    def test_artifact_cleanup_removes_tiny_detached_solid(self) -> None:
        main = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        fragment = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
        fragment.apply_translation((0.0, 0.0, 3.0))
        damaged = trimesh.util.concatenate((main, fragment))

        candidate, record = detached_artifact_cleanup_candidate(damaged)

        self.assertTrue(record["accepted"], record["rejection_reasons"])
        self.assertEqual(record["discarded_component_count"], 1)
        self.assertEqual(metrics(candidate)["components"], 1)
        self.assertTrue(candidate.is_watertight)
        self.assertLess(len(candidate.faces), len(damaged.faces))

    def test_artifact_cleanup_rejects_meaningful_second_part(self) -> None:
        main = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        second = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        second.apply_translation((2.0, 0.0, 0.0))
        source = trimesh.util.concatenate((main, second))

        candidate, record = detached_artifact_cleanup_candidate(source)

        self.assertFalse(record["accepted"])
        self.assertEqual(len(candidate.faces), len(source.faces))
        self.assertTrue(record["rejection_reasons"])

    def test_local_edge_flip_repairs_degenerate_face_without_moving_vertices(self) -> None:
        source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        face = np.asarray(source.faces[0], dtype=int)
        a, b, c = np.asarray(source.vertices)[face]
        midpoint = (a + b) / 2.0
        inserted = midpoint + 1e-12 * (c - midpoint)
        vertices = np.vstack((np.asarray(source.vertices), inserted))
        point = len(vertices) - 1
        faces = np.delete(np.asarray(source.faces), 0, axis=0)
        faces = np.vstack(
            (
                faces,
                [face[0], face[1], point],
                [face[1], face[2], point],
                [face[2], face[0], point],
            )
        )
        damaged = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.assertTrue(damaged.is_watertight)
        self.assertGreater(metrics(damaged)["degenerate_faces"], 0)

        candidate, record = repair_degenerate_faces_by_edge_flip(damaged)

        self.assertTrue(record["accepted"], record["rejection_reasons"])
        self.assertEqual(metrics(candidate)["degenerate_faces"], 0)
        self.assertTrue(candidate.is_watertight)
        self.assertTrue(
            np.array_equal(np.asarray(candidate.vertices), np.asarray(damaged.vertices))
        )

    def test_voxel_reconstruction_closes_dominant_open_surface(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        damaged = source.copy()
        damaged.update_faces(np.arange(len(damaged.faces)) >= 12)
        self.assertFalse(damaged.is_watertight)

        candidate, record = voxel_reconstruction_candidate(
            damaged,
            resolution=96,
            closing_iterations=1,
            distance_samples=2000,
        )

        self.assertTrue(record["accepted"], record["rejection_reasons"])
        result = metrics(candidate)
        self.assertEqual(result["components"], 1)
        self.assertTrue(result["watertight"])
        self.assertEqual(result["boundary_edges"], 0)
        self.assertEqual(result["nonmanifold_edges"], 0)

    def test_voxel_reconstruction_rejects_two_meaningful_components(self) -> None:
        left = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        right = left.copy()
        right.apply_translation((2.0, 0.0, 0.0))
        source = trimesh.util.concatenate((left, right))

        candidate, record = voxel_reconstruction_candidate(
            source,
            resolution=96,
            distance_samples=1000,
        )

        self.assertFalse(record["accepted"])
        self.assertIn("低于 95%", record["rejection_reasons"][0])
        self.assertEqual(len(candidate.faces), len(source.faces))

    @unittest.skipIf(pymeshlab is None, "高级测试需要 Hunyuan 环境中的 pymeshlab")
    def test_advanced_candidate_repairs_duplicate_topology(self) -> None:
        from mesh_repair_tool import advanced_repair_candidate

        source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        damaged = trimesh.Trimesh(
            vertices=source.vertices,
            faces=np.vstack([source.faces, source.faces[0]]),
            process=False,
        )
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "damaged.glb"
            damaged.export(input_path)
            candidate, record = advanced_repair_candidate(
                damaged,
                input_path,
                max_hole_size=4,
            )

        self.assertTrue(record["accepted"])
        self.assertTrue(candidate.is_watertight)
        self.assertEqual(metrics(candidate)["nonmanifold_edges"], 0)

    @unittest.skipIf(pymeshlab is None, "重网格测试需要 Hunyuan 环境中的 pymeshlab")
    def test_surface_remesh_rejects_already_regular_mesh(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "regular.glb"
            source.export(input_path)
            candidate, record = surface_remesh_candidate(
                source,
                input_path,
                distance_samples=2000,
            )

        self.assertFalse(record["accepted"])
        self.assertIn("三角形长宽比 P99 未改善至少 20%", record["rejection_reasons"])
        self.assertEqual(len(candidate.faces), len(source.faces))


if __name__ == "__main__":
    unittest.main()
