"""Tests for the versioned CPU mesh repair adapter and workflow handoff."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import mesh_repair_adapter


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeshRepairAdapterTest(unittest.TestCase):
    def test_route_only_recommends_guarded_topology_modes(self):
        for route, expected in (
            ("SAFE_REPAIR_CANDIDATE", "safe"),
            ("STANDARD_HOLE_REPAIR_CANDIDATE", "standard"),
            ("DETACHED_ARTIFACT_CLEANUP_CANDIDATE", "artifact_cleanup"),
            ("ADVANCED_TOPOLOGY_REPAIR_CANDIDATE", "advanced"),
        ):
            self.assertEqual(
                mesh_repair_adapter.recommended_mode(
                    {"automatic_repairability_before": {"route": route}}
                ),
                expected,
            )
        self.assertEqual(
            mesh_repair_adapter.recommended_mode(
                {
                    "automatic_repairability_before": {
                        "route": "RECONSTRUCTION_RECOMMENDED"
                    }
                }
            ),
            "reconstruct",
        )
        self.assertIsNone(
            mesh_repair_adapter.recommended_mode(
                {
                    "automatic_repairability_before": {
                        "route": "SURFACE_QUALITY_VISUAL_REVIEW_REQUIRED"
                    }
                }
            )
        )

    def test_adapter_rejects_mesh_outside_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "workspace/runs/RUN-20260928-000000-abcdef"
            other_mesh = root / "workspace/runs/RUN-20260928-000001-fedcba/meshes/a.glb"
            other_mesh.parent.mkdir(parents=True)
            other_mesh.write_bytes(b"mesh")
            run_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "当前任务 meshes"):
                mesh_repair_adapter.run_mesh_repair(
                    project_root=root,
                    run_dir=run_dir,
                    source_mesh=other_mesh,
                    source_sha256=digest(other_mesh),
                )


class MeshRepairWorkflowTest(unittest.TestCase):
    def _create_run_with_selected_mesh(self, root: Path) -> tuple[str, Path, dict]:
        runs = root / "workspace/runs"
        with patch.object(app, "PROJECT_ROOT", root), patch.object(app, "RUNS_DIR", runs):
            run_id, _, _, _ = app.create_run(
                "齿轮", "数字化模型修复", "二十七齿", 27, 60, 1234
            )
            manifest_path, manifest = app.read_manifest(run_id)
            mesh = manifest_path.parent / "meshes/source_review.glb"
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_bytes(b"source-mesh")
            manifest["selected_mesh"] = {
                "path": str(mesh.relative_to(root)),
                "sha256": digest(mesh),
                "status": "REVIEW_READY_WAITING_CONFIRMATION",
            }
            app.write_manifest(manifest_path, manifest)
            return run_id, manifest_path, manifest

    def test_diagnosis_is_versioned_and_does_not_replace_selected_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, manifest_path, original_manifest = self._create_run_with_selected_mesh(root)
            run_dir = manifest_path.parent
            annotated = run_dir / "meshes/mesh_repair_v1_diagnostic.glb"
            report_path = run_dir / "reports/mesh_repair_v1.json"
            request = run_dir / "requests/mesh_repair_v1.json"
            log = run_dir / "logs/mesh_repair_v1.log"
            for path in (annotated, report_path, request, log):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            report = {
                "input_sha256": original_manifest["selected_mesh"]["sha256"],
                "before": {"components": 2},
                "automatic_repairability_before": {
                    "route": "ADVANCED_TOPOLOGY_REPAIR_CANDIDATE"
                },
                "status": "DIAGNOSIS_ONLY_NOT_PRINT_APPROVAL",
            }
            execution = {
                "version": 1,
                "request_path": str(request.relative_to(root)),
                "report_path": str(report_path.relative_to(root)),
                "annotated_path": str(annotated.relative_to(root)),
                "log_path": str(log.relative_to(root)),
                "report": report,
                "recommended_mode": "advanced",
            }
            with (
                patch.object(app, "PROJECT_ROOT", root),
                patch.object(app, "RUNS_DIR", root / "workspace/runs"),
                patch.object(app, "run_mesh_repair", return_value=execution),
            ):
                result = app.diagnose_current_mesh(run_id)
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(result[0].startswith("✅"))
            self.assertEqual(
                stored["selected_mesh"], original_manifest["selected_mesh"]
            )
            self.assertEqual(
                stored["selected_mesh_repair"]["status"], "DIAGNOSIS_READY"
            )

    def test_only_fully_gated_candidate_can_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, manifest_path, manifest = self._create_run_with_selected_mesh(root)
            run_dir = manifest_path.parent
            candidate = run_dir / "meshes/mesh_repair_v1_candidate.glb"
            report_path = run_dir / "reports/mesh_repair_v1.json"
            inspection_path = run_dir / "reports/mesh_repair_v1_candidate_inspection.json"
            preview = run_dir / "reports/mesh_repair_v1_candidate_preview.png"
            candidate.write_bytes(b"candidate-mesh")
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(b"preview")
            report_path.write_text(
                json.dumps(
                    {"status": "REPAIRED_CANDIDATE_REQUIRES_VISUAL_REVIEW"}
                ),
                encoding="utf-8",
            )
            inspection_path.write_text(
                json.dumps(
                    {
                        "input_sha256": digest(candidate),
                        "components": 1,
                        "watertight": True,
                        "winding_consistent": True,
                        "boundary_edges": 0,
                        "nonmanifold_edges": 0,
                        "degenerate_faces": 0,
                    }
                ),
                encoding="utf-8",
            )
            manifest["selected_mesh_repair"] = {
                "version": 1,
                "mode": "advanced",
                "status": "CANDIDATE_WAITING_CONFIRMATION",
                "source_path": manifest["selected_mesh"]["path"],
                "source_sha256": manifest["selected_mesh"]["sha256"],
                "candidate_path": str(candidate.relative_to(root)),
                "candidate_sha256": digest(candidate),
                "report_path": str(report_path.relative_to(root)),
                "inspection_path": str(inspection_path.relative_to(root)),
                "preview_path": str(preview.relative_to(root)),
            }
            app.write_manifest(manifest_path, manifest)
            with patch.object(app, "PROJECT_ROOT", root), patch.object(
                app, "RUNS_DIR", root / "workspace/runs"
            ):
                status, stored = app.decide_mesh_repair_candidate(run_id, "ACCEPT")
            self.assertTrue(status.startswith("✅"))
            self.assertEqual(stored["selected_mesh"]["path"], str(candidate.relative_to(root)))
            self.assertEqual(stored["stages"]["mesh"], "WAITING_CONFIRMATION")
            self.assertEqual(len(stored["mesh_selection_history"]), 1)

    def test_partial_candidate_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, manifest_path, manifest = self._create_run_with_selected_mesh(root)
            run_dir = manifest_path.parent
            candidate = run_dir / "meshes/candidate.glb"
            report = run_dir / "reports/report.json"
            inspection = run_dir / "reports/inspection.json"
            preview = run_dir / "reports/preview.png"
            candidate.write_bytes(b"candidate")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"status": "PARTIAL_REPAIR_NOT_PRINT_APPROVAL"}))
            inspection.write_text(
                json.dumps(
                    {
                        "input_sha256": digest(candidate),
                        "components": 2,
                        "watertight": False,
                        "winding_consistent": True,
                        "boundary_edges": 8,
                        "nonmanifold_edges": 0,
                        "degenerate_faces": 0,
                    }
                )
            )
            preview.write_bytes(b"preview")
            original_path = manifest["selected_mesh"]["path"]
            manifest["selected_mesh_repair"] = {
                "version": 1,
                "mode": "advanced",
                "status": "CANDIDATE_WAITING_CONFIRMATION",
                "source_path": original_path,
                "source_sha256": manifest["selected_mesh"]["sha256"],
                "candidate_path": str(candidate.relative_to(root)),
                "candidate_sha256": digest(candidate),
                "report_path": str(report.relative_to(root)),
                "inspection_path": str(inspection.relative_to(root)),
                "preview_path": str(preview.relative_to(root)),
            }
            app.write_manifest(manifest_path, manifest)
            with patch.object(app, "PROJECT_ROOT", root), patch.object(
                app, "RUNS_DIR", root / "workspace/runs"
            ):
                status, stored = app.decide_mesh_repair_candidate(run_id, "ACCEPT")
            self.assertTrue(status.startswith("⛔"))
            self.assertEqual(stored["selected_mesh"]["path"], original_path)


if __name__ == "__main__":
    unittest.main()
