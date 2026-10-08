"""Tests for external GLB import, repair history, and gear metrics UI logic."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import gear_analysis_adapter
import mesh_import_adapter


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeshImportAdapterTest(unittest.TestCase):
    def test_rejects_fake_glb_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "workspace/runs/RUN-20260928-000000-abcdef"
            run_dir.mkdir(parents=True)
            upload = root / "fake.glb"
            upload.write_bytes(b"not-a-real-glb")
            with self.assertRaisesRegex(ValueError, "文件头"):
                mesh_import_adapter.archive_and_inspect_glb(
                    project_root=root,
                    run_dir=run_dir,
                    upload=upload,
                )

    def test_archives_valid_header_and_verifies_inspection_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "workspace/runs/RUN-20260928-000000-abcdef"
            for name in ("meshes", "reports", "requests", "logs"):
                (run_dir / name).mkdir(parents=True, exist_ok=True)
            upload = root / "input.glb"
            upload.write_bytes(b"glTF" + b"test-binary")
            python_path = root / "apps/Hunyuan3D-2.1/.venv/bin/python"
            inspect_script = root / "apps/Hunyuan3D-2.1/hy3dshape/inspect_service_mesh.py"
            python_path.parent.mkdir(parents=True)
            inspect_script.parent.mkdir(parents=True)
            python_path.write_bytes(b"")
            inspect_script.write_bytes(b"")

            def fake_run(command, **_kwargs):
                mesh = Path(command[command.index("--input") + 1])
                report = Path(command[command.index("--report-output") + 1])
                preview = Path(command[command.index("--preview-output") + 1])
                report.write_text(
                    json.dumps(
                        {
                            "input_sha256": digest(mesh),
                            "components": 2,
                            "watertight": False,
                        }
                    ),
                    encoding="utf-8",
                )
                preview.write_bytes(b"png")
                return type("Completed", (), {"returncode": 0})()

            with patch.object(mesh_import_adapter.subprocess, "run", side_effect=fake_run):
                result = mesh_import_adapter.archive_and_inspect_glb(
                    project_root=root,
                    run_dir=run_dir,
                    upload=upload,
                )
            archived = root / result["mesh_path"]
            self.assertEqual(archived.read_bytes(), upload.read_bytes())
            self.assertEqual(result["mesh_sha256"], digest(archived))
            self.assertEqual(result["inspection"]["components"], 2)


class MeshWorkbenchPhase2Test(unittest.TestCase):
    def _run(self, root: Path, part_type: str = "直齿圆柱齿轮") -> tuple[str, Path, dict]:
        runs = root / "workspace/runs"
        with patch.object(app, "PROJECT_ROOT", root), patch.object(app, "RUNS_DIR", runs):
            run_id, _, _, manifest = app.create_run(
                part_type, "数字化修复", "二十七齿，中心孔贯通", 27, 60, 1234
            )
            return run_id, runs / run_id / "manifest.json", manifest

    def test_imported_broken_mesh_becomes_repair_source_not_approved_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, manifest_path, _ = self._run(root)
            run_dir = manifest_path.parent
            files = {
                "mesh_path": run_dir / "meshes/external_mesh_import_v1.glb",
                "report_path": run_dir / "reports/external_mesh_import_v1_inspection.json",
                "preview_path": run_dir / "reports/external_mesh_import_v1_preview.png",
                "request_path": run_dir / "requests/external_mesh_import_v1.json",
                "log_path": run_dir / "logs/external_mesh_import_v1.log",
            }
            for path in files.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            execution = {
                "version": 1,
                **{key: str(path.relative_to(root)) for key, path in files.items()},
                "mesh_sha256": digest(files["mesh_path"]),
                "inspection": {
                    "input_sha256": digest(files["mesh_path"]),
                    "components": 3,
                    "watertight": False,
                    "winding_consistent": True,
                    "boundary_edges": 12,
                    "nonmanifold_edges": 0,
                    "degenerate_faces": 0,
                },
                "original_name": "broken.glb",
            }
            with (
                patch.object(app, "PROJECT_ROOT", root),
                patch.object(app, "RUNS_DIR", root / "workspace/runs"),
                patch.object(app, "archive_and_inspect_glb", return_value=execution),
            ):
                result = app.import_external_mesh(run_id, "ignored.glb")
            selected = result[1]["selected_mesh"]
            self.assertEqual(selected["status"], "IMPORTED_WAITING_REPAIR")
            self.assertEqual(result[1]["stages"]["mesh"], "WAITING_REVIEW")
            self.assertEqual(selected["path"], execution["mesh_path"])

    def test_repair_history_can_be_restored_without_changing_selected_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, manifest_path, manifest = self._run(root)
            run_dir = manifest_path.parent
            selected = run_dir / "meshes/current.glb"
            selected.write_bytes(b"current")
            manifest["selected_mesh"] = {"path": str(selected.relative_to(root)), "sha256": digest(selected)}
            app.write_manifest(manifest_path, manifest)
            report = run_dir / "reports/mesh_repair_v3.json"
            annotated = run_dir / "meshes/mesh_repair_v3_diagnostic.glb"
            report.write_text(
                json.dumps(
                    {
                        "repair_mode": "safe",
                        "status": "NO_SAFE_REPAIR_AVAILABLE",
                        "automatic_repairability_before": {"route": "SAFE_REPAIR_CANDIDATE"},
                    }
                ),
                encoding="utf-8",
            )
            annotated.write_bytes(b"diagnostic")
            with patch.object(app, "PROJECT_ROOT", root), patch.object(
                app, "RUNS_DIR", root / "workspace/runs"
            ):
                rows = app.mesh_repair_history_table(run_id)
                restored = app.load_mesh_repair_history(run_id, 3)
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(rows[0][0], 3)
            self.assertEqual(restored[0], str(annotated.resolve()))
            self.assertEqual(stored["selected_mesh"]["path"], str(selected.relative_to(root)))

    def test_gear_summary_scales_only_with_known_outside_diameter(self):
        report = {
            "input": {
                "estimated_tooth_count": 27,
                "confidence_ratio_to_second": 2.0,
                "tooth_count_signal_confident": True,
                "outside_diameter_mesh_units": 2.0,
                "root_diameter_mesh_units": 1.6,
                "bore_diameter_mesh_units": 0.2,
                "thickness_mesh_units": 0.4,
                "module_from_outside_diameter_mesh_units": 2.0 / 29.0,
            }
        }
        raw = app.gear_analysis_summary(report, 0)
        scaled = app.gear_analysis_summary(report, 43.5)
        self.assertIn("模型单位", raw)
        self.assertIn("43.500 mm", scaled)
        self.assertIn("4.350 mm", scaled)

    def test_non_gear_task_cannot_run_gear_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, _, _ = self._run(root, "自定义零件")
            with patch.object(app, "PROJECT_ROOT", root), patch.object(
                app, "RUNS_DIR", root / "workspace/runs"
            ):
                result = app.analyze_current_spur_gear(run_id)
            self.assertTrue(result[0].startswith("⛔"))


class GearAnalysisAdapterTest(unittest.TestCase):
    def test_rejects_source_outside_current_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "workspace/runs/RUN-20260928-000000-abcdef"
            run_dir.mkdir(parents=True)
            source = root / "outside.glb"
            source.write_bytes(b"glTF")
            with self.assertRaisesRegex(ValueError, "当前任务 meshes"):
                gear_analysis_adapter.run_gear_analysis(
                    project_root=root,
                    run_dir=run_dir,
                    source_mesh=source,
                    source_sha256=digest(source),
                )


if __name__ == "__main__":
    unittest.main()
