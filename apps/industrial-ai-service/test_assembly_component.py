"""Standard-library tests for the verified assembly-component bundle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import assembly_component


class AssemblyComponentTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[str, Path, Path]:
        run_id = "RUN-20260901-120000-abc123"
        runs_root = root / "workspace/runs"
        run_dir = runs_root / run_id
        for directory in ("reports", "assembly", "stl", "reviews"):
            (run_dir / directory).mkdir(parents=True, exist_ok=True)
        design = {
            "run_id": run_id,
            "geometry_method": "CPU deterministic parametric mesh",
            "bom": [
                {"index": 1, "part": "16齿D形孔齿轮", "quantity": 1},
                {"index": 2, "part": "D形短轴", "quantity": 1},
                {"index": 3, "part": "限位帽", "quantity": 1},
            ],
            "parameters_mm": {
                "gear_outer_diameter_mm": 60.0,
                "shaft_base_height_mm": 4.0,
                "gear_thickness_mm": 10.0,
                "cap_height_mm": 6.0,
                "gear_bore_diametral_clearance_mm": 0.30,
                "cap_socket_diametral_clearance_mm": 0.10,
            },
            "combined_print": {
                "components": 2,
                "extents_mm": [81.0, 60.0, 10.0],
                "shaft_reprint_required": False,
            },
        }
        design_path = run_dir / "reports/design.json"
        design_path.write_text(json.dumps(design), encoding="utf-8")
        role_paths = {
            "design_report": design_path,
            "assembly_preview": run_dir / "reports/preview.png",
            "gear_print_stl": run_dir / "stl/gear.stl",
            "shaft_print_stl": run_dir / "stl/shaft.stl",
            "cap_recommended_print_orientation_stl": run_dir / "stl/cap.stl",
            "gear_cap_combined_print_stl": run_dir / "stl/combined.stl",
            "assembled_review_glb": run_dir / "assembly/assembled.glb",
            "exploded_review_glb": run_dir / "assembly/exploded.glb",
            "assembly_process_sheet": run_dir / "reports/process.png",
        }
        for role, path in role_paths.items():
            if role != "design_report":
                path.write_bytes(role.encode("utf-8"))
        artifacts = []
        for role, path in role_paths.items():
            artifacts.append(
                {
                    "role": role,
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": assembly_component.file_sha256(path),
                }
            )
        manifest = {
            "run_id": run_id,
            "mode": "cpu-parametric-gear-assembly-v3",
            "status": "PARAMETRIC_ASSEMBLY_REVIEW_READY",
            "stages": {"print": "WAITING_BAMBU_SLICER_REVIEW"},
            "artifacts": artifacts,
            "events": [],
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return run_id, runs_root, run_dir

    def test_load_bundle_and_save_versioned_fit_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, runs_root, run_dir = self.make_bundle(root)
            bundle = assembly_component.load_component_bundle(
                run_id, project_root=root, runs_root=runs_root
            )
            self.assertEqual(len(bundle["bom"]), 3)
            self.assertEqual(len(assembly_component.component_bom_rows(bundle)), 3)
            self.assertIn("0.30 mm", assembly_component.component_summary(bundle))

            record, path, manifest = assembly_component.save_fit_review(
                run_id,
                1.5,
                "顺畅",
                "略紧可接受",
                False,
                "可正常手动拆卸",
                "",
                "人工试装通过",
                project_root=root,
                runs_root=runs_root,
            )
            self.assertEqual(record["status"], "PASS")
            self.assertEqual(path.name, "assembly_fit_review_v1.json")
            self.assertEqual(manifest["stages"]["fit_review"], "PASS")
            self.assertTrue((run_dir / "reviews/assembly_fit_review_v1.json").is_file())

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id, runs_root, run_dir = self.make_bundle(root)
            (run_dir / "stl/gear.stl").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                assembly_component.load_component_bundle(
                    run_id, project_root=root, runs_root=runs_root
                )


if __name__ == "__main__":
    unittest.main()
