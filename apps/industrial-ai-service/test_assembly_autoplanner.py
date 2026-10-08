"""Standard-library tests for function-first assembly planning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import assembly_autoplanner as planner


class AssemblyAutoplannerTest(unittest.TestCase):
    def test_functional_input_resolves_to_validated_internal_parameters(self):
        plan = planner.build_auto_plan(
            planner.ASSEMBLY_PURPOSES[0],
            planner.FIT_GOALS[0],
            planner.PRINT_PROFILES[0],
            "用于教学展示，要求可反复拆装。",
        )
        self.assertEqual(plan["planner_mode"], "validated-template-function-first")
        self.assertNotIn("shaft_diameter_mm", plan["user_inputs"])
        self.assertEqual(
            plan["platform_decision"]["parameters_mm"]
            ["gear_bore_diametral_clearance_mm"],
            0.30,
        )
        self.assertTrue(plan["platform_decision"]["individual_stl_required"])

    def test_empty_or_unsupported_input_is_rejected(self):
        with self.assertRaises(ValueError):
            planner.build_auto_plan(
                planner.ASSEMBLY_PURPOSES[0],
                planner.FIT_GOALS[0],
                planner.PRINT_PROFILES[0],
                "",
            )
        with self.assertRaises(ValueError):
            planner.build_auto_plan(
                "动力传动",
                planner.FIT_GOALS[0],
                planner.PRINT_PROFILES[0],
                "测试",
            )

    def test_summary_does_not_require_user_to_enter_dimensions(self):
        plan = planner.build_auto_plan(
            planner.ASSEMBLY_PURPOSES[0],
            planner.FIT_GOALS[0],
            planner.PRINT_PROFILES[0],
            "装配展示",
        )
        summary = planner.auto_plan_summary(plan, "RUN-20260901-120000-abcdef")
        self.assertIn("选手无需手动填写工程参数", summary)
        self.assertIn("三个独立 STL", summary)

    def test_create_archives_functional_request_after_cpu_generators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "workspace/runs"
            source_dir = runs_root / planner.VALIDATED_SOURCE_RUN
            source_dir.mkdir(parents=True)
            (source_dir / "manifest.json").write_text(
                json.dumps({"stages": {"fit_review": "PASS"}}),
                encoding="utf-8",
            )
            fake_python = root / "python"
            fake_python.write_text("placeholder", encoding="utf-8")

            def fake_generator(command: list[str], *, timeout_seconds: int = 180):
                if command[1] == str(planner.BUILD_SCRIPT):
                    run_id = command[command.index("--run-id") + 1]
                    run_dir = runs_root / run_id
                    run_dir.mkdir(parents=True)
                    (run_dir / "manifest.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "request": {},
                                "events": [],
                                "stages": {},
                            }
                        ),
                        encoding="utf-8",
                    )

            with patch.object(planner, "_run_checked", side_effect=fake_generator):
                run_id, plan, manifest = planner.create_auto_planned_component(
                    planner.ASSEMBLY_PURPOSES[0],
                    planner.FIT_GOALS[0],
                    planner.PRINT_PROFILES[0],
                    "用户只提供功能目标。",
                    project_root=root,
                    runs_root=runs_root,
                    geometry_python=fake_python,
                )

            self.assertTrue(run_id.startswith("RUN-"))
            self.assertEqual(manifest["auto_planning"], plan)
            self.assertEqual(
                manifest["events"][-1]["event"],
                "FUNCTION_FIRST_AUTO_PLAN_CREATED",
            )


if __name__ == "__main__":
    unittest.main()
