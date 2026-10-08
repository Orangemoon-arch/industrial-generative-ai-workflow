"""Standard-library tests for manifest and prompt behavior."""

import json
import importlib.util
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import app
import assembly_planning
import device_center
import gpu_executor
import mesh_optimization_adapter
import multimodal_capture
import part_profiles
import qwen_adapter


QWEN_RULES_PATH = Path(__file__).resolve().parents[1] / "qwen3-vl/qwen_review_rules.py"
QWEN_RULES_SPEC = importlib.util.spec_from_file_location(
    "industrial_ai_qwen_review_rules", QWEN_RULES_PATH
)
qwen_review_rules = importlib.util.module_from_spec(QWEN_RULES_SPEC)
assert QWEN_RULES_SPEC.loader is not None
QWEN_RULES_SPEC.loader.exec_module(qwen_review_rules)


class WorkflowSkeletonTest(unittest.TestCase):
    def test_prompt_contains_user_constraints(self):
        prompt = app.build_prompt(
            "齿轮", "静态教学展示", "十二齿，中心孔清晰", 12, "多余轮齿"
        )
        self.assertIn("齿轮", prompt)
        self.assertIn("十二齿，中心孔清晰", prompt)
        self.assertIn("严格且仅有 12 个", prompt)
        self.assertIn("禁止出现：多余轮齿", prompt)

    def test_part_profiles_generate_type_specific_count_constraints(self):
        fan = app.build_prompt("涵道风扇", "验证", "叶片连续", 7)
        gear = app.build_prompt("直齿圆柱齿轮", "验证", "中心孔贯通", 12)
        flange = app.build_prompt("圆形法兰", "验证", "中心孔和螺栓孔贯通", 6)
        custom = app.build_prompt("异形定位夹具", "验证", "两个定位面连续", 0)
        self.assertIn("7 个叶片", fan)
        self.assertIn("12 个轮齿", gear)
        self.assertIn("6 个均布螺栓孔", flange)
        self.assertIn("不强制旋转对称", custom)
        self.assertNotIn("数量硬约束", custom)

    def test_profile_defaults_and_count_ranges(self):
        requirement, forbidden, count, guidance = part_profiles.profile_defaults("法兰")
        self.assertIn("中心通孔", requirement)
        self.assertIn("螺栓", forbidden)
        self.assertEqual(count, 6)
        self.assertIn("均布螺栓孔", guidance)
        self.assertEqual(
            part_profiles.resolve_part_profile("异形定位夹具").key,
            "custom",
        )
        with self.assertRaisesRegex(ValueError, "轮齿数量"):
            app.build_prompt("齿轮", "验证", "结构完整", 4)

    def test_requirement_count_conflicts_are_blocked(self):
        with self.assertRaisesRegex(ValueError, "结构要求中写出的轮齿数量为 16"):
            app.build_prompt("齿轮", "验证", "单个完整十六齿齿轮", 12)
        with self.assertRaisesRegex(ValueError, "均布螺栓孔数量为 8"):
            app.build_prompt("法兰", "验证", "中心孔和8个均布孔贯通", 6)
        prompt = app.build_prompt("法兰", "验证", "中心孔和六个均布孔贯通", 6)
        self.assertIn("6 个均布螺栓孔", prompt)
        custom = app.build_prompt("异形定位夹具", "验证", "两个定位面连续", 0)
        self.assertIn("不强制旋转对称", custom)

    def test_assembly_purpose_adds_profile_specific_constraints(self):
        prompt = app.build_prompt(
            "齿轮",
            "装配场景演示",
            "十二齿，中心孔贯通",
            12,
        )
        self.assertIn("装配表达约束", prompt)
        self.assertIn("中心孔作为与轴配合", prompt)
        self.assertIn("尺寸、公差与受力必须由工程资料确认", prompt)

    def test_create_and_confirm_run(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, prompt, _, manifest = app.create_run(
                    "涵道风扇",
                    "三维打印验证",
                    "七片叶片",
                    7,
                    60,
                    20260834,
                )
                self.assertEqual(manifest["status"], "DRAFT")
                self.assertEqual(
                    manifest["request"]["part_profile"]["key"],
                    "ducted_fan",
                )
                self.assertTrue((runs_dir / run_id / "manifest.json").is_file())

                _, confirmed = app.confirm_prompt(run_id, prompt)
                self.assertTrue(confirmed["prompt"]["confirmed"])
                self.assertEqual(confirmed["stages"]["prompt"], "CONFIRMED")

                stored = json.loads(
                    (runs_dir / run_id / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(stored["run_id"], run_id)

    def test_manifest_summary_and_history_table_support_legacy_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, _ = app.create_run(
                    "齿轮", "验证", "十二齿", 12, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                manifest["request"].pop("part_profile")
                manifest["status"] = "ERNIE_COMPLETED"
                app.write_manifest(path, manifest)
                before = path.read_bytes()
                summary = app.manifest_summary(manifest)
                rows = app.list_runs_table()
                after = path.read_bytes()
                self.assertIn("直齿圆柱齿轮配置", summary)
                self.assertIn("候选图已生成", summary)
                self.assertEqual(rows[0][0], run_id)
                self.assertEqual(rows[0][2], "候选图已生成")
                self.assertEqual(before, after)

    def test_next_action_is_human_readable(self):
        manifest = {
            "stages": {
                "prompt": "CONFIRMED",
                "candidate": "CONFIRMED",
                "mask": "CONFIRMED",
                "mesh": "CONFIRMED",
            },
            "artifacts": [],
            "selected_stl": {"path": "workspace/runs/example/stl/result.stl"},
        }
        self.assertIn("Bambu Studio", app.next_action_text(manifest))

    def test_rejects_invalid_run_id(self):
        with self.assertRaises(ValueError):
            app.manifest_path("../../outside")

    def test_existing_run_can_be_loaded_without_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, created = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                before = (runs_dir / run_id / "manifest.json").read_bytes()
                normalized, message, loaded = app.load_existing_run(f" {run_id} \n")
                after = (runs_dir / run_id / "manifest.json").read_bytes()
                self.assertEqual(normalized, run_id)
                self.assertIn(run_id, message)
                self.assertEqual(loaded["run_id"], created["run_id"])
                self.assertEqual(before, after)
                self.assertEqual(app.manifest_path(f"\t{run_id} "), runs_dir / run_id / "manifest.json")

    def test_existing_run_context_restores_latest_candidate_and_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, prompt, _, _ = app.create_run(
                    "齿轮", "三维打印验证", "十二齿，中心孔贯通", 12, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                old_candidate = path.parent / "images/attempt_001/old.png"
                current_candidate = path.parent / "images/attempt_002/current.png"
                old_candidate.parent.mkdir(parents=True)
                current_candidate.parent.mkdir(parents=True)
                old_candidate.write_bytes(b"old")
                current_candidate.write_bytes(b"current")
                relative = lambda item: str(item.relative_to(project_root))
                manifest["artifacts"].extend(
                    [
                        {
                            "role": "ernie_candidate",
                            "attempt": 1,
                            "index": 1,
                            "seed": 1234,
                            "path": relative(old_candidate),
                        },
                        {
                            "role": "ernie_candidate",
                            "attempt": 2,
                            "index": 1,
                            "seed": 1235,
                            "path": relative(current_candidate),
                        },
                    ]
                )
                qwen_result = path.parent / "reviews/qwen_review_002.json"
                qwen_result.write_text(
                    json.dumps(
                        {
                            "review": {
                                "decision": "REJECT",
                                "observed_repeat_count": 16,
                                "subject_complete": True,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                manifest["candidate_review"] = {
                    "candidate_index": 1,
                    "candidate_path": relative(current_candidate),
                    "decision": "REJECT",
                    "result_path": relative(qwen_result),
                }
                app.write_manifest(path, manifest)
                before = path.read_bytes()
                restored = app.load_existing_run_context(f" {run_id}\n")
                after = path.read_bytes()
                self.assertEqual(restored[0], run_id)
                self.assertIn("已恢复表单", restored[1])
                self.assertEqual(restored[3], "齿轮")
                self.assertEqual(restored[6], "")
                self.assertEqual(restored[7], 12)
                self.assertEqual(restored[10], prompt)
                self.assertEqual(restored[11], [(str(current_candidate), "候选 1 · seed 1235")])
                self.assertEqual(restored[13]["observed_repeat_count"], 16)
                self.assertIn("直齿圆柱齿轮", restored[14])
                self.assertIn("尚无专用确定性制造规则化", restored[15])
                self.assertEqual(before, after)

    def test_qwen_summary_is_human_readable(self):
        summary = app.qwen_review_summary(
            {
                "decision": "REJECT",
                "observed_repeat_count": 16,
                "subject_complete": True,
                "background_clean": True,
                "view_suitable_for_3d": True,
                "forbidden_elements_found": [],
                "structural_issues": [],
                "reason": "实际观察到16齿。",
            }
        )
        self.assertIn("已拦截，不能进入三维", summary)
        self.assertIn("实际观察数量：16", summary)
        self.assertNotIn("{", summary)

    def test_mask_and_mesh_summaries_are_human_readable(self):
        mask = app.mask_report_summary(
            {
                "corner_alpha": [0, 0, 0, 0],
                "final_foreground_components": 1,
                "large_transparent_hole_count": 7,
                "foreground_fraction": 0.4734,
                "method": "test-mask",
            }
        )
        self.assertIn("内部透明区域：7 个", mask)
        self.assertIn("47.34%", mask)
        self.assertNotIn("{", mask)
        mesh = app.mesh_report_summary(
            {
                "vertices": 100,
                "faces": 200,
                "components": 1,
                "watertight": True,
                "winding_consistent": True,
                "boundary_edges": 0,
                "nonmanifold_edges": 0,
                "degenerate_faces": 0,
                "extents": [1.0, 2.0, 0.5],
            }
        )
        self.assertIn("通用几何门禁：**通过", mesh)
        self.assertIn("1.0000 × 2.0000 × 0.5000", mesh)
        self.assertNotIn("{", mesh)

    def test_history_row_resolves_run_id(self):
        rows = [["RUN-20260828-101957-fe2c9e", "圆形法兰", "已完成", "", ""]]
        self.assertEqual(
            app.history_run_id(rows, 0), "RUN-20260828-101957-fe2c9e"
        )
        with self.assertRaisesRegex(ValueError, "有效任务编号"):
            app.history_run_id([["../../outside"]], 0)

    def test_assembly_plan_requires_distinct_symbolic_zones(self):
        defaults = assembly_planning.assembly_defaults("法兰")
        with self.assertRaisesRegex(ValueError, "必须使用不同名称"):
            assembly_planning.build_assembly_plan(
                run_id="RUN-20260827-000000-abcdef",
                part_type="法兰",
                counterpart=defaults[0],
                action=defaults[1],
                pickup_zone="工作区",
                assembly_zone="工作区",
                finished_zone="成品区",
                reject_zone="异常区",
                grasp_feature=defaults[6],
                approach_direction=defaults[7],
                mating_feature=defaults[8],
                assembly_direction=defaults[9],
                success_criteria=defaults[10],
            )

    def test_versioned_assembly_plan_is_archived_and_stays_non_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "法兰", "装配场景演示", "六个均布孔", 6, 60, 1234
                )
                defaults = assembly_planning.assembly_defaults("法兰")
                saved = app.save_assembly_plan(run_id, *defaults[:11])
                self.assertIn("不能控制机械臂", saved[0])
                manifest = saved[1]
                plan = saved[3]
                plan_path = Path(saved[4])
                self.assertTrue(plan_path.is_file())
                self.assertEqual(
                    plan["execution_readiness"],
                    "BLOCKED_DEVICE_DOCUMENTATION_AND_SIMULATION",
                )
                self.assertIn("SIMULATION_REQUIRED", plan["state_machine"])
                self.assertIn("关节角命令", plan["prohibited_outputs"])
                self.assertEqual(
                    manifest["selected_assembly_plan"]["status"],
                    "PLAN_DRAFT_WAITING_CONFIRMATION",
                )
                loaded = app.load_current_assembly_plan(run_id)
                self.assertEqual(loaded[4], str(plan_path))
                confirmed = app.confirm_assembly_plan(run_id)
                self.assertIn("不是机械臂执行批准", confirmed[0])
                self.assertEqual(confirmed[1]["status"], "ASSEMBLY_PLAN_CONFIRMED")
                self.assertEqual(
                    confirmed[1]["stages"]["assembly"],
                    "PLAN_CONFIRMED_SIMULATION_REQUIRED",
                )

    def test_tampered_assembly_plan_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "齿轮", "装配场景演示", "十二齿", 12, 60, 1234
                )
                defaults = assembly_planning.assembly_defaults("齿轮")
                saved = app.save_assembly_plan(run_id, *defaults[:11])
                Path(saved[4]).write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    app.confirm_assembly_plan(run_id)

    def test_device_center_is_explicitly_simulated_and_never_controllable(self):
        snapshot = device_center.build_device_snapshot(
            "只读在线演示（模拟）"
        )
        self.assertEqual(snapshot["source"], "LOCAL_DEVICE_SIMULATOR")
        self.assertFalse(snapshot["safety"]["real_hardware_contacted"])
        self.assertFalse(snapshot["safety"]["control_commands_available"])
        self.assertTrue(all(not item["control_allowed"] for item in snapshot["devices"]))
        robot = next(item for item in snapshot["devices"] if item["key"] == "rm65_b")
        self.assertIn("RM65-B", robot["label"])
        self.assertEqual(robot["connection"], "READ_ONLY_ONLINE_SIMULATED")
        with self.assertRaisesRegex(ValueError, "场景无效"):
            device_center.build_device_snapshot("连接并控制真机")

    def test_device_snapshot_is_versioned_and_does_not_change_workflow_status(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "齿轮", "装配场景演示", "十二齿", 12, 60, 1234
                )
                first = app.save_device_snapshot(run_id, device_center.DEVICE_SCENARIOS[0])
                second = app.save_device_snapshot(run_id, device_center.DEVICE_SCENARIOS[1])
                self.assertIn("不代表真机已连接", second[0])
                self.assertEqual(first[1]["status"], "DRAFT")
                self.assertEqual(second[1]["status"], "DRAFT")
                self.assertEqual(second[1]["selected_device_snapshot"]["version"], 2)
                self.assertFalse(second[1]["selected_device_snapshot"]["control_allowed"])
                self.assertTrue(Path(second[4]).is_file())
                loaded = app.load_current_device_snapshot(run_id)
                self.assertEqual(loaded[3]["version"], 2)
                self.assertEqual(loaded[3]["source"], "LOCAL_DEVICE_SIMULATOR")

    def test_manual_multimodal_capture_is_archived_without_device_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            upload_dir = project_root / "review_uploads"
            upload_dir.mkdir(parents=True)
            upload = upload_dir / "frame.png"
            upload.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test-frame")
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "齿轮", "装配场景演示", "十二齿", 12, 60, 1234
                )
                app.save_device_snapshot(run_id, device_center.DEVICE_SCENARIOS[0])
                saved = app.save_multimodal_capture(
                    run_id,
                    str(upload),
                    "before_pick",
                    "gear-001",
                    "取料区",
                    "PART_PRESENT",
                    "人工上传测试帧",
                )
                self.assertIn("已保存采集记录", saved[0])
                self.assertEqual(saved[1]["task3"]["capture_count"], 1)
                self.assertFalse(saved[1]["task3"]["live_camera_connected"])
                self.assertEqual(len(saved[3]), 1)
                self.assertEqual(len(saved[4]), 2)
                record_path = next(Path(item) for item in saved[4] if item.endswith("record.json"))
                record = json.loads(record_path.read_text(encoding="utf-8"))
                self.assertEqual(record["source"], "MANUAL_WEB_UPLOAD")
                self.assertEqual(record["calibration"]["status"], "UNAVAILABLE")
                self.assertEqual(record["model_review"]["qwen_status"], "NOT_RUN")
                self.assertFalse(record["safety"]["device_command_generated"])
                self.assertFalse(record["safety"]["robot_control_allowed"])
                loaded = app.load_current_multimodal_captures(run_id)
                self.assertIn("1 条", loaded[0])
                self.assertEqual(len(loaded[3]), 1)

    def test_manual_capture_rejects_invalid_checkpoint_and_media(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            run_dir = project_root / "workspace/runs/RUN-20260831-000000-abcdef"
            run_dir.mkdir(parents=True)
            bad = project_root / "bad.txt"
            bad.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "采集节点无效"):
                multimodal_capture.archive_manual_capture(
                    project_root=project_root,
                    run_dir=run_dir,
                    run_id=run_dir.name,
                    upload=str(bad),
                    checkpoint="move_robot",
                    part_id="part-1",
                    zone="取料区",
                    label="UNLABELED",
                )
            with self.assertRaisesRegex(ValueError, "只接受内容"):
                multimodal_capture.archive_manual_capture(
                    project_root=project_root,
                    run_dir=run_dir,
                    run_id=run_dir.name,
                    upload=str(bad),
                    checkpoint="before_pick",
                    part_id="part-1",
                    zone="取料区",
                    label="UNLABELED",
                )

    def test_ernie_success_updates_manifest(self):
        fake_execution = {
            "attempt": 1,
            "request_path": "workspace/runs/example/requests/ernie_attempt_001.json",
            "result_path": "workspace/runs/example/reports/ernie_attempt_001_result.json",
            "log_path": "workspace/runs/example/logs/ernie_attempt_001.log",
            "gpu_preflight": {
                "name": "Test GPU",
                "total_mib": 12288,
                "used_mib": 100,
                "free_mib": 12188,
                "active_compute_process_count": 0,
            },
            "result": {
                "load_seconds": 1.0,
                "total_seconds": 2.0,
                "peak_allocated_gib": 2.4,
                "peak_reserved_gib": 2.5,
                "outputs": [
                    {
                        "index": 1,
                        "seed": 1234,
                        "path": "workspace/runs/example/images/candidate.png",
                        "bytes": 10,
                        "sha256": "abc",
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, prompt, _, _ = app.create_run(
                    "齿轮", "三维打印验证", "十二齿", 12, 60, 1234
                )
                app.confirm_prompt(run_id, prompt)
                with patch.object(app, "run_ernie", return_value=fake_execution):
                    _, manifest, gallery = app.generate_ernie_candidates(run_id, 1)
                self.assertEqual(manifest["status"], "ERNIE_COMPLETED")
                self.assertEqual(len(gallery), 1)
                self.assertEqual(
                    manifest["metrics"]["ernie_attempts"][0]["peak_allocated_gib"],
                    2.4,
                )
                self.assertEqual(
                    manifest["metrics"]["ernie_attempts"][0]["width"], 1024
                )

    def test_ernie_screening_resolution_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, prompt, _, _ = app.create_run(
                    "涵道风扇", "候选筛选", "七片叶片", 7, 60, 1234
                )
                app.confirm_prompt(run_id, prompt)
                with patch.object(app, "run_ernie") as mocked:
                    mocked.return_value = {
                        "attempt": 1,
                        "request_path": "requests/ernie.json",
                        "result_path": "reports/ernie.json",
                        "log_path": "logs/ernie.log",
                        "gpu_preflight": {"name": "GPU", "free_mib": 12000},
                        "result": {
                            "load_seconds": 1,
                            "total_seconds": 2,
                            "peak_allocated_gib": 2.4,
                            "peak_reserved_gib": 2.5,
                            "outputs": [
                                {
                                    "index": 1,
                                    "seed": 1234,
                                    "path": "workspace/runs/example/images/candidate.png",
                                    "bytes": 10,
                                    "sha256": "abc",
                                }
                            ],
                        },
                    }
                    _, manifest, _ = app.generate_ernie_candidates(
                        run_id, 1, 768
                    )
                self.assertEqual(mocked.call_args.kwargs["width"], 768)
                self.assertEqual(mocked.call_args.kwargs["height"], 768)
                self.assertEqual(
                    manifest["metrics"]["ernie_attempts"][0]["width"], 768
                )

    def test_qwen_reject_blocks_human_confirmation(self):
        fake_ernie = {
            "attempt": 1,
            "request_path": "workspace/runs/example/requests/ernie.json",
            "result_path": "workspace/runs/example/reports/ernie.json",
            "log_path": "workspace/runs/example/logs/ernie.log",
            "gpu_preflight": {"name": "GPU", "free_mib": 12000},
            "result": {
                "load_seconds": 1,
                "total_seconds": 2,
                "peak_allocated_gib": 2.4,
                "peak_reserved_gib": 2.5,
                "outputs": [
                    {
                        "index": 1,
                        "seed": 1234,
                        "path": "workspace/runs/example/images/candidate.png",
                        "bytes": 10,
                        "sha256": "abc",
                    }
                ],
            },
        }
        fake_qwen = {
            "review_number": 1,
            "request_path": "workspace/runs/example/requests/qwen.json",
            "result_path": "workspace/runs/example/reviews/qwen.json",
            "log_path": "workspace/runs/example/logs/qwen.log",
            "gpu_preflight": {"name": "GPU", "free_mib": 12000},
            "result": {
                "review": {
                    "decision": "REJECT",
                    "observed_repeat_count": 8,
                    "count_matches": False,
                },
                "load_seconds": 3,
                "inference_seconds": 1,
                "total_seconds": 4,
                "peak_allocated_gib": 6.3,
                "peak_reserved_gib": 6.4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, prompt, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                app.confirm_prompt(run_id, prompt)
                with patch.object(app, "run_ernie", return_value=fake_ernie):
                    app.generate_ernie_candidates(run_id, 1)
                with patch.object(
                    app, "run_qwen_review", return_value=fake_qwen
                ) as mocked_qwen:
                    _, manifest, review = app.review_candidate_with_qwen(run_id, 1)
                self.assertEqual(mocked_qwen.call_args.kwargs["expected_count"], 7)
                self.assertEqual(mocked_qwen.call_args.kwargs["counted_feature"], "叶片")
                self.assertIn(
                    "风道孔洞贯通",
                    mocked_qwen.call_args.kwargs["visual_checks"],
                )
                self.assertEqual(review["decision"], "REJECT")
                self.assertEqual(manifest["status"], "CANDIDATE_REJECTED")
                message, confirmed_manifest = app.confirm_candidate_manually(run_id, 1)
                self.assertIn("仅供参考", message)
                self.assertEqual(
                    confirmed_manifest["stages"]["candidate"], "CONFIRMED"
                )
                self.assertEqual(
                    confirmed_manifest["candidate_review"]["human_decision"],
                    "PASS",
                )

    def test_qwen_pass_requires_and_records_human_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, _ = app.create_run(
                    "齿轮", "三维打印验证", "十二齿", 12, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                manifest["candidate_review"] = {
                    "decision": "PASS",
                    "human_confirmed": False,
                }
                manifest["stages"]["candidate"] = "QWEN_PASS_WAITING_HUMAN"
                app.write_manifest(path, manifest)
                _, confirmed = app.confirm_stage(run_id, "candidate")
                self.assertEqual(confirmed["stages"]["candidate"], "CONFIRMED")
                self.assertTrue(confirmed["candidate_review"]["human_confirmed"])

    def test_manual_confirmation_does_not_require_qwen_pass(self):
        for qwen_decision in (None, "REVIEW", "REJECT"):
            with self.subTest(qwen_decision=qwen_decision):
                with tempfile.TemporaryDirectory() as directory:
                    runs_dir = Path(directory) / "runs"
                    with patch.object(app, "RUNS_DIR", runs_dir):
                        run_id, _, _, _ = app.create_run(
                            "齿轮", "三维打印验证", "十二齿", 12, 60, 1234
                        )
                        path, manifest = app.read_manifest(run_id)
                        candidate_path = (
                            "workspace/runs/example/images/candidate_01.png"
                        )
                        manifest["artifacts"].append(
                            {
                                "role": "ernie_candidate",
                                "attempt": 1,
                                "index": 1,
                                "seed": 1234,
                                "path": candidate_path,
                            }
                        )
                        if qwen_decision is not None:
                            app.store_candidate_review(
                                manifest,
                                {
                                    "candidate_path": candidate_path,
                                    "candidate_index": 1,
                                    "decision": qwen_decision,
                                    "human_confirmed": False,
                                },
                            )
                        app.write_manifest(path, manifest)

                        message, confirmed = app.confirm_candidate_manually(
                            run_id, 1
                        )

                        self.assertIn("仅供参考", message)
                        self.assertEqual(
                            confirmed["stages"]["candidate"], "CONFIRMED"
                        )
                        self.assertEqual(
                            confirmed["selected_candidate"]["path"], candidate_path
                        )
                        self.assertEqual(
                            confirmed["candidate_review"]["decision"],
                            qwen_decision or "NOT_RUN",
                        )
                        self.assertEqual(
                            confirmed["candidate_review"]["human_decision"],
                            "PASS",
                        )

    def test_custom_part_qwen_does_not_require_repeat_count(self):
        fake_qwen = {
            "review_number": 1,
            "request_path": "workspace/runs/example/requests/qwen.json",
            "result_path": "workspace/runs/example/reviews/qwen.json",
            "log_path": "workspace/runs/example/logs/qwen.log",
            "gpu_preflight": {"name": "GPU", "free_mib": 12000},
            "result": {
                "review": {"decision": "PASS", "observed_repeat_count": None},
                "load_seconds": 3,
                "inference_seconds": 1,
                "total_seconds": 4,
                "peak_allocated_gib": 6.3,
                "peak_reserved_gib": 6.4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, _ = app.create_run(
                    "异形定位夹具", "验证", "两个定位面连续", 0, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                manifest["artifacts"].append(
                    {
                        "role": "ernie_candidate",
                        "attempt": 1,
                        "index": 1,
                        "path": "workspace/runs/example/images/candidate.png",
                    }
                )
                app.write_manifest(path, manifest)
                with patch.object(
                    app, "run_qwen_review", return_value=fake_qwen
                ) as mocked_qwen:
                    _, updated, _ = app.review_candidate_with_qwen(run_id, 1)
                self.assertIsNone(mocked_qwen.call_args.kwargs["expected_count"])
                self.assertEqual(
                    mocked_qwen.call_args.kwargs["counted_feature"],
                    "主要重复结构",
                )
                self.assertEqual(updated["status"], "QWEN_PASS")

    def test_manual_reject_overrides_qwen_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                manifest["artifacts"].append(
                    {
                        "role": "ernie_candidate",
                        "attempt": 1,
                        "index": 1,
                        "seed": 1234,
                        "path": "workspace/runs/example/images/candidate.png",
                    }
                )
                manifest["candidate_review"] = {
                    "candidate_path": "workspace/runs/example/images/candidate.png",
                    "candidate_index": 1,
                    "decision": "PASS",
                    "human_confirmed": False,
                }
                app.write_manifest(path, manifest)
                _, rejected = app.reject_candidate_manually(
                    run_id, 1, "人工发现数量不符"
                )
                self.assertEqual(rejected["status"], "CANDIDATE_REJECTED")
                message, blocked = app.confirm_stage(run_id, "candidate")
                self.assertIn("人工拒绝", message)
                self.assertNotEqual(blocked["stages"]["candidate"], "CONFIRMED")

    def test_multiple_candidate_reviews_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                candidate_paths = [
                    "workspace/runs/example/images/candidate_01.png",
                    "workspace/runs/example/images/candidate_02.png",
                ]
                for index, candidate_path in enumerate(candidate_paths, start=1):
                    manifest["artifacts"].append(
                        {
                            "role": "ernie_candidate",
                            "attempt": 1,
                            "index": index,
                            "seed": 1233 + index,
                            "path": candidate_path,
                        }
                    )
                    app.store_candidate_review(
                        manifest,
                        {
                            "candidate_path": candidate_path,
                            "candidate_index": index,
                            "decision": "REJECT",
                            "result_path": f"reviews/qwen_{index}.json",
                            "human_confirmed": False,
                        },
                    )
                app.write_manifest(path, manifest)

                app.reject_candidate_manually(run_id, 1, "候选一人工拒绝")
                _, rejected = app.reject_candidate_manually(
                    run_id, 2, "候选二人工拒绝"
                )
                reviews = {
                    item["candidate_index"]: item
                    for item in rejected["candidate_reviews"]
                }
                self.assertEqual(set(reviews), {1, 2})
                self.assertEqual(reviews[1]["decision"], "REJECT")
                self.assertEqual(reviews[2]["decision"], "REJECT")
                self.assertEqual(reviews[1]["human_reason"], "候选一人工拒绝")
                self.assertEqual(reviews[2]["human_reason"], "候选二人工拒绝")

    def test_confirm_specific_candidate_by_index(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory) / "runs"
            with patch.object(app, "RUNS_DIR", runs_dir):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "候选筛选", "七片叶片", 7, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                candidate_paths = [
                    "workspace/runs/example/images/candidate_01.png",
                    "workspace/runs/example/images/candidate_02.png",
                ]
                for index, candidate_path in enumerate(candidate_paths, start=1):
                    manifest["artifacts"].append(
                        {
                            "role": "ernie_candidate",
                            "attempt": 1,
                            "index": index,
                            "seed": 1233 + index,
                            "path": candidate_path,
                        }
                    )
                app.store_candidate_review(
                    manifest,
                    {
                        "candidate_path": candidate_paths[0],
                        "candidate_index": 1,
                        "decision": "PASS",
                        "human_confirmed": False,
                    },
                )
                app.store_candidate_review(
                    manifest,
                    {
                        "candidate_path": candidate_paths[1],
                        "candidate_index": 2,
                        "decision": "REJECT",
                        "human_confirmed": False,
                    },
                )
                app.write_manifest(path, manifest)

                message, confirmed = app.confirm_candidate_manually(run_id, 1)
                self.assertIn("指定候选", message)
                self.assertEqual(confirmed["selected_candidate"]["candidate_index"], 1)
                self.assertEqual(confirmed["stages"]["candidate"], "CONFIRMED")
                self.assertEqual(
                    app.candidate_review_for(confirmed, candidate_paths[0])[
                        "human_decision"
                    ],
                    "PASS",
                )

    def test_mask_confirmation_requires_approved_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                manifest["stages"]["candidate"] = "CONFIRMED"
                app.write_manifest(path, manifest)

                message, blocked = app.confirm_stage(run_id, "mask")
                self.assertIn("没有通过检查", message)
                self.assertNotEqual(blocked["stages"]["mask"], "CONFIRMED")

                path, manifest = app.read_manifest(run_id)
                rgba = path.parent / "masks/approved.png"
                report = path.parent / "masks/report.json"
                rgba.write_bytes(b"approved-rgba")
                rgba_sha = app.file_sha256(rgba)
                report.write_text(
                    json.dumps(
                        {
                            "rgba_sha256": rgba_sha,
                            "corner_alpha": [0, 0, 0, 0],
                            "final_foreground_components": 1,
                        }
                    ),
                    encoding="utf-8",
                )
                relative_rgba = str(rgba.relative_to(project_root))
                manifest["selected_mask"] = {
                    "status": "REVIEW_READY_WAITING_CONFIRMATION",
                    "rgba_path": relative_rgba,
                    "report_path": str(report.relative_to(project_root)),
                }
                manifest["artifacts"].append(
                    {
                        "role": "approved_candidate_mask_rgba",
                        "path": relative_rgba,
                        "sha256": rgba_sha,
                    }
                )
                app.write_manifest(path, manifest)
                _, confirmed = app.confirm_stage(run_id, "mask")
                self.assertEqual(confirmed["stages"]["mask"], "CONFIRMED")
                self.assertEqual(
                    confirmed["selected_mask"]["status"], "CONFIRMED"
                )
                self.assertTrue(
                    any(
                        item.get("role") == "approved_rgba_mask"
                        and item.get("sha256") == rgba_sha
                        for item in confirmed["artifacts"]
                    )
                )

    def test_load_current_mask_uses_selected_task_files(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                run_dir = path.parent
                candidate = run_dir / "images/candidate.png"
                rgba = run_dir / "masks/approved.png"
                preview = run_dir / "masks/preview.png"
                report = run_dir / "masks/report.json"
                candidate.write_bytes(b"candidate")
                rgba.write_bytes(b"rgba")
                preview.write_bytes(b"preview")
                report.write_text('{"topology_pass": true}', encoding="utf-8")
                manifest["selected_candidate"] = {
                    "candidate_index": 1,
                    "path": str(candidate.relative_to(project_root)),
                }
                manifest["selected_mask"] = {
                    "status": "CONFIRMED",
                    "rgba_path": str(rgba.relative_to(project_root)),
                    "preview_path": str(preview.relative_to(project_root)),
                    "report_path": str(report.relative_to(project_root)),
                }
                app.write_manifest(path, manifest)

                loaded_candidate, loaded_rgba, loaded_preview, loaded_report, message = (
                    app.load_current_mask(run_id)
                )
                self.assertEqual(loaded_candidate, str(candidate))
                self.assertEqual(loaded_rgba, str(rgba))
                self.assertEqual(loaded_preview, str(preview))
                self.assertTrue(loaded_report["topology_pass"])
                self.assertIn("候选 1", message)

    def test_mask_generation_archives_reviewable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "法兰", "三维打印验证", "六个均布孔", 6, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                run_dir = path.parent
                candidate = run_dir / "images/ernie_attempt_001/candidate.png"
                candidate.parent.mkdir(parents=True)
                candidate.write_bytes(b"candidate")
                candidate_relative = str(candidate.relative_to(project_root))
                candidate_sha = app.file_sha256(candidate)
                manifest["stages"]["candidate"] = "CONFIRMED"
                manifest["selected_candidate"] = {
                    "candidate_index": 1,
                    "path": candidate_relative,
                }
                manifest["artifacts"].append(
                    {
                        "role": "ernie_candidate",
                        "path": candidate_relative,
                        "sha256": candidate_sha,
                    }
                )
                app.write_manifest(path, manifest)

                outputs = {
                    "rgba_path": run_dir / "masks/candidate_rgba_auto_v1.png",
                    "preview_path": run_dir / "masks/candidate_mask_auto_v1_preview.png",
                    "report_path": run_dir / "reports/candidate_mask_auto_v1.json",
                    "request_path": run_dir / "requests/candidate_mask_auto_v1.json",
                    "log_path": run_dir / "logs/candidate_mask_auto_v1.log",
                }
                for output in outputs.values():
                    output.write_bytes(b"test-output")
                rgba_sha = app.file_sha256(outputs["rgba_path"])
                execution = {
                    "version": 1,
                    **{
                        key: str(value.relative_to(project_root))
                        for key, value in outputs.items()
                    },
                    "rgba_sha256": rgba_sha,
                    "report": {
                        "rgba_sha256": rgba_sha,
                        "corner_alpha": [0, 0, 0, 0],
                        "final_foreground_components": 1,
                    },
                }
                with patch.object(
                    app, "run_mask_preparation", return_value=execution
                ):
                    message, updated, original, rgba, preview, report = (
                        app.generate_candidate_mask(run_id, 40, 20, 100, 0)
                    )
                self.assertIn("后再确认", message)
                self.assertEqual(updated["status"], "MASK_REVIEW_READY")
                self.assertEqual(updated["stages"]["mask"], "WAITING_CONFIRMATION")
                self.assertEqual(
                    updated["selected_mask"]["status"],
                    "REVIEW_READY_WAITING_CONFIRMATION",
                )
                self.assertEqual(original, str(candidate))
                self.assertEqual(rgba, str(outputs["rgba_path"]))
                self.assertEqual(preview, str(outputs["preview_path"]))
                self.assertEqual(report["rgba_sha256"], rgba_sha)

    def test_hunyuan_raw_mesh_review_archives_output(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "法兰", "三维打印验证", "六个均布孔", 6, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                run_dir = path.parent
                raw = run_dir / "meshes/hunyuan_attempt_001_raw.glb"
                raw.write_bytes(b"raw-mesh")
                raw_relative = str(raw.relative_to(project_root))
                raw_sha = app.file_sha256(raw)
                manifest["selected_mesh"] = {
                    "attempt": 1,
                    "status": "RAW_WAITING_REVIEW",
                    "path": raw_relative,
                    "sha256": raw_sha,
                }
                manifest["artifacts"].append(
                    {"role": "hunyuan_raw_glb", "path": raw_relative, "sha256": raw_sha}
                )
                app.write_manifest(path, manifest)

                names = {
                    "review_path": "meshes/hunyuan_attempt_001_review_v1.glb",
                    "raw_inspection_path": "reports/raw_inspection_v1.json",
                    "raw_preview_path": "reports/raw_fourview_v1.png",
                    "process_path": "reports/review_process_v1.json",
                    "inspection_path": "reports/review_inspection_v1.json",
                    "preview_path": "reports/review_fourview_v1.png",
                    "request_path": "requests/review_v1.json",
                    "log_path": "logs/review_v1.log",
                }
                execution = {"version": 1}
                for key, relative in names.items():
                    output = run_dir / relative
                    output.write_bytes(b"review-output")
                    execution[key] = str(output.relative_to(project_root))
                review = project_root / execution["review_path"]
                execution["review_sha256"] = app.file_sha256(review)
                execution["inspection"] = {
                    "components": 1,
                    "watertight": True,
                    "winding_consistent": True,
                    "boundary_edges": 0,
                    "nonmanifold_edges": 0,
                }

                with patch.object(
                    app, "run_mesh_review_preparation", return_value=execution
                ):
                    message, updated, model, preview, report, files = (
                        app.prepare_current_hunyuan_mesh_review(run_id)
                    )
                self.assertIn("CPU 网格门禁", message)
                self.assertEqual(updated["status"], "MESH_REVIEW_READY")
                self.assertEqual(updated["stages"]["mesh"], "WAITING_CONFIRMATION")
                self.assertEqual(
                    updated["selected_mesh"]["status"],
                    "REVIEW_READY_WAITING_CONFIRMATION",
                )
                self.assertEqual(model, str(review))
                self.assertEqual(preview, str(project_root / execution["preview_path"]))
                self.assertEqual(len(files), 8)
                self.assertTrue(report["watertight"])

    def test_hunyuan_requires_confirmed_mask_and_archives_result(self):
        fake_execution = {
            "attempt": 1,
            "request_path": "workspace/runs/example/requests/hunyuan.json",
            "output_path": "workspace/runs/example/meshes/raw.glb",
            "result_path": "workspace/runs/example/reports/hunyuan.json",
            "log_path": "workspace/runs/example/logs/hunyuan.log",
            "gpu_preflight": {"name": "GPU", "free_mib": 12000},
            "result": {
                "output_bytes": 100,
                "output_sha256": "a" * 64,
                "load_seconds": 2,
                "generation_seconds": 3,
                "total_seconds": 5,
                "peak_allocated_gib": 6.4,
                "peak_reserved_gib": 6.5,
                "mesh": {
                    "vertices": 10,
                    "faces": 20,
                    "components": 1,
                    "watertight": True,
                    "winding_consistent": True,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                message, blocked, output = app.generate_hunyuan_mesh(run_id)
                self.assertIn("必须先确认", message)
                self.assertIsNone(output)
                self.assertNotEqual(blocked["stages"]["mesh"], "WAITING_REVIEW")

                path, manifest = app.read_manifest(run_id)
                mask = path.parent / "masks/approved.png"
                mask.write_bytes(b"approved-mask")
                mask_relative = str(mask.relative_to(project_root))
                mask_sha = app.file_sha256(mask)
                manifest["stages"]["mask"] = "CONFIRMED"
                manifest["selected_mask"] = {
                    "status": "CONFIRMED",
                    "rgba_path": mask_relative,
                }
                manifest["artifacts"].append(
                    {
                        "role": "approved_rgba_mask",
                        "path": mask_relative,
                        "sha256": mask_sha,
                    }
                )
                app.write_manifest(path, manifest)
                with patch.object(app, "run_hunyuan", return_value=fake_execution):
                    _, completed, output = app.generate_hunyuan_mesh(run_id)
                self.assertEqual(completed["status"], "HUNYUAN_COMPLETED")
                self.assertEqual(completed["stages"]["mesh"], "WAITING_REVIEW")
                self.assertEqual(completed["selected_mesh"]["status"], "RAW_WAITING_REVIEW")
                self.assertTrue(output.endswith("workspace/runs/example/meshes/raw.glb"))
                message, blocked = app.confirm_stage(run_id, "mesh")
                self.assertIn("必须先完成网格检查", message)
                self.assertNotEqual(blocked["stages"]["mesh"], "CONFIRMED")

    def test_multiview_hunyuan_archives_inputs_and_records_2mv_backend(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            uploads_dir = project_root / "uploads"
            uploads_dir.mkdir(parents=True)
            uploads = {}
            for view in ("front", "left", "back"):
                upload = uploads_dir / f"{view}.png"
                image = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
                image.putpixel((8, 8), (10, 10, 10, 255))
                image.save(upload)
                uploads[view] = str(upload)

            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "齿轮", "多视图验证", "十二齿，中心孔贯通", 12, 60, 1234
                )
                run_dir = runs_dir / run_id
                output = run_dir / "meshes/hunyuan_attempt_001_raw.glb"
                output.write_bytes(b"glTF-test")
                fake_execution = {
                    "attempt": 1,
                    "output_path": str(output.relative_to(project_root)),
                    "request_path": str((run_dir / "requests/request.json").relative_to(project_root)),
                    "result_path": str((run_dir / "reports/result.json").relative_to(project_root)),
                    "log_path": str((run_dir / "logs/run.log").relative_to(project_root)),
                    "gpu_preflight": {"free_mib": 10000},
                    "result": {
                        "interface_name": "Hunyuan3D-2.1-compatible",
                        "backend_model": "Hunyuan3D-2mv",
                        "input_mode": "multi_view",
                        "inputs": {view: {} for view in ("front", "left", "back")},
                        "output_bytes": output.stat().st_size,
                        "output_sha256": app.file_sha256(output),
                        "load_seconds": 1.0,
                        "generation_seconds": 2.0,
                        "total_seconds": 3.0,
                        "peak_allocated_gib": 5.0,
                        "peak_reserved_gib": 5.5,
                        "mesh": {"components": 1, "watertight": True},
                    },
                }
                with patch.object(
                    app,
                    "run_hunyuan_multiview",
                    return_value=fake_execution,
                ):
                    message, manifest, model_path = app.generate_hunyuan_multiview_mesh(
                        run_id,
                        uploads["front"],
                        uploads["left"],
                        uploads["back"],
                        None,
                    )
                self.assertIn("Hunyuan3D-2mv", message)
                self.assertEqual(manifest["selected_mesh"]["backend_model"], "Hunyuan3D-2mv")
                self.assertEqual(manifest["stages"]["mesh"], "WAITING_REVIEW")
                self.assertEqual(
                    len(
                        [
                            item
                            for item in manifest["artifacts"]
                            if item["role"] == "hunyuan_multiview_input"
                        ]
                    ),
                    3,
                )
                self.assertEqual(model_path, str(output))

    def test_review_mesh_load_and_confirmation_require_verified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                path, manifest = app.read_manifest(run_id)
                run_dir = path.parent
                mesh = run_dir / "meshes/review.glb"
                inspection = run_dir / "reports/inspection.json"
                projection = run_dir / "reports/projection.png"
                cleanup = run_dir / "reports/cleanup.json"
                mesh.write_bytes(b"review-mesh")
                projection.write_bytes(b"projection")
                cleanup.write_text('{"removed_components": 4}', encoding="utf-8")
                mesh_sha = app.file_sha256(mesh)
                inspection.write_text(
                    json.dumps(
                        {
                            "input_sha256": mesh_sha,
                            "components": 1,
                            "watertight": True,
                            "winding_consistent": True,
                            "boundary_edges": 0,
                            "nonmanifold_edges": 0,
                        }
                    ),
                    encoding="utf-8",
                )
                relative = lambda item: str(item.relative_to(project_root))
                manifest["stages"]["mask"] = "CONFIRMED"
                manifest["stages"]["mesh"] = "WAITING_CONFIRMATION"
                manifest["selected_mesh"] = {
                    "attempt": 1,
                    "status": "REVIEW_READY_WAITING_CONFIRMATION",
                    "path": relative(mesh),
                    "inspection_path": relative(inspection),
                    "projection_path": relative(projection),
                    "cleanup_report_path": relative(cleanup),
                    "sha256": mesh_sha,
                }
                manifest["artifacts"].append(
                    {
                        "role": "hunyuan_review_glb",
                        "path": relative(mesh),
                        "sha256": mesh_sha,
                    }
                )
                app.write_manifest(path, manifest)

                loaded_mesh, loaded_projection, report, files, message = (
                    app.load_current_mesh_review(run_id)
                )
                self.assertEqual(loaded_mesh, str(mesh))
                self.assertEqual(loaded_projection, str(projection))
                self.assertTrue(report["watertight"])
                self.assertEqual(len(files), 4)
                self.assertIn("review GLB", message)

                _, confirmed = app.confirm_stage(run_id, "mesh")
                self.assertEqual(confirmed["stages"]["mesh"], "CONFIRMED")
                self.assertEqual(confirmed["selected_mesh"]["status"], "CONFIRMED")

                path, regularized = app.read_manifest(run_id)
                regularized["stages"]["mesh"] = "WAITING_CONFIRMATION"
                regularized["selected_mesh"]["status"] = (
                    "REVIEW_READY_WAITING_CONFIRMATION"
                )
                next(
                    item
                    for item in regularized["artifacts"]
                    if item.get("role") == "hunyuan_review_glb"
                )["role"] = "manufacturing_review_glb"
                app.write_manifest(path, regularized)
                _, reconfirmed = app.confirm_stage(run_id, "mesh")
                self.assertEqual(reconfirmed["stages"]["mesh"], "CONFIRMED")

    def test_regularization_parameters_reject_impossible_blade_depth(self):
        parameters = {
            "outer_diameter_mm": 60,
            "ring_inner_diameter_mm": 47,
            "ring_depth_mm": 6,
            "hub_diameter_mm": 20,
            "hub_depth_mm": 6,
            "blade_thickness_mm": 6,
            "blade_sweep_deg": 20,
            "blade_pitch_camber_mm": 2,
            "blade_radial_arch_mm": 1,
            "resolution_mm": 0.15,
        }
        with self.assertRaisesRegex(ValueError, "超出外环"):
            mesh_optimization_adapter.validate_regularization_parameters(parameters)

    def test_cpu_regularization_archives_new_version_and_supersedes_stl(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "七叶片环形转子教学模型", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                manifest_path, manifest = app.read_manifest(run_id)
                run_dir = manifest_path.parent
                source = run_dir / "meshes/source.glb"
                source.write_bytes(b"source-mesh")
                source_relative = str(source.relative_to(project_root))
                source_sha = app.file_sha256(source)
                manifest["stages"]["mask"] = "CONFIRMED"
                manifest["selected_mesh"] = {
                    "status": "REVIEW_READY_WAITING_CONFIRMATION",
                    "path": source_relative,
                    "sha256": source_sha,
                }
                manifest["selected_stl"] = {"status": "WAITING_BAMBU_SLICER"}
                manifest["artifacts"].append(
                    {
                        "role": "hunyuan_review_glb",
                        "path": source_relative,
                        "sha256": source_sha,
                    }
                )

                mesh = run_dir / "meshes/fan_7blade_regularized_review_v1.glb"
                build = run_dir / "reports/build.json"
                inspection = run_dir / "reports/inspection.json"
                preview = run_dir / "reports/preview.png"
                request = run_dir / "requests/request.json"
                log = run_dir / "logs/regularization.log"
                mesh.write_bytes(b"regularized-mesh")
                mesh_sha = app.file_sha256(mesh)
                inspection_value = {
                    "input_sha256": mesh_sha,
                    "components": 1,
                    "watertight": True,
                    "winding_consistent": True,
                    "boundary_edges": 0,
                    "nonmanifold_edges": 0,
                    "degenerate_faces": 0,
                    "extents": [60.0, 60.0, 12.0],
                }
                for output, contents in (
                    (build, "{}"),
                    (inspection, json.dumps(inspection_value)),
                    (request, "{}"),
                    (log, "ok"),
                ):
                    output.write_text(contents, encoding="utf-8")
                preview.write_bytes(b"preview")
                relative = lambda item: str(item.relative_to(project_root))
                execution = {
                    "version": 1,
                    "request_path": relative(request),
                    "mesh_path": relative(mesh),
                    "build_report_path": relative(build),
                    "inspection_path": relative(inspection),
                    "preview_path": relative(preview),
                    "log_path": relative(log),
                    "mesh_sha256": mesh_sha,
                    "mesh_bytes": mesh.stat().st_size,
                    "inspection": inspection_value,
                }
                app.write_manifest(manifest_path, manifest)
                with patch.object(app, "run_regularization", return_value=execution):
                    message, completed, loaded_mesh, _, report, files = (
                        app.generate_regularized_fan(
                            run_id, 60, 47, 12, 20, 8, 4.4, 20, 1.2, 0.45, 0.15
                        )
                    )
                self.assertIn("通过自动网格门禁", message)
                self.assertEqual(completed["status"], "REGULARIZED_MESH_REVIEW_READY")
                self.assertEqual(completed["stages"]["mesh"], "WAITING_CONFIRMATION")
                self.assertEqual(
                    completed["selected_stl"]["status"], "SUPERSEDED_BY_NEW_MESH"
                )
                self.assertEqual(loaded_mesh, str(mesh))
                self.assertTrue(report["watertight"])
                self.assertEqual(len(files), 6)

    def test_stl_export_requires_human_mesh_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "涵道风扇", "三维打印验证", "七片叶片", 7, 60, 1234
                )
                message, _, output = app.export_confirmed_mesh_stl(run_id, True)
                self.assertIn("必须先人工确认", message)
                self.assertIsNone(output)

    def test_generic_hunyuan_stl_accepts_versioned_export_target_size(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            runs_dir = project_root / "workspace/runs"
            with (
                patch.object(app, "PROJECT_ROOT", project_root),
                patch.object(app, "RUNS_DIR", runs_dir),
            ):
                run_id, _, _, _ = app.create_run(
                    "圆形法兰", "流程验证", "中心孔和六个均布孔贯通", 6, 60, 1234
                )
                manifest_path, manifest = app.read_manifest(run_id)
                run_dir = manifest_path.parent
                mesh = run_dir / "meshes/hunyuan_review_v1.glb"
                inspection = run_dir / "reports/hunyuan_review_v1.json"
                output = run_dir / "stl/hunyuan_review_v1_print_candidate_v1.stl"
                report = run_dir / "reports/hunyuan_review_v1_stl_export_v1.json"
                log = run_dir / "logs/hunyuan_review_v1_stl_export_v1.log"
                mesh.write_bytes(b"unitless-hunyuan-mesh")
                mesh_sha = app.file_sha256(mesh)
                inspection.write_text(
                    json.dumps({"extents": [1.9, 1.8, 0.4]}), encoding="utf-8"
                )
                output.write_bytes(b"scaled-stl")
                report.write_text("{}", encoding="utf-8")
                log.write_text("ok", encoding="utf-8")
                relative = lambda item: str(item.relative_to(project_root))
                manifest["stages"]["mesh"] = "CONFIRMED"
                manifest["selected_mesh"] = {
                    "version": 1,
                    "origin": "Hunyuan 原始网格保守 review v1",
                    "status": "CONFIRMED",
                    "path": relative(mesh),
                    "inspection_path": relative(inspection),
                    "sha256": mesh_sha,
                }
                manifest["artifacts"].append(
                    {
                        "role": "hunyuan_review_glb",
                        "path": relative(mesh),
                        "sha256": mesh_sha,
                    }
                )
                app.write_manifest(manifest_path, manifest)
                execution = {
                    "version": 1,
                    "output_path": relative(output),
                    "report_path": relative(report),
                    "log_path": relative(log),
                    "output_sha256": app.file_sha256(output),
                    "report": {"support_required": False},
                }
                with patch.object(app, "run_stl_export", return_value=execution) as run:
                    message, completed, downloaded = app.export_confirmed_mesh_stl(
                        run_id, True, 100
                    )
                self.assertIn("STL 已导出", message)
                self.assertEqual(downloaded, str(output))
                self.assertEqual(run.call_args.kwargs["target_longest_mm"], 100.0)
                self.assertEqual(
                    completed["selected_stl"]["target_longest_mm"], 100.0
                )
                self.assertEqual(
                    completed["selected_stl"]["task_original_target_longest_mm"],
                    60.0,
                )
                self.assertEqual(completed["request"]["target_size_mm"], 60.0)

    def test_stl_size_preview_uses_uniform_mesh_proportions(self):
        preview = app.stl_size_preview({"extents": [2.0, 1.5, 0.5]}, 100)
        self.assertIn("100.000 × 75.000 × 25.000 mm", preview)
        self.assertIn("不覆盖已有 STL", preview)

        invalid = app.stl_size_preview({"extents": [2.0, 1.5, 0.5]}, 10)
        self.assertIn("20–200 mm", invalid)

    def test_failed_stl_export_log_advances_retry_version(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "logs").mkdir()
            (run_dir / "logs/hunyuan_review_v1_stl_export_v1.log").write_text(
                "failed", encoding="utf-8"
            )
            self.assertEqual(
                mesh_optimization_adapter._next_stl_export_version(
                    run_dir, "hunyuan_review_v1"
                ),
                2,
            )

    def test_stl_failure_detail_prefers_runtime_error(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "failed.log"
            log.write_text(
                "Traceback (most recent call last):\nRuntimeError: 网格尺寸无效。\n",
                encoding="utf-8",
            )
            self.assertEqual(
                mesh_optimization_adapter._log_failure_detail(log),
                "RuntimeError: 网格尺寸无效。",
            )


class GpuExecutorTest(unittest.TestCase):
    @staticmethod
    def completed(stdout=""):
        return type("Completed", (), {"stdout": stdout})()

    def test_available_gpu_snapshot(self):
        results = [
            self.completed(),
            self.completed("NVIDIA Test GPU, 12288, 200, 12088\n"),
            self.completed(""),
        ]
        with patch.object(gpu_executor, "_run", side_effect=results):
            snapshot = gpu_executor.require_available_gpu(3500)
        self.assertEqual(snapshot.free_mib, 12088)
        self.assertEqual(snapshot.active_compute_process_count, 0)

    def test_active_compute_process_blocks_launch(self):
        results = [
            self.completed(),
            self.completed("NVIDIA Test GPU, 12288, 4000, 8288\n"),
            self.completed("12345\n"),
        ]
        with patch.object(gpu_executor, "_run", side_effect=results):
            with self.assertRaises(gpu_executor.GpuUnavailableError):
                gpu_executor.require_available_gpu(3500)


class QwenReviewRulesTest(unittest.TestCase):
    @staticmethod
    def observed(visible_elements):
        return {
            "observed_repeat_count": 6,
            "subject_complete": True,
            "background_clean": True,
            "view_suitable_for_3d": True,
            "structure_continuous": True,
            "visible_auxiliary_elements": visible_elements,
            "structural_issues": [],
            "observation_reason": "法兰主体完整。",
        }

    def test_flange_bolt_holes_are_not_treated_as_installed_bolts(self):
        review = qwen_review_rules.normalize_review(
            self.observed(["中心孔", "6个均布螺栓孔"]),
            6,
            "螺栓、螺母、管道",
        )
        self.assertEqual(review["decision"], "PASS")
        self.assertEqual(review["forbidden_elements_found"], [])

    def test_installed_fastener_is_still_rejected(self):
        review = qwen_review_rules.normalize_review(
            self.observed(["中心孔", "安装有6个螺栓"]),
            6,
            "螺栓、螺母、管道",
        )
        self.assertEqual(review["decision"], "REJECT")
        self.assertEqual(review["forbidden_elements_found"], ["螺栓"])

    def test_custom_part_without_count_can_pass_and_prompt_stays_blind(self):
        observed = self.observed([])
        observed["observed_repeat_count"] = None
        review = qwen_review_rules.normalize_review(observed, None, "文字、底座")
        instruction = qwen_review_rules.build_instruction(
            {
                "part_type": "异形定位夹具",
                "expected_repeat_count": None,
                "counted_feature": "主要重复结构",
                "visual_checks": ["主体完整", "连接连续"],
            }
        )
        self.assertEqual(review["decision"], "PASS")
        self.assertIn("主要重复结构", instruction)
        self.assertNotIn("6 个", instruction)
        self.assertNotIn("严格且仅有", instruction)


class QwenAdapterTest(unittest.TestCase):
    def test_optional_count_and_profile_metadata_are_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            qwen_dir = project_root / "apps/qwen3-vl"
            qwen_dir.mkdir(parents=True)
            (qwen_dir / ".venv/bin").mkdir(parents=True)
            (qwen_dir / ".venv/bin/python").write_bytes(b"python")
            (qwen_dir / "run_qwen_service_review.py").write_bytes(b"worker")
            (project_root / "models/qwen3-vl-8b-instruct-ms").mkdir(parents=True)
            run_dir = project_root / "workspace/runs/RUN-20260827-000000-abcdef"
            image = run_dir / "images/attempt/candidate.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")

            class Snapshot:
                @staticmethod
                def to_dict():
                    return {"name": "GPU", "free_mib": 12000}

            def fake_subprocess(command, **_kwargs):
                request_path = Path(command[-1])
                request = json.loads(request_path.read_text(encoding="utf-8"))
                result_path = project_root / request["result_path"]
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(
                        {
                            "image_path": request["image_path"],
                            "review": {"decision": "PASS"},
                        }
                    ),
                    encoding="utf-8",
                )
                return type("Completed", (), {"returncode": 0})()

            with (
                patch.object(qwen_adapter, "project_gpu_lock", return_value=nullcontext()),
                patch.object(qwen_adapter, "require_available_gpu", return_value=Snapshot()),
                patch.object(qwen_adapter.subprocess, "run", side_effect=fake_subprocess),
            ):
                execution = qwen_adapter.run_qwen_review(
                    project_root=project_root,
                    run_dir=run_dir,
                    image_path=image,
                    part_type="异形定位夹具",
                    expected_count=None,
                    counted_feature="主要重复结构",
                    visual_checks=["主体完整", "连接连续"],
                    required_structure="两个定位面连续",
                    forbidden_elements="文字、底座",
                )
            request = json.loads(
                (project_root / execution["request_path"]).read_text(encoding="utf-8")
            )
            self.assertIsNone(request["expected_repeat_count"])
            self.assertEqual(request["counted_feature"], "主要重复结构")
            self.assertEqual(request["visual_checks"], ["主体完整", "连接连续"])


if __name__ == "__main__":
    unittest.main()
