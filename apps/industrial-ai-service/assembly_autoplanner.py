"""Function-first auto-planning for the validated three-part teaching assembly.

The user selects functional intent. Exact dimensions remain an archived,
traceable platform decision and are never inferred from an unscaled image.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
GEOMETRY_PYTHON = PROJECT_ROOT / "apps/Hunyuan3D-2.1/.venv/bin/python"
BUILD_SCRIPT = Path(__file__).with_name("build_parametric_gear_assembly.py")
PROCESS_SHEET_SCRIPT = Path(__file__).with_name("build_gear_assembly_process_sheet.py")

VALIDATED_SOURCE_RUN = "RUN-20260901-115822-a30c10"
ASSEMBLY_PURPOSES = ["教学装配与流程验证"]
FIT_GOALS = ["可手动拆卸、齿轮不能明显空转、限位帽倒置不脱落"]
PRINT_PROFILES = ["Bambu Lab A1 / PLA（已实测）"]

PLATFORM_PARAMETERS_MM = {
    "gear_outer_diameter_mm": 60.0,
    "shaft_diameter_mm": 12.0,
    "retaining_peg_diameter_mm": 8.0,
    "gear_bore_diametral_clearance_mm": 0.30,
    "cap_socket_diametral_clearance_mm": 0.10,
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _new_run_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"RUN-{stamp}-{uuid.uuid4().hex[:6]}"


def build_auto_plan(
    purpose: str,
    fit_goal: str,
    print_profile: str,
    requirement: str,
) -> dict[str, Any]:
    """Resolve user-facing intent to one validated, traceable template."""
    if purpose not in ASSEMBLY_PURPOSES:
        raise ValueError("当前版本只支持教学装配与流程验证。")
    if fit_goal not in FIT_GOALS:
        raise ValueError("当前版本只支持已完成实物验证的可拆稳定配合。")
    if print_profile not in PRINT_PROFILES:
        raise ValueError("当前版本只支持已实测的 Bambu Lab A1 / PLA 配置。")
    requirement_text = str(requirement or "").strip()
    if not requirement_text:
        raise ValueError("请用一句话描述装配用途或希望达到的效果。")
    if len(requirement_text) > 500:
        raise ValueError("装配需求不能超过 500 个字符。")
    return {
        "schema_version": "1.0",
        "planner_mode": "validated-template-function-first",
        "template_id": "gear-d-shaft-removable-cap-v1",
        "template_name": "16齿教学齿轮 + D形短轴 + 可拆限位帽",
        "user_inputs": {
            "purpose": purpose,
            "fit_goal": fit_goal,
            "print_profile": print_profile,
            "requirement": requirement_text,
        },
        "platform_decision": {
            "source_run": VALIDATED_SOURCE_RUN,
            "parameter_source": "同尺寸 Bambu Lab A1 / PLA 实物试装通过版本",
            "parameters_mm": dict(PLATFORM_PARAMETERS_MM),
            "parts": ["16齿D形孔齿轮", "带底座D形短轴", "圆孔可拆限位帽"],
            "individual_stl_required": True,
            "combined_print_layout_optional": True,
        },
        "limits": [
            "当前仅支持一个已验证模板，不代表任意装配体均可自动规划。",
            "同板打印只是排版选择，各零件仍输出独立 STL。",
            "每次新打印仍须通过 Bambu Studio 人工切片检查和实物验收。",
            "不生成或执行机械臂、灵巧手控制命令。",
        ],
    }


def auto_plan_summary(plan: dict[str, Any] | None, run_id: str = "") -> str:
    if not plan:
        return (
            "### 平台自动规划结果\n\n"
            "填写功能需求后点击“自动规划并创建组件”。选手无需输入齿轮外径、"
            "轴径、孔径或配合间隙；这些参数由平台从已验证模板中选择并归档。"
        )
    user = plan["user_inputs"]
    decision = plan["platform_decision"]
    run_line = f"- 新任务：`{run_id}`\n" if run_id else ""
    return (
        "### 平台自动规划结果\n\n"
        f"{run_line}"
        f"- 识别用途：{user['purpose']}\n"
        f"- 配合目标：{user['fit_goal']}\n"
        f"- 自动选择：{plan['template_name']}\n"
        f"- 参数依据：{decision['parameter_source']}\n"
        "- 输出方式：三个独立 STL，并提供可选同板排版 STL\n"
        "- 下一步：查看装配预览和工艺图，再进行 Bambu Studio 人工切片检查。\n\n"
        "平台已在后台记录具体尺寸；选手无需手动填写工程参数。"
    )


def _run_checked(command: list[str], *, timeout_seconds: int = 180) -> None:
    completed = subprocess.run(
        command,
        cwd=str(BUILD_SCRIPT.parent),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "未知错误").strip()
        raise RuntimeError(detail[-2000:])


def create_auto_planned_component(
    purpose: str,
    fit_goal: str,
    print_profile: str,
    requirement: str,
    *,
    project_root: Path = PROJECT_ROOT,
    runs_root: Path = RUNS_ROOT,
    geometry_python: Path = GEOMETRY_PYTHON,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Create a new versioned component without asking the user for dimensions."""
    plan = build_auto_plan(purpose, fit_goal, print_profile, requirement)
    source_manifest_path = runs_root / VALIDATED_SOURCE_RUN / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("平台缺少已验证的三件套参数来源任务。")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("stages", {}).get("fit_review") != "PASS":
        raise ValueError("三件套参数来源尚未归档为实物验收通过。")
    if not geometry_python.is_file():
        raise FileNotFoundError("CPU 参数化几何环境不可用。")

    run_id = _new_run_id()
    _run_checked(
        [
            str(geometry_python),
            str(BUILD_SCRIPT),
            "--run-id",
            run_id,
            "--scale",
            "1.0",
            "--version",
            "1",
            "--gear-bore-clearance-mm",
            "0.30",
            "--cap-socket-clearance-mm",
            "0.10",
            "--fit-validation-source-run",
            VALIDATED_SOURCE_RUN,
            "--shaft-reprint-required",
        ]
    )
    _run_checked(
        [
            str(geometry_python),
            str(PROCESS_SHEET_SCRIPT),
            "--run-id",
            run_id,
            "--version",
            "1",
        ]
    )

    manifest_path = runs_root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    now = _now_iso()
    manifest["auto_planning"] = plan
    manifest.setdefault("request", {})["user_functional_requirement"] = plan[
        "user_inputs"
    ]
    manifest["updated_at"] = now
    manifest.setdefault("events", []).append(
        {
            "time": now,
            "event": "FUNCTION_FIRST_AUTO_PLAN_CREATED",
            "detail": "用户只提交功能目标；平台引用实测模板自动确定尺寸、配合与BOM。",
        }
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return run_id, plan, manifest
