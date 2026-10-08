"""Verified loading and fit-review records for parametric assembly components."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
RUN_ID_PATTERN = re.compile(r"^RUN-[0-9]{8}-[0-9]{6}-[0-9a-z]{6}$")

INSERTION_RESULTS = ["无法插入", "过紧", "略紧可接受", "顺畅", "过松"]
REMOVAL_RESULTS = ["可正常手动拆卸", "需要较大力才能拆卸", "无法无损拆卸"]

REQUIRED_ROLES = {
    "design_report",
    "assembly_preview",
    "gear_print_stl",
    "shaft_print_stl",
    "cap_recommended_print_orientation_stl",
    "assembled_review_glb",
    "exploded_review_glb",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_archived_artifact(
    artifact: dict[str, Any], project_root: Path, run_dir: Path
) -> Path:
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("装配组件归档缺少文件路径。")
    path = (project_root / relative).resolve()
    if run_dir.resolve() not in path.parents or not path.is_file():
        raise ValueError("装配组件文件路径无效或越出任务目录。")
    if artifact.get("sha256") != file_sha256(path):
        raise ValueError(f"装配组件文件 SHA256 不匹配：{path.name}")
    return path


def load_component_bundle(
    run_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Any]:
    normalized = str(run_id).strip()
    if not RUN_ID_PATTERN.fullmatch(normalized):
        raise ValueError("装配组件任务 ID 格式无效。")
    run_dir = (runs_root / normalized).resolve()
    if run_dir.parent != runs_root.resolve():
        raise ValueError("装配组件任务路径越界。")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("装配组件任务不存在。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != normalized:
        raise ValueError("装配组件清单与任务 ID 不一致。")
    if not str(manifest.get("mode", "")).startswith("cpu-parametric-gear-assembly-"):
        raise ValueError("当前任务不是受支持的参数化三件式装配任务。")

    archived: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for artifact in manifest.get("artifacts", []):
        role = artifact.get("role")
        if isinstance(role, str):
            archived[role] = artifact
    missing = sorted(REQUIRED_ROLES - archived.keys())
    if missing:
        raise ValueError(f"装配组件缺少归档：{', '.join(missing)}")
    for role, artifact in archived.items():
        paths[role] = _resolve_archived_artifact(artifact, project_root, run_dir)

    design = json.loads(paths["design_report"].read_text(encoding="utf-8"))
    if design.get("run_id") != normalized:
        raise ValueError("装配设计报告关联了错误任务。")
    if design.get("geometry_method") != "CPU deterministic parametric mesh":
        raise ValueError("装配组件几何来源不受支持。")
    bom = design.get("bom")
    if not isinstance(bom, list) or len(bom) != 3:
        raise ValueError("装配组件 BOM 必须包含三个零件。")
    parameters = design.get("parameters_mm", {})
    if not isinstance(parameters, dict):
        raise ValueError("装配设计报告缺少参数。")

    return {
        "run_id": normalized,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "design": design,
        "bom": bom,
        "paths": paths,
    }


def component_bom_rows(bundle: dict[str, Any]) -> list[list[Any]]:
    paths = bundle["paths"]
    role_by_index = {
        1: "gear_print_stl",
        2: "shaft_print_stl",
        3: "cap_recommended_print_orientation_stl",
    }
    return [
        [
            int(item.get("index", 0)),
            str(item.get("part", "")),
            int(item.get("quantity", 1)),
            paths[role_by_index[int(item.get("index", 0))]].name,
        ]
        for item in bundle["bom"]
    ]


def component_summary(bundle: dict[str, Any]) -> str:
    design = bundle["design"]
    params = design["parameters_mm"]
    combined = design.get("combined_print", {})
    stages = bundle["manifest"].get("stages", {})
    return (
        "### 三件式装配组件摘要\n\n"
        f"- 任务：`{bundle['run_id']}`\n"
        f"- 组件：16齿教学齿轮 + D形短轴 + 可拆限位帽\n"
        f"- 组件尺寸：`{float(params['gear_outer_diameter_mm']):g} × "
        f"{float(params['gear_outer_diameter_mm']):g} × "
        f"{float(sum([params['shaft_base_height_mm'], params['gear_thickness_mm'], params['cap_height_mm']])):g} mm`\n"
        f"- D孔直径间隙：`{float(params['gear_bore_diametral_clearance_mm']):.2f} mm`\n"
        f"- 帽孔直径间隙：`{float(params['cap_socket_diametral_clearance_mm']):.2f} mm`\n"
        f"- 同板打印：`{int(combined.get('components', 0))}` 个独立实体，"
        f"`{' × '.join(f'{float(v):g}' for v in combined.get('extents_mm', []))} mm`\n"
        f"- 短轴是否需重打：`{'是' if combined.get('shaft_reprint_required') else '否'}`\n"
        f"- 打印阶段：`{stages.get('print', 'UNKNOWN')}`\n"
        "- 边界：教学装配与流程验证，不用于动力传动或承载；当前无机器人控制。"
    )


def _next_review_version(review_dir: Path) -> int:
    versions: list[int] = []
    for path in review_dir.glob("assembly_fit_review_v*.json"):
        match = re.fullmatch(r"assembly_fit_review_v(\d+)\.json", path.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def save_fit_review(
    run_id: str,
    gear_rotation_deg: float,
    gear_insertion: str,
    cap_insertion: str,
    cap_falls_when_inverted: bool,
    removal_result: str,
    visible_defects: str,
    notes: str,
    *,
    project_root: Path = PROJECT_ROOT,
    runs_root: Path = RUNS_ROOT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    bundle = load_component_bundle(
        run_id, project_root=project_root, runs_root=runs_root
    )
    if not math.isfinite(float(gear_rotation_deg)) or not 0.0 <= float(gear_rotation_deg) <= 180.0:
        raise ValueError("齿轮空转角度必须在 0–180°。")
    if gear_insertion not in INSERTION_RESULTS:
        raise ValueError("齿轮插入结果无效。")
    if cap_insertion not in INSERTION_RESULTS:
        raise ValueError("限位帽插入结果无效。")
    if removal_result not in REMOVAL_RESULTS:
        raise ValueError("拆卸结果无效。")
    defects = str(visible_defects).strip()
    note_text = str(notes).strip()
    acceptable_insertions = {"略紧可接受", "顺畅"}
    passed = (
        float(gear_rotation_deg) <= 2.0
        and gear_insertion in acceptable_insertions
        and cap_insertion in acceptable_insertions
        and not bool(cap_falls_when_inverted)
        and removal_result != "无法无损拆卸"
        and not defects
    )
    version = _next_review_version(bundle["run_dir"] / "reviews")
    now = _now_iso()
    record = {
        "schema_version": "1.0",
        "created_at": now,
        "run_id": bundle["run_id"],
        "version": version,
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "source_design_report": str(
            bundle["paths"]["design_report"].relative_to(project_root)
        ),
        "measurements": {
            "gear_free_rotation_deg": float(gear_rotation_deg),
            "gear_insertion": gear_insertion,
            "cap_insertion": cap_insertion,
            "cap_falls_when_inverted": bool(cap_falls_when_inverted),
            "removal_result": removal_result,
            "visible_defects": defects,
            "notes": note_text,
        },
        "acceptance": {
            "maximum_gear_free_rotation_deg": 2.0,
            "cap_must_not_fall_when_inverted": True,
            "result": "PASS" if passed else "REVIEW_REQUIRED",
        },
        "robot_control": "PLANNING_ONLY_NO_ROBOT_CONTROL",
    }
    review_path = bundle["run_dir"] / "reviews" / f"assembly_fit_review_v{version}.json"
    _write_json_new(review_path, record)

    manifest = bundle["manifest"]
    relative = str(review_path.relative_to(project_root))
    manifest.setdefault("artifacts", []).append(
        {
            "role": "assembly_fit_review",
            "version": version,
            "path": relative,
            "bytes": review_path.stat().st_size,
            "sha256": file_sha256(review_path),
        }
    )
    manifest.setdefault("stages", {})["fit_review"] = record["status"]
    manifest["updated_at"] = now
    manifest.setdefault("events", []).append(
        {
            "time": now,
            "event": "ASSEMBLY_FIT_REVIEW_RECORDED",
            "detail": f"记录三件套试装 v{version}：{record['status']}。",
        }
    )
    manifest_path = bundle["manifest_path"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record, review_path, manifest


def fit_review_summary(record: dict[str, Any] | None) -> str:
    if not record:
        return "### 试装验收\n\n尚未保存本次打印的试装结果。"
    values = record["measurements"]
    result = "✅ 通过" if record["status"] == "PASS" else "⚠️ 需要继续调整"
    return (
        "### 试装验收\n\n"
        f"- 结论：**{result}**\n"
        f"- 齿轮剩余空转：`{float(values['gear_free_rotation_deg']):g}°`\n"
        f"- 齿轮插入：{values['gear_insertion']}\n"
        f"- 限位帽插入：{values['cap_insertion']}\n"
        f"- 倒置脱落：{'是' if values['cap_falls_when_inverted'] else '否'}\n"
        f"- 拆卸：{values['removal_result']}\n"
        f"- 可见缺陷：{values['visible_defects'] or '无'}"
    )
