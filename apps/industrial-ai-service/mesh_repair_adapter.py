"""Versioned, CPU-only adapter for guarded mesh diagnosis and repair."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class MeshRepairExecutionError(RuntimeError):
    """Raised when the isolated mesh repair worker fails."""


ROUTE_TO_MODE = {
    "SAFE_REPAIR_CANDIDATE": "safe",
    "STANDARD_HOLE_REPAIR_CANDIDATE": "standard",
    "DETACHED_ARTIFACT_CLEANUP_CANDIDATE": "artifact_cleanup",
    "ADVANCED_TOPOLOGY_REPAIR_CANDIDATE": "advanced",
    "RECONSTRUCTION_RECOMMENDED": "reconstruct",
}

MODE_FLAGS = {
    "safe": "--safe-repair",
    "standard": "--standard-repair",
    "artifact_cleanup": "--artifact-cleanup-candidate",
    "advanced": "--advanced-repair",
    "reconstruct": "--reconstruction-candidate",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _next_version(run_dir: Path) -> int:
    pattern = re.compile(r"mesh_repair_v(\d+)\.json")
    versions: list[int] = []
    reports_dir = run_dir / "reports"
    if reports_dir.is_dir():
        for path in reports_dir.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def recommended_mode(report: dict[str, Any]) -> str | None:
    route = report.get("automatic_repairability_before", {}).get("route")
    return ROUTE_TO_MODE.get(str(route))


def run_mesh_repair(
    *,
    project_root: Path,
    run_dir: Path,
    source_mesh: Path,
    source_sha256: str,
    mode: str = "diagnose",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Diagnose or repair one GLB owned by the current versioned RUN."""
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    source_mesh = source_mesh.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if run_dir.parent != runs_root:
        raise ValueError("网格修复任务必须是 workspace/runs 下的直接子目录。")
    if source_mesh.parent != run_dir / "meshes" or not source_mesh.is_file():
        raise ValueError("只能读取当前任务 meshes 目录中的源 GLB。")
    if source_mesh.suffix.lower() != ".glb":
        raise ValueError("网格修复源文件必须是 GLB。")
    if len(source_sha256) != 64 or _sha256(source_mesh) != source_sha256.lower():
        raise ValueError("源 GLB 的 SHA256 与任务档案不一致。")
    if mode not in {"diagnose", *MODE_FLAGS}:
        raise ValueError(
            "修复模式必须是 diagnose、safe、standard、artifact_cleanup、advanced 或 reconstruct。"
        )

    python_path = project_root / "apps/Hunyuan3D-2.1/.venv/bin/python"
    repair_script = project_root / "apps/industrial-ai-service/mesh_repair_tool.py"
    inspect_script = project_root / "apps/Hunyuan3D-2.1/hy3dshape/inspect_service_mesh.py"
    for required in (python_path, repair_script, inspect_script):
        if not required.is_file():
            raise FileNotFoundError(f"网格修复依赖不存在：{required}")

    version = _next_version(run_dir)
    stem = f"mesh_repair_v{version}"
    request_path = run_dir / "requests" / f"{stem}.json"
    report_path = run_dir / "reports" / f"{stem}.json"
    annotated_path = run_dir / "meshes" / f"{stem}_diagnostic.glb"
    candidate_path = run_dir / "meshes" / f"{stem}_candidate.glb"
    inspection_path = run_dir / "reports" / f"{stem}_candidate_inspection.json"
    preview_path = run_dir / "reports" / f"{stem}_candidate_preview.png"
    comparison_path = run_dir / "reports" / f"{stem}_before_after_comparison.png"
    log_path = run_dir / "logs" / f"{stem}.log"
    outputs = [request_path, report_path, annotated_path, log_path]
    if mode != "diagnose":
        outputs.extend([candidate_path, inspection_path, preview_path, comparison_path])
    if any(path.exists() for path in outputs):
        raise FileExistsError("版本化网格修复输出已存在，拒绝覆盖。")

    request = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operation": "guarded_mesh_diagnosis_and_repair",
        "version": version,
        "mode": mode,
        "source_mesh": str(source_mesh.relative_to(project_root)),
        "source_sha256": source_sha256.lower(),
        "outputs": {
            "report": str(report_path.relative_to(project_root)),
            "annotated_glb": str(annotated_path.relative_to(project_root)),
            "candidate_glb": (
                str(candidate_path.relative_to(project_root))
                if mode != "diagnose"
                else None
            ),
        },
    }
    _write_json(request_path, request)

    command = [
        str(python_path),
        str(repair_script),
        "--input",
        str(source_mesh),
        "--report",
        str(report_path),
        "--annotated-glb",
        str(annotated_path),
    ]
    if mode != "diagnose":
        command.extend([MODE_FLAGS[mode], "--repaired-glb", str(candidate_path)])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=project_root / "apps/industrial-ai-service",
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MeshRepairExecutionError(
                f"CPU 网格修复超过 {timeout_seconds} 秒。"
            ) from exc
    if completed.returncode != 0:
        raise MeshRepairExecutionError(
            f"CPU 网格修复失败（退出码 {completed.returncode}），请查看版本化日志。"
        )
    if not report_path.is_file() or not annotated_path.is_file():
        raise MeshRepairExecutionError("网格修复没有生成完整的诊断报告和标记 GLB。")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("input_sha256") != source_sha256.lower():
        raise MeshRepairExecutionError("修复报告与源 GLB 的 SHA256 不一致。")

    candidate_exists = candidate_path.is_file()
    inspection: dict[str, Any] | None = None
    if candidate_exists:
        inspect_command = [
            str(python_path),
            str(inspect_script),
            "--input",
            str(candidate_path),
            "--report-output",
            str(inspection_path),
            "--preview-output",
            str(preview_path),
            "--include-back",
        ]
        with log_path.open("a", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                inspect_command,
                cwd=project_root / "apps/Hunyuan3D-2.1",
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0 or not inspection_path.is_file() or not preview_path.is_file():
            raise MeshRepairExecutionError("修复候选已生成，但复核报告或预览生成失败。")
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        if inspection.get("input_sha256") != _sha256(candidate_path):
            raise MeshRepairExecutionError("候选检查报告与修复 GLB 不匹配。")
        comparison_script = (
            project_root
            / "apps/Hunyuan3D-2.1/hy3dshape/render_mesh_repair_comparison.py"
        )
        comparison_command = [
            str(python_path),
            str(comparison_script),
            "--source",
            str(source_mesh),
            "--candidate",
            str(candidate_path),
            "--output",
            str(comparison_path),
        ]
        with log_path.open("a", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                comparison_command,
                cwd=project_root / "apps/Hunyuan3D-2.1/hy3dshape",
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0 or not comparison_path.is_file():
            raise MeshRepairExecutionError("候选已生成，但同视角前后对照图生成失败。")
    elif mode != "diagnose" and report.get("repair_applied") is True:
        raise MeshRepairExecutionError("报告声称已修复，但候选 GLB 不存在。")

    def relative(path: Path) -> str:
        return str(path.relative_to(project_root))

    return {
        "version": version,
        "mode": mode,
        "request_path": relative(request_path),
        "report_path": relative(report_path),
        "annotated_path": relative(annotated_path),
        "candidate_path": relative(candidate_path) if candidate_exists else None,
        "inspection_path": relative(inspection_path) if inspection else None,
        "preview_path": relative(preview_path) if inspection else None,
        "comparison_path": relative(comparison_path) if comparison_path.is_file() else None,
        "log_path": relative(log_path),
        "candidate_sha256": _sha256(candidate_path) if candidate_exists else None,
        "report": report,
        "inspection": inspection,
        "recommended_mode": recommended_mode(report),
    }
