"""CPU-only, versioned adapters for fan regularization and STL export."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class MeshOptimizationError(RuntimeError):
    """Raised when a CPU mesh operation fails or returns incomplete outputs."""


def _inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    return resolved == parent.resolve() or parent.resolve() in resolved.parents


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


def _log_failure_detail(log_path: Path) -> str:
    try:
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return "请查看任务日志"
    meaningful = [line for line in lines if line]
    if not meaningful:
        return "任务日志为空"
    preferred = next(
        (
            line
            for line in reversed(meaningful)
            if any(marker in line for marker in ("RuntimeError:", "ValueError:", "Error:"))
        ),
        meaningful[-1],
    )
    return preferred[:500]


def _next_version(directory: Path, pattern: str) -> int:
    regex = re.compile(pattern)
    numbers = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = regex.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _next_stl_export_version(run_dir: Path, source_stem: str) -> int:
    escaped_stem = re.escape(source_stem)
    patterns = (
        (run_dir / "stl", re.compile(rf"{escaped_stem}_print_candidate_v(\d+)\.stl")),
        (run_dir / "reports", re.compile(rf"{escaped_stem}_stl_export_v(\d+)\.json")),
        (run_dir / "logs", re.compile(rf"{escaped_stem}_stl_export_v(\d+)\.log")),
    )
    versions: list[int] = []
    for directory, pattern in patterns:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def validate_regularization_parameters(parameters: dict[str, float]) -> None:
    outer = float(parameters["outer_diameter_mm"])
    inner = float(parameters["ring_inner_diameter_mm"])
    ring_depth = float(parameters["ring_depth_mm"])
    hub = float(parameters["hub_diameter_mm"])
    hub_depth = float(parameters["hub_depth_mm"])
    thickness = float(parameters["blade_thickness_mm"])
    sweep = float(parameters["blade_sweep_deg"])
    pitch = float(parameters["blade_pitch_camber_mm"])
    arch = float(parameters["blade_radial_arch_mm"])
    resolution = float(parameters.get("resolution_mm", 0.15))
    if not 40.0 <= outer <= 120.0:
        raise ValueError("外径必须在 40–120 mm。")
    if not 4.0 <= ring_depth <= 30.0:
        raise ValueError("外环深度必须在 4–30 mm。")
    if not 4.0 <= hub_depth <= ring_depth:
        raise ValueError("轮毂深度必须在 4 mm 到外环深度之间。")
    if inner <= hub + 8.0:
        raise ValueError("外环内径与轮毂直径之间至少相差 8 mm。")
    if outer <= inner + 4.0:
        raise ValueError("外径与外环内径至少相差 4 mm。")
    if not 1.6 <= thickness <= 8.0:
        raise ValueError("叶片厚度必须在 1.6–8 mm。")
    if not 0.0 <= sweep <= 35.0:
        raise ValueError("叶片掠角必须在 0–35°。")
    if not 0.0 <= pitch <= 3.0 or not 0.0 <= arch <= 2.0:
        raise ValueError("叶片弯度超出允许范围。")
    if thickness / 2.0 + pitch + arch + 0.4 > ring_depth / 2.0:
        raise ValueError("叶片厚度与弯度超出外环轴向包容空间。")
    if not 0.10 <= resolution <= 0.30:
        raise ValueError("体素分辨率必须在 0.10–0.30 mm。")


def run_regularization(
    *,
    project_root: Path,
    run_dir: Path,
    source_mesh: Path,
    source_sha256: str,
    parameters: dict[str, float],
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    source_mesh = source_mesh.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if run_dir.parent != runs_root:
        raise ValueError("规则化任务必须是 workspace/runs 下的直接子目录。")
    if source_mesh.parent != run_dir / "meshes" or not source_mesh.is_file():
        raise ValueError("只能读取当前任务 meshes 目录中的源网格。")
    if source_mesh.suffix.lower() != ".glb":
        raise ValueError("规则化源网格必须是 GLB 文件。")
    if len(source_sha256) != 64 or _sha256(source_mesh) != source_sha256.lower():
        raise ValueError("源网格 SHA256 与任务档案不一致。")
    validate_regularization_parameters(parameters)

    app_dir = project_root / "apps/Hunyuan3D-2.1"
    python_path = app_dir / ".venv/bin/python"
    build_script = app_dir / "hy3dshape/build_regularized_fan_review.py"
    inspect_script = app_dir / "hy3dshape/inspect_service_mesh.py"
    for required in (python_path, build_script, inspect_script):
        if not required.is_file():
            raise FileNotFoundError(f"规则化依赖不存在：{required}")

    version = _next_version(
        run_dir / "meshes", r"fan_7blade_regularized_review_v(\d+)\.glb"
    )
    stem = f"fan_7blade_regularized_review_v{version}"
    request_path = run_dir / "requests" / f"fan_regularization_v{version}.json"
    mesh_path = run_dir / "meshes" / f"{stem}.glb"
    build_report_path = run_dir / "reports" / f"fan_7blade_regularized_build_v{version}.json"
    inspection_path = run_dir / "reports" / f"fan_7blade_regularized_inspection_v{version}.json"
    preview_path = run_dir / "reports" / f"fan_7blade_regularized_projection_v{version}.png"
    log_path = run_dir / "logs" / f"fan_regularization_v{version}.log"
    outputs = (request_path, mesh_path, build_report_path, inspection_path, preview_path, log_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("版本化输出已存在，拒绝覆盖。")

    request = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operation": "seven_blade_fan_cpu_regularization",
        "version": version,
        "source_mesh": str(source_mesh.relative_to(project_root)),
        "source_sha256": source_sha256,
        "parameters": {key: float(value) for key, value in parameters.items()},
        "outputs": {
            "mesh": str(mesh_path.relative_to(project_root)),
            "build_report": str(build_report_path.relative_to(project_root)),
            "inspection": str(inspection_path.relative_to(project_root)),
            "preview": str(preview_path.relative_to(project_root)),
        },
    }
    _write_json(request_path, request)
    command = [
        str(python_path), str(build_script), "--run-dir", str(run_dir),
        "--source-mesh", str(source_mesh), "--output", str(mesh_path),
        "--report-output", str(build_report_path),
    ]
    cli_names = {
        "resolution_mm": "--resolution-mm",
        "outer_diameter_mm": "--outer-diameter-mm",
        "ring_inner_diameter_mm": "--ring-inner-diameter-mm",
        "ring_depth_mm": "--ring-depth-mm",
        "hub_diameter_mm": "--hub-diameter-mm",
        "hub_depth_mm": "--hub-depth-mm",
        "blade_thickness_mm": "--blade-thickness-mm",
        "blade_sweep_deg": "--blade-sweep-deg",
        "blade_pitch_camber_mm": "--blade-pitch-camber-mm",
        "blade_radial_arch_mm": "--blade-radial-arch-mm",
    }
    for key, flag in cli_names.items():
        command.extend([flag, str(float(parameters[key]))])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        for operation in (
            command,
            [
                str(python_path), str(inspect_script), "--input", str(mesh_path),
                "--report-output", str(inspection_path), "--preview-output",
                str(preview_path), "--include-back",
            ],
        ):
            try:
                completed = subprocess.run(
                    operation,
                    cwd=app_dir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise MeshOptimizationError(
                    f"CPU 网格处理超过 {timeout_seconds} 秒。"
                ) from exc
            if completed.returncode != 0:
                raise MeshOptimizationError(
                    f"CPU 网格处理失败（退出码 {completed.returncode}），请查看日志。"
                )

    for output in (mesh_path, build_report_path, inspection_path, preview_path):
        if not output.is_file():
            raise MeshOptimizationError("规则化没有生成完整的网格和检查档案。")
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    mesh_sha = _sha256(mesh_path)
    if inspection.get("input_sha256") != mesh_sha:
        raise MeshOptimizationError("自动检查报告与新 GLB 不匹配。")
    if not (
        inspection.get("components") == 1
        and inspection.get("watertight") is True
        and inspection.get("winding_consistent") is True
        and inspection.get("boundary_edges") == 0
        and inspection.get("nonmanifold_edges") == 0
        and inspection.get("degenerate_faces") == 0
    ):
        raise MeshOptimizationError("新规则化网格未通过自动几何门禁。")
    return {
        "version": version,
        "request_path": str(request_path.relative_to(project_root)),
        "mesh_path": str(mesh_path.relative_to(project_root)),
        "build_report_path": str(build_report_path.relative_to(project_root)),
        "inspection_path": str(inspection_path.relative_to(project_root)),
        "preview_path": str(preview_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "mesh_sha256": mesh_sha,
        "mesh_bytes": mesh_path.stat().st_size,
        "inspection": inspection,
    }


def run_stl_export(
    *,
    project_root: Path,
    run_dir: Path,
    source_mesh: Path,
    source_sha256: str,
    target_longest_mm: float,
    allow_supported_overhangs: bool,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    source_mesh = source_mesh.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if run_dir.parent != runs_root:
        raise ValueError("STL 导出任务必须是 workspace/runs 下的直接子目录。")
    if source_mesh.parent != run_dir / "meshes" or not source_mesh.is_file():
        raise ValueError("STL 源网格必须位于当前任务 meshes 目录。")
    if source_mesh.suffix.lower() != ".glb":
        raise ValueError("STL 源网格必须是 GLB 文件。")
    if len(source_sha256) != 64 or _sha256(source_mesh) != source_sha256.lower():
        raise ValueError("STL 源网格 SHA256 与批准档案不一致。")
    if not 20.0 <= float(target_longest_mm) <= 200.0:
        raise ValueError("目标最长尺寸必须在 20–200 mm。")

    app_dir = project_root / "apps/Hunyuan3D-2.1"
    python_path = app_dir / ".venv/bin/python"
    export_script = app_dir / "hy3dshape/export_regularized_print_stl.py"
    for required in (python_path, export_script):
        if not required.is_file():
            raise FileNotFoundError(f"STL 导出依赖不存在：{required}")
    version = _next_stl_export_version(run_dir, source_mesh.stem)
    output_path = run_dir / "stl" / f"{source_mesh.stem}_print_candidate_v{version}.stl"
    report_path = run_dir / "reports" / f"{source_mesh.stem}_stl_export_v{version}.json"
    log_path = run_dir / "logs" / f"{source_mesh.stem}_stl_export_v{version}.log"
    if any(path.exists() for path in (output_path, report_path, log_path)):
        raise FileExistsError("STL 版本化输出已存在，拒绝覆盖。")
    command = [
        str(python_path), str(export_script), "--input", str(source_mesh),
        "--expected-sha256", source_sha256, "--output", str(output_path),
        "--report-output", str(report_path), "--target-longest-mm",
        str(float(target_longest_mm)),
    ]
    if allow_supported_overhangs:
        command.append("--allow-supported-overhangs")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            completed = subprocess.run(
                command, cwd=app_dir, stdout=log_handle, stderr=subprocess.STDOUT,
                text=True, timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MeshOptimizationError(f"STL 导出超过 {timeout_seconds} 秒。") from exc
    if completed.returncode != 0 or not output_path.is_file() or not report_path.is_file():
        raise MeshOptimizationError(
            f"STL 导出失败（退出码 {completed.returncode}）；"
            f"原因：{_log_failure_detail(log_path)}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("input_sha256") != source_sha256 or report.get("output_sha256") != _sha256(output_path):
        raise MeshOptimizationError("STL 导出报告与实际文件不匹配。")
    return {
        "version": version,
        "output_path": str(output_path.relative_to(project_root)),
        "report_path": str(report_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "output_sha256": report["output_sha256"],
        "output_bytes": output_path.stat().st_size,
        "report": report,
    }
