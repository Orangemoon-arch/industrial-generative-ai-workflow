"""Versioned adapter for CPU spur-gear geometry analysis."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


class GearAnalysisError(RuntimeError):
    """Raised when the isolated gear analysis worker fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _next_version(run_dir: Path) -> int:
    version = 1
    while (run_dir / "reports" / f"gear_analysis_v{version}.json").exists():
        version += 1
    return version


def run_gear_analysis(
    *,
    project_root: Path,
    run_dir: Path,
    source_mesh: Path,
    source_sha256: str,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    source_mesh = source_mesh.resolve()
    if run_dir.parent != (project_root / "workspace/runs").resolve():
        raise ValueError("齿轮分析任务必须是 workspace/runs 下的直接子目录。")
    if source_mesh.parent != run_dir / "meshes" or not source_mesh.is_file():
        raise ValueError("只能分析当前任务 meshes 目录中的 GLB。")
    if source_mesh.suffix.lower() != ".glb":
        raise ValueError("齿轮分析输入必须是 GLB。")
    if len(source_sha256) != 64 or _sha256(source_mesh) != source_sha256.lower():
        raise ValueError("齿轮分析输入 SHA256 与任务档案不一致。")

    python_path = project_root / "apps/Hunyuan3D-2.1/.venv/bin/python"
    analysis_script = project_root / "apps/industrial-ai-service/extract_spur_gear_parameters.py"
    for required in (python_path, analysis_script):
        if not required.is_file():
            raise FileNotFoundError(f"齿轮分析依赖不存在：{required}")

    version = _next_version(run_dir)
    report_path = run_dir / "reports" / f"gear_analysis_v{version}.json"
    preview_path = run_dir / "reports" / f"gear_analysis_v{version}_radial_profile.png"
    log_path = run_dir / "logs" / f"gear_analysis_v{version}.log"
    if any(path.exists() for path in (report_path, preview_path, log_path)):
        raise FileExistsError("版本化齿轮分析输出已存在，拒绝覆盖。")
    command = [
        str(python_path),
        str(analysis_script),
        "--input",
        str(source_mesh),
        "--report",
        str(report_path),
        "--profile-preview",
        str(preview_path),
    ]
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
            raise GearAnalysisError(f"齿轮参数分析超过 {timeout_seconds} 秒。") from exc
    if completed.returncode != 0 or not report_path.is_file() or not preview_path.is_file():
        raise GearAnalysisError(
            f"齿轮参数分析失败（退出码 {completed.returncode}），请查看日志。"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("input", {}).get("sha256") != source_sha256.lower():
        raise GearAnalysisError("齿轮参数报告与当前 GLB 的 SHA256 不一致。")

    def relative(path: Path) -> str:
        return str(path.relative_to(project_root))

    return {
        "version": version,
        "report_path": relative(report_path),
        "preview_path": relative(preview_path),
        "log_path": relative(log_path),
        "report": report,
    }
