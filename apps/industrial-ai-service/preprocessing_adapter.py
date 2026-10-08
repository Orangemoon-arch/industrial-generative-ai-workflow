"""CPU-only adapters for generic mask creation and conservative mesh review."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


class PreprocessingError(RuntimeError):
    """Raised when a CPU preprocessing task fails or returns unsafe output."""


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


def _next_version(directory: Path, pattern: str) -> int:
    regex = re.compile(pattern)
    versions = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = regex.fullmatch(path.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def _validate_run(project_root: Path, run_dir: Path) -> tuple[Path, Path]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if run_dir.parent != runs_root or not run_dir.is_dir():
        raise ValueError("任务必须是 workspace/runs 下的直接子目录。")
    return project_root, run_dir


def _run_logged(
    command: list[str], *, cwd: Path, log_handle, timeout_seconds: int
) -> None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreprocessingError(f"CPU 处理超过 {timeout_seconds} 秒。") from exc
    if completed.returncode != 0:
        raise PreprocessingError(
            f"CPU 处理失败（退出码 {completed.returncode}），请查看任务日志。"
        )


def run_mask_preparation(
    *,
    project_root: Path,
    run_dir: Path,
    image_path: Path,
    image_sha256: str,
    threshold: float = 40.0,
    border_width: int = 20,
    minimum_hole_pixels: int = 100,
    trim_below_row: int | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    project_root, run_dir = _validate_run(project_root, run_dir)
    image_path = image_path.resolve()
    if image_path.parent.parent != run_dir / "images" or not image_path.is_file():
        raise ValueError("遮罩输入必须位于当前任务 images 目录的候选子目录。")
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("遮罩输入只支持 PNG/JPEG。")
    if len(image_sha256) != 64 or _sha256(image_path) != image_sha256.lower():
        raise ValueError("候选图片 SHA256 与批准档案不一致。")
    if not 5.0 <= float(threshold) <= 120.0:
        raise ValueError("背景颜色距离阈值必须在 5–120。")
    if not 1 <= int(border_width) <= 100:
        raise ValueError("背景边缘宽度必须在 1–100 像素。")
    if not 1 <= int(minimum_hole_pixels) <= 1_000_000:
        raise ValueError("最小保留孔洞面积必须为正整数。")
    if trim_below_row is not None and trim_below_row <= 0:
        trim_below_row = None

    app_dir = project_root / "apps/Hunyuan3D-2.1"
    python_path = app_dir / ".venv/bin/python"
    script_path = app_dir / "hy3dshape/prepare_service_white_background_mask.py"
    for required in (python_path, script_path):
        if not required.is_file():
            raise FileNotFoundError(f"遮罩处理依赖不存在：{required}")

    escaped_stem = re.escape(image_path.stem)
    version = _next_version(
        run_dir / "masks", rf"{escaped_stem}_rgba_auto_v(\d+)\.png"
    )
    stem = f"{image_path.stem}_mask_auto_v{version}"
    rgba_path = run_dir / "masks" / f"{image_path.stem}_rgba_auto_v{version}.png"
    preview_path = run_dir / "masks" / f"{stem}_preview.png"
    report_path = run_dir / "masks" / f"{stem}_report.json"
    request_path = run_dir / "requests" / f"{stem}.json"
    log_path = run_dir / "logs" / f"{stem}.log"
    outputs = (rgba_path, preview_path, report_path, request_path, log_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("遮罩版本化输出已存在，拒绝覆盖。")
    request = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operation": "white_background_mask_preparation",
        "version": version,
        "input_path": str(image_path.relative_to(project_root)),
        "input_sha256": image_sha256,
        "parameters": {
            "threshold": float(threshold),
            "border_width": int(border_width),
            "minimum_hole_pixels": int(minimum_hole_pixels),
            "trim_below_row": trim_below_row,
        },
    }
    _write_json(request_path, request)
    command = [
        str(python_path),
        str(script_path),
        "--input",
        str(image_path),
        "--rgba-output",
        str(rgba_path),
        "--preview-output",
        str(preview_path),
        "--report-output",
        str(report_path),
        "--threshold",
        str(float(threshold)),
        "--border-width",
        str(int(border_width)),
        "--minimum-hole-pixels",
        str(int(minimum_hole_pixels)),
    ]
    if trim_below_row is not None:
        command.extend(["--trim-below-row", str(int(trim_below_row))])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        _run_logged(
            command, cwd=app_dir, log_handle=log_handle, timeout_seconds=timeout_seconds
        )
    for output in (rgba_path, preview_path, report_path):
        if not output.is_file():
            raise PreprocessingError("遮罩处理未生成完整输出。")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("input_sha256") != image_sha256:
        raise PreprocessingError("遮罩报告与候选图片不匹配。")
    if report.get("rgba_sha256") != _sha256(rgba_path):
        raise PreprocessingError("遮罩报告与 RGBA 文件不匹配。")
    if report.get("corner_alpha") != [0, 0, 0, 0]:
        raise PreprocessingError("遮罩四角未全部透明，拒绝进入人工确认。")
    if report.get("final_foreground_components") != 1:
        raise PreprocessingError("遮罩前景不是单一连通主体。")
    fraction = float(report.get("foreground_fraction", 0.0))
    if not 0.02 <= fraction <= 0.95:
        raise PreprocessingError("遮罩前景占比异常，需调整阈值。")
    return {
        "version": version,
        "request_path": str(request_path.relative_to(project_root)),
        "rgba_path": str(rgba_path.relative_to(project_root)),
        "preview_path": str(preview_path.relative_to(project_root)),
        "report_path": str(report_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "rgba_sha256": _sha256(rgba_path),
        "report": report,
    }


def _mesh_gate(report: dict[str, Any]) -> bool:
    return bool(
        report.get("components") == 1
        and report.get("watertight") is True
        and report.get("winding_consistent") is True
        and report.get("boundary_edges") == 0
        and report.get("nonmanifold_edges") == 0
        and float(report.get("volume", 0.0)) > 0.0
    )


def run_mesh_review_preparation(
    *,
    project_root: Path,
    run_dir: Path,
    raw_mesh_path: Path,
    raw_mesh_sha256: str,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    project_root, run_dir = _validate_run(project_root, run_dir)
    raw_mesh_path = raw_mesh_path.resolve()
    if raw_mesh_path.parent != run_dir / "meshes" or not raw_mesh_path.is_file():
        raise ValueError("原始网格必须位于当前任务 meshes 目录。")
    if raw_mesh_path.suffix.lower() != ".glb":
        raise ValueError("原始网格必须是 GLB。")
    if len(raw_mesh_sha256) != 64 or _sha256(raw_mesh_path) != raw_mesh_sha256.lower():
        raise ValueError("原始 GLB SHA256 与任务档案不一致。")

    app_dir = project_root / "apps/Hunyuan3D-2.1"
    python_path = app_dir / ".venv/bin/python"
    inspect_script = app_dir / "hy3dshape/inspect_service_mesh.py"
    prepare_script = app_dir / "hy3dshape/prepare_service_mesh_review.py"
    for required in (python_path, inspect_script, prepare_script):
        if not required.is_file():
            raise FileNotFoundError(f"网格检查依赖不存在：{required}")

    base = raw_mesh_path.stem.removesuffix("_raw")
    escaped_base = re.escape(base)
    version = _next_version(
        run_dir / "meshes", rf"{escaped_base}_main_review_v(\d+)\.glb"
    )
    review_path = run_dir / "meshes" / f"{base}_main_review_v{version}.glb"
    raw_inspection_path = run_dir / "reports" / f"{base}_raw_inspection_v{version}.json"
    raw_preview_path = run_dir / "reports" / f"{base}_raw_projection_v{version}.png"
    process_path = run_dir / "reports" / f"{base}_review_process_v{version}.json"
    inspection_path = run_dir / "reports" / f"{base}_review_inspection_v{version}.json"
    preview_path = run_dir / "reports" / f"{base}_review_projection_v{version}.png"
    request_path = run_dir / "requests" / f"{base}_mesh_review_v{version}.json"
    log_path = run_dir / "logs" / f"{base}_mesh_review_v{version}.log"
    outputs = (
        review_path,
        raw_inspection_path,
        raw_preview_path,
        process_path,
        inspection_path,
        preview_path,
        request_path,
        log_path,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("网格 review 版本化输出已存在，拒绝覆盖。")
    _write_json(
        request_path,
        {
            "schema_version": "1.0",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "operation": "inspect_and_conservatively_prepare_mesh_review",
            "version": version,
            "input_path": str(raw_mesh_path.relative_to(project_root)),
            "input_sha256": raw_mesh_sha256,
        },
    )
    inspect_raw = [
        str(python_path),
        str(inspect_script),
        "--input",
        str(raw_mesh_path),
        "--report-output",
        str(raw_inspection_path),
        "--preview-output",
        str(raw_preview_path),
        "--include-back",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        _run_logged(
            inspect_raw,
            cwd=app_dir,
            log_handle=log_handle,
            timeout_seconds=timeout_seconds,
        )
        raw_report = json.loads(raw_inspection_path.read_text(encoding="utf-8"))
        if raw_report.get("input_sha256") != raw_mesh_sha256:
            raise PreprocessingError("原始网格检查报告与 GLB 不匹配。")
        if _mesh_gate(raw_report):
            shutil.copyfile(raw_mesh_path, review_path)
            _write_json(
                process_path,
                {
                    "schema_version": "1.0",
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "operation": "byte_copy_already_valid_raw_glb_to_versioned_review",
                    "input_path": str(raw_mesh_path.relative_to(project_root)),
                    "input_sha256": raw_mesh_sha256,
                    "output_path": str(review_path.relative_to(project_root)),
                    "output_sha256": _sha256(review_path),
                    "removed_components": 0,
                    "note": "原始 GLB 已通过几何门禁，仅创建版本化 review 副本。",
                },
            )
            log_handle.write("\nRaw mesh already passed; created byte-identical review copy.\n")
        else:
            _run_logged(
                [
                    str(python_path),
                    str(prepare_script),
                    "--input",
                    str(raw_mesh_path),
                    "--output",
                    str(review_path),
                    "--report-output",
                    str(process_path),
                ],
                cwd=app_dir,
                log_handle=log_handle,
                timeout_seconds=timeout_seconds,
            )
        _run_logged(
            [
                str(python_path),
                str(inspect_script),
                "--input",
                str(review_path),
                "--report-output",
                str(inspection_path),
                "--preview-output",
                str(preview_path),
                "--include-back",
            ],
            cwd=app_dir,
            log_handle=log_handle,
            timeout_seconds=timeout_seconds,
        )
    for output in (review_path, process_path, inspection_path, preview_path):
        if not output.is_file():
            raise PreprocessingError("网格 review 没有生成完整档案。")
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    review_sha = _sha256(review_path)
    if inspection.get("input_sha256") != review_sha:
        raise PreprocessingError("review 检查报告与 GLB 不匹配。")
    if not _mesh_gate(inspection):
        raise PreprocessingError("review GLB 未通过通用几何门禁。")
    return {
        "version": version,
        "request_path": str(request_path.relative_to(project_root)),
        "review_path": str(review_path.relative_to(project_root)),
        "raw_inspection_path": str(raw_inspection_path.relative_to(project_root)),
        "raw_preview_path": str(raw_preview_path.relative_to(project_root)),
        "process_path": str(process_path.relative_to(project_root)),
        "inspection_path": str(inspection_path.relative_to(project_root)),
        "preview_path": str(preview_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "review_sha256": review_sha,
        "inspection": inspection,
    }
