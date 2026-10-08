"""Safe, versioned GLB import for the CPU mesh-repair workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


MAX_GLB_UPLOAD_BYTES = 200 * 1024 * 1024


class MeshImportError(RuntimeError):
    """Raised when an uploaded GLB cannot be archived and inspected safely."""


def _inside(path: Path, root: Path) -> bool:
    root = root.resolve()
    path = path.resolve()
    return path == root or root in path.parents


def _upload_path(upload: Any) -> Path:
    value = upload
    if hasattr(value, "path"):
        value = value.path
    elif isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("请先上传一个 GLB 文件。")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError("上传的 GLB 文件不存在。")
    return path


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
    version = 1
    while (run_dir / "meshes" / f"external_mesh_import_v{version}.glb").exists():
        version += 1
    return version


def archive_and_inspect_glb(
    *,
    project_root: Path,
    run_dir: Path,
    upload: Any,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    if run_dir.parent != (project_root / "workspace/runs").resolve():
        raise ValueError("外部 GLB 必须归档到 workspace/runs 下的直接子任务。")
    source = _upload_path(upload)
    gradio_root = (Path(tempfile.gettempdir()) / "gradio").resolve()
    if not (_inside(source, project_root) or _inside(source, gradio_root)):
        raise ValueError("上传源文件不在项目目录或 Gradio 临时上传目录中。")
    size = source.stat().st_size
    if size <= 0 or size > MAX_GLB_UPLOAD_BYTES:
        raise ValueError("GLB 文件必须大于 0 且不超过 200 MiB。")
    if source.suffix.lower() != ".glb":
        raise ValueError("只接受扩展名为 .glb 的文件。")
    with source.open("rb") as handle:
        if handle.read(4) != b"glTF":
            raise ValueError("文件头不是 glTF 2.0 binary，拒绝按 GLB 导入。")

    python_path = project_root / "apps/Hunyuan3D-2.1/.venv/bin/python"
    inspect_script = project_root / "apps/Hunyuan3D-2.1/hy3dshape/inspect_service_mesh.py"
    for required in (python_path, inspect_script):
        if not required.is_file():
            raise FileNotFoundError(f"外部 GLB 检查依赖不存在：{required}")

    version = _next_version(run_dir)
    mesh_path = run_dir / "meshes" / f"external_mesh_import_v{version}.glb"
    report_path = run_dir / "reports" / f"external_mesh_import_v{version}_inspection.json"
    preview_path = run_dir / "reports" / f"external_mesh_import_v{version}_preview.png"
    request_path = run_dir / "requests" / f"external_mesh_import_v{version}.json"
    log_path = run_dir / "logs" / f"external_mesh_import_v{version}.log"
    if any(path.exists() for path in (mesh_path, report_path, preview_path, request_path, log_path)):
        raise FileExistsError("版本化外部 GLB 输出已存在，拒绝覆盖。")

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mesh_path.with_suffix(".glb.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(mesh_path)
    archived_sha = _sha256(mesh_path)
    request = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operation": "external_glb_import_and_inspection",
        "version": version,
        "original_name": source.name,
        "source_bytes": size,
        "archived_path": str(mesh_path.relative_to(project_root)),
        "archived_sha256": archived_sha,
    }
    _write_json(request_path, request)

    command = [
        str(python_path),
        str(inspect_script),
        "--input",
        str(mesh_path),
        "--report-output",
        str(report_path),
        "--preview-output",
        str(preview_path),
        "--include-back",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=project_root / "apps/Hunyuan3D-2.1",
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MeshImportError(f"外部 GLB 检查超过 {timeout_seconds} 秒。") from exc
    if completed.returncode != 0 or not report_path.is_file() or not preview_path.is_file():
        raise MeshImportError(
            f"外部 GLB 已归档但检查失败（退出码 {completed.returncode}），请查看日志。"
        )
    inspection = json.loads(report_path.read_text(encoding="utf-8"))
    if inspection.get("input_sha256") != archived_sha:
        raise MeshImportError("外部 GLB 检查报告与归档文件 SHA256 不一致。")

    def relative(path: Path) -> str:
        return str(path.relative_to(project_root))

    return {
        "version": version,
        "mesh_path": relative(mesh_path),
        "mesh_sha256": archived_sha,
        "report_path": relative(report_path),
        "preview_path": relative(preview_path),
        "request_path": relative(request_path),
        "log_path": relative(log_path),
        "inspection": inspection,
        "original_name": source.name,
    }
