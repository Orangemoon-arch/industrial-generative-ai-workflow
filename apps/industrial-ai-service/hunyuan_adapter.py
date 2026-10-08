"""Guarded subprocess adapter for single-view 2.1 and multi-view 2mv Shape."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from gpu_executor import project_gpu_lock, require_available_gpu


HUNYUAN_BACKENDS: dict[str, dict[str, str]] = {
    "single_view_2_1": {
        "display_name": "Hunyuan3D-2.1 Shape",
        "model_name": "Hunyuan3D-2.1",
        "model_relative_path": "models/hunyuan3d-2.1-shape-ms",
        "subfolder": "hunyuan3d-dit-v2-1",
        "input_mode": "single_view",
    },
    "multi_view_2mv": {
        "display_name": "Hunyuan3D-2.1 兼容多视图（2mv 后端）",
        "model_name": "Hunyuan3D-2mv",
        "model_relative_path": "models/hunyuan3d-2mv-ms",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "input_mode": "multi_view",
    },
}

MULTIVIEW_ORDER = ("front", "left", "back", "right")
MULTIVIEW_REQUIRED = ("front", "left", "back")


class HunyuanExecutionError(RuntimeError):
    """Raised when a Hunyuan worker attempt fails."""


def backend_is_available(project_root: Path, backend_key: str) -> bool:
    """Return whether the configured local backend has config and FP16 weights."""
    backend = HUNYUAN_BACKENDS.get(backend_key)
    if backend is None:
        return False
    folder = project_root / backend["model_relative_path"] / backend["subfolder"]
    return (folder / "config.yaml").is_file() and any(
        (folder / name).is_file()
        for name in ("model.fp16.ckpt", "model.fp16.safetensors")
    )


def _inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    return resolved == parent.resolve() or parent.resolve() in resolved.parents


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def next_attempt(run_dir: Path) -> int:
    request_dir = run_dir / "requests"
    existing = (
        list(request_dir.glob("hunyuan_attempt_*.json"))
        if request_dir.is_dir()
        else []
    )
    numbers: list[int] = []
    for path in existing:
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def run_hunyuan(
    *,
    project_root: Path,
    run_dir: Path,
    image_path: Path,
    image_sha256: str,
    seed: int = 1234,
    inference_steps: int = 30,
    octree_resolution: int = 256,
    num_chunks: int = 8000,
    minimum_free_mib: int = 8000,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the existing single-view Hunyuan3D-2.1 backend."""
    return _run_hunyuan_request(
        project_root=project_root,
        run_dir=run_dir,
        backend_key="single_view_2_1",
        image_inputs={"front": (image_path, image_sha256)},
        seed=seed,
        inference_steps=inference_steps,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        minimum_free_mib=minimum_free_mib,
        timeout_seconds=timeout_seconds,
    )


def run_hunyuan_multiview(
    *,
    project_root: Path,
    run_dir: Path,
    image_paths: dict[str, Path],
    image_sha256: dict[str, str],
    seed: int = 1234,
    inference_steps: int = 30,
    octree_resolution: int = 256,
    num_chunks: int = 8000,
    minimum_free_mib: int = 8000,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run Hunyuan3D-2mv behind the same audited GLB contract as 2.1."""
    if set(image_paths) != set(image_sha256):
        raise ValueError("多视图路径和 SHA256 的视角集合不一致。")
    return _run_hunyuan_request(
        project_root=project_root,
        run_dir=run_dir,
        backend_key="multi_view_2mv",
        image_inputs={view: (path, image_sha256[view]) for view, path in image_paths.items()},
        seed=seed,
        inference_steps=inference_steps,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        minimum_free_mib=minimum_free_mib,
        timeout_seconds=timeout_seconds,
    )


def _run_hunyuan_request(
    *,
    project_root: Path,
    run_dir: Path,
    backend_key: str,
    image_inputs: dict[str, tuple[Path, str]],
    seed: int,
    inference_steps: int,
    octree_resolution: int,
    num_chunks: int,
    minimum_free_mib: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if not _inside(run_dir, runs_root):
        raise ValueError("Hunyuan 任务目录必须位于 workspace/runs 内。")
    if backend_key not in HUNYUAN_BACKENDS:
        raise ValueError(f"未知 Hunyuan 后端：{backend_key}")
    backend = HUNYUAN_BACKENDS[backend_key]
    views = tuple(image_inputs)
    if backend["input_mode"] == "single_view":
        if views != ("front",):
            raise ValueError("Hunyuan3D-2.1 单图后端必须且只能接收一个 front 输入。")
    else:
        unknown = set(views) - set(MULTIVIEW_ORDER)
        missing = set(MULTIVIEW_REQUIRED) - set(views)
        if unknown:
            raise ValueError("不支持的多视图标签：" + ", ".join(sorted(unknown)))
        if missing:
            raise ValueError("多视图输入缺少：" + ", ".join(sorted(missing)))
        if len(views) not in {3, 4}:
            raise ValueError("Hunyuan3D-2mv 只接受 3 或 4 张标准视角图。")

    normalized_inputs: dict[str, tuple[Path, str]] = {}
    for view in MULTIVIEW_ORDER:
        if view not in image_inputs:
            continue
        image_path, digest = image_inputs[view]
        image_path = image_path.resolve()
        if not _inside(image_path, run_dir / "masks") or not image_path.is_file():
            raise ValueError(f"Hunyuan {view} 输入必须位于当前任务 masks 目录。")
        if image_path.suffix.lower() != ".png":
            raise ValueError(f"Hunyuan {view} 输入必须是 PNG RGBA。")
        if len(digest) != 64:
            raise ValueError(f"Hunyuan {view} 输入 SHA256 无效。")
        normalized_inputs[view] = (image_path, digest)
    if not 1 <= inference_steps <= 50:
        raise ValueError("Hunyuan 推理步数必须为 1 到 50。")
    if octree_resolution not in {128, 256, 384}:
        raise ValueError("octree resolution 只允许 128、256 或 384。")
    if not 1000 <= num_chunks <= 20000:
        raise ValueError("num_chunks 必须在 1000 到 20000 之间。")

    hunyuan_dir = project_root / "apps/Hunyuan3D-2.1"
    python_path = hunyuan_dir / ".venv/bin/python"
    worker_path = hunyuan_dir / "hy3dshape/run_hunyuan_service_task.py"
    model_path = project_root / backend["model_relative_path"]
    for required in (python_path, worker_path, model_path):
        if not required.exists():
            raise FileNotFoundError(f"Hunyuan 运行依赖不存在：{required}")
    model_subfolder = model_path / backend["subfolder"]
    config_path = model_subfolder / "config.yaml"
    safetensors_path = model_subfolder / "model.fp16.safetensors"
    checkpoint_path = model_subfolder / "model.fp16.ckpt"
    if not model_subfolder.is_dir():
        raise FileNotFoundError(f"Hunyuan 模型子目录不存在：{model_subfolder}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Hunyuan 模型配置不存在：{config_path}")
    if safetensors_path.is_file():
        use_safetensors = True
    elif checkpoint_path.is_file():
        use_safetensors = False
    else:
        raise FileNotFoundError(
            f"Hunyuan FP16 权重不存在：{safetensors_path} 或 {checkpoint_path}"
        )

    attempt = next_attempt(run_dir)
    attempt_name = f"hunyuan_attempt_{attempt:03d}"
    request_path = run_dir / "requests" / f"{attempt_name}.json"
    output_path = run_dir / "meshes" / f"{attempt_name}_raw.glb"
    result_path = run_dir / "reports" / f"{attempt_name}_result.json"
    log_path = run_dir / "logs" / f"{attempt_name}.log"
    request = {
        "schema_version": "1.1",
        "interface_name": "Hunyuan3D-2.1-compatible",
        "backend_key": backend_key,
        "backend_model": backend["model_name"],
        "input_mode": backend["input_mode"],
        "model_path": str(model_path.relative_to(project_root)),
        "model_subfolder": backend["subfolder"],
        "use_safetensors": use_safetensors,
        "images": {
            view: {
                "path": str(path.relative_to(project_root)),
                "sha256": digest,
            }
            for view, (path, digest) in normalized_inputs.items()
        },
        "output_path": str(output_path.relative_to(project_root)),
        "result_path": str(result_path.relative_to(project_root)),
        "seed": seed,
        "num_inference_steps": inference_steps,
        "octree_resolution": octree_resolution,
        "num_chunks": num_chunks,
        "dtype": "float16",
        "low_vram_mode": "model_cpu_offload",
        "remove_background": False,
    }
    _write_json(request_path, request)

    lock_path = runs_root / ".gpu-model.lock"
    with project_gpu_lock(lock_path):
        snapshot = require_available_gpu(minimum_free_mib)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                completed = subprocess.run(
                    [str(python_path), str(worker_path), "--request", str(request_path)],
                    cwd=hunyuan_dir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise HunyuanExecutionError(
                    f"Hunyuan 任务超过 {timeout_seconds} 秒，已由执行器停止。"
                ) from exc

    if completed.returncode != 0:
        raise HunyuanExecutionError(
            f"Hunyuan 工作进程退出码为 {completed.returncode}，详见项目内日志。"
        )
    if not result_path.is_file() or not output_path.is_file():
        raise HunyuanExecutionError("Hunyuan 工作进程没有生成完整结果。")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("output_path") != str(output_path.relative_to(project_root)):
        raise HunyuanExecutionError("Hunyuan 返回的输出路径与请求不一致。")
    if result.get("backend_model") != backend["model_name"]:
        raise HunyuanExecutionError("Hunyuan 返回的真实后端标识与请求不一致。")
    result_inputs = result.get("inputs", {})
    expected_hashes = {view: digest for view, (_, digest) in normalized_inputs.items()}
    actual_hashes = {
        view: item.get("sha256")
        for view, item in result_inputs.items()
        if isinstance(item, dict)
    }
    if actual_hashes != expected_hashes:
        raise HunyuanExecutionError("Hunyuan 实际输入与已确认图片 SHA256 不一致。")
    if result.get("output_sha256") is None:
        raise HunyuanExecutionError("Hunyuan 结果缺少输出 SHA256。")

    return {
        "attempt": attempt,
        "request_path": str(request_path.relative_to(project_root)),
        "output_path": str(output_path.relative_to(project_root)),
        "result_path": str(result_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "gpu_preflight": snapshot.to_dict(),
        "result": result,
    }
