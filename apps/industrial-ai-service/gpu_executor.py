"""Shared-GPU preflight and a project-local non-blocking execution lock."""

from __future__ import annotations

import fcntl
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


class GpuUnavailableError(RuntimeError):
    """Raised when a safe model launch cannot be established."""


@dataclass(frozen=True)
class GpuSnapshot:
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    active_compute_process_count: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as exc:
        raise GpuUnavailableError("系统中找不到 nvidia-smi。") from exc
    except subprocess.TimeoutExpired as exc:
        raise GpuUnavailableError("nvidia-smi 检查超时。") from exc
    except subprocess.CalledProcessError as exc:
        raise GpuUnavailableError("nvidia-smi 检查失败。") from exc


def inspect_gpu() -> GpuSnapshot:
    # Project policy requires the standard overview command before every model.
    _run(["nvidia-smi"])

    gpu_result = _run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    lines = [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GpuUnavailableError("无法唯一识别 GPU 0。")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise GpuUnavailableError("无法解析 GPU 显存状态。")

    process_result = _run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    process_count = len(
        [line for line in process_result.stdout.splitlines() if line.strip()]
    )
    try:
        return GpuSnapshot(
            name=fields[0],
            total_mib=int(fields[1]),
            used_mib=int(fields[2]),
            free_mib=int(fields[3]),
            active_compute_process_count=process_count,
        )
    except ValueError as exc:
        raise GpuUnavailableError("GPU 显存状态不是有效数字。") from exc


def require_available_gpu(minimum_free_mib: int) -> GpuSnapshot:
    snapshot = inspect_gpu()
    if snapshot.active_compute_process_count:
        raise GpuUnavailableError(
            "检测到其他 GPU 计算进程。系统不会启动模型，也不会干扰该进程。"
        )
    if snapshot.free_mib < minimum_free_mib:
        raise GpuUnavailableError(
            f"GPU 可用显存为 {snapshot.free_mib} MiB，"
            f"低于本任务要求的 {minimum_free_mib} MiB。"
        )
    return snapshot


@contextmanager
def project_gpu_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GpuUnavailableError(
                "本服务已有一个 GPU 模型任务正在运行，请等待完成后重试。"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
