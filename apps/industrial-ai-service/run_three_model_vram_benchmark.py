"""Sequentially benchmark the three project models with nvidia-smi sampling.

This script creates a dedicated versioned RUN directory, refuses to start when
another compute process is present, and never runs two models concurrently.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
SOURCE_RUN = RUNS_ROOT / "RUN-20260828-101957-fe2c9e"
SOURCE_MASK = SOURCE_RUN / "masks/candidate_02_seed_20260851_rgba_auto_v2.png"
RUN_ID_PATTERN = re.compile(r"RUN-\d{8}-\d{6}-[0-9a-f]{6}")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def display_command_item(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        # Do not resolve symlinks here: a virtual-environment Python executable
        # may resolve to /usr/bin even though the invoked path is project-local.
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return value


def nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def gpu_snapshot() -> dict[str, int]:
    line = nvidia_smi(
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).strip().splitlines()[0]
    values = [int(part.strip()) for part in line.split(",")]
    return {
        "total_mib": values[0],
        "used_mib": values[1],
        "free_mib": values[2],
        "utilization_percent": values[3],
    }


def compute_processes() -> list[dict[str, int]]:
    output = nvidia_smi(
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    records: list[dict[str, int]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        try:
            used = int(fields[1])
        except ValueError:
            continue
        records.append({"pid": int(fields[0]), "used_mib": used})
    return records


def process_rss_mib(pid: int) -> float | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


def preflight(label: str, report_dir: Path) -> dict[str, Any]:
    full = nvidia_smi()
    (report_dir / f"{label}_nvidia_smi_preflight.txt").write_text(full, encoding="utf-8")
    processes = compute_processes()
    if processes:
        raise RuntimeError(f"{label} 启动前发现其他 GPU 计算进程，拒绝启动。")
    snapshot = gpu_snapshot()
    snapshot["time"] = now()
    print(
        f"[{label}] preflight: used={snapshot['used_mib']} MiB, "
        f"free={snapshot['free_mib']} MiB, compute_processes=0",
        flush=True,
    )
    return snapshot


def wait_for_gpu_release(label: str, timeout_seconds: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        processes = compute_processes()
        if not processes:
            snapshot = gpu_snapshot()
            snapshot["time"] = now()
            print(f"[{label}] GPU released: used={snapshot['used_mib']} MiB", flush=True)
            return snapshot
        time.sleep(0.5)
    raise RuntimeError(f"{label} 进程结束后 GPU 计算进程未在 {timeout_seconds} 秒内释放。")


def monitor_command(
    *,
    label: str,
    command: list[str],
    log_path: Path,
    report_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    before = preflight(label, report_path.parent)
    started_wall = now()
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    peak_device_used = before["used_mib"]
    peak_compute_used = 0
    peak_rss = 0.0
    last_console = -10.0
    last_saved = -1.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            snapshot = gpu_snapshot()
            compute = compute_processes()
            compute_used = sum(record["used_mib"] for record in compute)
            rss = process_rss_mib(process.pid)
            peak_device_used = max(peak_device_used, snapshot["used_mib"])
            peak_compute_used = max(peak_compute_used, compute_used)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
            if elapsed - last_saved >= 1.0:
                samples.append(
                    {
                        "elapsed_seconds": round(elapsed, 3),
                        "device_used_mib": snapshot["used_mib"],
                        "device_free_mib": snapshot["free_mib"],
                        "gpu_utilization_percent": snapshot["utilization_percent"],
                        "compute_process_used_mib": compute_used,
                        "worker_rss_mib": round(rss, 1) if rss is not None else None,
                    }
                )
                last_saved = elapsed
            if elapsed - last_console >= 10.0:
                print(
                    f"[{label}] {elapsed:.0f}s, device={snapshot['used_mib']} MiB, "
                    f"compute={compute_used} MiB, util={snapshot['utilization_percent']}%",
                    flush=True,
                )
                last_console = elapsed
            time.sleep(0.25)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    after = wait_for_gpu_release(label)
    torch_result = None
    if result_path.is_file():
        torch_result = json.loads(result_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": "1.0",
        "model": label,
        "started_at": started_wall,
        "completed_at": now(),
        "elapsed_seconds": round(elapsed, 3),
        "return_code": return_code,
        "command": [display_command_item(item) for item in command],
        "preflight": before,
        "postflight": after,
        "sampling_interval_seconds": 0.25,
        "nvidia_smi": {
            "peak_device_used_mib": peak_device_used,
            "baseline_device_used_mib": before["used_mib"],
            "peak_device_increase_mib": peak_device_used - before["used_mib"],
            "peak_compute_process_used_mib": peak_compute_used,
        },
        "worker_peak_rss_mib": round(peak_rss, 1),
        "torch_metrics": (
            {
                "peak_allocated_gib": torch_result.get("peak_allocated_gib"),
                "peak_reserved_gib": torch_result.get("peak_reserved_gib"),
            }
            if isinstance(torch_result, dict)
            else None
        ),
        "samples_1s": samples,
        "log_path": project_relative(log_path),
        "result_path": project_relative(result_path),
    }
    write_json(report_path, report)
    if return_code != 0:
        raise RuntimeError(f"{label} 运行失败，退出码 {return_code}；查看 {log_path}")
    print(
        f"[{label}] completed: nvidia-smi compute peak={peak_compute_used} MiB, "
        f"device peak={peak_device_used} MiB, worker RSS peak={peak_rss:.1f} MiB",
        flush=True,
    )
    return report


def artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": project_relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="三个模型顺序显存基准")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run-id 格式无效。")
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有任务：{run_dir}")
    for directory in ("requests", "images", "masks", "meshes", "reports", "logs"):
        (run_dir / directory).mkdir(parents=True, exist_ok=False)

    source_manifest = json.loads((SOURCE_RUN / "manifest.json").read_text(encoding="utf-8"))
    prompt = source_manifest["prompt"]["text"]
    copied_mask = run_dir / "masks/flange_confirmed_rgba_v1.png"
    shutil.copy2(SOURCE_MASK, copied_mask)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at": now(),
        "updated_at": now(),
        "status": "VRAM_BENCHMARK_RUNNING",
        "mode": "three-model-sequential-vram-benchmark-v1",
        "source_run": SOURCE_RUN.name,
        "safety": {"models_sequential": True, "preflight_before_each": True},
        "stages": {"ernie": "PENDING", "qwen": "PENDING", "hunyuan": "PENDING"},
        "artifacts": [],
        "events": [{"time": now(), "event": "BENCHMARK_CREATED", "detail": "开始三个模型顺序显存复测。"}],
    }
    write_json(manifest_path, manifest)

    ernie_request_path = run_dir / "requests/ernie_benchmark_v1.json"
    ernie_result_path = run_dir / "reports/ernie_result_v1.json"
    ernie_output_dir = run_dir / "images/ernie_benchmark_v1"
    write_json(
        ernie_request_path,
        {
            "model_path": "models/ernie-image-turbo-ms",
            "output_dir": project_relative(ernie_output_dir),
            "result_path": project_relative(ernie_result_path),
            "prompt": prompt,
            "seeds": [20260870],
            "width": 768,
            "height": 768,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "use_pe": True,
        },
    )
    ernie_report_path = run_dir / "reports/ernie_vram_monitor_v1.json"
    ernie_report = monitor_command(
        label="ERNIE-Image-Turbo",
        command=[
            str(PROJECT_ROOT / "apps/ernie-image/.venv/bin/python"),
            str(PROJECT_ROOT / "apps/ernie-image/run_ernie_service_task.py"),
            "--request",
            str(ernie_request_path),
        ],
        log_path=run_dir / "logs/ernie_benchmark_v1.log",
        report_path=ernie_report_path,
        result_path=ernie_result_path,
    )
    manifest["stages"]["ernie"] = "COMPLETED"
    manifest["updated_at"] = now()
    manifest["events"].append({"time": now(), "event": "ERNIE_COMPLETED", "detail": "ERNIE 显存采样完成。"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ernie_result = json.loads(ernie_result_path.read_text(encoding="utf-8"))
    generated_image = PROJECT_ROOT / ernie_result["outputs"][0]["path"]
    qwen_request_path = run_dir / "requests/qwen_benchmark_v1.json"
    qwen_result_path = run_dir / "reports/qwen_result_v1.json"
    write_json(
        qwen_request_path,
        {
            "model_path": "models/qwen3-vl-8b-instruct-ms",
            "image_path": project_relative(generated_image),
            "result_path": project_relative(qwen_result_path),
            "part_type": "圆形法兰",
            "expected_repeat_count": 6,
            "counted_feature": "均布螺栓孔",
            "visual_checks": ["中心孔贯通", "螺栓孔均匀", "主体完整", "背景干净"],
            "required_structure": source_manifest["request"]["requirement"],
            "forbidden_elements": source_manifest["request"]["forbidden_elements"],
            "image_max_size": 768,
            "max_new_tokens": 512,
        },
    )
    qwen_report_path = run_dir / "reports/qwen_vram_monitor_v1.json"
    qwen_report = monitor_command(
        label="Qwen3-VL-8B-Instruct",
        command=[
            str(PROJECT_ROOT / "apps/qwen3-vl/.venv/bin/python"),
            str(PROJECT_ROOT / "apps/qwen3-vl/run_qwen_service_review.py"),
            "--request",
            str(qwen_request_path),
        ],
        log_path=run_dir / "logs/qwen_benchmark_v1.log",
        report_path=qwen_report_path,
        result_path=qwen_result_path,
    )
    manifest["stages"]["qwen"] = "COMPLETED"
    manifest["updated_at"] = now()
    manifest["events"].append({"time": now(), "event": "QWEN_COMPLETED", "detail": "Qwen 显存采样完成。"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hunyuan_request_path = run_dir / "requests/hunyuan_benchmark_v1.json"
    hunyuan_result_path = run_dir / "reports/hunyuan_result_v1.json"
    hunyuan_output_path = run_dir / "meshes/hunyuan_benchmark_v1_raw.glb"
    write_json(
        hunyuan_request_path,
        {
            "model_path": "models/hunyuan3d-2.1-shape-ms",
            "image_path": project_relative(copied_mask),
            "image_sha256": sha256(copied_mask),
            "output_path": project_relative(hunyuan_output_path),
            "result_path": project_relative(hunyuan_result_path),
            "seed": 1234,
            "num_inference_steps": 30,
            "octree_resolution": 256,
            "num_chunks": 8000,
            "dtype": "float16",
            "low_vram_mode": "model_cpu_offload",
        },
    )
    hunyuan_report_path = run_dir / "reports/hunyuan_vram_monitor_v1.json"
    hunyuan_report = monitor_command(
        label="Hunyuan3D-2.1-Shape",
        command=[
            str(PROJECT_ROOT / "apps/Hunyuan3D-2.1/.venv/bin/python"),
            str(PROJECT_ROOT / "apps/Hunyuan3D-2.1/hy3dshape/run_hunyuan_service_task.py"),
            "--request",
            str(hunyuan_request_path),
        ],
        log_path=run_dir / "logs/hunyuan_benchmark_v1.log",
        report_path=hunyuan_report_path,
        result_path=hunyuan_result_path,
    )
    manifest["stages"]["hunyuan"] = "COMPLETED"

    summary = {
        "schema_version": "1.0",
        "created_at": now(),
        "run_id": args.run_id,
        "measurement_scope": {
            "nvidia_smi": "250 ms sampling; device total and all compute-process memory",
            "torch": "worker-reported max_memory_allocated/max_memory_reserved",
            "cpu": "worker /proc VmRSS sampling; indicative, not whole-system memory",
        },
        "models_sequential": True,
        "models": {
            "ernie": ernie_report,
            "qwen": qwen_report,
            "hunyuan": hunyuan_report,
        },
    }
    summary_path = run_dir / "reports/three_model_vram_summary_v1.json"
    write_json(summary_path, summary)
    paths = [
        ernie_request_path,
        ernie_result_path,
        ernie_report_path,
        generated_image,
        qwen_request_path,
        qwen_result_path,
        qwen_report_path,
        copied_mask,
        hunyuan_request_path,
        hunyuan_result_path,
        hunyuan_report_path,
        hunyuan_output_path,
        summary_path,
    ]
    manifest["artifacts"] = [artifact(path, path.stem) for path in paths]
    manifest["status"] = "VRAM_BENCHMARK_COMPLETED"
    manifest["updated_at"] = now()
    manifest["events"].append({"time": now(), "event": "BENCHMARK_COMPLETED", "detail": "三个模型严格顺序复跑并完成显存采样。"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "summary": project_relative(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
