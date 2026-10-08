"""Gradio prototype for the industrial AI workflow.

This process deliberately does not import torch or a model package. ERNIE runs
on demand in its own validated virtual environment through a guarded subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from assembly_planning import (
    ASSEMBLY_DIRECTIONS,
    assembly_defaults,
    assembly_plan_summary,
    build_assembly_plan,
)
from assembly_component import (
    INSERTION_RESULTS,
    REMOVAL_RESULTS,
    component_bom_rows,
    component_summary,
    fit_review_summary,
    load_component_bundle,
    save_fit_review as archive_assembly_fit_review,
)
from assembly_autoplanner import (
    ASSEMBLY_PURPOSES,
    FIT_GOALS,
    PRINT_PROFILES,
    auto_plan_summary,
    create_auto_planned_component,
)
from device_center import (
    DEVICE_SCENARIOS,
    build_device_snapshot,
    device_snapshot_summary,
)
from ernie_adapter import ErnieExecutionError, run_ernie
from gpu_executor import GpuUnavailableError
from gear_analysis_adapter import GearAnalysisError, run_gear_analysis
from hunyuan_adapter import (
    HunyuanExecutionError,
    run_hunyuan,
    run_hunyuan_multiview,
)
from mesh_optimization_adapter import (
    MeshOptimizationError,
    run_regularization,
    run_stl_export,
)
from mesh_import_adapter import MeshImportError, archive_and_inspect_glb
from mesh_repair_adapter import (
    MeshRepairExecutionError,
    recommended_mode,
    run_mesh_repair,
)
from part_profiles import (
    PROFILE_CHOICES,
    build_profile_prompt,
    profile_defaults,
    profile_manifest,
    resolve_part_profile,
)
from preprocessing_adapter import (
    PreprocessingError,
    run_mask_preparation,
    run_mesh_review_preparation,
)
from qwen_adapter import QwenExecutionError, run_qwen_review
from multimodal_capture import (
    CAPTURE_CHECKPOINTS,
    CAPTURE_LABELS,
    CAPTURE_ZONES,
    archive_manual_capture,
    capture_summary,
    capture_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT_ROOT / "workspace"
RUNS_DIR = WORKSPACE / "runs"

DEMO_IMAGE = WORKSPACE / "images/ducted_fan_candidate_v4_seed_20260834.png"
DEMO_MASK = (
    WORKSPACE
    / "meshes/ducted_fan_v4_seed_20260834_input_rgba_fixed_v3_strict.png"
)
DEMO_GLB = (
    WORKSPACE
    / "meshes/ducted_fan_v4_seed_20260834_main_review_v1.glb"
)
DEMO_STL = (
    WORKSPACE
    / "stl/ducted_fan_v4_60mm_onepiece_print_candidate_v1.stl"
)
DEMO_REPORT = WORKSPACE / "docs/industrial_ai_phase1_validation_report_v1.md"
PRINT_PHOTOS = [
    WORKSPACE / "images/print_results/ducted_fan_print_01.jpg",
    WORKSPACE / "images/print_results/ducted_fan_print_02.jpg",
]

RUN_ID_PATTERN = re.compile(r"^RUN-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
STAGES = {
    "prompt": "提示词已确认",
    "candidate": "候选图片已确认",
    "mask": "透明遮罩已确认",
    "mesh": "已检查三维模型已确认",
}

STATUS_LABELS = {
    "DRAFT": "等待确认提示词",
    "PROMPT_CONFIRMED": "提示词已确认",
    "ERNIE_QUEUED": "文生图排队中",
    "ERNIE_COMPLETED": "候选图已生成",
    "QWEN_QUEUED": "图片检查排队中",
    "QWEN_PASS": "Qwen 通过，等待人工确认",
    "CANDIDATE_NEEDS_REVIEW": "候选图需要复核",
    "CANDIDATE_REJECTED": "候选图已拦截",
    "CANDIDATE_CONFIRMED": "候选图已确认",
    "MASK_REVIEW_READY": "遮罩等待人工确认",
    "MASK_CONFIRMED": "遮罩已确认",
    "HUNYUAN_COMPLETED": "原始三维已生成",
    "MESH_REVIEW_READY": "三维检查结果待确认",
    "EXTERNAL_MESH_IMPORTED_NEEDS_REPAIR": "外部 GLB 已归档，等待诊断修复",
    "REGULARIZED_MESH_REVIEW_READY": "规则化三维待确认",
    "MESH_CONFIRMED": "三维模型已确认",
    "ASSEMBLY_PLAN_REVIEW_READY": "装配方案等待人工确认",
    "ASSEMBLY_PLAN_CONFIRMED": "装配方案已确认，等待仿真与设备资料",
    "WAITING_GPU": "等待 GPU",
}

STAGE_STATUS_LABELS = {
    "NOT_STARTED": "尚未开始",
    "READY": "可以开始",
    "WAITING_CONFIRMATION": "等待人工确认",
    "WAITING_REVIEW": "等待网格检查",
    "QWEN_PASS_WAITING_HUMAN": "Qwen 已通过，等待人工确认",
    "QWEN_REVIEW": "需要人工复核",
    "QWEN_REJECTED": "Qwen 已拦截",
    "MANUAL_REJECTED": "已人工拒绝",
    "CONFIRMED": "已确认",
    "PLAN_CONFIRMED_SIMULATION_REQUIRED": "草案已确认，仍需仿真",
}

APP_CSS = """
.status-strip {padding: 10px 14px; border-radius: 10px; background: #eef6ff;}
.warning-strip {padding: 10px 14px; border-radius: 10px; background: #fff7e6;}
.small-note {font-size: 0.92rem; color: #52606d;}
"""

DEMO_REVIEW = """### 历史 Qwen/人工检查摘要

- 主体完整，约 7 片叶片，背景简洁。
- 初始背景遮罩曾错误填满内部孔洞，可能导致三维大型薄板。
- 严格 Alpha 遮罩修复后，内部孔洞保持透明。
- 此处只展示已有检查结果，**没有运行 Qwen 模型**。
"""

DEMO_MESH_REPORT = """### 已验证网格

| 检查项 | 结果 |
|---|---|
| 顶点 | 285,030 |
| 三角面 | 570,112 |
| 连通部件 | 1 |
| 水密 | 是 |
| 法向一致 | 是 |
| 正体积 | 是 |
| 60 mm STL 尺寸 | 59.891 × 60.000 × 31.714 mm |

用途限制：静态展示原型，不用于电机驱动或高速旋转。
"""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_demo_assets() -> list[Path]:
    assets = [DEMO_IMAGE, DEMO_MASK, DEMO_GLB, DEMO_STL, DEMO_REPORT, *PRINT_PHOTOS]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少演示文件：" + "；".join(missing))
    return assets


def build_prompt(
    part_type: str,
    purpose: str,
    requirement: str,
    structure_count: float,
    forbidden_elements: str = "",
) -> str:
    prompt, _, _ = build_profile_prompt(
        part_type,
        purpose,
        requirement,
        structure_count,
        forbidden_elements,
    )
    return prompt


def new_run_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"RUN-{stamp}-{uuid.uuid4().hex[:6]}"


def manifest_path(run_id: str) -> Path:
    normalized_run_id = (run_id or "").strip()
    if not RUN_ID_PATTERN.fullmatch(normalized_run_id):
        raise ValueError("任务 ID 无效，请先在“创建与提示词”页面创建任务。")
    path = (RUNS_DIR / normalized_run_id / "manifest.json").resolve()
    if RUNS_DIR.resolve() not in path.parents:
        raise ValueError("任务路径越界。")
    return path


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_manifest(run_id: str) -> tuple[Path, dict[str, Any]]:
    path = manifest_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"没有找到任务：{run_id}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def append_event(manifest: dict[str, Any], event: str, detail: str) -> None:
    manifest.setdefault("events", []).append(
        {"time": now_iso(), "event": event, "detail": detail}
    )
    manifest["updated_at"] = now_iso()


def create_run(
    part_type: str,
    purpose: str,
    requirement: str,
    structure_count: float,
    target_size_mm: float,
    seed: float,
    forbidden_elements: str = "",
) -> tuple[str, str, str, dict[str, Any]]:
    run_id = new_run_id()
    prompt, profile, count = build_profile_prompt(
        part_type,
        purpose,
        requirement,
        structure_count,
        forbidden_elements,
    )
    run_dir = RUNS_DIR / run_id
    for name in (
        "requests",
        "prompts",
        "images",
        "reviews",
        "masks",
        "meshes",
        "stl",
        "reports",
        "logs",
        "assembly",
        "devices",
        "multimodal",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    created = now_iso()
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created,
        "updated_at": created,
        "status": "DRAFT",
        "mode": "service-prototype-ernie-qwen-mask-hunyuan-on-demand",
        "request": {
            "part_type": part_type,
            "purpose": purpose,
            "requirement": requirement.strip(),
            "forbidden_elements": forbidden_elements.strip(),
            "structure_count": count,
            "target_size_mm": float(target_size_mm),
            "part_profile": profile_manifest(profile),
        },
        "prompt": {"text": prompt, "confirmed": False},
        "parameters": {"seed": int(seed)},
        "stages": {
            "prompt": "WAITING_CONFIRMATION",
            "candidate": "NOT_STARTED",
            "mask": "NOT_STARTED",
            "mesh": "NOT_STARTED",
            "assembly": "NOT_STARTED",
        },
        "models": {
            "ernie": {
                "name": "ERNIE-Image",
                "variant": "ernie-image",
                "status": "READY",
                "execution": "guarded_subprocess",
                "low_vram_mode": "sequential_cpu_offload",
            },
            "qwen": {
                "status": "READY",
                "execution": "guarded_subprocess",
                "quantization": "NF4_4bit_double_quant",
            },
            "hunyuan": {
                "interface_name": "Hunyuan3D-2.1-compatible",
                "single_view_backend": "Hunyuan3D-2.1",
                "multi_view_backend": "Hunyuan3D-2mv",
                "status": "READY",
                "execution": "guarded_subprocess",
                "low_vram_mode": "model_cpu_offload",
            },
        },
        "artifacts": [],
        "candidate_reviews": [],
        "metrics": {},
        "events": [
            {
                "time": created,
                "event": "RUN_CREATED",
                "detail": "由服务原型创建任务，ERNIE 可按需运行。",
            }
        ],
    }
    path = run_dir / "manifest.json"
    write_manifest(path, manifest)
    status = f"✅ 已创建 `{run_id}`。当前等待确认最终提示词。"
    return run_id, prompt, status, manifest


def confirm_prompt(run_id: str, prompt: str) -> tuple[str, dict[str, Any]]:
    if not prompt.strip():
        raise ValueError("最终提示词不能为空。")
    path, manifest = read_manifest(run_id)
    manifest["prompt"] = {
        "text": prompt.strip(),
        "confirmed": True,
        "confirmed_at": now_iso(),
    }
    manifest["stages"]["prompt"] = "CONFIRMED"
    manifest["stages"]["candidate"] = "READY"
    manifest["status"] = "PROMPT_CONFIRMED"
    append_event(manifest, "PROMPT_CONFIRMED", "用户确认最终提示词。")
    write_manifest(path, manifest)
    return "✅ 提示词已确认。可进入候选图片页面启动 ERNIE。", manifest


def generate_ernie_candidates(
    run_id: str,
    candidate_count: float,
    resolution: float = 1024,
) -> tuple[str, dict[str, Any], list[tuple[str, str]]]:
    path, manifest = read_manifest(run_id)
    prompt_data = manifest.get("prompt", {})
    if not prompt_data.get("confirmed"):
        return "⚠️ 请先确认最终提示词，再启动 ERNIE。", manifest, []

    count = int(candidate_count)
    image_size = int(resolution)
    if image_size not in {512, 768, 1024}:
        return "⚠️ 候选分辨率只允许 512、768 或 1024。", manifest, []
    manifest["status"] = "ERNIE_QUEUED"
    manifest["models"]["ernie"]["status"] = "QUEUED"
    append_event(
        manifest,
        "ERNIE_QUEUED",
        f"请求生成 {count} 张 {image_size}×{image_size} 候选图片。",
    )
    write_manifest(path, manifest)

    try:
        execution = run_ernie(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            prompt=prompt_data["text"],
            base_seed=int(manifest["parameters"]["seed"]),
            candidate_count=count,
            width=image_size,
            height=image_size,
        )
    except GpuUnavailableError as exc:
        manifest["status"] = "WAITING_GPU"
        manifest["models"]["ernie"]["status"] = "WAITING_GPU"
        append_event(manifest, "ERNIE_WAITING_GPU", str(exc))
        write_manifest(path, manifest)
        return f"⏳ ERNIE 未启动：{exc}", manifest, []
    except (ErnieExecutionError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "ERNIE_FAILED"
        manifest["models"]["ernie"]["status"] = "FAILED"
        append_event(manifest, "ERNIE_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ ERNIE 执行失败：{exc} 已保留请求和日志。", manifest, []

    result = execution["result"]
    gallery: list[tuple[str, str]] = []
    for output in result["outputs"]:
        absolute = PROJECT_ROOT / output["path"]
        gallery.append((str(absolute), f"候选 {output['index']} · seed {output['seed']}"))
        manifest["artifacts"].append(
            {
                "role": "ernie_candidate",
                "attempt": execution["attempt"],
                "index": output["index"],
                "seed": output["seed"],
                "path": output["path"],
                "bytes": output["bytes"],
                "sha256": output["sha256"],
            }
        )

    manifest["artifacts"].extend(
        [
            {"role": "ernie_request", "path": execution["request_path"]},
            {"role": "ernie_result", "path": execution["result_path"]},
            {"role": "ernie_log", "path": execution["log_path"]},
        ]
    )
    manifest.setdefault("metrics", {}).setdefault("ernie_attempts", []).append(
        {
            "attempt": execution["attempt"],
            "candidate_count": len(result["outputs"]),
            "width": image_size,
            "height": image_size,
            "load_seconds": result["load_seconds"],
            "total_seconds": result["total_seconds"],
            "peak_allocated_gib": result["peak_allocated_gib"],
            "peak_reserved_gib": result["peak_reserved_gib"],
            "model": result.get("model", "ERNIE-Image"),
            "gpu_preflight": execution["gpu_preflight"],
        }
    )
    manifest["models"]["ernie"]["status"] = "COMPLETED"
    manifest["stages"]["candidate"] = "WAITING_CONFIRMATION"
    manifest["status"] = "ERNIE_COMPLETED"
    append_event(
        manifest,
        "ERNIE_COMPLETED",
        f"第 {execution['attempt']} 次尝试生成 {len(gallery)} 张候选图片。",
    )
    write_manifest(path, manifest)
    status = (
        f"✅ ERNIE 已生成 {len(gallery)} 张 {image_size}×{image_size} 候选图；"
        f"耗时 {result['total_seconds']:.1f} 秒，"
        f"峰值显存 {result['peak_allocated_gib']:.2f} GiB。"
    )
    return status, manifest, gallery


def candidate_for_review(
    manifest: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    candidates = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("role") == "ernie_candidate"
    ]
    if not candidates:
        raise ValueError("当前任务还没有 ERNIE 候选图片。")
    latest_attempt = max(int(item.get("attempt", 0)) for item in candidates)
    latest = [item for item in candidates if int(item.get("attempt", 0)) == latest_attempt]
    latest.sort(key=lambda item: (int(item.get("index", 0)), int(item.get("seed", 0))))
    if not 1 <= candidate_index <= len(latest):
        raise ValueError(f"候选编号应为 1 到 {len(latest)}。")
    return latest[candidate_index - 1]


def candidate_review_for(
    manifest: dict[str, Any], candidate_path: str
) -> dict[str, Any]:
    for review in manifest.get("candidate_reviews", []):
        if review.get("candidate_path") == candidate_path:
            return dict(review)
    legacy = manifest.get("candidate_review", {})
    if legacy.get("candidate_path") == candidate_path:
        return dict(legacy)
    return {}


def store_candidate_review(
    manifest: dict[str, Any], review: dict[str, Any]
) -> None:
    stored = dict(review)
    candidate_path = stored.get("candidate_path")
    reviews = manifest.setdefault("candidate_reviews", [])
    if candidate_path:
        for index, existing in enumerate(reviews):
            if existing.get("candidate_path") == candidate_path:
                reviews[index] = stored
                break
        else:
            reviews.append(stored)
    manifest["candidate_review"] = stored


def review_candidate_with_qwen(
    run_id: str,
    candidate_index: float,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    path, manifest = read_manifest(run_id)
    try:
        candidate = candidate_for_review(manifest, int(candidate_index))
    except ValueError as exc:
        return f"⚠️ {exc}", manifest, {}

    image_path = (PROJECT_ROOT / candidate["path"]).resolve()
    manifest["status"] = "QWEN_QUEUED"
    manifest["models"]["qwen"]["status"] = "QUEUED"
    append_event(
        manifest,
        "QWEN_QUEUED",
        f"请求检查候选 {int(candidate_index)}：{candidate['path']}",
    )
    write_manifest(path, manifest)

    request_data = manifest.get("request", {})
    part_name = str(request_data.get("part_type", "工业零部件"))
    resolved_profile = resolve_part_profile(part_name)
    stored_profile = request_data.get("part_profile", {})
    if not isinstance(stored_profile, dict):
        stored_profile = {}
    stored_count = int(request_data.get("structure_count", 0) or 0)
    expected_count = (
        stored_count
        if resolved_profile.count_required or stored_count > 0
        else None
    )
    try:
        execution = run_qwen_review(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            image_path=image_path,
            part_type=part_name,
            expected_count=expected_count,
            counted_feature=str(
                stored_profile.get("count_feature", resolved_profile.count_feature)
            ),
            visual_checks=list(
                stored_profile.get("visual_checks", resolved_profile.visual_checks)
            ),
            required_structure=str(request_data.get("requirement", "")),
            forbidden_elements=str(request_data.get("forbidden_elements", "")),
        )
    except GpuUnavailableError as exc:
        manifest["status"] = "WAITING_GPU"
        manifest["models"]["qwen"]["status"] = "WAITING_GPU"
        append_event(manifest, "QWEN_WAITING_GPU", str(exc))
        write_manifest(path, manifest)
        return f"⏳ Qwen 未启动：{exc}", manifest, {}
    except (QwenExecutionError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "QWEN_FAILED"
        manifest["models"]["qwen"]["status"] = "FAILED"
        append_event(manifest, "QWEN_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ Qwen 检查失败：{exc} 已保留请求和日志。", manifest, {}

    result = execution["result"]
    review = result["review"]
    manifest["artifacts"].extend(
        [
            {
                "role": "qwen_candidate_review",
                "review_number": execution["review_number"],
                "candidate_path": candidate["path"],
                "decision": review["decision"],
                "path": execution["result_path"],
            },
            {"role": "qwen_request", "path": execution["request_path"]},
            {"role": "qwen_log", "path": execution["log_path"]},
        ]
    )
    manifest.setdefault("metrics", {}).setdefault("qwen_reviews", []).append(
        {
            "review_number": execution["review_number"],
            "candidate_path": candidate["path"],
            "decision": review["decision"],
            "load_seconds": result["load_seconds"],
            "inference_seconds": result["inference_seconds"],
            "total_seconds": result["total_seconds"],
            "peak_allocated_gib": result["peak_allocated_gib"],
            "peak_reserved_gib": result["peak_reserved_gib"],
            "gpu_preflight": execution["gpu_preflight"],
        }
    )
    previous_human: dict[str, Any] = {}
    previous_review = candidate_review_for(manifest, candidate["path"])
    if previous_review:
        previous_human = {
            key: previous_review[key]
            for key in (
                "human_confirmed",
                "human_decision",
                "human_reason",
                "human_decided_at",
                "human_confirmed_at",
            )
            if key in previous_review
        }
    previous_human_decision = previous_human.get("human_decision")
    current_review = {
        "candidate_path": candidate["path"],
        "candidate_index": int(candidate_index),
        "decision": review["decision"],
        "result_path": execution["result_path"],
        "reviewed_at": now_iso(),
        "human_confirmed": False,
        **previous_human,
    }
    store_candidate_review(manifest, current_review)
    manifest["models"]["qwen"]["status"] = "COMPLETED"
    if review["decision"] == "REJECT":
        manifest["stages"]["candidate"] = "QWEN_REJECTED"
        manifest["status"] = "CANDIDATE_REJECTED"
    elif review["decision"] == "REVIEW":
        manifest["stages"]["candidate"] = "QWEN_REVIEW"
        manifest["status"] = "CANDIDATE_NEEDS_REVIEW"
    else:
        manifest["stages"]["candidate"] = "QWEN_PASS_WAITING_HUMAN"
        manifest["status"] = "QWEN_PASS"
    if previous_human_decision == "REJECT":
        manifest["stages"]["candidate"] = "MANUAL_REJECTED"
        manifest["status"] = "CANDIDATE_REJECTED"
    elif previous_human_decision == "PASS":
        manifest["stages"]["candidate"] = "CONFIRMED"
        manifest["status"] = "CANDIDATE_CONFIRMED"
    append_event(
        manifest,
        "QWEN_COMPLETED",
        f"候选 {int(candidate_index)} 的结构化检查等级为 {review['decision']}。",
    )
    write_manifest(path, manifest)
    status = (
        f"✅ Qwen 检查完成：**{review['decision']}**；"
        f"耗时 {result['total_seconds']:.1f} 秒，"
        f"峰值显存 {result['peak_allocated_gib']:.2f} GiB。"
    )
    if previous_human_decision == "REJECT":
        status += " 该候选已有人工拒绝记录，仍禁止进入遮罩步骤。"
    elif previous_human_decision == "PASS":
        status += " 该候选已由人工确认，Qwen 结果仅供参考，仍可继续遮罩步骤。"
    else:
        status += " Qwen 结果仅供参考；请人工复核后决定确认或拒绝候选。"
    return status, manifest, review


def reject_candidate_manually(
    run_id: str,
    candidate_index: float,
    reason: str,
) -> tuple[str, dict[str, Any]]:
    path, manifest = read_manifest(run_id)
    try:
        candidate = candidate_for_review(manifest, int(candidate_index))
    except ValueError as exc:
        return f"⚠️ {exc}", manifest
    reason = reason.strip()
    if not reason:
        return "⚠️ 请填写人工拒绝原因，便于追溯和改进提示词。", manifest

    current = candidate_review_for(manifest, candidate["path"])
    if not current:
        current = {
            "candidate_path": candidate["path"],
            "candidate_index": int(candidate_index),
            "decision": "NOT_RUN",
            "result_path": None,
            "reviewed_at": None,
        }
    current.update(
        {
            "human_confirmed": False,
            "human_decision": "REJECT",
            "human_reason": reason,
            "human_decided_at": now_iso(),
        }
    )
    store_candidate_review(manifest, current)
    manifest["stages"]["candidate"] = "MANUAL_REJECTED"
    manifest["status"] = "CANDIDATE_REJECTED"
    append_event(
        manifest,
        "CANDIDATE_MANUALLY_REJECTED",
        f"候选 {int(candidate_index)}：{reason}",
    )
    write_manifest(path, manifest)
    return "⛔ 已记录人工拒绝。该候选不能进入遮罩或三维步骤。", manifest


def confirm_candidate_manually(
    run_id: str,
    candidate_index: float,
) -> tuple[str, dict[str, Any]]:
    path, manifest = read_manifest(run_id)
    try:
        candidate = candidate_for_review(manifest, int(candidate_index))
    except ValueError as exc:
        return f"⚠️ {exc}", manifest
    review = candidate_review_for(manifest, candidate["path"])
    if not review:
        review = {
            "candidate_path": candidate["path"],
            "candidate_index": int(candidate_index),
            "decision": "NOT_RUN",
            "result_path": None,
            "reviewed_at": None,
        }
    if review.get("human_decision") == "REJECT":
        return "⛔ 该候选已被人工拒绝，不能重新确认进入后续步骤。", manifest

    review["human_confirmed"] = True
    review["human_decision"] = "PASS"
    review["human_confirmed_at"] = now_iso()
    store_candidate_review(manifest, review)
    qwen_decision = review.get("decision", "NOT_RUN")
    manifest["selected_candidate"] = {
        "candidate_index": int(candidate_index),
        "path": candidate["path"],
        "selected_at": now_iso(),
        "qwen_decision_at_confirmation": qwen_decision,
    }
    manifest["stages"]["candidate"] = "CONFIRMED"
    manifest["status"] = "CANDIDATE_CONFIRMED"
    append_event(
        manifest,
        "CANDIDATE_CONFIRMED",
        f"人工确认候选 {int(candidate_index)}，允许进入遮罩步骤；"
        f"Qwen 等级 {qwen_decision} 仅供参考。",
    )
    write_manifest(path, manifest)
    return (
        "✅ 已人工确认指定候选。流程允许进入遮罩步骤。"
        f"Qwen 等级：{qwen_decision}（仅供参考，不作为门禁）。",
        manifest,
    )


def confirm_stage(run_id: str, stage: str) -> tuple[str, dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError("未知确认阶段。")
    path, manifest = read_manifest(run_id)
    if stage == "candidate":
        review = manifest.get("candidate_review")
        if not review:
            return "⚠️ 请使用候选编号旁的“人工确认候选通过”按钮。", manifest
        if review.get("human_decision") == "REJECT":
            return "⛔ 该候选已被人工拒绝，不能重新确认进入后续步骤。", manifest
        review["human_confirmed"] = True
        review["human_decision"] = "PASS"
        review["human_confirmed_at"] = now_iso()
        store_candidate_review(manifest, review)
    elif stage == "mask":
        if manifest.get("stages", {}).get("candidate") != "CONFIRMED":
            return "⛔ 必须先人工确认候选图片；Qwen 检查为可选参考。", manifest
        selected_mask = manifest.get("selected_mask")
        if not selected_mask or selected_mask.get("status") not in {
            "PASS_WAITING_CONFIRMATION",
            "REVIEW_READY_WAITING_CONFIRMATION",
            "CONFIRMED",
        }:
            return "⛔ 当前任务没有通过检查的透明遮罩。", manifest
        relative_rgba = selected_mask.get("rgba_path")
        relative_report = selected_mask.get("report_path")
        if not relative_rgba or not relative_report:
            return "⛔ 当前遮罩缺少 RGBA 或检查报告。", manifest
        rgba_path = (PROJECT_ROOT / relative_rgba).resolve()
        report_path = (PROJECT_ROOT / relative_report).resolve()
        if any(
            path.parent.resolve() not in item.parents or not item.is_file()
            for item in (rgba_path, report_path)
        ):
            return "⛔ 遮罩档案路径无效或越出当前任务。", manifest
        actual_sha = file_sha256(rgba_path)
        archived = next(
            (
                item
                for item in manifest.get("artifacts", [])
                if item.get("role") in {
                    "approved_candidate_mask_rgba",
                    "approved_rgba_mask",
                }
                and item.get("path") == relative_rgba
                and item.get("sha256") == actual_sha
            ),
            None,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not archived or not (
            report.get("rgba_sha256") == actual_sha
            and report.get("corner_alpha") == [0, 0, 0, 0]
            and report.get("final_foreground_components") == 1
        ):
            return "⛔ 遮罩 SHA256 或基础连通性报告未通过。", manifest
        selected_mask["status"] = "CONFIRMED"
        selected_mask["confirmed_at"] = now_iso()
        selected_mask["sha256"] = actual_sha
        if not any(
            item.get("role") == "approved_rgba_mask"
            and item.get("path") == relative_rgba
            and item.get("sha256") == actual_sha
            for item in manifest.get("artifacts", [])
        ):
            manifest["artifacts"].append(
                {
                    "role": "approved_rgba_mask",
                    "version": selected_mask.get("version"),
                    "path": relative_rgba,
                    "bytes": rgba_path.stat().st_size,
                    "sha256": actual_sha,
                    "approved_at": now_iso(),
                }
            )
    elif stage == "mesh":
        if manifest.get("stages", {}).get("mask") != "CONFIRMED":
            return "⛔ 必须先确认透明遮罩。", manifest
        selected_mesh = manifest.get("selected_mesh", {})
        if selected_mesh.get("status") not in {
            "REVIEW_READY_WAITING_CONFIRMATION",
            "CONFIRMED",
        }:
            return "⛔ 当前只有原始 GLB，必须先完成网格检查和版本化整理。", manifest
        relative_mesh = selected_mesh.get("path")
        relative_report = selected_mesh.get("inspection_path")
        if not relative_mesh or not relative_report:
            return "⛔ 当前三维缺少 review GLB 或检查报告。", manifest
        mesh_path = (PROJECT_ROOT / relative_mesh).resolve()
        report_path = (PROJECT_ROOT / relative_report).resolve()
        if any(
            path.parent.resolve() not in item.parents or not item.is_file()
            for item in (mesh_path, report_path)
        ):
            return "⛔ 三维检查档案路径无效或越出当前任务。", manifest
        actual_sha = file_sha256(mesh_path)
        approved_artifact = next(
            (
                item
                for item in manifest.get("artifacts", [])
                if item.get("role") in {
                    "hunyuan_review_glb",
                    "manufacturing_review_glb",
                }
                and item.get("path") == relative_mesh
            ),
            None,
        )
        if (
            actual_sha != selected_mesh.get("sha256")
            or not approved_artifact
            or approved_artifact.get("sha256") != actual_sha
        ):
            return "⛔ review GLB 的 SHA256 与批准档案不一致。", manifest
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not (
            report.get("input_sha256") == actual_sha
            and report.get("components") == 1
            and report.get("watertight") is True
            and report.get("winding_consistent") is True
            and report.get("boundary_edges") == 0
            and report.get("nonmanifold_edges") == 0
        ):
            return "⛔ 网格检查指标未全部通过，不能人工确认。", manifest
        selected_mesh["status"] = "CONFIRMED"
        selected_mesh["confirmed_at"] = now_iso()
    manifest["stages"][stage] = "CONFIRMED"
    manifest["status"] = f"{stage.upper()}_CONFIRMED"
    append_event(manifest, f"{stage.upper()}_CONFIRMED", STAGES[stage])
    write_manifest(path, manifest)
    return f"✅ {STAGES[stage]}。流程门禁已更新。", manifest


def generate_candidate_mask(
    run_id: str,
    threshold: float,
    border_width: float,
    minimum_hole_pixels: float,
    trim_below_row: float,
) -> tuple[
    str,
    dict[str, Any],
    str | None,
    str | None,
    str | None,
    dict[str, Any],
]:
    """Generate a reviewable white-background mask for the selected candidate."""
    path, manifest = read_manifest(run_id)
    if manifest.get("stages", {}).get("candidate") != "CONFIRMED":
        return "⛔ 必须先人工确认候选图；Qwen 检查为可选参考。", manifest, None, None, None, {}
    selected = manifest.get("selected_candidate", {})
    relative_image = selected.get("path")
    if not relative_image:
        return "⛔ 当前任务没有 selected_candidate。", manifest, None, None, None, {}
    image_path = (PROJECT_ROOT / relative_image).resolve()
    if path.parent.resolve() not in image_path.parents or not image_path.is_file():
        return "⛔ 已确认候选图路径无效或越出任务目录。", manifest, None, None, None, {}
    image_sha = file_sha256(image_path)
    approved = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") == "ernie_candidate"
            and item.get("path") == relative_image
            and item.get("sha256") == image_sha
        ),
        None,
    )
    if not approved:
        return "⛔ 候选图 SHA256 与 ERNIE 归档不一致。", manifest, None, None, None, {}
    trim_value = int(trim_below_row) if float(trim_below_row or 0) > 0 else None
    manifest["status"] = "MASK_PREPARATION_RUNNING"
    append_event(
        manifest,
        "MASK_PREPARATION_STARTED",
        f"为已确认候选生成白背景遮罩：{relative_image}",
    )
    write_manifest(path, manifest)
    try:
        execution = run_mask_preparation(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            image_path=image_path,
            image_sha256=image_sha,
            threshold=float(threshold),
            border_width=int(border_width),
            minimum_hole_pixels=int(minimum_hole_pixels),
            trim_below_row=trim_value,
        )
    except (PreprocessingError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "MASK_PREPARATION_FAILED"
        append_event(manifest, "MASK_PREPARATION_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ CPU 遮罩生成失败：{exc}", manifest, str(image_path), None, None, {}

    previous_mask = manifest.get("selected_mask")
    if previous_mask:
        manifest.setdefault("mask_history", []).append(dict(previous_mask))
    version = execution["version"]
    for role, relative in (
        ("approved_candidate_mask_rgba", execution["rgba_path"]),
        ("candidate_mask_preview", execution["preview_path"]),
        ("candidate_mask_report", execution["report_path"]),
        ("candidate_mask_request", execution["request_path"]),
        ("candidate_mask_log", execution["log_path"]),
    ):
        artifact_path = PROJECT_ROOT / relative
        manifest["artifacts"].append(
            {
                "role": role,
                "version": version,
                "candidate_path": relative_image,
                "path": relative,
                "bytes": artifact_path.stat().st_size,
                "sha256": file_sha256(artifact_path),
            }
        )
    manifest["selected_mask"] = {
        "version": version,
        "status": "REVIEW_READY_WAITING_CONFIRMATION",
        "source_candidate_path": relative_image,
        "source_candidate_sha256": image_sha,
        "rgba_path": execution["rgba_path"],
        "preview_path": execution["preview_path"],
        "report_path": execution["report_path"],
        "request_path": execution["request_path"],
        "sha256": execution["rgba_sha256"],
    }
    manifest["stages"]["mask"] = "WAITING_CONFIRMATION"
    manifest["status"] = "MASK_REVIEW_READY"
    append_event(
        manifest,
        "MASK_REVIEW_READY",
        f"遮罩 v{version} 通过基础连通性检查，等待人工检查孔洞和阴影。",
    )
    write_manifest(path, manifest)
    return (
        f"✅ 遮罩 v{version} 已生成。请检查透明背景、孔洞和底部阴影后再确认。",
        manifest,
        str(image_path),
        str(PROJECT_ROOT / execution["rgba_path"]),
        str(PROJECT_ROOT / execution["preview_path"]),
        execution["report"],
    )


def load_current_mask(
    run_id: str,
) -> tuple[str, str, str, dict[str, Any], str]:
    manifest_file, manifest = read_manifest(run_id)
    selected_candidate = manifest.get("selected_candidate", {})
    selected_mask = manifest.get("selected_mask", {})
    candidate_path = selected_candidate.get("path")
    rgba_path = selected_mask.get("rgba_path")
    preview_path = selected_mask.get("preview_path")
    report_path = selected_mask.get("report_path")
    if not candidate_path or not rgba_path or not preview_path or not report_path:
        raise ValueError("当前任务还没有可加载的已检查遮罩。")

    resolved: list[Path] = []
    for relative_path in (candidate_path, rgba_path, preview_path, report_path):
        absolute = (PROJECT_ROOT / relative_path).resolve()
        if manifest_file.parent.resolve() not in absolute.parents or not absolute.is_file():
            raise ValueError("遮罩档案包含无效或越界的文件路径。")
        resolved.append(absolute)
    report = json.loads(resolved[3].read_text(encoding="utf-8"))
    status = (
        f"✅ 已加载候选 {selected_candidate.get('candidate_index')} 的当前遮罩；"
        f"状态：{selected_mask.get('status', 'UNKNOWN')}。"
    )
    return str(resolved[0]), str(resolved[1]), str(resolved[2]), report, status


def load_current_mesh_review(
    run_id: str,
) -> tuple[str, str, dict[str, Any], list[str], str]:
    manifest_file, manifest = read_manifest(run_id)
    selected_mesh = manifest.get("selected_mesh", {})
    relative_paths = {
        "mesh": selected_mesh.get("path"),
        "inspection": selected_mesh.get("inspection_path"),
        "projection": selected_mesh.get("projection_path"),
        "process": selected_mesh.get("process_report_path")
        or selected_mesh.get("cleanup_report_path"),
    }
    if selected_mesh.get("status") not in {
        "REVIEW_READY_WAITING_CONFIRMATION",
        "CONFIRMED",
        "IMPORTED_WAITING_REPAIR",
    } or any(not value for value in relative_paths.values()):
        raise ValueError("当前任务还没有可供人工确认的 review GLB。")

    resolved: dict[str, Path] = {}
    for key, relative_path in relative_paths.items():
        absolute = (PROJECT_ROOT / relative_path).resolve()
        if manifest_file.parent.resolve() not in absolute.parents or not absolute.is_file():
            raise ValueError("网格档案包含无效或越界的文件路径。")
        resolved[key] = absolute
    orientation_path = selected_mesh.get("orientation_report_path")
    if orientation_path:
        absolute = (PROJECT_ROOT / orientation_path).resolve()
        if manifest_file.parent.resolve() not in absolute.parents or not absolute.is_file():
            raise ValueError("打印方向档案包含无效或越界的文件路径。")
        resolved["orientation"] = absolute
    actual_sha = file_sha256(resolved["mesh"])
    if actual_sha != selected_mesh.get("sha256"):
        raise ValueError("review GLB 的 SHA256 与任务清单不一致。")
    report = json.loads(resolved["inspection"].read_text(encoding="utf-8"))
    if report.get("input_sha256") != actual_sha:
        raise ValueError("检查报告与当前 review GLB 不匹配。")
    origin = selected_mesh.get("origin", "Hunyuan")
    status = (
        f"✅ 已加载 {origin} 的 review GLB；"
        f"状态：{selected_mesh.get('status')}。请检查三视图后再人工确认。"
    )
    files = [str(resolved[key]) for key in ("mesh", "inspection", "projection", "process")]
    if "orientation" in resolved:
        files.append(str(resolved["orientation"]))
    return (
        str(resolved["mesh"]),
        str(resolved["projection"]),
        report,
        files,
        status,
    )


def import_external_mesh(
    run_id: str,
    upload: Any,
) -> tuple[str, dict[str, Any], str | None, str | None, dict[str, Any], list[str]]:
    """Archive an uploaded GLB inside the current RUN and make it repairable."""
    manifest_file, manifest = read_manifest(run_id)
    try:
        execution = archive_and_inspect_glb(
            project_root=PROJECT_ROOT,
            run_dir=manifest_file.parent,
            upload=upload,
        )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, MeshImportError) as exc:
        return f"❌ 外部 GLB 导入失败：{exc}", manifest, None, None, {}, []

    inspection = execution["inspection"]
    topology_passed = (
        inspection.get("components") == 1
        and inspection.get("watertight") is True
        and inspection.get("winding_consistent") is True
        and inspection.get("boundary_edges") == 0
        and inspection.get("nonmanifold_edges") == 0
        and inspection.get("degenerate_faces") == 0
    )
    previous = manifest.get("selected_mesh")
    if isinstance(previous, dict) and previous.get("path"):
        archived_previous = dict(previous)
        archived_previous["superseded_by_external_import_at"] = now_iso()
        manifest.setdefault("mesh_selection_history", []).append(archived_previous)
    version = execution["version"]
    for role, key in (
        ("external_mesh_import_glb", "mesh_path"),
        ("external_mesh_import_inspection", "report_path"),
        ("external_mesh_import_preview", "preview_path"),
        ("external_mesh_import_request", "request_path"),
        ("external_mesh_import_log", "log_path"),
    ):
        _append_mesh_repair_artifact(manifest, role, version, execution[key])
    selected_status = (
        "REVIEW_READY_WAITING_CONFIRMATION"
        if topology_passed
        else "IMPORTED_WAITING_REPAIR"
    )
    manifest["selected_mesh"] = {
        "version": version,
        "origin": f"外部 GLB 导入 v{version}：{execution['original_name']}",
        "status": selected_status,
        "path": execution["mesh_path"],
        "inspection_path": execution["report_path"],
        "projection_path": execution["preview_path"],
        "process_report_path": execution["request_path"],
        "sha256": execution["mesh_sha256"],
    }
    manifest.pop("selected_mesh_repair", None)
    manifest.pop("selected_gear_analysis", None)
    manifest["stages"]["mesh"] = (
        "WAITING_CONFIRMATION" if topology_passed else "WAITING_REVIEW"
    )
    manifest["status"] = (
        "MESH_REVIEW_READY"
        if topology_passed
        else "EXTERNAL_MESH_IMPORTED_NEEDS_REPAIR"
    )
    if manifest.get("selected_stl"):
        manifest["selected_stl"]["status"] = "SUPERSEDED_BY_EXTERNAL_MESH"
        manifest["selected_stl"]["superseded_at"] = now_iso()
    append_event(
        manifest,
        "EXTERNAL_MESH_IMPORTED",
        f"外部 GLB v{version} 已归档并检查；拓扑门禁："
        f"{'通过' if topology_passed else '未通过，等待修复'}。",
    )
    write_manifest(manifest_file, manifest)
    files = [
        str(PROJECT_ROOT / execution[key])
        for key in ("mesh_path", "report_path", "preview_path", "request_path", "log_path")
    ]
    return (
        f"✅ 外部 GLB v{version} 已安全归档。"
        f"拓扑门禁{'通过' if topology_passed else '未通过，请继续执行诊断和修复'}；"
        "原上传文件未修改。",
        manifest,
        str(PROJECT_ROOT / execution["mesh_path"]),
        str(PROJECT_ROOT / execution["preview_path"]),
        inspection,
        files,
    )


def _current_mesh_repair_source(
    manifest_file: Path,
    manifest: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    selected = manifest.get("selected_mesh", {})
    relative = selected.get("path") if isinstance(selected, dict) else None
    expected_sha = selected.get("sha256") if isinstance(selected, dict) else None
    if not relative or not expected_sha:
        raise ValueError("当前任务没有已检查的 selected_mesh；请先完成 CPU 网格检查。")
    source = (PROJECT_ROOT / str(relative)).resolve()
    if source.parent != manifest_file.parent / "meshes" or not source.is_file():
        raise ValueError("selected_mesh 路径无效或不属于当前任务。")
    actual_sha = file_sha256(source)
    if actual_sha != expected_sha:
        raise ValueError("selected_mesh 的 SHA256 与任务档案不一致。")
    return source, actual_sha, selected


def _append_mesh_repair_artifact(
    manifest: dict[str, Any],
    role: str,
    version: int,
    relative: str | None,
) -> None:
    if not relative:
        return
    artifact = (PROJECT_ROOT / relative).resolve()
    manifest.setdefault("artifacts", []).append(
        {
            "role": role,
            "version": version,
            "path": relative,
            "bytes": artifact.stat().st_size,
            "sha256": file_sha256(artifact),
        }
    )


def diagnose_current_mesh(
    run_id: str,
) -> tuple[str, dict[str, Any], str | None, dict[str, Any], list[str]]:
    """Run versioned geometry-only diagnosis for the current selected GLB."""
    manifest_file, manifest = read_manifest(run_id)
    try:
        source, source_sha, selected = _current_mesh_repair_source(
            manifest_file, manifest
        )
        execution = run_mesh_repair(
            project_root=PROJECT_ROOT,
            run_dir=manifest_file.parent,
            source_mesh=source,
            source_sha256=source_sha,
            mode="diagnose",
        )
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        MeshRepairExecutionError,
    ) as exc:
        return f"❌ 网格诊断失败：{exc}", manifest, None, {}, []

    version = execution["version"]
    for role, key in (
        ("mesh_repair_request", "request_path"),
        ("mesh_repair_diagnosis", "report_path"),
        ("mesh_repair_diagnostic_glb", "annotated_path"),
        ("mesh_repair_log", "log_path"),
    ):
        _append_mesh_repair_artifact(manifest, role, version, execution[key])
    manifest["selected_mesh_repair"] = {
        "version": version,
        "status": "DIAGNOSIS_READY",
        "mode": "diagnose",
        "source_path": selected["path"],
        "source_sha256": source_sha,
        "report_path": execution["report_path"],
        "annotated_path": execution["annotated_path"],
        "recommended_mode": execution["recommended_mode"],
        "created_at": now_iso(),
    }
    route = execution["report"].get("automatic_repairability_before", {}).get(
        "route", "UNKNOWN"
    )
    append_event(
        manifest,
        "MESH_REPAIR_DIAGNOSIS_READY",
        f"网格修复诊断 v{version} 完成；自动分流：{route}。",
    )
    write_manifest(manifest_file, manifest)
    files = [
        str(PROJECT_ROOT / execution[key])
        for key in ("annotated_path", "report_path", "request_path", "log_path")
    ]
    return (
        f"✅ 网格诊断 v{version} 完成。自动分流：{route}；尚未修改当前模型。",
        manifest,
        str(PROJECT_ROOT / execution["annotated_path"]),
        execution["report"],
        files,
    )


def repair_current_mesh(
    run_id: str,
    requested_mode: str,
) -> tuple[
    str,
    dict[str, Any],
    str | None,
    str | None,
    dict[str, Any],
    list[str],
]:
    """Create a versioned repair candidate; never replace selected_mesh here."""
    manifest_file, manifest = read_manifest(run_id)
    try:
        source, source_sha, selected = _current_mesh_repair_source(
            manifest_file, manifest
        )
        mode = str(requested_mode or "auto")
        if mode == "auto":
            diagnosis = manifest.get("selected_mesh_repair", {})
            report_path = diagnosis.get("report_path")
            if not report_path or diagnosis.get("source_sha256") != source_sha:
                return (
                    "⛔ 自动推荐需要先对当前 selected_mesh 执行诊断。",
                    manifest,
                    None,
                    None,
                    {},
                    [],
                )
            resolved_report = (PROJECT_ROOT / report_path).resolve()
            if (
                resolved_report.parent != manifest_file.parent / "reports"
                or not resolved_report.is_file()
            ):
                raise ValueError("自动推荐诊断报告路径无效。")
            diagnosis_report = json.loads(resolved_report.read_text(encoding="utf-8"))
            mode = recommended_mode(diagnosis_report) or ""
            if not mode:
                route = diagnosis_report.get("automatic_repairability_before", {}).get(
                    "route", "UNKNOWN"
                )
                return (
                    f"⛔ 当前自动分流为 {route}，没有可安全自动执行的拓扑修复；"
                    "请保留诊断结果并进行视觉复核或结构重建。",
                    manifest,
                    None,
                    None,
                    diagnosis_report,
                    [str(resolved_report)],
                )
        execution = run_mesh_repair(
            project_root=PROJECT_ROOT,
            run_dir=manifest_file.parent,
            source_mesh=source,
            source_sha256=source_sha,
            mode=mode,
        )
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        MeshRepairExecutionError,
    ) as exc:
        return f"❌ 网格修复失败：{exc}", manifest, None, None, {}, []

    version = execution["version"]
    role_keys = (
        ("mesh_repair_request", "request_path"),
        ("mesh_repair_report", "report_path"),
        ("mesh_repair_diagnostic_glb", "annotated_path"),
        ("mesh_repair_candidate_glb", "candidate_path"),
        ("mesh_repair_candidate_inspection", "inspection_path"),
        ("mesh_repair_candidate_preview", "preview_path"),
        ("mesh_repair_before_after_comparison", "comparison_path"),
        ("mesh_repair_log", "log_path"),
    )
    for role, key in role_keys:
        _append_mesh_repair_artifact(manifest, role, version, execution.get(key))

    candidate_path = execution.get("candidate_path")
    candidate_status = (
        "CANDIDATE_WAITING_CONFIRMATION"
        if candidate_path
        else "NO_SAFE_REPAIR_AVAILABLE"
    )
    manifest["selected_mesh_repair"] = {
        "version": version,
        "status": candidate_status,
        "mode": mode,
        "source_path": selected["path"],
        "source_sha256": source_sha,
        "report_path": execution["report_path"],
        "annotated_path": execution["annotated_path"],
        "candidate_path": candidate_path,
        "candidate_sha256": execution.get("candidate_sha256"),
        "inspection_path": execution.get("inspection_path"),
        "preview_path": execution.get("preview_path"),
        "comparison_path": execution.get("comparison_path"),
        "created_at": now_iso(),
    }
    append_event(
        manifest,
        "MESH_REPAIR_CANDIDATE_READY" if candidate_path else "MESH_REPAIR_NO_CHANGE",
        f"网格修复 v{version}（{mode}）完成；状态：{candidate_status}。",
    )
    write_manifest(manifest_file, manifest)
    files = [
        str(PROJECT_ROOT / execution[key])
        for _, key in role_keys
        if execution.get(key)
    ]
    if not candidate_path:
        return (
            f"ℹ️ {mode} 修复未找到能通过安全门禁的改动；没有生成候选 GLB。",
            manifest,
            str(PROJECT_ROOT / execution["annotated_path"]),
            None,
            execution["report"],
            files,
        )
    mode_label = {
        "safe": "低风险清理",
        "standard": "小孔修补",
        "artifact_cleanup": "独立小伪影清理",
        "advanced": "高级拓扑修复",
        "reconstruct": "严重模型体素重建",
    }.get(mode, mode)
    review_warning = (
        "体素重建已改变拓扑，只代表数字化门禁通过；必须重点核对真实孔洞、"
        "薄壁、配合面和整体轮廓。"
        if mode == "reconstruct"
        else "请比较原模型、候选和指标后选择接受或拒绝。"
    )
    return (
        f"✅ {mode_label}候选 v{version} 已生成。当前 selected_mesh 尚未替换；"
        f"{review_warning}",
        manifest,
        str(PROJECT_ROOT / candidate_path),
        str(
            PROJECT_ROOT
            / (execution.get("comparison_path") or execution["preview_path"])
        ),
        execution["report"],
        files,
    )


def decide_mesh_repair_candidate(
    run_id: str,
    decision: str,
) -> tuple[str, dict[str, Any]]:
    """Accept a fully gated candidate as selected_mesh, or archive its rejection."""
    manifest_file, manifest = read_manifest(run_id)
    repair = manifest.get("selected_mesh_repair", {})
    if repair.get("status") != "CANDIDATE_WAITING_CONFIRMATION":
        return "⛔ 当前没有等待确认的修复候选。", manifest
    normalized = str(decision).upper()
    if normalized == "REJECT":
        repair["status"] = "REJECTED_BY_HUMAN"
        repair["decided_at"] = now_iso()
        append_event(manifest, "MESH_REPAIR_REJECTED", "人工拒绝当前修复候选。")
        write_manifest(manifest_file, manifest)
        return "✅ 已拒绝修复候选；当前 selected_mesh 保持不变。", manifest
    if normalized != "ACCEPT":
        raise ValueError("修复候选决定必须是 ACCEPT 或 REJECT。")

    required = ("candidate_path", "candidate_sha256", "inspection_path", "preview_path", "report_path")
    if any(not repair.get(key) for key in required):
        return "⛔ 修复候选档案不完整，不能接受。", manifest
    paths = {key: (PROJECT_ROOT / repair[key]).resolve() for key in required if key.endswith("_path")}
    if any(
        path.parent not in {manifest_file.parent / "meshes", manifest_file.parent / "reports"}
        or not path.is_file()
        for path in paths.values()
    ):
        return "⛔ 修复候选路径无效或越出当前任务。", manifest
    if file_sha256(paths["candidate_path"]) != repair["candidate_sha256"]:
        return "⛔ 修复候选 SHA256 与任务档案不一致。", manifest
    repair_report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    inspection = json.loads(paths["inspection_path"].read_text(encoding="utf-8"))
    gate = (
        repair_report.get("status") in {
            "REPAIRED_CANDIDATE_REQUIRES_VISUAL_REVIEW",
            "RECONSTRUCTION_CANDIDATE_REQUIRES_VISUAL_REVIEW",
        }
        and inspection.get("input_sha256") == repair["candidate_sha256"]
        and inspection.get("components") == 1
        and inspection.get("watertight") is True
        and inspection.get("winding_consistent") is True
        and inspection.get("boundary_edges") == 0
        and inspection.get("nonmanifold_edges") == 0
        and inspection.get("degenerate_faces") == 0
    )
    if not gate:
        return "⛔ 候选未通过完整数字化拓扑门禁，只能保留为研究结果，不能接受。", manifest

    previous = manifest.get("selected_mesh", {})
    if isinstance(previous, dict):
        previous = dict(previous)
        previous["superseded_by_repair_at"] = now_iso()
        manifest.setdefault("mesh_selection_history", []).append(previous)
    manifest["selected_mesh"] = {
        "version": repair["version"],
        "origin": f"通用网格 {repair['mode']} 修复候选 v{repair['version']}",
        "status": "REVIEW_READY_WAITING_CONFIRMATION",
        "path": repair["candidate_path"],
        "source_raw_path": repair["source_path"],
        "source_raw_sha256": repair["source_sha256"],
        "inspection_path": repair["inspection_path"],
        "projection_path": repair["preview_path"],
        "process_report_path": repair["report_path"],
        "sha256": repair["candidate_sha256"],
    }
    repair["status"] = "ACCEPTED_AS_SELECTED_MESH"
    repair["decided_at"] = now_iso()
    manifest["stages"]["mesh"] = "WAITING_CONFIRMATION"
    manifest["status"] = "MESH_REVIEW_READY"
    if manifest.get("selected_stl"):
        manifest["selected_stl"]["status"] = "SUPERSEDED_BY_REPAIRED_MESH"
        manifest["selected_stl"]["superseded_at"] = now_iso()
    append_event(
        manifest,
        "MESH_REPAIR_ACCEPTED",
        "人工接受修复候选为新的 selected_mesh；仍需完成三维人工确认。",
    )
    write_manifest(manifest_file, manifest)
    return "✅ 已接受修复候选为当前 review GLB；原始模型仍保留，尚未导出或打印。", manifest


def accept_mesh_repair_candidate(
    run_id: str,
) -> tuple[str, dict[str, Any], str | None, str | None, dict[str, Any], list[str]]:
    status, manifest = decide_mesh_repair_candidate(run_id, "ACCEPT")
    if not status.startswith("✅"):
        return status, manifest, None, None, {}, []
    mesh, projection, report, files, _ = load_current_mesh_review(run_id)
    return status, manifest, mesh, projection, report, files


def mesh_repair_history_table(run_id: str) -> list[list[Any]]:
    manifest_file, _ = read_manifest(run_id)
    rows: list[list[Any]] = []
    pattern = re.compile(r"mesh_repair_v(\d+)\.json")
    for report_path in sorted((manifest_file.parent / "reports").glob("mesh_repair_v*.json")):
        match = pattern.fullmatch(report_path.name)
        if not match:
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        version = int(match.group(1))
        route = report.get("automatic_repairability_before", {}).get("route", "UNKNOWN")
        candidate = manifest_file.parent / "meshes" / f"mesh_repair_v{version}_candidate.glb"
        rows.append(
            [
                version,
                report.get("repair_mode", "diagnose"),
                route,
                report.get("status", "UNKNOWN"),
                "是" if candidate.is_file() else "否",
                datetime.fromtimestamp(report_path.stat().st_mtime).astimezone().isoformat(
                    timespec="seconds"
                ),
            ]
        )
    return rows


def load_mesh_repair_history(
    run_id: str,
    version_value: float,
) -> tuple[str | None, str | None, dict[str, Any], list[str], str]:
    manifest_file, _ = read_manifest(run_id)
    try:
        version = int(float(version_value))
    except (TypeError, ValueError) as exc:
        raise ValueError("修复历史版本必须是正整数。") from exc
    if version < 1:
        raise ValueError("修复历史版本必须是正整数。")
    run_dir = manifest_file.parent
    report_path = (run_dir / "reports" / f"mesh_repair_v{version}.json").resolve()
    annotated_path = (run_dir / "meshes" / f"mesh_repair_v{version}_diagnostic.glb").resolve()
    candidate_path = (run_dir / "meshes" / f"mesh_repair_v{version}_candidate.glb").resolve()
    preview_path = (run_dir / "reports" / f"mesh_repair_v{version}_candidate_preview.png").resolve()
    comparison_path = (
        run_dir / "reports" / f"mesh_repair_v{version}_before_after_comparison.png"
    ).resolve()
    inspection_path = (run_dir / "reports" / f"mesh_repair_v{version}_candidate_inspection.json").resolve()
    request_path = (run_dir / "requests" / f"mesh_repair_v{version}.json").resolve()
    log_path = (run_dir / "logs" / f"mesh_repair_v{version}.log").resolve()
    if report_path.parent != run_dir / "reports" or not report_path.is_file():
        raise ValueError(f"当前任务不存在修复历史 v{version}。")
    if annotated_path.parent != run_dir / "meshes" or not annotated_path.is_file():
        raise ValueError("修复历史缺少诊断标记 GLB。")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_path = candidate_path if candidate_path.is_file() else annotated_path
    files = [report_path, annotated_path]
    for optional in (
        candidate_path,
        preview_path,
        comparison_path,
        inspection_path,
        request_path,
        log_path,
    ):
        if optional.is_file():
            files.append(optional)
    return (
        str(model_path),
        str(comparison_path)
        if comparison_path.is_file()
        else str(preview_path)
        if preview_path.is_file()
        else None,
        report,
        [str(path) for path in files],
        f"✅ 已恢复模型修复历史 v{version}；只更新页面显示，不改变 selected_mesh。",
    )


def analyze_current_spur_gear(
    run_id: str,
) -> tuple[str, dict[str, Any], dict[str, Any], str | None, list[str]]:
    manifest_file, manifest = read_manifest(run_id)
    profile = resolve_part_profile(str(manifest.get("request", {}).get("part_type", "")))
    if profile.key != "spur_gear":
        return "⛔ 齿轮专用指标只适用于直齿圆柱齿轮任务。", manifest, {}, None, []
    try:
        source, source_sha, selected = _current_mesh_repair_source(
            manifest_file, manifest
        )
        execution = run_gear_analysis(
            project_root=PROJECT_ROOT,
            run_dir=manifest_file.parent,
            source_mesh=source,
            source_sha256=source_sha,
        )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, GearAnalysisError) as exc:
        return f"❌ 齿轮参数分析失败：{exc}", manifest, {}, None, []
    version = execution["version"]
    for role, key in (
        ("spur_gear_parameter_analysis", "report_path"),
        ("spur_gear_radial_profile", "preview_path"),
        ("spur_gear_analysis_log", "log_path"),
    ):
        _append_mesh_repair_artifact(manifest, role, version, execution[key])
    manifest["selected_gear_analysis"] = {
        "version": version,
        "status": "PARAMETER_DRAFT_REQUIRES_ENGINEERING_CONFIRMATION",
        "source_path": selected["path"],
        "source_sha256": source_sha,
        "report_path": execution["report_path"],
        "preview_path": execution["preview_path"],
        "created_at": now_iso(),
    }
    count = execution["report"].get("input", {}).get("estimated_tooth_count")
    append_event(
        manifest,
        "SPUR_GEAR_ANALYSIS_READY",
        f"齿轮专用指标 v{version} 完成；FFT 估算齿数：{count}。",
    )
    write_manifest(manifest_file, manifest)
    files = [str(PROJECT_ROOT / execution[key]) for key in ("report_path", "preview_path", "log_path")]
    return (
        f"✅ 齿轮指标 v{version} 已生成；估算齿数 {count}。"
        "结果是数字化参数草案，不等于工程尺寸确认。",
        manifest,
        execution["report"],
        str(PROJECT_ROOT / execution["preview_path"]),
        files,
    )


def generate_hunyuan_mesh(
    run_id: str,
) -> tuple[str, dict[str, Any], str | None]:
    path, manifest = read_manifest(run_id)
    if manifest.get("stages", {}).get("mask") != "CONFIRMED":
        return "⛔ 必须先确认当前任务的透明遮罩。", manifest, None
    selected_mask = manifest.get("selected_mask", {})
    if selected_mask.get("status") != "CONFIRMED":
        return "⛔ 当前任务没有已确认的 selected_mask。", manifest, None
    relative_mask = selected_mask.get("rgba_path")
    if not relative_mask:
        return "⛔ 已确认遮罩缺少 RGBA 路径。", manifest, None
    mask_path = (PROJECT_ROOT / relative_mask).resolve()
    if path.parent.resolve() not in mask_path.parents or not mask_path.is_file():
        return "⛔ 已确认遮罩路径无效或越出当前任务。", manifest, None
    mask_sha = file_sha256(mask_path)
    approved_artifact = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") == "approved_rgba_mask"
            and item.get("path") == relative_mask
        ),
        None,
    )
    if not approved_artifact or approved_artifact.get("sha256") != mask_sha:
        return "⛔ 遮罩 SHA256 与批准记录不一致，拒绝启动 Hunyuan。", manifest, None

    manifest["status"] = "HUNYUAN_QUEUED"
    manifest["models"]["hunyuan"]["status"] = "QUEUED"
    append_event(manifest, "HUNYUAN_QUEUED", f"使用已确认遮罩：{relative_mask}")
    write_manifest(path, manifest)
    try:
        execution = run_hunyuan(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            image_path=mask_path,
            image_sha256=mask_sha,
        )
    except GpuUnavailableError as exc:
        manifest["status"] = "WAITING_GPU"
        manifest["models"]["hunyuan"]["status"] = "WAITING_GPU"
        append_event(manifest, "HUNYUAN_WAITING_GPU", str(exc))
        write_manifest(path, manifest)
        return f"⏳ Hunyuan 未启动：{exc}", manifest, None
    except (HunyuanExecutionError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "HUNYUAN_FAILED"
        manifest["models"]["hunyuan"]["status"] = "FAILED"
        append_event(manifest, "HUNYUAN_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ Hunyuan 执行失败：{exc} 已保留请求和日志。", manifest, None

    result = execution["result"]
    manifest["artifacts"].extend(
        [
            {
                "role": "hunyuan_raw_glb",
                "attempt": execution["attempt"],
                "path": execution["output_path"],
                "bytes": result["output_bytes"],
                "sha256": result["output_sha256"],
            },
            {"role": "hunyuan_request", "path": execution["request_path"]},
            {"role": "hunyuan_result", "path": execution["result_path"]},
            {"role": "hunyuan_log", "path": execution["log_path"]},
        ]
    )
    manifest.setdefault("metrics", {}).setdefault("hunyuan_attempts", []).append(
        {
            "attempt": execution["attempt"],
            "load_seconds": result["load_seconds"],
            "generation_seconds": result["generation_seconds"],
            "total_seconds": result["total_seconds"],
            "peak_allocated_gib": result["peak_allocated_gib"],
            "peak_reserved_gib": result["peak_reserved_gib"],
            "gpu_preflight": execution["gpu_preflight"],
            "mesh": result["mesh"],
        }
    )
    manifest["selected_mesh"] = {
        "attempt": execution["attempt"],
        "status": "RAW_WAITING_REVIEW",
        "path": execution["output_path"],
        "result_path": execution["result_path"],
        "sha256": result["output_sha256"],
    }
    manifest["models"]["hunyuan"]["status"] = "COMPLETED"
    manifest["stages"]["mesh"] = "WAITING_REVIEW"
    manifest["status"] = "HUNYUAN_COMPLETED"
    append_event(
        manifest,
        "HUNYUAN_COMPLETED",
        f"第 {execution['attempt']} 次生成完成，原始 GLB 等待网格检查。",
    )
    write_manifest(path, manifest)
    status = (
        f"✅ Hunyuan 原始 GLB 已生成；耗时 {result['total_seconds']:.1f} 秒，"
        f"峰值显存 {result['peak_allocated_gib']:.2f} GiB。"
        " 当前仅供网格检查，尚未批准 STL 或打印。"
    )
    return status, manifest, str(PROJECT_ROOT / execution["output_path"])


def generate_hunyuan_multiview_mesh(
    run_id: str,
    front_upload: str | None,
    left_upload: str | None,
    back_upload: str | None,
    right_upload: str | None,
) -> tuple[str, dict[str, Any], str | None]:
    """Archive approved canonical views and run the 2mv compatibility backend."""
    path, manifest = read_manifest(run_id)
    uploads = {
        "front": front_upload,
        "left": left_upload,
        "back": back_upload,
        "right": right_upload,
    }
    missing = [view for view in ("front", "left", "back") if not uploads[view]]
    if missing:
        return "⛔ 多视图至少需要正面、左侧、背面三张 RGBA PNG。", manifest, None

    masks_dir = path.parent / "masks"
    version = 1
    while (masks_dir / f"multiview_input_v{version}").exists():
        version += 1
    archive_dir = masks_dir / f"multiview_input_v{version}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived_paths: dict[str, Path] = {}
    archived_hashes: dict[str, str] = {}
    try:
        for view, uploaded in uploads.items():
            if not uploaded:
                continue
            source = Path(uploaded).resolve()
            if not source.is_file() or source.suffix.lower() != ".png":
                raise ValueError(f"{view} 视角必须上传 PNG 文件。")
            with Image.open(source) as opened:
                opened.load()
                if opened.mode != "RGBA":
                    raise ValueError(f"{view} 视角必须是带 Alpha 通道的 RGBA PNG。")
                alpha = opened.getchannel("A")
                alpha_min, alpha_max = alpha.getextrema()
                if alpha_min != 0 or alpha_max != 255:
                    raise ValueError(f"{view} 视角必须同时包含透明背景和不透明主体。")
                destination = archive_dir / f"{view}.png"
                opened.save(destination, format="PNG")
            archived_paths[view] = destination
            archived_hashes[view] = file_sha256(destination)
    except Exception as exc:
        # Keep an empty/partial versioned folder as evidence; never overwrite or delete user inputs.
        return f"⛔ 多视图输入归档失败：{exc}", manifest, None

    for view, archived in archived_paths.items():
        manifest["artifacts"].append(
            {
                "role": "hunyuan_multiview_input",
                "version": version,
                "view": view,
                "path": project_relative(archived),
                "sha256": archived_hashes[view],
            }
        )
    manifest["status"] = "HUNYUAN_QUEUED"
    manifest["models"]["hunyuan"]["status"] = "QUEUED"
    append_event(
        manifest,
        "HUNYUAN_MULTIVIEW_QUEUED",
        f"已归档并确认多视图 v{version}：{', '.join(archived_paths)}；真实后端 Hunyuan3D-2mv。",
    )
    write_manifest(path, manifest)

    try:
        execution = run_hunyuan_multiview(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            image_paths=archived_paths,
            image_sha256=archived_hashes,
        )
    except GpuUnavailableError as exc:
        manifest["status"] = "WAITING_GPU"
        manifest["models"]["hunyuan"]["status"] = "WAITING_GPU"
        append_event(manifest, "HUNYUAN_MULTIVIEW_WAITING_GPU", str(exc))
        write_manifest(path, manifest)
        return f"⏳ Hunyuan 多视图未启动：{exc}", manifest, None
    except (HunyuanExecutionError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "HUNYUAN_FAILED"
        manifest["models"]["hunyuan"]["status"] = "FAILED"
        append_event(manifest, "HUNYUAN_MULTIVIEW_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ Hunyuan 多视图执行失败：{exc} 已保留输入、请求和日志。", manifest, None

    result = execution["result"]
    manifest["artifacts"].extend(
        [
            {
                "role": "hunyuan_raw_glb",
                "attempt": execution["attempt"],
                "backend_model": result["backend_model"],
                "input_mode": result["input_mode"],
                "path": execution["output_path"],
                "bytes": result["output_bytes"],
                "sha256": result["output_sha256"],
            },
            {"role": "hunyuan_request", "path": execution["request_path"]},
            {"role": "hunyuan_result", "path": execution["result_path"]},
            {"role": "hunyuan_log", "path": execution["log_path"]},
        ]
    )
    manifest.setdefault("metrics", {}).setdefault("hunyuan_attempts", []).append(
        {
            "attempt": execution["attempt"],
            "interface_name": result["interface_name"],
            "backend_model": result["backend_model"],
            "input_mode": result["input_mode"],
            "views": list(result["inputs"]),
            "load_seconds": result["load_seconds"],
            "generation_seconds": result["generation_seconds"],
            "total_seconds": result["total_seconds"],
            "peak_allocated_gib": result["peak_allocated_gib"],
            "peak_reserved_gib": result["peak_reserved_gib"],
            "gpu_preflight": execution["gpu_preflight"],
            "mesh": result["mesh"],
        }
    )
    manifest["selected_mesh"] = {
        "attempt": execution["attempt"],
        "status": "RAW_WAITING_REVIEW",
        "origin": "Hunyuan3D-2mv 多视图后端",
        "backend_model": result["backend_model"],
        "input_mode": result["input_mode"],
        "path": execution["output_path"],
        "result_path": execution["result_path"],
        "sha256": result["output_sha256"],
    }
    manifest["models"]["hunyuan"]["status"] = "COMPLETED"
    manifest["stages"]["mesh"] = "WAITING_REVIEW"
    manifest["status"] = "HUNYUAN_COMPLETED"
    append_event(
        manifest,
        "HUNYUAN_MULTIVIEW_COMPLETED",
        f"第 {execution['attempt']} 次生成完成；真实后端 Hunyuan3D-2mv，原始 GLB 等待网格检查。",
    )
    write_manifest(path, manifest)
    status = (
        f"✅ 多视图原始 GLB 已生成；真实后端 {result['backend_model']}，"
        f"输入 {len(result['inputs'])} 个视角，耗时 {result['total_seconds']:.1f} 秒，"
        f"峰值显存 {result['peak_allocated_gib']:.2f} GiB。"
        " 当前仅供网格检查，尚未批准 STL 或打印。"
    )
    return status, manifest, str(PROJECT_ROOT / execution["output_path"])


def prepare_current_hunyuan_mesh_review(
    run_id: str,
) -> tuple[str, dict[str, Any], str | None, str | None, dict[str, Any], list[str]]:
    """Inspect a Hunyuan raw GLB and prepare a conservative review version."""
    path, manifest = read_manifest(run_id)
    selected = manifest.get("selected_mesh", {})
    if selected.get("status") != "RAW_WAITING_REVIEW":
        return "⛔ 当前选中项不是等待检查的 Hunyuan 原始 GLB。", manifest, None, None, {}, []
    relative_raw = selected.get("path")
    expected_sha = selected.get("sha256")
    if not relative_raw or not expected_sha:
        return "⛔ Hunyuan 原始 GLB 缺少路径或 SHA256。", manifest, None, None, {}, []
    raw_path = (PROJECT_ROOT / relative_raw).resolve()
    if raw_path.parent != path.parent / "meshes" or not raw_path.is_file():
        return "⛔ Hunyuan 原始 GLB 路径无效或越出任务目录。", manifest, None, None, {}, []
    actual_sha = file_sha256(raw_path)
    archived = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") == "hunyuan_raw_glb"
            and item.get("path") == relative_raw
            and item.get("sha256") == actual_sha
        ),
        None,
    )
    if expected_sha != actual_sha or not archived:
        return "⛔ Hunyuan 原始 GLB 与生成档案的 SHA256 不一致。", manifest, None, None, {}, []
    manifest["status"] = "MESH_REVIEW_PREPARATION_RUNNING"
    append_event(manifest, "MESH_REVIEW_PREPARATION_STARTED", relative_raw)
    write_manifest(path, manifest)
    try:
        execution = run_mesh_review_preparation(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            raw_mesh_path=raw_path,
            raw_mesh_sha256=actual_sha,
        )
    except (PreprocessingError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "MESH_REVIEW_PREPARATION_FAILED"
        append_event(manifest, "MESH_REVIEW_PREPARATION_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ CPU 网格检查/整理失败：{exc}", manifest, None, None, {}, []

    version = execution["version"]
    for role, relative in (
        ("hunyuan_review_glb", execution["review_path"]),
        ("hunyuan_raw_mesh_inspection", execution["raw_inspection_path"]),
        ("hunyuan_raw_projection", execution["raw_preview_path"]),
        ("hunyuan_review_process", execution["process_path"]),
        ("hunyuan_review_mesh_inspection", execution["inspection_path"]),
        ("hunyuan_review_projection", execution["preview_path"]),
        ("hunyuan_mesh_review_request", execution["request_path"]),
        ("hunyuan_mesh_review_log", execution["log_path"]),
    ):
        artifact_path = PROJECT_ROOT / relative
        manifest["artifacts"].append(
            {
                "role": role,
                "version": version,
                "path": relative,
                "bytes": artifact_path.stat().st_size,
                "sha256": file_sha256(artifact_path),
            }
        )
    if manifest.get("selected_stl"):
        manifest["selected_stl"]["status"] = "SUPERSEDED_BY_NEW_MESH"
        manifest["selected_stl"]["superseded_at"] = now_iso()
    manifest["selected_mesh"] = {
        "attempt": selected.get("attempt"),
        "version": version,
        "origin": f"Hunyuan 原始网格保守 review v{version}",
        "status": "REVIEW_READY_WAITING_CONFIRMATION",
        "path": execution["review_path"],
        "source_raw_path": relative_raw,
        "source_raw_sha256": actual_sha,
        "inspection_path": execution["inspection_path"],
        "projection_path": execution["preview_path"],
        "process_report_path": execution["process_path"],
        "raw_inspection_path": execution["raw_inspection_path"],
        "raw_projection_path": execution["raw_preview_path"],
        "sha256": execution["review_sha256"],
    }
    manifest["stages"]["mesh"] = "WAITING_CONFIRMATION"
    manifest["status"] = "MESH_REVIEW_READY"
    append_event(
        manifest,
        "MESH_REVIEW_READY",
        f"Hunyuan review v{version} 通过通用几何门禁，等待人工视觉检查。",
    )
    write_manifest(path, manifest)
    files = [
        str(PROJECT_ROOT / execution[key])
        for key in (
            "review_path",
            "inspection_path",
            "preview_path",
            "process_path",
            "raw_inspection_path",
            "raw_preview_path",
            "request_path",
            "log_path",
        )
    ]
    return (
        f"✅ Hunyuan review v{version} 已通过 CPU 网格门禁。请检查 GLB 和四视图。",
        manifest,
        str(PROJECT_ROOT / execution["review_path"]),
        str(PROJECT_ROOT / execution["preview_path"]),
        execution["inspection"],
        files,
    )


def generate_regularized_fan(
    run_id: str,
    outer_diameter_mm: float,
    ring_inner_diameter_mm: float,
    ring_depth_mm: float,
    hub_diameter_mm: float,
    hub_depth_mm: float,
    blade_thickness_mm: float,
    blade_sweep_deg: float,
    blade_pitch_camber_mm: float,
    blade_radial_arch_mm: float,
    resolution_mm: float,
) -> tuple[str, dict[str, Any], str | None, str | None, dict[str, Any], list[str]]:
    """Create a new deterministic review mesh without loading a GPU model."""
    path, manifest = read_manifest(run_id)
    request = manifest.get("request", {})
    part_type = str(request.get("part_type", ""))
    if not any(name in part_type for name in ("风扇", "转子")) or int(
        request.get("structure_count", 0)
    ) != 7:
        return (
            "⛔ 当前规则化模板仅支持七叶片风扇任务。",
            manifest,
            None,
            None,
            {},
            [],
        )
    if manifest.get("stages", {}).get("mask") != "CONFIRMED":
        return "⛔ 必须先确认透明遮罩。", manifest, None, None, {}, []
    selected_mesh = manifest.get("selected_mesh", {})
    relative_source = selected_mesh.get("path")
    source_sha = selected_mesh.get("sha256")
    if not relative_source or not source_sha:
        return "⛔ 当前任务还没有可作为外形参考的 GLB。", manifest, None, None, {}, []
    source_path = (PROJECT_ROOT / relative_source).resolve()
    if source_path.parent != path.parent / "meshes" or not source_path.is_file():
        return "⛔ 当前源 GLB 路径无效或越出任务目录。", manifest, None, None, {}, []
    actual_sha = file_sha256(source_path)
    approved = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") in {
                "hunyuan_raw_glb",
                "hunyuan_review_glb",
                "manufacturing_review_glb",
            }
            and item.get("path") == relative_source
            and item.get("sha256") == actual_sha
        ),
        None,
    )
    if source_sha != actual_sha or not approved:
        return "⛔ 源 GLB 的 SHA256 与任务档案不一致。", manifest, None, None, {}, []
    parameters = {
        "outer_diameter_mm": float(outer_diameter_mm),
        "ring_inner_diameter_mm": float(ring_inner_diameter_mm),
        "ring_depth_mm": float(ring_depth_mm),
        "hub_diameter_mm": float(hub_diameter_mm),
        "hub_depth_mm": float(hub_depth_mm),
        "blade_thickness_mm": float(blade_thickness_mm),
        "blade_sweep_deg": float(blade_sweep_deg),
        "blade_pitch_camber_mm": float(blade_pitch_camber_mm),
        "blade_radial_arch_mm": float(blade_radial_arch_mm),
        "resolution_mm": float(resolution_mm),
    }
    manifest["status"] = "MESH_OPTIMIZATION_RUNNING"
    append_event(
        manifest,
        "MESH_OPTIMIZATION_STARTED",
        f"以 {relative_source} 为外形参考启动 CPU 七叶片规则化。",
    )
    write_manifest(path, manifest)
    try:
        execution = run_regularization(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            source_mesh=source_path,
            source_sha256=actual_sha,
            parameters=parameters,
        )
    except (MeshOptimizationError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "MESH_OPTIMIZATION_FAILED"
        append_event(manifest, "MESH_OPTIMIZATION_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ CPU 规则化失败：{exc}", manifest, None, None, {}, []

    version = execution["version"]
    artifact_specs = [
        ("manufacturing_review_glb", execution["mesh_path"]),
        ("regularized_build_report", execution["build_report_path"]),
        ("manufacturing_review_inspection", execution["inspection_path"]),
        ("manufacturing_review_projection", execution["preview_path"]),
        ("regularization_request", execution["request_path"]),
        ("regularization_log", execution["log_path"]),
    ]
    for role, relative in artifact_specs:
        artifact_path = PROJECT_ROOT / relative
        record = {"role": role, "version": version, "path": relative}
        if artifact_path.is_file():
            record.update(
                {"bytes": artifact_path.stat().st_size, "sha256": file_sha256(artifact_path)}
            )
        manifest["artifacts"].append(record)
    previous_stl = manifest.get("selected_stl")
    if previous_stl:
        previous_stl["status"] = "SUPERSEDED_BY_NEW_MESH"
        previous_stl["superseded_at"] = now_iso()
    manifest.setdefault("metrics", {}).setdefault(
        "mesh_optimization_attempts", []
    ).append(
        {
            "version": version,
            "source_path": relative_source,
            "source_sha256": actual_sha,
            "parameters": parameters,
            "inspection": execution["inspection"],
        }
    )
    manifest["selected_mesh"] = {
        "version": version,
        "origin": f"CPU 参数化七叶片规则化 v{version}",
        "status": "REVIEW_READY_WAITING_CONFIRMATION",
        "path": execution["mesh_path"],
        "source_mesh_path": relative_source,
        "source_mesh_sha256": actual_sha,
        "inspection_path": execution["inspection_path"],
        "projection_path": execution["preview_path"],
        "process_report_path": execution["build_report_path"],
        "request_path": execution["request_path"],
        "sha256": execution["mesh_sha256"],
    }
    manifest["stages"]["mesh"] = "WAITING_CONFIRMATION"
    manifest["status"] = "REGULARIZED_MESH_REVIEW_READY"
    append_event(
        manifest,
        "REGULARIZED_MESH_REVIEW_READY",
        f"v{version} 已通过自动几何检查，等待人工检查四视图。",
    )
    write_manifest(path, manifest)
    files = [
        str(PROJECT_ROOT / execution[key])
        for key in (
            "mesh_path",
            "inspection_path",
            "preview_path",
            "build_report_path",
            "request_path",
            "log_path",
        )
    ]
    status = (
        f"✅ 规则化 v{version} 已生成并通过自动网格门禁。"
        "请先检查 GLB 和四视图，然后点击“确认已检查三维模型”。"
    )
    return (
        status,
        manifest,
        str(PROJECT_ROOT / execution["mesh_path"]),
        str(PROJECT_ROOT / execution["preview_path"]),
        execution["inspection"],
        files,
    )


def export_confirmed_mesh_stl(
    run_id: str,
    support_acknowledged: bool,
    requested_target_longest_mm: float | None = None,
) -> tuple[str, dict[str, Any], str | None]:
    """Export only the exact mesh version that passed the human gate."""
    path, manifest = read_manifest(run_id)
    selected = manifest.get("selected_mesh", {})
    if (
        manifest.get("stages", {}).get("mesh") != "CONFIRMED"
        or selected.get("status") != "CONFIRMED"
    ):
        return "⛔ 必须先人工确认当前三维模型。", manifest, None
    if not support_acknowledged:
        return (
            "⛔ 当前模型在某些打印朝向下可能需要支撑；请先勾选切片检查确认。",
            manifest,
            None,
        )
    relative_mesh = selected.get("path")
    expected_sha = selected.get("sha256")
    mesh_path = (PROJECT_ROOT / str(relative_mesh)).resolve()
    if mesh_path.parent != path.parent / "meshes" or not mesh_path.is_file():
        return "⛔ 已确认 GLB 路径无效。", manifest, None
    actual_sha = file_sha256(mesh_path)
    approved = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") in {"hunyuan_review_glb", "manufacturing_review_glb"}
            and item.get("path") == relative_mesh
            and item.get("sha256") == actual_sha
        ),
        None,
    )
    if expected_sha != actual_sha or not approved:
        return "⛔ 已确认 GLB 的 SHA256 与批准档案不一致。", manifest, None
    inspection_path = (PROJECT_ROOT / selected["inspection_path"]).resolve()
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    task_target_longest = float(
        manifest.get("request", {}).get("target_size_mm", 0) or 0
    )
    target_longest = float(
        task_target_longest
        if requested_target_longest_mm is None
        else requested_target_longest_mm
    )
    if not 20.0 <= target_longest <= 200.0:
        return (
            "⛔ 本次 STL 目标最长尺寸必须在 20–200 mm 范围内。",
            manifest,
            None,
        )
    manifest["status"] = "STL_EXPORT_RUNNING"
    append_event(
        manifest,
        "STL_EXPORT_STARTED",
        f"导出已确认 GLB：{relative_mesh}；本次目标最长尺寸 {target_longest:.3f} mm。",
    )
    write_manifest(path, manifest)
    try:
        execution = run_stl_export(
            project_root=PROJECT_ROOT,
            run_dir=path.parent,
            source_mesh=mesh_path,
            source_sha256=actual_sha,
            target_longest_mm=target_longest,
            allow_supported_overhangs=True,
        )
    except (MeshOptimizationError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "STL_EXPORT_FAILED"
        append_event(manifest, "STL_EXPORT_FAILED", str(exc))
        write_manifest(path, manifest)
        return f"❌ STL 导出失败：{exc}", manifest, None
    report = execution["report"]
    for role, relative in (
        ("print_candidate_stl", execution["output_path"]),
        ("stl_export_report", execution["report_path"]),
        ("stl_export_log", execution["log_path"]),
    ):
        artifact_path = PROJECT_ROOT / relative
        manifest["artifacts"].append(
            {
                "role": role,
                "version": execution["version"],
                "source_mesh_version": selected.get("version"),
                "path": relative,
                "bytes": artifact_path.stat().st_size,
                "sha256": file_sha256(artifact_path),
            }
        )
    manifest["selected_stl"] = {
        "version": execution["version"],
        "source_mesh_version": selected.get("version"),
        "status": (
            "WAITING_BAMBU_SLICER_SUPPORT_REQUIRED"
            if report.get("support_required")
            else "WAITING_BAMBU_SLICER"
        ),
        "path": execution["output_path"],
        "report_path": execution["report_path"],
        "sha256": execution["output_sha256"],
        "target_longest_mm": target_longest,
        "task_original_target_longest_mm": task_target_longest,
        "support_required": bool(report.get("support_required")),
    }
    manifest["status"] = "STL_READY_WAITING_BAMBU_SLICER"
    append_event(
        manifest,
        "STL_READY_WAITING_BAMBU_SLICER",
        "STL 仅完成几何导出，尚未切片或发送打印。",
    )
    write_manifest(path, manifest)
    reloaded_metrics = report.get("reloaded_stl_metrics", {})
    extents = reloaded_metrics.get("extents_mm") if isinstance(reloaded_metrics, dict) else None
    extent_text = (
        " × ".join(f"{float(value):.3f}" for value in extents) + " mm"
        if isinstance(extents, list) and len(extents) == 3
        else f"最长边 {target_longest:.3f} mm"
    )
    return (
        f"✅ STL 已导出新版本，验证尺寸：{extent_text}。"
        "它仍必须在 Bambu Studio 中逐层检查，当前没有发送打印。",
        manifest,
        str(PROJECT_ROOT / execution["output_path"]),
    )


def _next_assembly_version(run_dir: Path) -> int:
    versions: list[int] = []
    for item in (run_dir / "assembly").glob("assembly_plan_v*.json"):
        match = re.fullmatch(r"assembly_plan_v(\d+)\.json", item.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def save_assembly_plan(
    run_id: str,
    counterpart: str,
    action: str,
    pickup_zone: str,
    assembly_zone: str,
    finished_zone: str,
    reject_zone: str,
    grasp_feature: str,
    approach_direction: str,
    mating_feature: str,
    assembly_direction: str,
    success_criteria: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str | None]:
    manifest_path_value, manifest = read_manifest(run_id)
    request = manifest.get("request", {})
    try:
        plan = build_assembly_plan(
            run_id=manifest["run_id"],
            part_type=str(request.get("part_type", "自定义零件")),
            counterpart=counterpart,
            action=action,
            pickup_zone=pickup_zone,
            assembly_zone=assembly_zone,
            finished_zone=finished_zone,
            reject_zone=reject_zone,
            grasp_feature=grasp_feature,
            approach_direction=approach_direction,
            mating_feature=mating_feature,
            assembly_direction=assembly_direction,
            success_criteria=success_criteria,
        )
    except ValueError as exc:
        return f"⛔ 装配方案未保存：{exc}", manifest, assembly_plan_summary(None), {}, None

    run_dir = manifest_path_value.parent
    version = _next_assembly_version(run_dir)
    output_path = run_dir / "assembly" / f"assembly_plan_v{version}.json"
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有装配方案：{output_path}")
    selected_mesh = manifest.get("selected_mesh", {})
    plan["version"] = version
    plan["source_artifacts"] = {
        "selected_mesh_path": (
            selected_mesh.get("path") if isinstance(selected_mesh, dict) else None
        ),
        "selected_mesh_status": (
            selected_mesh.get("status") if isinstance(selected_mesh, dict) else None
        ),
        "physical_part_verified": False,
        "note": "真实零件尺寸、质量、材料、配合公差和可抓取性尚未验证。",
    }
    write_manifest(output_path, plan)
    relative = project_relative(output_path)
    plan_sha = file_sha256(output_path)
    manifest.setdefault("artifacts", []).append(
        {
            "role": "assembly_plan",
            "version": version,
            "approval_status": "WAITING_CONFIRMATION",
            "path": relative,
            "bytes": output_path.stat().st_size,
            "sha256": plan_sha,
        }
    )
    manifest["selected_assembly_plan"] = {
        "version": version,
        "status": "PLAN_DRAFT_WAITING_CONFIRMATION",
        "path": relative,
        "sha256": plan_sha,
    }
    manifest.setdefault("stages", {})["assembly"] = "WAITING_CONFIRMATION"
    manifest["status"] = "ASSEMBLY_PLAN_REVIEW_READY"
    append_event(
        manifest,
        "ASSEMBLY_PLAN_SAVED",
        f"装配工艺草案 v{version} 已保存；仅供规划，禁止下发真机。",
    )
    write_manifest(manifest_path_value, manifest)
    return (
        f"✅ 装配方案 v{version} 已保存。请人工复核；设备资料和仿真未完成，不能控制机械臂。",
        manifest,
        assembly_plan_summary(plan),
        plan,
        str(output_path),
    )


def _verified_assembly_plan(
    manifest_file: Path,
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    selected = manifest.get("selected_assembly_plan", {})
    if not isinstance(selected, dict) or not selected.get("path"):
        raise ValueError("当前任务还没有装配方案。")
    plan_path = (PROJECT_ROOT / selected["path"]).resolve()
    if plan_path.parent != manifest_file.parent / "assembly" or not plan_path.is_file():
        raise ValueError("装配方案路径无效或越出当前任务目录。")
    actual_sha = file_sha256(plan_path)
    if selected.get("sha256") != actual_sha:
        raise ValueError("装配方案 SHA256 与任务清单不一致。")
    archived = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") == "assembly_plan"
            and item.get("path") == selected["path"]
            and item.get("sha256") == actual_sha
        ),
        None,
    )
    if not archived:
        raise ValueError("装配方案缺少匹配的归档记录。")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("run_id") != manifest.get("run_id")
        or plan.get("mode") != "PLANNING_ONLY_NO_ROBOT_CONTROL"
        or plan.get("execution_readiness")
        != "BLOCKED_DEVICE_DOCUMENTATION_AND_SIMULATION"
    ):
        raise ValueError("装配方案安全模式或任务关联无效。")
    return plan_path, plan, archived


def load_current_assembly_plan(
    run_id: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str]:
    manifest_file, manifest = read_manifest(run_id)
    plan_path, plan, _ = _verified_assembly_plan(manifest_file, manifest)
    return (
        "✅ 已加载当前装配工艺草案。该文件不包含也不能生成真机运动命令。",
        manifest,
        assembly_plan_summary(plan),
        plan,
        str(plan_path),
    )


def confirm_assembly_plan(
    run_id: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str]:
    manifest_file, manifest = read_manifest(run_id)
    plan_path, plan, artifact = _verified_assembly_plan(manifest_file, manifest)
    selected = manifest["selected_assembly_plan"]
    selected["status"] = "PLAN_CONFIRMED_SIMULATION_REQUIRED"
    selected["confirmed_at"] = now_iso()
    artifact["approval_status"] = "PLAN_CONFIRMED_SIMULATION_REQUIRED"
    manifest.setdefault("stages", {})["assembly"] = "PLAN_CONFIRMED_SIMULATION_REQUIRED"
    manifest["status"] = "ASSEMBLY_PLAN_CONFIRMED"
    append_event(
        manifest,
        "ASSEMBLY_PLAN_CONFIRMED",
        "人工确认工艺草案；仍需设备资料、标定、仿真、空载和真机授权。",
    )
    write_manifest(manifest_file, manifest)
    return (
        "✅ 装配工艺草案已确认。注意：这不是机械臂执行批准，下一步仍是设备资料和离线仿真。",
        manifest,
        assembly_plan_summary(plan),
        plan,
        str(plan_path),
    )


def load_parametric_assembly_component(
    component_run_id: str,
) -> tuple[
    str,
    dict[str, Any],
    str,
    list[list[Any]],
    str | None,
    str | None,
    list[str],
    str,
    dict[str, Any],
    str | None,
]:
    try:
        bundle = load_component_bundle(
            component_run_id,
            project_root=PROJECT_ROOT,
            runs_root=RUNS_DIR,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return (
            f"⛔ 装配组件未加载：{exc}",
            {},
            "### 三件式装配组件摘要\n\n尚未加载有效组件。",
            [],
            None,
            None,
            [],
            fit_review_summary(None),
            {},
            None,
        )

    paths = bundle["paths"]
    download_roles = [
        "gear_print_stl",
        "shaft_print_stl",
        "cap_recommended_print_orientation_stl",
        "gear_cap_combined_print_stl",
        "assembled_review_glb",
        "exploded_review_glb",
        "assembly_process_guide",
        "assembly_fit_acceptance_template",
    ]
    downloads = [str(paths[role]) for role in download_roles if role in paths]
    latest_review: dict[str, Any] = {}
    review_path: str | None = None
    if "assembly_fit_review" in paths:
        review_path = str(paths["assembly_fit_review"])
        latest_review = json.loads(paths["assembly_fit_review"].read_text(encoding="utf-8"))
    return (
        "✅ 已加载并校验三件式参数化装配组件；所有文件均来自该任务并通过 SHA256 校验。",
        bundle["manifest"],
        component_summary(bundle),
        component_bom_rows(bundle),
        str(paths["assembly_preview"]),
        str(paths["assembly_process_sheet"]) if "assembly_process_sheet" in paths else None,
        downloads,
        fit_review_summary(latest_review or None),
        latest_review,
        review_path,
    )


def create_function_first_assembly_component(
    purpose: str,
    fit_goal: str,
    print_profile: str,
    requirement: str,
) -> tuple[Any, ...]:
    """Create and immediately load the validated template from functional intent."""
    try:
        run_id, plan, _ = create_auto_planned_component(
            purpose,
            fit_goal,
            print_profile,
            requirement,
        )
        loaded = load_parametric_assembly_component(run_id)
        return (
            run_id,
            "✅ 平台已根据功能需求自动规划并创建三件套组件；请查看预览和工艺图。",
            loaded[1],
            auto_plan_summary(plan, run_id),
            plan,
            *loaded[2:],
        )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        return (
            "",
            f"⛔ 自动规划未完成：{exc}",
            {},
            auto_plan_summary(None),
            {},
            "### 三件式装配组件摘要\n\n尚未加载有效组件。",
            [],
            None,
            None,
            [],
            fit_review_summary(None),
            {},
            None,
        )


def save_parametric_assembly_fit_review(
    component_run_id: str,
    gear_rotation_deg: float,
    gear_insertion: str,
    cap_insertion: str,
    cap_falls_when_inverted: bool,
    removal_result: str,
    visible_defects: str,
    notes: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str | None]:
    try:
        record, review_path, manifest = archive_assembly_fit_review(
            component_run_id,
            gear_rotation_deg,
            gear_insertion,
            cap_insertion,
            cap_falls_when_inverted,
            removal_result,
            visible_defects,
            notes,
            project_root=PROJECT_ROOT,
            runs_root=RUNS_DIR,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return f"⛔ 试装结果未保存：{exc}", {}, fit_review_summary(None), {}, None
    message = (
        "✅ 试装结果已归档，当前满足验收判据。"
        if record["status"] == "PASS"
        else "⚠️ 试装结果已归档，但仍需继续调整或人工复核。"
    )
    return message, manifest, fit_review_summary(record), record, str(review_path)


def preview_device_center(scenario: str) -> tuple[str, str, dict[str, Any]]:
    snapshot = build_device_snapshot(scenario)
    return (
        "✅ 已刷新安全设备视图；没有探测、连接或控制任何真实设备。",
        device_snapshot_summary(snapshot),
        snapshot,
    )


def _next_device_snapshot_version(run_dir: Path) -> int:
    versions: list[int] = []
    for item in (run_dir / "devices").glob("device_snapshot_v*.json"):
        match = re.fullmatch(r"device_snapshot_v(\d+)\.json", item.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def save_device_snapshot(
    run_id: str,
    scenario: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str | None]:
    manifest_file, manifest = read_manifest(run_id)
    try:
        snapshot = build_device_snapshot(scenario)
    except ValueError as exc:
        return f"⛔ 设备快照未保存：{exc}", manifest, device_snapshot_summary(None), {}, None
    run_dir = manifest_file.parent
    device_dir = run_dir / "devices"
    device_dir.mkdir(parents=True, exist_ok=True)
    version = _next_device_snapshot_version(run_dir)
    output_path = device_dir / f"device_snapshot_v{version}.json"
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有设备快照：{output_path}")
    snapshot["run_id"] = manifest["run_id"]
    snapshot["version"] = version
    write_manifest(output_path, snapshot)
    relative = project_relative(output_path)
    snapshot_sha = file_sha256(output_path)
    manifest.setdefault("artifacts", []).append(
        {
            "role": "device_snapshot",
            "version": version,
            "path": relative,
            "bytes": output_path.stat().st_size,
            "sha256": snapshot_sha,
            "source": snapshot["source"],
        }
    )
    manifest["selected_device_snapshot"] = {
        "version": version,
        "path": relative,
        "sha256": snapshot_sha,
        "source": snapshot["source"],
        "mode": snapshot["mode"],
        "control_allowed": False,
    }
    task3 = manifest.setdefault("task3", {})
    task3["device_center"] = "SIMULATION_OR_KNOWN_FACTS_ONLY"
    task3["real_devices_connected"] = False
    append_event(
        manifest,
        "DEVICE_SNAPSHOT_SAVED",
        f"设备快照 v{version} 已保存；来源 {snapshot['source']}，未连接真机。",
    )
    write_manifest(manifest_file, manifest)
    return (
        f"✅ 设备快照 v{version} 已归档到当前任务；它不代表真机已连接。",
        manifest,
        device_snapshot_summary(snapshot),
        snapshot,
        str(output_path),
    )


def _verified_device_snapshot(
    manifest_file: Path,
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    selected = manifest.get("selected_device_snapshot", {})
    if not isinstance(selected, dict) or not selected.get("path"):
        raise ValueError("当前任务还没有设备快照。")
    snapshot_path = (PROJECT_ROOT / str(selected["path"])).resolve()
    if snapshot_path.parent != manifest_file.parent / "devices" or not snapshot_path.is_file():
        raise ValueError("设备快照路径无效或越出当前任务目录。")
    actual_sha = file_sha256(snapshot_path)
    if actual_sha != selected.get("sha256"):
        raise ValueError("设备快照 SHA256 与任务清单不一致。")
    archived = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("role") == "device_snapshot"
            and item.get("path") == selected.get("path")
            and item.get("sha256") == actual_sha
        ),
        None,
    )
    if not archived:
        raise ValueError("设备快照缺少匹配的归档记录。")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    safety = snapshot.get("safety", {})
    if (
        snapshot.get("run_id") != manifest.get("run_id")
        or snapshot.get("mode") != "READ_ONLY_NO_DEVICE_COMMANDS"
        or safety.get("real_hardware_contacted") is not False
        or safety.get("control_commands_available") is not False
        or safety.get("qwen_may_control_devices") is not False
        or any(item.get("control_allowed") is not False for item in snapshot.get("devices", []))
    ):
        raise ValueError("设备快照的安全模式无效。")
    return snapshot_path, snapshot


def load_current_device_snapshot(
    run_id: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str]:
    manifest_file, manifest = read_manifest(run_id)
    snapshot_path, snapshot = _verified_device_snapshot(manifest_file, manifest)
    return (
        "✅ 已加载当前任务的设备快照；没有进行新的网络或串口探测。",
        manifest,
        device_snapshot_summary(snapshot),
        snapshot,
        str(snapshot_path),
    )


def _verified_capture_records(
    manifest_file: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    files: list[str] = []
    for selected in manifest.get("multimodal_captures", []):
        if not isinstance(selected, dict):
            raise ValueError("多模态采集清单格式无效。")
        record_path = (PROJECT_ROOT / str(selected.get("record_path", ""))).resolve()
        media_path = (PROJECT_ROOT / str(selected.get("media_path", ""))).resolve()
        captures_root = (manifest_file.parent / "multimodal" / "captures").resolve()
        if (
            captures_root not in record_path.parents
            or captures_root not in media_path.parents
            or not record_path.is_file()
            or not media_path.is_file()
        ):
            raise ValueError("多模态采集路径无效或越出当前任务目录。")
        if (
            file_sha256(record_path) != selected.get("record_sha256")
            or file_sha256(media_path) != selected.get("media_sha256")
        ):
            raise ValueError("多模态采集文件 SHA256 与任务清单不一致。")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        safety = record.get("safety", {})
        if (
            record.get("run_id") != manifest.get("run_id")
            or safety.get("device_command_generated") is not False
            or safety.get("robot_control_allowed") is not False
        ):
            raise ValueError("多模态采集记录的安全模式无效。")
        records.append(record)
        files.extend([str(media_path), str(record_path)])
    return records, files


def save_multimodal_capture(
    run_id: str,
    upload: Any,
    checkpoint: str,
    part_id: str,
    zone: str,
    label: str,
    notes: str,
) -> tuple[str, dict[str, Any], str, list[list[str]], list[str]]:
    manifest_file, manifest = read_manifest(run_id)
    existing, existing_files = _verified_capture_records(manifest_file, manifest)
    try:
        archived = archive_manual_capture(
            project_root=PROJECT_ROOT,
            run_dir=manifest_file.parent,
            run_id=manifest["run_id"],
            upload=upload,
            checkpoint=checkpoint,
            part_id=part_id,
            zone=zone,
            label=label,
            notes=notes,
            device_snapshot=manifest.get("selected_device_snapshot"),
        )
    except ValueError as exc:
        return (
            f"⛔ 采集记录未保存：{exc}",
            manifest,
            capture_summary(existing),
            capture_table(existing),
            existing_files,
        )

    record = archived["record"]
    selected = {
        "capture_id": record["capture_id"],
        "record_path": archived["record_path"],
        "record_sha256": archived["record_sha256"],
        "media_path": archived["media_path"],
        "media_sha256": archived["media_sha256"],
    }
    manifest.setdefault("multimodal_captures", []).append(selected)
    for role, relative, digest in (
        ("multimodal_capture_media", archived["media_path"], archived["media_sha256"]),
        ("multimodal_capture_record", archived["record_path"], archived["record_sha256"]),
    ):
        artifact_path = PROJECT_ROOT / relative
        manifest.setdefault("artifacts", []).append(
            {
                "role": role,
                "capture_id": record["capture_id"],
                "path": relative,
                "bytes": artifact_path.stat().st_size,
                "sha256": digest,
            }
        )
    task3 = manifest.setdefault("task3", {})
    task3["capture_framework"] = "MANUAL_UPLOAD_VERSIONED_V1"
    task3["capture_count"] = len(manifest["multimodal_captures"])
    task3["live_camera_connected"] = False
    task3["dataset_quality"] = "MANUAL_LABELS_WAITING_QA"
    append_event(
        manifest,
        "MULTIMODAL_CAPTURE_SAVED",
        f"{record['capture_id']} 已归档；来源为人工上传，未连接相机或控制设备。",
    )
    write_manifest(manifest_file, manifest)
    records, files = _verified_capture_records(manifest_file, manifest)
    return (
        f"✅ 已保存采集记录 {record['capture_id']}；当前标签仍需人工质检。",
        manifest,
        capture_summary(records),
        capture_table(records),
        files,
    )


def load_current_multimodal_captures(
    run_id: str,
) -> tuple[str, dict[str, Any], str, list[list[str]], list[str]]:
    manifest_file, manifest = read_manifest(run_id)
    records, files = _verified_capture_records(manifest_file, manifest)
    return (
        f"✅ 已加载并校验 {len(records)} 条多模态采集记录。",
        manifest,
        capture_summary(records),
        capture_table(records),
        files,
    )


def attach_demo_assets(run_id: str) -> tuple[str, dict[str, Any]]:
    assets = checked_demo_assets()
    path, manifest = read_manifest(run_id)
    records = []
    for asset in assets:
        records.append(
            {
                "role": "phase1_demo",
                "path": project_relative(asset),
                "bytes": asset.stat().st_size,
                "sha256": file_sha256(asset),
            }
        )
    manifest["artifacts"] = records
    append_event(manifest, "DEMO_ASSETS_ATTACHED", "关联第一阶段已验证结果。")
    write_manifest(path, manifest)
    return "✅ 已关联第一阶段涡扇结果，未复制或修改原文件。", manifest


def list_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("RUN-*/manifest.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_id": data.get("run_id"),
                    "status": data.get("status"),
                    "part_type": data.get("request", {}).get("part_type"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return summaries


def status_label(status: Any) -> str:
    raw = str(status or "UNKNOWN")
    return STATUS_LABELS.get(raw, raw)


def stage_status_label(status: Any) -> str:
    raw = str(status or "UNKNOWN")
    return STAGE_STATUS_LABELS.get(raw, raw)


def next_action_text(manifest: dict[str, Any]) -> str:
    stages = manifest.get("stages", {})
    artifacts = manifest.get("artifacts", [])
    if stages.get("prompt") != "CONFIRMED":
        return "确认最终提示词"
    if stages.get("candidate") != "CONFIRMED":
        if not any(
            isinstance(item, dict) and item.get("role") == "ernie_candidate"
            for item in artifacts
        ):
            return "启动 ERNIE 生成候选图片"
        if stages.get("candidate") == "MANUAL_REJECTED":
            return "选择其他候选或重新生成候选图片"
        return "人工复核并确认候选图片（Qwen 检查可选）"
    if stages.get("mask") != "CONFIRMED":
        if manifest.get("selected_mask"):
            return "检查三联图并人工确认透明遮罩"
        return "CPU 生成透明遮罩"
    if stages.get("mesh") != "CONFIRMED":
        selected_mesh = manifest.get("selected_mesh", {})
        mesh_status = selected_mesh.get("status") if isinstance(selected_mesh, dict) else None
        if mesh_status == "RAW_WAITING_REVIEW":
            return "CPU 检查并整理 Hunyuan 原始 GLB"
        if mesh_status == "REVIEW_READY_WAITING_CONFIRMATION":
            return "检查 GLB 与多视图并人工确认"
        return "启动 Hunyuan Shape"
    if not manifest.get("selected_stl"):
        return "勾选切片检查说明并导出 STL"
    return "在 Bambu Studio 中检查尺寸、朝向、首层、支撑和逐层路径"


def _markdown_cell(value: Any) -> str:
    return str(value if value not in (None, "") else "—").replace("|", "\\|").replace("\n", " ")


def manifest_summary(manifest: dict[str, Any] | None) -> str:
    """Render a compact human-readable task summary; tolerate legacy manifests."""
    if not isinstance(manifest, dict) or not manifest:
        return "### 当前任务摘要\n\n尚未创建或加载任务。"
    request = manifest.get("request", {})
    if not isinstance(request, dict):
        request = {}
    stages = manifest.get("stages", {})
    if not isinstance(stages, dict):
        stages = {}
    stage_labels = {
        "prompt": "提示词",
        "candidate": "候选图片",
        "mask": "透明遮罩",
        "mesh": "三维模型",
        "assembly": "装配方案",
    }
    stage_text = " · ".join(
        f"{stage_labels[key]}：{stage_status_label(stages.get(key))}"
        for key in ("prompt", "candidate", "mask", "mesh", "assembly")
    )
    artifacts = manifest.get("artifacts", [])
    candidate_count = sum(
        1
        for item in artifacts
        if isinstance(item, dict) and item.get("role") == "ernie_candidate"
    )
    review = manifest.get("candidate_review", {})
    qwen_decision = review.get("decision", "未检查") if isinstance(review, dict) else "未检查"
    profile_data = request.get("part_profile", {})
    if isinstance(profile_data, dict) and profile_data.get("label"):
        profile_name = profile_data["label"]
    else:
        try:
            profile_name = resolve_part_profile(str(request.get("part_type", "自定义零件"))).label
        except ValueError:
            profile_name = "旧任务/未知"
    count = request.get("structure_count", "—")
    return (
        "### 当前任务摘要\n\n"
        "| 项目 | 内容 |\n|---|---|\n"
        f"| 任务编号 | `{_markdown_cell(manifest.get('run_id'))}` |\n"
        f"| 零件 | {_markdown_cell(request.get('part_type'))}（{_markdown_cell(profile_name)}配置） |\n"
        f"| 当前状态 | **{_markdown_cell(status_label(manifest.get('status')))}** |\n"
        f"| 重复结构数量 | {_markdown_cell(count)} |\n"
        f"| 已生成候选 | {candidate_count} 张 |\n"
        f"| Qwen 结果 | {_markdown_cell(qwen_decision)} |\n"
        f"| 建议下一步 | **{_markdown_cell(next_action_text(manifest))}** |\n"
        f"| 更新时间 | {_markdown_cell(manifest.get('updated_at'))} |\n\n"
        f"流程：{_markdown_cell(stage_text)}"
    )


def list_runs_table() -> list[list[str]]:
    rows: list[list[str]] = []
    for item in list_runs():
        rows.append(
            [
                str(item.get("run_id") or ""),
                str(item.get("part_type") or ""),
                status_label(item.get("status")),
                str(item.get("created_at") or ""),
                str(item.get("updated_at") or ""),
            ]
        )
    return rows


def history_run_id(table: Any, row_index: int) -> str:
    """Resolve a selected history row without trusting a browser-provided path."""
    index = int(row_index)
    if hasattr(table, "iloc"):
        value = table.iloc[index, 0]
    else:
        value = table[index][0]
    normalized = str(value or "").strip()
    if not RUN_ID_PATTERN.fullmatch(normalized):
        raise ValueError("历史任务行中没有有效任务编号。")
    return normalized


def load_existing_run(run_id: str) -> tuple[str, str, dict[str, Any]]:
    """Load an archived task without changing its manifest."""
    normalized_run_id = (run_id or "").strip()
    _, manifest = read_manifest(normalized_run_id)
    return (
        normalized_run_id,
        f"✅ 已加载 `{manifest['run_id']}`；当前状态："
        f"`{manifest.get('status', 'UNKNOWN')}`。",
        manifest,
    )


def _current_candidate_gallery(
    manifest_file: Path,
    manifest: dict[str, Any],
) -> list[tuple[str, str]]:
    candidates = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("role") == "ernie_candidate"
    ]
    numbered_attempts = [
        int(item["attempt"])
        for item in candidates
        if isinstance(item.get("attempt"), int)
    ]
    if numbered_attempts:
        latest_attempt = max(numbered_attempts)
        candidates = [item for item in candidates if item.get("attempt") == latest_attempt]
    candidates.sort(key=lambda item: int(item.get("index", 0)))
    gallery: list[tuple[str, str]] = []
    for item in candidates:
        relative = item.get("path")
        if not relative:
            continue
        absolute = (PROJECT_ROOT / relative).resolve()
        if manifest_file.parent.resolve() not in absolute.parents or not absolute.is_file():
            continue
        caption = f"候选 {item.get('index', '?')}"
        if item.get("seed") is not None:
            caption += f" · seed {item['seed']}"
        gallery.append((str(absolute), caption))
    return gallery


def _current_qwen_review(
    manifest_file: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    review_record = manifest.get("candidate_review", {})
    if not isinstance(review_record, dict):
        return {}
    relative = review_record.get("result_path")
    if not relative:
        return {
            key: review_record[key]
            for key in ("decision", "human_decision", "human_reason")
            if key in review_record
        }
    result_path = (PROJECT_ROOT / relative).resolve()
    if manifest_file.parent.resolve() not in result_path.parents or not result_path.is_file():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    value = payload.get("review", payload)
    return value if isinstance(value, dict) else {}


def qwen_review_summary(review: dict[str, Any] | None) -> str:
    if not isinstance(review, dict) or not review:
        return "### Qwen 检查摘要\n\n尚未检查当前候选。"
    decision = str(review.get("decision", "UNKNOWN"))
    decision_text = {
        "PASS": "通过自动检查，仍需人工确认",
        "REVIEW": "需要人工复核",
        "REJECT": "已拦截，不能进入三维",
    }.get(decision, decision)
    count = review.get("observed_repeat_count")
    count_text = "未能可靠计数" if count is None else str(count)
    forbidden = review.get("forbidden_elements_found") or []
    issues = review.get("structural_issues") or []
    return (
        "### Qwen 检查摘要\n\n"
        f"- 结论：**{_markdown_cell(decision_text)}**\n"
        f"- 实际观察数量：{_markdown_cell(count_text)}\n"
        f"- 主体完整：{_markdown_cell(review.get('subject_complete'))}\n"
        f"- 背景干净：{_markdown_cell(review.get('background_clean'))}\n"
        f"- 适合三维观察：{_markdown_cell(review.get('view_suitable_for_3d'))}\n"
        f"- 禁止元素：{_markdown_cell('、'.join(map(str, forbidden)) if forbidden else '未发现')}\n"
        f"- 结构问题：{_markdown_cell('、'.join(map(str, issues)) if issues else '未报告')}\n"
        f"- 说明：{_markdown_cell(review.get('reason'))}"
    )


def mask_report_summary(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict) or not report:
        return "### 遮罩检查摘要\n\n尚未生成或加载当前任务遮罩。"
    corners = report.get("corner_alpha")
    corners_clear = corners == [0, 0, 0, 0]
    components = report.get("final_foreground_components")
    holes = report.get("large_transparent_hole_count")
    fraction = report.get("foreground_fraction")
    fraction_text = (
        f"{float(fraction) * 100:.2f}%" if isinstance(fraction, (int, float)) else "未知"
    )
    basic_pass = corners_clear and components == 1
    return (
        "### 遮罩检查摘要\n\n"
        f"- 基础检查：**{'通过，仍需人工看孔洞' if basic_pass else '未通过'}**\n"
        f"- 外部四角透明：{'是' if corners_clear else '否'}\n"
        f"- 零件主体数量：{_markdown_cell(components)} 个\n"
        f"- 保留的内部透明区域：{_markdown_cell(holes)} 个\n"
        f"- 零件前景占比：{fraction_text}\n"
        f"- 当前方法：{_markdown_cell(report.get('method'))}\n\n"
        "请在三联图中确认孔洞透明、实体没有被挖空；数量正确也不能代替人工检查。"
    )


def mesh_report_summary(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict) or not report:
        return "### 网格检查摘要\n\n尚未生成或加载当前任务的 review GLB。"
    components = report.get("components")
    watertight = report.get("watertight") is True
    winding = report.get("winding_consistent") is True
    boundary = int(report.get("boundary_edges", -1) or 0)
    nonmanifold = int(report.get("nonmanifold_edges", -1) or 0)
    degenerate = int(report.get("degenerate_faces", -1) or 0)
    gate = components == 1 and watertight and winding and boundary == 0 and nonmanifold == 0
    extents = report.get("extents")
    extent_text = (
        " × ".join(f"{float(value):.4f}" for value in extents)
        if isinstance(extents, list) and len(extents) == 3
        else "未知"
    )
    return (
        "### 网格检查摘要\n\n"
        f"- 通用几何门禁：**{'通过，仍需人工看多视图' if gate else '未通过'}**\n"
        f"- 顶点 / 三角面：{_markdown_cell(report.get('vertices'))} / "
        f"{_markdown_cell(report.get('faces'))}\n"
        f"- 连通部件：{_markdown_cell(components)}\n"
        f"- 水密 / 法向一致：{'是' if watertight else '否'} / "
        f"{'是' if winding else '否'}\n"
        f"- 边界边 / 非流形边：{boundary} / {nonmanifold}\n"
        f"- 极小退化面：{degenerate}\n"
        f"- 当前坐标包围盒：{extent_text}\n\n"
        "包围盒在导出前可能仍是无单位坐标；最终毫米尺寸以 STL 导出报告为准。"
    )


def mesh_repair_report_summary(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict) or not report:
        return "### 模型修复摘要\n\n尚未对当前模型执行修复诊断。"
    before = report.get("before", {})
    after = report.get("after_export_reload") or report.get("after_in_memory")
    triage = report.get("automatic_repairability_before", {})
    route = triage.get("route", "UNKNOWN")
    reasons = triage.get("reasons") or []

    def metric_line(values: dict[str, Any]) -> str:
        return (
            f"部件 {values.get('components', '未知')}；"
            f"水密 {'是' if values.get('watertight') is True else '否'}；"
            f"边界边 {values.get('boundary_edges', '未知')}；"
            f"非流形边 {values.get('nonmanifold_edges', '未知')}；"
            f"退化面 {values.get('degenerate_faces', '未知')}"
        )

    operations = report.get("accepted_operations") or []
    lines = [
        "### 模型修复摘要",
        "",
        f"- 自动分流：**{_markdown_cell(route)}**",
        f"- 原模型：{metric_line(before)}",
        f"- 报告状态：**{_markdown_cell(report.get('status'))}**",
    ]
    if after:
        lines.append(f"- 修复候选：{metric_line(after)}")
    if operations:
        lines.append(f"- 已接受的操作：{_markdown_cell('、'.join(map(str, operations)))}")
    if reasons:
        lines.append(f"- 推荐原因：{_markdown_cell('；'.join(map(str, reasons)))}")
    reconstruction = next(
        (
            item
            for item in report.get("repair_operations") or []
            if item.get("name")
            == "dominant_component_voxel_reconstruction_candidate"
        ),
        None,
    )
    if reconstruction:
        distance = reconstruction.get("surface_distance") or {}
        lines.extend(
            [
                "",
                "#### ⚠️ 严重模型重建说明",
                "",
                f"- 主体表面积占比：{float(reconstruction.get('main_component_area_share', 0)):.2%}",
                f"- 丢弃的微小输入碎片：{_markdown_cell(reconstruction.get('discarded_input_component_count'))} 个",
                f"- 最大包围盒尺寸变化：{float(reconstruction.get('max_extent_relative_change', 0)):.2%}",
                f"- 双向表面距离 P99 / 对角线：{float(distance.get('p99_to_diagonal_ratio', 0)):.2%}",
                f"- 风险：**{_markdown_cell(reconstruction.get('semantic_warning'))}**",
                "- 结论：拓扑通过不等于结构正确；预览不符合目标外形时应拒绝候选。",
            ]
        )
    artifact_cleanup = next(
        (
            item
            for item in report.get("repair_operations") or []
            if item.get("name") == "remove_tiny_detached_artifacts_candidate"
        ),
        None,
    )
    if artifact_cleanup:
        retriangulation = artifact_cleanup.get("local_degenerate_retriangulation") or {}
        lines.extend(
            [
                "",
                "#### 独立小伪影清理证据",
                "",
                f"- 主体表面积占比：{float(artifact_cleanup.get('main_area_share', 0)):.4%}",
                f"- 清理的独立小部件：{_markdown_cell(artifact_cleanup.get('discarded_component_count'))} 个",
                f"- 清理部分表面积占比：{float(artifact_cleanup.get('discarded_area_share', 0)):.6%}",
                f"- 清理部分体积占比：{float(artifact_cleanup.get('discarded_volume_share', 0)):.6%}",
                f"- 局部退化面重划分：{'通过' if retriangulation.get('accepted') else '未执行或未通过'}；不移动顶点",
                f"- 风险：**{_markdown_cell(artifact_cleanup.get('semantic_warning'))}**",
            ]
        )
    lines.extend(
        [
            "",
            "诊断和拓扑门禁不识别齿数、孔径、公差或配合面。"
            "只有人工接受后，候选才会成为新的 review GLB；原始 GLB 永不覆盖。",
        ]
    )
    return "\n".join(lines)


def gear_analysis_summary(
    report: dict[str, Any] | None,
    known_outside_diameter_mm: float = 0,
) -> str:
    if not isinstance(report, dict) or not isinstance(report.get("input"), dict):
        return "### 齿轮专用指标\n\n尚未分析当前模型。"
    values = report["input"]
    try:
        known_mm = float(known_outside_diameter_mm or 0)
    except (TypeError, ValueError):
        known_mm = 0.0
    outside = float(values.get("outside_diameter_mesh_units", 0) or 0)
    scale = known_mm / outside if known_mm > 0 and outside > 0 else None

    def dimension(field: str) -> str:
        raw = values.get(field)
        if not isinstance(raw, (int, float)):
            return "未知"
        if scale is None:
            return f"{float(raw):.5f} 模型单位"
        return f"{float(raw) * scale:.3f} mm"

    expected = values.get("estimated_tooth_count")
    confidence = values.get("confidence_ratio_to_second")
    confidence_text = (
        f"{float(confidence):.2f}"
        if isinstance(confidence, (int, float))
        else "未知"
    )
    unit_note = (
        f"已按外径 **{known_mm:.3f} mm** 等比换算；该输入值需由图纸或实测确认。"
        if scale is not None
        else "当前显示模型单位；填写已知外径后可等比换算为毫米。"
    )
    return (
        "### 齿轮专用指标\n\n"
        f"- FFT 估算齿数：**{_markdown_cell(expected)}**\n"
        f"- 齿数信号置信比：**{confidence_text}**；"
        f"{'达到当前阈值' if values.get('tooth_count_signal_confident') else '低于当前阈值，必须人工复核'}\n"
        f"- 外径：{dimension('outside_diameter_mesh_units')}\n"
        f"- 齿根径：{dimension('root_diameter_mesh_units')}\n"
        f"- 中心孔径：{dimension('bore_diameter_mesh_units')}\n"
        f"- 厚度：{dimension('thickness_mesh_units')}\n"
        f"- 由外径反推模数：{dimension('module_from_outside_diameter_mesh_units')}\n\n"
        f"{unit_note} 压力角、侧隙、公差和渐开线精度不能由本面板自动确认。"
    )


def stl_size_preview(
    report: dict[str, Any] | None,
    target_longest_mm: float,
) -> str:
    """Preview uniform STL scaling from the currently loaded mesh report."""
    try:
        target = float(target_longest_mm)
    except (TypeError, ValueError):
        return "### STL 尺寸预览\n\n请输入 20–200 mm 的目标最长尺寸。"
    if not 20.0 <= target <= 200.0:
        return "### STL 尺寸预览\n\n目标最长尺寸必须在 20–200 mm。"
    if not isinstance(report, dict) or not report:
        return (
            "### STL 尺寸预览\n\n"
            f"本次目标最长尺寸：**{target:.3f} mm**。加载网格检查后显示预计 X/Y/Z。"
        )
    extents = report.get("extents")
    if not (
        isinstance(extents, list)
        and len(extents) == 3
        and all(isinstance(value, (int, float)) for value in extents)
    ):
        return (
            "### STL 尺寸预览\n\n"
            f"本次目标最长尺寸：**{target:.3f} mm**；当前报告缺少有效包围盒。"
        )
    source_longest = max(float(value) for value in extents)
    if source_longest <= 0:
        return "### STL 尺寸预览\n\n当前网格包围盒无效，禁止导出。"
    predicted = [float(value) * target / source_longest for value in extents]
    predicted_text = " × ".join(f"{value:.3f}" for value in predicted)
    return (
        "### STL 尺寸预览\n\n"
        f"- 本次目标最长尺寸：**{target:.3f} mm**\n"
        f"- 预计 X × Y × Z：**{predicted_text} mm**\n"
        "- 缩放方式：三轴等比例；不会自动修正孔径、壁厚或齿形。\n"
        "- 每次导出都会创建新版本，不覆盖已有 STL。"
    )


def local_action_status(status: Any) -> str:
    return f"### 本页最近操作\n\n{str(status or '本页暂无操作。')}"


def manufacturing_guidance(part_type: str) -> str:
    profile = resolve_part_profile(part_type)
    if profile.key == "ducted_fan":
        return (
            "✅ 当前零件可使用下方七叶片 CPU 制造规则化，但它只适用于单体七叶片"
            "风扇，不代表 Hunyuan 原始输出。"
        )
    return (
        f"⚠️ 当前为{profile.label}配置：通用 Hunyuan 生成、网格检查和 STL 门禁可用，"
        "但尚无专用确定性制造规则化。不要使用下方七叶片参数面板生成此类零件。"
    )


def profile_ui_defaults(part_type: str) -> tuple[str, str, int, str, str]:
    requirement, forbidden, count, guidance = profile_defaults(part_type)
    return requirement, forbidden, count, guidance, manufacturing_guidance(part_type)


def current_artifact_context(run_id: str) -> tuple[Any, ...]:
    """Restore only verified artifacts owned by the selected run; otherwise clear UI."""
    mask_values: tuple[Any, ...] = (None, None, None, {})
    mesh_values: tuple[Any, ...] = (None, None, {}, [])
    stl_value: str | None = None
    try:
        original, rgba, preview, report, _ = load_current_mask(run_id)
        mask_values = (original, rgba, preview, report)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        pass
    try:
        mesh, projection, report, files, _ = load_current_mesh_review(run_id)
        mesh_values = (mesh, projection, report, files)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        pass

    manifest_file, manifest = read_manifest(run_id)
    selected_stl = manifest.get("selected_stl", {})
    if isinstance(selected_stl, dict) and selected_stl.get("path"):
        candidate = (PROJECT_ROOT / str(selected_stl["path"])).resolve()
        if (
            candidate.parent == manifest_file.parent / "stl"
            and candidate.is_file()
            and file_sha256(candidate) == selected_stl.get("sha256")
        ):
            stl_value = str(candidate)
    return (*mask_values, *mesh_values, stl_value)


def load_existing_run_context(run_id: str) -> tuple[Any, ...]:
    """Load a task and restore its form/gallery/review without changing the manifest."""
    normalized, status, manifest = load_existing_run(run_id)
    manifest_file = manifest_path(normalized)
    request = manifest.get("request", {})
    prompt = manifest.get("prompt", {})
    parameters = manifest.get("parameters", {})
    part_type = str(request.get("part_type", "自定义零件"))
    selected_candidate = manifest.get("selected_candidate", {})
    review_record = manifest.get("candidate_review", {})
    candidate_index = (
        selected_candidate.get("candidate_index")
        if isinstance(selected_candidate, dict)
        else None
    ) or (
        review_record.get("candidate_index")
        if isinstance(review_record, dict)
        else None
    ) or 1
    assembly_values = assembly_defaults(part_type)
    artifact_values = current_artifact_context(normalized)
    return (
        normalized,
        status + " 已恢复表单、最新候选图和已有 Qwen 结果。",
        manifest,
        part_type,
        str(request.get("purpose", "三维打印验证")),
        str(request.get("requirement", "")),
        str(request.get("forbidden_elements", "")),
        int(request.get("structure_count", 0) or 0),
        float(request.get("target_size_mm", 60) or 60),
        int(parameters.get("seed", 0) or 0),
        str(prompt.get("text", "")) if isinstance(prompt, dict) else "",
        _current_candidate_gallery(manifest_file, manifest),
        int(candidate_index),
        _current_qwen_review(manifest_file, manifest),
        profile_defaults(part_type)[3],
        manufacturing_guidance(part_type),
        *assembly_values,
        *artifact_values,
        float(request.get("target_size_mm", 60) or 60),
    )


def build_app():
    import gradio as gr

    checked_demo_assets()
    with gr.Blocks(
        title="工业生成式 AI 工作台",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# 工业生成式 AI 工作台 · 服务原型")
        gr.Markdown(
            "界面：**可用**　|　ERNIE-Image：**默认后端，按需加载**　|　"
            "Qwen：**已接入，按需加载**　|　Hunyuan：**已接入，按需加载**　|　"
            "打印机：**未连接**　|　机械臂：**未连接**　|　"
            "设备中心：**模拟/只读框架**",
            elem_classes="status-strip",
        )
        gr.Markdown(
            "点击“启动 ERNIE”会在执行 `nvidia-smi` 安全检查后运行真实文生图；"
            "Qwen 检查使用相同的串行 GPU 门禁。检测到其他 GPU 计算进程时"
            "不会启动。Hunyuan 只接受已确认遮罩；七叶片规则化使用 CPU 并"
            "自动产生四视图。打印机和机械臂仍未接入；设备中心的模拟状态不代表真机在线。",
            elem_classes="warning-strip",
        )

        with gr.Row():
            run_id = gr.Textbox(
                label="当前/已有任务 ID", interactive=True, scale=2
            )
            load_run_button = gr.Button("加载已有任务", scale=1)
            workflow_status = gr.Markdown("尚未创建任务。", scale=3)

        with gr.Tabs():
            with gr.Tab("1 任务与提示词"):
                tab1_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "**本页完成条件：**生成任务编号，并点击“确认最终提示词”；然后进入第 2 页。",
                    elem_classes="small-note",
                )
                with gr.Row():
                    with gr.Column():
                        part_type = gr.Dropdown(
                            PROFILE_CHOICES,
                            value="涵道风扇",
                            label="零部件类型（可选择或直接输入）",
                            allow_custom_value=True,
                        )
                        profile_guidance = gr.Markdown(
                            profile_defaults("涵道风扇")[3],
                            elem_classes="small-note",
                        )
                        purpose = gr.Dropdown(
                            ["静态教学展示", "三维打印验证", "装配场景演示"],
                            value="三维打印验证",
                            label="用途",
                        )
                        requirement = gr.Textbox(
                            value=(
                                "七片宽而厚的叶片，叶片与短而封闭的中心轮毂和"
                                "浅圆筒涵道连续连接，轮毂无轴孔"
                            ),
                            label="结构要求",
                            lines=4,
                        )
                        forbidden_elements = gr.Textbox(
                            value=(
                                "多余叶片、螺丝、螺栓、轴孔、细杆、长轴、格栅、"
                                "第二层叶片、悬浮碎片、文字、底座、支架"
                            ),
                            label="禁止元素",
                            lines=3,
                        )
                        structure_count = gr.Number(
                            value=7,
                            precision=0,
                            label="主要重复结构数量（自定义零件可填 0）",
                        )
                        target_size = gr.Number(value=60, label="目标最长尺寸（mm）")
                        seed = gr.Number(value=20260834, precision=0, label="随机种子")
                        create_button = gr.Button("创建任务并生成提示词", variant="primary")
                    with gr.Column():
                        final_prompt = gr.Textbox(
                            label="最终提示词预览",
                            lines=12,
                            placeholder="创建任务后在此显示。",
                        )
                        confirm_prompt_button = gr.Button("确认最终提示词")
                        manifest_summary_view = gr.Markdown(
                            manifest_summary(None),
                            elem_classes="status-strip",
                        )
                        with gr.Accordion("高级信息：完整任务 JSON", open=False):
                            manifest_view = gr.JSON(label="任务原始档案")

            with gr.Tab("2 文生图与检查"):
                tab2_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "**本页完成条件：**先运行 ERNIE，再由人工检查并确认候选；"
                    "Qwen 检查是可选参考，不作为进入第 3 页的门禁。模型在独立进程中顺序运行。",
                    elem_classes="small-note",
                )
                with gr.Row():
                    candidate_count = gr.Slider(
                        1, 4, value=1, step=1, label="本次候选图片数量"
                    )
                    candidate_resolution = gr.Dropdown(
                        [512, 768, 1024],
                        value=768,
                        label="候选筛选分辨率",
                    )
                    generate_ernie_button = gr.Button(
                        "启动 ERNIE 文生图", variant="primary"
                    )
                generated_gallery = gr.Gallery(
                    value=[],
                    label="本任务新生成候选图",
                    columns=2,
                    rows=2,
                    height=520,
                )
                with gr.Row():
                    candidate_index = gr.Number(
                        value=1,
                        precision=0,
                        label="要检查的候选编号（按最新一次生成）",
                    )
                    review_qwen_button = gr.Button("使用 Qwen 检查候选（可选）")
                qwen_summary_view = gr.Markdown(
                    qwen_review_summary(None),
                    elem_classes="status-strip",
                )
                with gr.Accordion("高级信息：Qwen 原始 JSON", open=False):
                    qwen_review_result = gr.JSON(label="Qwen 结构化检查结果")
                gr.Markdown(
                    "人工确认是进入遮罩步骤的唯一候选门禁。Qwen 的 PASS、REVIEW、"
                    "REJECT 均作为风险提示保留，不会覆盖人工确认。",
                    elem_classes="small-note",
                )
                human_review_reason = gr.Textbox(
                    label="人工复核意见",
                    placeholder="例如：实际约8片叶片，且轮毂出现轴孔和紧固件。",
                )
                with gr.Row():
                    confirm_candidate_button = gr.Button("人工确认候选通过")
                    reject_candidate_button = gr.Button(
                        "人工拒绝候选", variant="stop"
                    )
                gr.Markdown("### 第一阶段历史参考")
                with gr.Row():
                    candidate_image = gr.Image(
                        value=str(DEMO_IMAGE), label="历史候选图", interactive=False
                    )
                    review_result = gr.Markdown(DEMO_REVIEW)
                attach_button = gr.Button("关联历史演示结果到当前任务")

            with gr.Tab("3 透明遮罩"):
                tab3_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "**本页完成条件：**生成遮罩，检查三联图中的孔洞、实体和阴影，"
                    "确认无误后点击“确认遮罩”，再进入第 4 页。适用于纯白或浅色简洁背景。"
                )
                with gr.Row():
                    mask_threshold = gr.Slider(
                        5, 120, value=40, step=1, label="背景颜色距离阈值"
                    )
                    mask_border_width = gr.Slider(
                        1, 100, value=20, step=1, label="背景边缘采样宽度（px）"
                    )
                    mask_minimum_hole = gr.Number(
                        value=100, precision=0, label="小于此面积的透明噪点将填补（px）"
                    )
                    mask_trim_row = gr.Number(
                        value=0,
                        precision=0,
                        label="可选底部裁剪行（0=不裁剪）",
                    )
                with gr.Row():
                    generate_mask_button = gr.Button(
                        "CPU 生成新遮罩", variant="primary"
                    )
                    load_mask_button = gr.Button("加载当前任务遮罩")
                with gr.Row():
                    original_image = gr.Image(
                        value=None, label="当前任务候选原图", interactive=False
                    )
                    mask_image = gr.Image(
                        value=None, label="当前任务 RGBA 遮罩", interactive=False
                    )
                mask_preview = gr.Image(
                    label="原图 / 棋盘格合成 / Alpha 三联检查图",
                    interactive=False,
                )
                mask_summary_view = gr.Markdown(
                    mask_report_summary(None), elem_classes="status-strip"
                )
                with gr.Accordion("高级信息：遮罩原始统计 JSON", open=False):
                    mask_report_view = gr.JSON(label="当前遮罩原始统计")
                confirm_mask_button = gr.Button("确认遮罩")

            with gr.Tab("4 三维生成与基础检查"):
                tab4_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "**本页顺序：**启动 Hunyuan → CPU 网格检查 → 查看基础检查结果。"
                    "如需诊断或修复，请进入第 5 页“模型修复”。",
                    elem_classes="small-note",
                )
                generate_hunyuan_button = gr.Button(
                    "启动 Hunyuan Shape", variant="primary"
                )
                with gr.Accordion(
                    "Hunyuan 多视图输入（2.1 兼容接口 / 真实 2mv 后端）",
                    open=False,
                ):
                    gr.Markdown(
                        "上传同一零件、同一尺度与光照下的透明背景 RGBA PNG。"
                        "正面、左侧、背面必填，右侧可选。点击生成即表示确认这些视角；"
                        "结果档案会如实记录 `backend_model=Hunyuan3D-2mv`。"
                    )
                    with gr.Row():
                        multiview_front = gr.Image(
                            label="正面 front（必填）", type="filepath", sources=["upload"]
                        )
                        multiview_left = gr.Image(
                            label="左侧 left（必填）", type="filepath", sources=["upload"]
                        )
                    with gr.Row():
                        multiview_back = gr.Image(
                            label="背面 back（必填）", type="filepath", sources=["upload"]
                        )
                        multiview_right = gr.Image(
                            label="右侧 right（可选）", type="filepath", sources=["upload"]
                        )
                    generate_hunyuan_multiview_button = gr.Button(
                        "使用多视图生成 GLB（Hunyuan3D-2mv）"
                    )
                prepare_mesh_review_button = gr.Button(
                    "CPU 检查并保守整理 Hunyuan 原始 GLB"
                )
                load_mesh_review_button = gr.Button("加载当前网格检查")
                with gr.Row():
                    model_view = gr.Model3D(
                        value=None,
                        label="当前任务 GLB",
                        display_mode="solid",
                        interactive=False,
                        height=500,
                    )
                    mesh_summary_view = gr.Markdown(
                        mesh_report_summary(None), elem_classes="status-strip"
                    )
                mesh_projection = gr.Image(
                    label="正视 / 侧视 / 斜视检查图", interactive=False
                )
                with gr.Accordion("高级信息：网格原始检查 JSON", open=False):
                    mesh_report_view = gr.JSON(label="当前网格原始检查报告")
                result_files = gr.File(
                    value=[],
                    label="当前任务网格检查文件",
                    file_count="multiple",
                    interactive=False,
                )
                confirm_mesh_button = gr.Button("确认已检查三维模型")
                manufacturing_note = gr.Markdown(
                    manufacturing_guidance("涵道风扇"),
                    elem_classes="warning-strip",
                )
                with gr.Accordion("七叶片制造规则化（CPU，可调参）", open=False):
                    gr.Markdown(
                        "当 Hunyuan 外形可以参考、但叶片不规整时，可用下列参数生成"
                        "新版本单体网格。它不覆盖旧版本，不运行 GPU，也不会自动打印。"
                    )
                    with gr.Row():
                        regular_outer = gr.Slider(
                            40, 120, value=60, step=1, label="外径（mm）"
                        )
                        regular_inner = gr.Slider(
                            28, 100, value=47, step=1, label="外环内径（mm）"
                        )
                        regular_ring_depth = gr.Slider(
                            4, 30, value=12, step=0.5, label="外环轴向深度（mm）"
                        )
                    with gr.Row():
                        regular_hub = gr.Slider(
                            10, 60, value=20, step=1, label="轮毂直径（mm）"
                        )
                        regular_hub_depth = gr.Slider(
                            4, 30, value=8, step=0.5, label="轮毂深度（mm）"
                        )
                        regular_blade_thickness = gr.Slider(
                            1.6, 8, value=4.4, step=0.2, label="叶片厚度（mm）"
                        )
                    with gr.Row():
                        regular_sweep = gr.Slider(
                            0, 35, value=20, step=1, label="叶片掠角（°）"
                        )
                        regular_pitch = gr.Slider(
                            0, 3, value=1.2, step=0.1, label="叶片横向弯度（mm）"
                        )
                        regular_arch = gr.Slider(
                            0, 2, value=0.45, step=0.05, label="叶片径向拱高（mm）"
                        )
                        regular_resolution = gr.Dropdown(
                            choices=[0.20, 0.15, 0.12],
                            value=0.15,
                            label="网格精度（数值越小越慢）",
                        )
                    regularize_button = gr.Button(
                        "CPU 生成新版本并自动检查", variant="primary"
                    )

                with gr.Accordion("可选出口：通用 STL 导出", open=False):
                    gr.Markdown(
                        "该区域不是模型修复的完成条件。仅在确实需要下游 STL 时使用；"
                        "每次导出新版本，不覆盖旧文件。"
                    )
                    stl_target_size = gr.Number(
                        value=60,
                        minimum=20,
                        maximum=200,
                        label="本次 STL 目标最长尺寸（mm）",
                        info="默认继承任务创建时的目标值；允许改为 20–200 mm 并导出新版本。",
                    )
                    stl_size_preview_view = gr.Markdown(
                        stl_size_preview(None, 60),
                        elem_classes="status-strip",
                    )
                    support_ack = gr.Checkbox(
                        value=False,
                        label="我会在 Bambu Studio 中检查尺寸、朝向、首层和支撑",
                    )
                    export_stl_button = gr.Button("导出已确认模型的 STL")
                    stl_download = gr.File(
                        label="当前任务 STL 下载",
                        file_count="single",
                        interactive=False,
                    )

            with gr.Tab("5 模型修复"):
                repair_tab_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "## 模型修复工作台\n"
                    "按照 **选择模型 → 自动诊断 → 生成候选 → 对比确认** 的顺序操作。"
                    "全部步骤使用 CPU，原始 GLB 始终保留，实物打印不是验收条件。"
                )

                gr.Markdown("### ① 选择修复源")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        load_repair_source_button = gr.Button(
                            "加载当前任务的 review GLB", variant="secondary"
                        )
                        external_mesh_upload = gr.File(
                            label="或上传外部 GLB（最大 200 MiB）",
                            file_types=[".glb"],
                            file_count="single",
                            type="filepath",
                        )
                        import_external_mesh_button = gr.Button(
                            "安全归档并设为修复源"
                        )
                    with gr.Column(scale=3):
                        repair_source_summary_view = gr.Markdown(
                            mesh_report_summary(None),
                            elem_classes="status-strip",
                        )
                        gr.Markdown(
                            "外部文件会先检查文件头、大小和 SHA256，再归档到当前 RUN；"
                            "不会直接覆盖任何已有模型。",
                            elem_classes="small-note",
                        )

                gr.Markdown("### ② 自动诊断与推荐")
                with gr.Row():
                    diagnose_mesh_button = gr.Button(
                        "诊断当前模型并自动分流", variant="primary", scale=2
                    )
                    repair_mesh_button = gr.Button(
                        "按推荐方案生成候选", scale=2
                    )
                with gr.Accordion("高级设置：手动指定修复等级", open=False):
                    repair_mode = gr.Dropdown(
                        choices=[
                            ("自动推荐（建议）", "auto"),
                            ("Safe：重复/退化/法向", "safe"),
                            ("Standard：Safe + 受限小孔", "standard"),
                            ("Artifact cleanup：受控独立小伪影清理", "artifact_cleanup"),
                            ("Advanced：非流形/多部件候选", "advanced"),
                            ("Reconstruct：严重开放/碎裂体素重建", "reconstruct"),
                        ],
                        value="auto",
                        label="修复等级",
                    )
                    gr.Markdown(
                        "默认保持“自动推荐”。主体占比至少 99.5% 且其余部件极小时，"
                        "会生成独立伪影清理候选；严重碎裂才进入体素重建。仅有曲面"
                        "质量问题时仍停止自动处理。",
                        elem_classes="small-note",
                    )

                gr.Markdown("### ③ 原模型与候选对比")
                with gr.Row(equal_height=True):
                    repair_source_view = gr.Model3D(
                        value=None,
                        label="原模型 / 当前修复源（原始颜色）",
                        display_mode="solid",
                        interactive=False,
                        height=460,
                    )
                    repair_model_view = gr.Model3D(
                        value=None,
                        label="诊断标记 / 修复候选（红色=退化面，橙色=疑似异常面）",
                        display_mode="solid",
                        interactive=False,
                        height=460,
                    )
                repair_summary_view = gr.Markdown(
                    mesh_repair_report_summary(None),
                    elem_classes="status-strip",
                )
                gr.Markdown(
                    "提示：左侧显示原模型；右侧若出现红/橙色，只是诊断标记，不代表导入失败。"
                    "只有出现 Gradio 的 Error 提示时，才表示文件加载或回调失败。",
                    elem_classes="small-note",
                )
                with gr.Row(equal_height=True):
                    repair_source_preview = gr.Image(
                        label="原模型四视图",
                        interactive=False,
                    )
                    repair_preview = gr.Image(
                        label="修复前后同视角、同缩放对照（诊断阶段为空）",
                        interactive=False,
                    )
                with gr.Accordion("高级信息：检查报告与版本化文件", open=False):
                    with gr.Row():
                        repair_source_report_view = gr.JSON(label="修复源检查报告")
                        repair_report_view = gr.JSON(label="诊断 / 修复报告")
                    with gr.Row():
                        repair_source_files = gr.File(
                            value=[],
                            label="修复源档案",
                            file_count="multiple",
                            interactive=False,
                        )
                        repair_result_files = gr.File(
                            value=[],
                            label="修复结果档案",
                            file_count="multiple",
                            interactive=False,
                        )

                gr.Markdown("### ④ 结果管理")
                with gr.Row():
                    accept_repair_button = gr.Button(
                        "接受候选为新的 review GLB", variant="primary"
                    )
                    reject_repair_button = gr.Button(
                        "拒绝候选并保留当前模型", variant="stop"
                    )

                with gr.Accordion("直齿轮专用数字化指标", open=False):
                    gr.Markdown(
                        "从当前 GLB 的径向轮廓估算齿数、外径、齿根径、中心孔径和厚度。"
                        "仅适用于直齿圆柱齿轮任务；结果是参数草案。"
                    )
                    with gr.Row():
                        gear_known_outside_mm = gr.Number(
                            value=0,
                            minimum=0,
                            label="可选：已知外径（mm，0=显示模型单位）",
                        )
                        analyze_gear_button = gr.Button(
                            "分析当前齿轮 GLB", variant="primary"
                        )
                    with gr.Row():
                        gear_profile_preview = gr.Image(
                            label="0–360° 径向外轮廓",
                            interactive=False,
                        )
                        gear_summary_view = gr.Markdown(
                            gear_analysis_summary(None),
                            elem_classes="status-strip",
                        )
                    with gr.Accordion("齿轮参数原始 JSON", open=False):
                        gear_report_view = gr.JSON(label="齿轮参数草案")
                    gear_result_files = gr.File(
                        value=[],
                        label="齿轮分析档案",
                        file_count="multiple",
                        interactive=False,
                    )

                with gr.Accordion("修复历史记录", open=False):
                    repair_history_table = gr.Dataframe(
                        headers=["版本", "模式", "自动分流", "状态", "有候选", "时间"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Row():
                        repair_history_version = gr.Number(
                            value=1,
                            minimum=1,
                            precision=0,
                            label="要恢复查看的修复版本",
                        )
                        refresh_repair_history_button = gr.Button("刷新历史列表")
                        load_repair_history_button = gr.Button("恢复该版本到页面")
                    gr.Markdown(
                        "恢复历史只更新本页显示，不会改变当前 selected_mesh。",
                        elem_classes="small-note",
                    )

            with gr.Tab("6 打印证据"):
                tab5_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Gallery(
                    value=[
                        (str(PRINT_PHOTOS[0]), "60 mm 打印结果 1"),
                        (str(PRINT_PHOTOS[1]), "60 mm 打印结果 2"),
                    ],
                    label="第一阶段打印证据",
                    columns=2,
                    rows=1,
                    height=360,
                )
                gr.Markdown(
                    "打印已完成，但内部存在支撑粘连。此页面只做证据展示，"
                    "当前系统不会连接 Bambu Studio 或自动发送打印。"
                )

            with gr.Tab("7 历史任务"):
                tab6_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "点击表格中的任意一行即可直接加载该任务；无需再复制任务编号。",
                    elem_classes="small-note",
                )
                refresh_runs_button = gr.Button("刷新任务列表")
                runs_view = gr.Dataframe(
                    value=list_runs_table(),
                    headers=["任务编号", "零件类型", "当前状态", "创建时间", "更新时间"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="历史任务",
                )

            with gr.Tab("8 三件套装配验收"):
                tab7_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "### 当前推荐：先完成三件套闭环\n\n"
                    "加载组件后查看 BOM、装配预览和工艺图，再保存实物试装结果。"
                    "这条路线不需要重新运行三个 AI 模型。",
                    elem_classes="status-strip",
                )
                gr.Markdown(
                    "### 按功能自动规划组件（试用）\n\n"
                    "选手只描述用途和装配效果，不填写毫米尺寸。平台当前会选择唯一一个"
                    "已在 Bambu Lab A1 / PLA 上实测通过的三件套模板。",
                )
                with gr.Row():
                    auto_assembly_purpose = gr.Dropdown(
                        choices=ASSEMBLY_PURPOSES,
                        value=ASSEMBLY_PURPOSES[0],
                        label="装配用途",
                    )
                    auto_fit_goal = gr.Dropdown(
                        choices=FIT_GOALS,
                        value=FIT_GOALS[0],
                        label="希望达到的装配效果",
                    )
                    auto_print_profile = gr.Dropdown(
                        choices=PRINT_PROFILES,
                        value=PRINT_PROFILES[0],
                        label="平台采用的打印配置",
                        interactive=False,
                    )
                auto_requirement = gr.Textbox(
                    value="生成一套可反复拆装的教学组件，用于展示齿轮与D形轴的方向配合和限位帽装配。",
                    label="用一句话描述需求",
                    lines=2,
                )
                auto_plan_button = gr.Button(
                    "自动规划并创建组件",
                    variant="primary",
                )
                auto_plan_summary_view = gr.Markdown(
                    auto_plan_summary(None),
                    elem_classes="status-strip",
                )
                with gr.Accordion("高级信息：平台内部规划参数", open=False):
                    auto_plan_view = gr.JSON(label="自动规划档案")

                gr.Markdown(
                    "### 三件式参数化装配组件（MVP）\n\n"
                    "这一部分把齿轮、D形短轴和限位帽作为一个BOM管理，同时保留三个"
                    "独立零件和同板排版文件。它是当前唯一受支持的多零件模板，不代表"
                    "任意装配体已经自动化。**同板打印只是排版选择；装配接口、尺寸和"
                    "配合间隙必须在建模前规划，三个零件始终保留独立 STL。**",
                )
                with gr.Row():
                    assembly_component_run_id = gr.Textbox(
                        value="RUN-20260901-115822-a30c10",
                        label="装配组件任务编号",
                        scale=3,
                    )
                    load_assembly_component_button = gr.Button(
                        "加载三件套组件", variant="primary", scale=1
                    )
                assembly_component_summary = gr.Markdown(
                    "### 三件式装配组件摘要\n\n尚未加载有效组件。",
                    elem_classes="status-strip",
                )
                assembly_component_bom = gr.Dataframe(
                    headers=["序号", "零件", "数量", "打印文件"],
                    datatype=["number", "str", "number", "str"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="组件BOM",
                )
                with gr.Row():
                    assembly_component_preview = gr.Image(
                        value=None,
                        label="装配完成预览",
                        interactive=False,
                    )
                    assembly_process_sheet = gr.Image(
                        value=None,
                        label="装配工艺设计图",
                        interactive=False,
                    )
                assembly_component_files = gr.File(
                    value=None,
                    label="组件文件下载（独立STL、同板STL、装配/爆炸GLB、说明与验收模板）",
                    file_count="multiple",
                    interactive=False,
                )

                with gr.Accordion("打印完成后填写试装验收", open=True):
                    gr.Markdown(
                        "只记录人工实测，不连接打印机或机器人。通过建议：齿轮空转不大于"
                        "2°、限位帽倒置轻晃不脱落、两件可手动装入并无损拆卸。",
                        elem_classes="small-note",
                    )
                    with gr.Row():
                        fit_gear_rotation = gr.Number(
                            value=0.0,
                            minimum=0.0,
                            maximum=180.0,
                            step=0.5,
                            label="齿轮剩余空转角度（°）",
                        )
                        fit_cap_falls = gr.Checkbox(
                            value=False,
                            label="倒置轻晃时限位帽会脱落",
                        )
                    with gr.Row():
                        fit_gear_insertion = gr.Dropdown(
                            choices=INSERTION_RESULTS,
                            value="顺畅",
                            label="齿轮插入情况",
                        )
                        fit_cap_insertion = gr.Dropdown(
                            choices=INSERTION_RESULTS,
                            value="顺畅",
                            label="限位帽插入情况",
                        )
                        fit_removal = gr.Dropdown(
                            choices=REMOVAL_RESULTS,
                            value="可正常手动拆卸",
                            label="拆卸情况",
                        )
                    fit_visible_defects = gr.Textbox(
                        value="",
                        label="可见缺陷（无缺陷时留空）",
                        placeholder="例如：象脚、翘边、断齿、孔变形",
                    )
                    fit_notes = gr.Textbox(
                        value="",
                        label="试装备注",
                        lines=2,
                    )
                    save_assembly_fit_button = gr.Button(
                        "保存版本化试装结果", variant="primary"
                    )
                assembly_fit_summary = gr.Markdown(
                    fit_review_summary(None), elem_classes="status-strip"
                )
                with gr.Accordion("高级信息：试装记录 JSON", open=False):
                    assembly_fit_view = gr.JSON(label="当前试装记录")
                assembly_fit_file = gr.File(
                    label="试装记录下载",
                    file_count="single",
                    interactive=False,
                )

                assembly_initial = assembly_defaults("涵道风扇")
                with gr.Accordion(
                    "后续功能：机械臂装配工艺草案（当前不用填写）",
                    open=False,
                ):
                    gr.Markdown(
                        "这里只定义取料、抓取、对准、装配、复核和分拣流程。当前项目缺少"
                        "设备型号、坐标标定、负载、速度/力限制和急停资料，因此**不能输出或"
                        "执行真实机械臂/灵巧手命令**。",
                        elem_classes="warning-strip",
                    )
                    assembly_guidance_view = gr.Markdown(
                        assembly_initial[11], elem_classes="small-note"
                    )
                    with gr.Row():
                        assembly_counterpart = gr.Textbox(
                            value=assembly_initial[0], label="配合件", lines=2
                        )
                        assembly_action = gr.Textbox(
                            value=assembly_initial[1], label="装配动作流程", lines=2
                        )
                    with gr.Row():
                        pickup_zone = gr.Textbox(value=assembly_initial[2], label="取料区")
                        assembly_zone = gr.Textbox(value=assembly_initial[3], label="装配区")
                        finished_zone = gr.Textbox(value=assembly_initial[4], label="成品区")
                        reject_zone = gr.Textbox(value=assembly_initial[5], label="异常区")
                    with gr.Row():
                        grasp_feature = gr.Textbox(
                            value=assembly_initial[6], label="候选抓取特征", lines=2
                        )
                        approach_direction = gr.Dropdown(
                            ASSEMBLY_DIRECTIONS,
                            value=assembly_initial[7],
                            label="抓取接近方向",
                        )
                    with gr.Row():
                        mating_feature = gr.Textbox(
                            value=assembly_initial[8], label="配合/定位特征", lines=2
                        )
                        assembly_direction = gr.Dropdown(
                            ASSEMBLY_DIRECTIONS,
                            value=assembly_initial[9],
                            label="装配方向",
                        )
                    success_criteria = gr.Textbox(
                        value=assembly_initial[10], label="装配成功判据", lines=3
                    )
                    with gr.Row():
                        save_assembly_button = gr.Button(
                            "保存版本化装配方案", variant="primary"
                        )
                        load_assembly_button = gr.Button("加载当前装配方案")
                        confirm_assembly_button = gr.Button("人工确认工艺草案")
                    assembly_summary_view = gr.Markdown(
                        assembly_plan_summary(None), elem_classes="status-strip"
                    )
                    with gr.Accordion("高级信息：装配方案 JSON", open=False):
                        assembly_plan_view = gr.JSON(label="装配方案原始档案")
                    assembly_plan_file = gr.File(
                        label="装配方案下载",
                        file_count="single",
                        interactive=False,
                    )

            with gr.Tab("9 设备中心与多模态采集"):
                tab8_status = gr.Markdown(
                    local_action_status(None), elem_classes="status-strip"
                )
                gr.Markdown(
                    "### 任务 3 安全开发入口\n\n"
                    "当前只提供设备事实/模拟状态和人工上传采集框架。刷新或保存快照"
                    "**不会探测网络、打开串口、加载厂商 SDK 或控制设备**；人工上传"
                    "图片/短视频也不能解释为相机已经接入。",
                    elem_classes="warning-strip",
                )
                with gr.Tabs():
                    with gr.Tab("设备中心（模拟/只读）"):
                        device_scenario = gr.Dropdown(
                            choices=DEVICE_SCENARIOS,
                            value=DEVICE_SCENARIOS[0],
                            label="安全演示场景",
                        )
                        with gr.Row():
                            preview_devices_button = gr.Button("刷新安全设备视图")
                            save_device_snapshot_button = gr.Button(
                                "保存快照到当前任务", variant="primary"
                            )
                            load_device_snapshot_button = gr.Button("加载当前任务快照")
                        initial_device_snapshot = build_device_snapshot(DEVICE_SCENARIOS[0])
                        device_summary_view = gr.Markdown(
                            device_snapshot_summary(initial_device_snapshot),
                            elem_classes="status-strip",
                        )
                        with gr.Accordion("高级信息：设备快照 JSON", open=False):
                            device_snapshot_view = gr.JSON(
                                value=initial_device_snapshot,
                                label="设备状态（不会下发控制）",
                            )
                        device_snapshot_file = gr.File(
                            label="当前任务设备快照下载",
                            file_count="single",
                            interactive=False,
                        )

                    with gr.Tab("多模态采集（人工上传 V1）"):
                        gr.Markdown(
                            "先用人工上传验证数据目录、标签和审计链路。实时相机、机器人"
                            "状态同步、标定和 Qwen 复核尚未接入。单个文件不超过 20 MiB。"
                        )
                        capture_media = gr.File(
                            label="图片或短视频",
                            file_types=["image", "video"],
                            file_count="single",
                            type="filepath",
                        )
                        with gr.Row():
                            capture_checkpoint = gr.Dropdown(
                                choices=CAPTURE_CHECKPOINTS,
                                value="before_pick",
                                label="采集节点",
                            )
                            capture_part_id = gr.Textbox(
                                value="part-001", label="零件 ID"
                            )
                            capture_zone = gr.Dropdown(
                                choices=CAPTURE_ZONES,
                                value="取料区",
                                label="区域",
                            )
                            capture_label = gr.Dropdown(
                                choices=CAPTURE_LABELS,
                                value="UNLABELED",
                                label="人工标签",
                            )
                        capture_notes = gr.Textbox(
                            value="",
                            label="备注（不能填写设备运动命令）",
                            lines=2,
                        )
                        with gr.Row():
                            save_capture_button = gr.Button(
                                "归档到当前任务", variant="primary"
                            )
                            load_captures_button = gr.Button("加载并校验已有采集")
                        capture_summary_view = gr.Markdown(
                            capture_summary([]), elem_classes="status-strip"
                        )
                        capture_table_view = gr.Dataframe(
                            value=[],
                            headers=[
                                "采集编号",
                                "节点",
                                "零件 ID",
                                "区域",
                                "人工标签",
                                "媒体类型",
                                "时间",
                            ],
                            datatype=["str", "str", "str", "str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            label="当前任务采集记录",
                        )
                        capture_files = gr.File(
                            label="已校验的媒体与记录文件",
                            file_count="multiple",
                            interactive=False,
                        )

        load_context_outputs = [
            run_id,
            workflow_status,
            manifest_view,
            part_type,
            purpose,
            requirement,
            forbidden_elements,
            structure_count,
            target_size,
            seed,
            final_prompt,
            generated_gallery,
            candidate_index,
            qwen_review_result,
            profile_guidance,
            manufacturing_note,
            assembly_counterpart,
            assembly_action,
            pickup_zone,
            assembly_zone,
            finished_zone,
            reject_zone,
            grasp_feature,
            approach_direction,
            mating_feature,
            assembly_direction,
            success_criteria,
            assembly_guidance_view,
            original_image,
            mask_image,
            mask_preview,
            mask_report_view,
            model_view,
            mesh_projection,
            mesh_report_view,
            result_files,
            stl_download,
            stl_target_size,
        ]

        def load_history_selection(table: Any, evt: gr.SelectData):
            selected_index = evt.index
            row_index = (
                selected_index[0]
                if isinstance(selected_index, (list, tuple))
                else selected_index
            )
            result = load_existing_run_context(history_run_id(table, int(row_index)))
            return (*result, local_action_status(result[1]))

        # ``from __future__ import annotations`` stores the local Gradio type as
        # text; replace it with the runtime class so Gradio injects SelectData.
        load_history_selection.__annotations__["evt"] = gr.SelectData

        def begin_long_action(message: str, busy_label: str):
            return (
                message,
                local_action_status(message),
                gr.update(value=busy_label, interactive=False),
            )

        def finish_long_action(ready_label: str):
            return gr.update(value=ready_label, interactive=True)

        def with_page_status(function, status_index: int = 0):
            def wrapped(*args):
                result = function(*args)
                return (*result, local_action_status(result[status_index]))

            return wrapped

        part_type.input(
            fn=profile_ui_defaults,
            inputs=[part_type],
            outputs=[
                requirement,
                forbidden_elements,
                structure_count,
                profile_guidance,
                manufacturing_note,
            ],
            api_visibility="private",
        )
        part_type.input(
            fn=assembly_defaults,
            inputs=[part_type],
            outputs=[
                assembly_counterpart,
                assembly_action,
                pickup_zone,
                assembly_zone,
                finished_zone,
                reject_zone,
                grasp_feature,
                approach_direction,
                mating_feature,
                assembly_direction,
                success_criteria,
                assembly_guidance_view,
            ],
            api_visibility="private",
        )
        manifest_view.change(
            fn=manifest_summary,
            inputs=[manifest_view],
            outputs=[manifest_summary_view],
            api_visibility="private",
        )
        qwen_review_result.change(
            fn=qwen_review_summary,
            inputs=[qwen_review_result],
            outputs=[qwen_summary_view],
            api_visibility="private",
        )
        mask_report_view.change(
            fn=mask_report_summary,
            inputs=[mask_report_view],
            outputs=[mask_summary_view],
            api_visibility="private",
        )
        mesh_report_view.change(
            fn=mesh_report_summary,
            inputs=[mesh_report_view],
            outputs=[mesh_summary_view],
            api_visibility="private",
        )
        repair_source_report_view.change(
            fn=mesh_report_summary,
            inputs=[repair_source_report_view],
            outputs=[repair_source_summary_view],
            api_visibility="private",
        )
        repair_report_view.change(
            fn=mesh_repair_report_summary,
            inputs=[repair_report_view],
            outputs=[repair_summary_view],
            api_visibility="private",
        )
        gear_report_view.change(
            fn=gear_analysis_summary,
            inputs=[gear_report_view, gear_known_outside_mm],
            outputs=[gear_summary_view],
            api_visibility="private",
        )
        gear_known_outside_mm.change(
            fn=gear_analysis_summary,
            inputs=[gear_report_view, gear_known_outside_mm],
            outputs=[gear_summary_view],
            api_visibility="private",
        )
        mesh_report_view.change(
            fn=stl_size_preview,
            inputs=[mesh_report_view, stl_target_size],
            outputs=[stl_size_preview_view],
            api_visibility="private",
        )
        stl_target_size.change(
            fn=stl_size_preview,
            inputs=[mesh_report_view, stl_target_size],
            outputs=[stl_size_preview_view],
            api_visibility="private",
        )
        target_size.change(
            fn=lambda value: value,
            inputs=[target_size],
            outputs=[stl_target_size],
            api_visibility="private",
        )
        create_button.click(
            fn=with_page_status(create_run, 2),
            inputs=[
                part_type,
                purpose,
                requirement,
                structure_count,
                target_size,
                seed,
                forbidden_elements,
            ],
            outputs=[run_id, final_prompt, workflow_status, manifest_view, tab1_status],
            api_visibility="private",
        )
        load_run_button.click(
            fn=with_page_status(load_existing_run_context, 1),
            inputs=[run_id],
            outputs=[*load_context_outputs, tab1_status],
            api_visibility="private",
        )
        confirm_prompt_button.click(
            fn=with_page_status(confirm_prompt),
            inputs=[run_id, final_prompt],
            outputs=[workflow_status, manifest_view, tab1_status],
            api_visibility="private",
        )
        attach_button.click(
            fn=with_page_status(attach_demo_assets),
            inputs=[run_id],
            outputs=[workflow_status, manifest_view, tab2_status],
            api_visibility="private",
        )
        ernie_result = generate_ernie_button.click(
            fn=lambda: begin_long_action(
                "⏳ ERNIE-Image 已进入队列，正在检查 GPU 并生成图片；正式版步数更多，可能需要数分钟，请勿重复点击或刷新。",
                "⏳ ERNIE 生成中，请勿重复点击",
            ),
            outputs=[workflow_status, tab2_status, generate_ernie_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(generate_ernie_candidates),
            inputs=[run_id, candidate_count, candidate_resolution],
            outputs=[workflow_status, manifest_view, generated_gallery, tab2_status],
            api_visibility="private",
            concurrency_id="gpu-model",
            concurrency_limit=1,
        )
        ernie_result.then(
            fn=lambda: finish_long_action("启动 ERNIE 文生图"),
            outputs=[generate_ernie_button],
            queue=False,
            api_visibility="private",
        )
        qwen_result = review_qwen_button.click(
            fn=lambda: begin_long_action(
                "⏳ Qwen 已进入队列，正在检查 GPU、加载模型并分析候选；通常需要 20–40 秒，请勿重复点击。",
                "⏳ Qwen 检查中，请勿重复点击",
            ),
            outputs=[workflow_status, tab2_status, review_qwen_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(review_candidate_with_qwen),
            inputs=[run_id, candidate_index],
            outputs=[workflow_status, manifest_view, qwen_review_result, tab2_status],
            api_visibility="private",
            concurrency_id="gpu-model",
            concurrency_limit=1,
        )
        qwen_result.then(
            fn=lambda: finish_long_action("使用 Qwen 检查候选"),
            outputs=[review_qwen_button],
            queue=False,
            api_visibility="private",
        )
        confirm_candidate_button.click(
            fn=with_page_status(confirm_candidate_manually),
            inputs=[run_id, candidate_index],
            outputs=[workflow_status, manifest_view, tab2_status],
            api_visibility="private",
        )
        reject_candidate_button.click(
            fn=with_page_status(reject_candidate_manually),
            inputs=[run_id, candidate_index, human_review_reason],
            outputs=[workflow_status, manifest_view, tab2_status],
            api_visibility="private",
        )
        load_mask_button.click(
            fn=with_page_status(load_current_mask, 4),
            inputs=[run_id],
            outputs=[
                original_image,
                mask_image,
                mask_preview,
                mask_report_view,
                workflow_status,
                tab3_status,
            ],
            api_visibility="private",
        )
        mask_result = generate_mask_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在用 CPU 生成版本化遮罩和三联检查图，请勿重复点击。",
                "⏳ 遮罩生成中，请勿重复点击",
            ),
            outputs=[workflow_status, tab3_status, generate_mask_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(generate_candidate_mask),
            inputs=[
                run_id,
                mask_threshold,
                mask_border_width,
                mask_minimum_hole,
                mask_trim_row,
            ],
            outputs=[
                workflow_status,
                manifest_view,
                original_image,
                mask_image,
                mask_preview,
                mask_report_view,
                tab3_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-preprocess",
            concurrency_limit=1,
        )
        mask_result.then(
            fn=lambda: finish_long_action("CPU 生成新遮罩"),
            outputs=[generate_mask_button],
            queue=False,
            api_visibility="private",
        )
        confirm_mask_button.click(
            fn=with_page_status(lambda current_run: confirm_stage(current_run, "mask")),
            inputs=[run_id],
            outputs=[workflow_status, manifest_view, tab3_status],
            api_visibility="private",
        )
        hunyuan_result = generate_hunyuan_button.click(
            fn=lambda: begin_long_action(
                "⏳ Hunyuan 已进入队列，正在检查 GPU 并生成三维；通常需要 2–4 分钟，请勿重复点击或刷新。",
                "⏳ Hunyuan 生成中，请勿重复点击",
            ),
            outputs=[workflow_status, tab4_status, generate_hunyuan_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(generate_hunyuan_mesh),
            inputs=[run_id],
            outputs=[workflow_status, manifest_view, model_view, tab4_status],
            api_visibility="private",
            concurrency_id="gpu-model",
            concurrency_limit=1,
        )
        hunyuan_result.then(
            fn=lambda: finish_long_action("启动 Hunyuan Shape"),
            outputs=[generate_hunyuan_button],
            queue=False,
            api_visibility="private",
        )
        hunyuan_multiview_result = generate_hunyuan_multiview_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在归档多视图并检查 GPU；真实后端为 Hunyuan3D-2mv，请勿重复点击或刷新。",
                "⏳ Hunyuan3D-2mv 生成中，请勿重复点击",
            ),
            outputs=[workflow_status, tab4_status, generate_hunyuan_multiview_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(generate_hunyuan_multiview_mesh),
            inputs=[
                run_id,
                multiview_front,
                multiview_left,
                multiview_back,
                multiview_right,
            ],
            outputs=[workflow_status, manifest_view, model_view, tab4_status],
            api_visibility="private",
            concurrency_id="gpu-model",
            concurrency_limit=1,
        )
        hunyuan_multiview_result.then(
            fn=lambda: finish_long_action("使用多视图生成 GLB（Hunyuan3D-2mv）"),
            outputs=[generate_hunyuan_multiview_button],
            queue=False,
            api_visibility="private",
        )
        mesh_review_result = prepare_mesh_review_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在用 CPU 检查网格并生成多视图；模型面数较多时可能需要几十秒，请勿重复点击。",
                "⏳ 网格检查中，请勿重复点击",
            ),
            outputs=[workflow_status, tab4_status, prepare_mesh_review_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(prepare_current_hunyuan_mesh_review),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                model_view,
                mesh_projection,
                mesh_report_view,
                result_files,
                tab4_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-preprocess",
            concurrency_limit=1,
        )
        mesh_review_result.then(
            fn=lambda: finish_long_action("CPU 检查并保守整理 Hunyuan 原始 GLB"),
            outputs=[prepare_mesh_review_button],
            queue=False,
            api_visibility="private",
        )
        load_mesh_review_button.click(
            fn=with_page_status(load_current_mesh_review, 4),
            inputs=[run_id],
            outputs=[
                model_view,
                mesh_projection,
                mesh_report_view,
                result_files,
                workflow_status,
                tab4_status,
            ],
            api_visibility="private",
        )
        confirm_mesh_button.click(
            fn=with_page_status(lambda current_run: confirm_stage(current_run, "mesh")),
            inputs=[run_id],
            outputs=[workflow_status, manifest_view, tab4_status],
            api_visibility="private",
        )
        load_repair_source_button.click(
            fn=with_page_status(load_current_mesh_review, 4),
            inputs=[run_id],
            outputs=[
                repair_source_view,
                repair_source_preview,
                repair_source_report_view,
                repair_source_files,
                workflow_status,
                repair_tab_status,
            ],
            api_visibility="private",
        )
        external_import_result = import_external_mesh_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在归档并检查外部 GLB；原上传文件不会被修改。",
                "⏳ 外部 GLB 检查中，请勿重复点击",
            ),
            outputs=[workflow_status, repair_tab_status, import_external_mesh_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(import_external_mesh),
            inputs=[run_id, external_mesh_upload],
            outputs=[
                workflow_status,
                manifest_view,
                repair_source_view,
                repair_source_preview,
                repair_source_report_view,
                repair_source_files,
                repair_tab_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-mesh-repair",
            concurrency_limit=1,
        )
        external_import_result.then(
            fn=lambda: finish_long_action("安全归档并检查外部 GLB"),
            outputs=[import_external_mesh_button],
            queue=False,
            api_visibility="private",
        )
        mesh_diagnosis_result = diagnose_mesh_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在用 CPU 诊断当前 review GLB；不会修改或替换模型。",
                "⏳ 模型诊断中，请勿重复点击",
            ),
            outputs=[workflow_status, repair_tab_status, diagnose_mesh_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(diagnose_current_mesh),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                repair_model_view,
                repair_report_view,
                repair_result_files,
                repair_tab_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-mesh-repair",
            concurrency_limit=1,
        )
        mesh_diagnosis_result.then(
            fn=lambda: finish_long_action("1. 诊断并自动分流"),
            outputs=[diagnose_mesh_button],
            queue=False,
            api_visibility="private",
        )
        mesh_repair_result = repair_mesh_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在试算修复、执行安全门禁并生成版本化候选；原始 GLB 保持不变。",
                "⏳ 模型修复中，请勿重复点击",
            ),
            outputs=[workflow_status, repair_tab_status, repair_mesh_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(repair_current_mesh),
            inputs=[run_id, repair_mode],
            outputs=[
                workflow_status,
                manifest_view,
                repair_model_view,
                repair_preview,
                repair_report_view,
                repair_result_files,
                repair_tab_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-mesh-repair",
            concurrency_limit=1,
        )
        mesh_repair_result.then(
            fn=lambda: finish_long_action("2. 生成修复候选"),
            outputs=[repair_mesh_button],
            queue=False,
            api_visibility="private",
        )
        accept_repair_button.click(
            fn=with_page_status(accept_mesh_repair_candidate),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                repair_source_view,
                repair_source_preview,
                repair_source_report_view,
                repair_source_files,
                repair_tab_status,
            ],
            api_visibility="private",
        )
        reject_repair_button.click(
            fn=with_page_status(
                lambda current_run: decide_mesh_repair_candidate(current_run, "REJECT")
            ),
            inputs=[run_id],
            outputs=[workflow_status, manifest_view, repair_tab_status],
            api_visibility="private",
        )
        refresh_repair_history_button.click(
            fn=mesh_repair_history_table,
            inputs=[run_id],
            outputs=[repair_history_table],
            api_visibility="private",
        )
        load_repair_history_button.click(
            fn=with_page_status(load_mesh_repair_history, 4),
            inputs=[run_id, repair_history_version],
            outputs=[
                repair_model_view,
                repair_preview,
                repair_report_view,
                repair_result_files,
                workflow_status,
                repair_tab_status,
            ],
            api_visibility="private",
        )
        gear_analysis_result = analyze_gear_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在用 CPU 提取齿轮径向轮廓和尺寸比例；不会修改模型。",
                "⏳ 齿轮指标分析中，请勿重复点击",
            ),
            outputs=[workflow_status, repair_tab_status, analyze_gear_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(analyze_current_spur_gear),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                gear_report_view,
                gear_profile_preview,
                gear_result_files,
                repair_tab_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-mesh-repair",
            concurrency_limit=1,
        )
        gear_analysis_result.then(
            fn=lambda: finish_long_action("分析当前齿轮 GLB"),
            outputs=[analyze_gear_button],
            queue=False,
            api_visibility="private",
        )
        regularize_result = regularize_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在用 CPU 生成并检查新的七叶片规则化网格；可能需要数分钟，请勿重复点击。",
                "⏳ 规则化处理中，请勿重复点击",
            ),
            outputs=[workflow_status, tab4_status, regularize_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(generate_regularized_fan),
            inputs=[
                run_id,
                regular_outer,
                regular_inner,
                regular_ring_depth,
                regular_hub,
                regular_hub_depth,
                regular_blade_thickness,
                regular_sweep,
                regular_pitch,
                regular_arch,
                regular_resolution,
            ],
            outputs=[
                workflow_status,
                manifest_view,
                model_view,
                mesh_projection,
                mesh_report_view,
                result_files,
                tab4_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-mesh",
            concurrency_limit=1,
        )
        regularize_result.then(
            fn=lambda: finish_long_action("CPU 生成新版本并自动检查"),
            outputs=[regularize_button],
            queue=False,
            api_visibility="private",
        )
        stl_result = export_stl_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在缩放、对齐、导出并重新加载验证 STL，请勿重复点击。",
                "⏳ STL 导出验证中，请勿重复点击",
            ),
            outputs=[workflow_status, tab4_status, export_stl_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(export_confirmed_mesh_stl),
            inputs=[run_id, support_ack, stl_target_size],
            outputs=[workflow_status, manifest_view, stl_download, tab4_status],
            api_visibility="private",
            concurrency_id="cpu-mesh",
            concurrency_limit=1,
        )
        stl_result.then(
            fn=lambda: finish_long_action("导出已确认模型的 STL"),
            outputs=[export_stl_button],
            queue=False,
            api_visibility="private",
        )
        save_assembly_button.click(
            fn=with_page_status(save_assembly_plan),
            inputs=[
                run_id,
                assembly_counterpart,
                assembly_action,
                pickup_zone,
                assembly_zone,
                finished_zone,
                reject_zone,
                grasp_feature,
                approach_direction,
                mating_feature,
                assembly_direction,
                success_criteria,
            ],
            outputs=[
                workflow_status,
                manifest_view,
                assembly_summary_view,
                assembly_plan_view,
                assembly_plan_file,
                tab7_status,
            ],
            api_visibility="private",
        )
        load_assembly_button.click(
            fn=with_page_status(load_current_assembly_plan),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                assembly_summary_view,
                assembly_plan_view,
                assembly_plan_file,
                tab7_status,
            ],
            api_visibility="private",
        )
        confirm_assembly_button.click(
            fn=with_page_status(confirm_assembly_plan),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                assembly_summary_view,
                assembly_plan_view,
                assembly_plan_file,
                tab7_status,
            ],
            api_visibility="private",
        )
        auto_plan_result = auto_plan_button.click(
            fn=lambda: begin_long_action(
                "⏳ 正在根据功能需求创建版本化三件套、网格检查和工艺图；只运行 CPU，请勿重复点击。",
                "⏳ 自动规划与生成中，请勿重复点击",
            ),
            outputs=[workflow_status, tab7_status, auto_plan_button],
            queue=False,
            api_visibility="private",
        ).then(
            fn=with_page_status(create_function_first_assembly_component, 1),
            inputs=[
                auto_assembly_purpose,
                auto_fit_goal,
                auto_print_profile,
                auto_requirement,
            ],
            outputs=[
                assembly_component_run_id,
                workflow_status,
                manifest_view,
                auto_plan_summary_view,
                auto_plan_view,
                assembly_component_summary,
                assembly_component_bom,
                assembly_component_preview,
                assembly_process_sheet,
                assembly_component_files,
                assembly_fit_summary,
                assembly_fit_view,
                assembly_fit_file,
                tab7_status,
            ],
            api_visibility="private",
            concurrency_id="cpu-mesh",
            concurrency_limit=1,
        )
        auto_plan_result.then(
            fn=lambda: finish_long_action("自动规划并创建组件"),
            outputs=[auto_plan_button],
            queue=False,
            api_visibility="private",
        )
        load_assembly_component_button.click(
            fn=with_page_status(load_parametric_assembly_component),
            inputs=[assembly_component_run_id],
            outputs=[
                workflow_status,
                manifest_view,
                assembly_component_summary,
                assembly_component_bom,
                assembly_component_preview,
                assembly_process_sheet,
                assembly_component_files,
                assembly_fit_summary,
                assembly_fit_view,
                assembly_fit_file,
                tab7_status,
            ],
            api_visibility="private",
        )
        save_assembly_fit_button.click(
            fn=with_page_status(save_parametric_assembly_fit_review),
            inputs=[
                assembly_component_run_id,
                fit_gear_rotation,
                fit_gear_insertion,
                fit_cap_insertion,
                fit_cap_falls,
                fit_removal,
                fit_visible_defects,
                fit_notes,
            ],
            outputs=[
                workflow_status,
                manifest_view,
                assembly_fit_summary,
                assembly_fit_view,
                assembly_fit_file,
                tab7_status,
            ],
            api_visibility="private",
        )
        preview_devices_button.click(
            fn=with_page_status(preview_device_center),
            inputs=[device_scenario],
            outputs=[
                workflow_status,
                device_summary_view,
                device_snapshot_view,
                tab8_status,
            ],
            api_visibility="private",
        )
        save_device_snapshot_button.click(
            fn=with_page_status(save_device_snapshot),
            inputs=[run_id, device_scenario],
            outputs=[
                workflow_status,
                manifest_view,
                device_summary_view,
                device_snapshot_view,
                device_snapshot_file,
                tab8_status,
            ],
            api_visibility="private",
        )
        load_device_snapshot_button.click(
            fn=with_page_status(load_current_device_snapshot),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                device_summary_view,
                device_snapshot_view,
                device_snapshot_file,
                tab8_status,
            ],
            api_visibility="private",
        )
        save_capture_button.click(
            fn=with_page_status(save_multimodal_capture),
            inputs=[
                run_id,
                capture_media,
                capture_checkpoint,
                capture_part_id,
                capture_zone,
                capture_label,
                capture_notes,
            ],
            outputs=[
                workflow_status,
                manifest_view,
                capture_summary_view,
                capture_table_view,
                capture_files,
                tab8_status,
            ],
            api_visibility="private",
        )
        load_captures_button.click(
            fn=with_page_status(load_current_multimodal_captures),
            inputs=[run_id],
            outputs=[
                workflow_status,
                manifest_view,
                capture_summary_view,
                capture_table_view,
                capture_files,
                tab8_status,
            ],
            api_visibility="private",
        )
        refresh_runs_button.click(
            fn=list_runs_table,
            outputs=[runs_view],
            api_visibility="private",
        )
        runs_view.select(
            fn=load_history_selection,
            inputs=[runs_view],
            outputs=[*load_context_outputs, tab6_status],
            api_visibility="private",
        )

    return demo.queue(max_size=8, default_concurrency_limit=1, api_open=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工业生成式 AI 服务原型")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=17860, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_app()
    allowed_paths = [
        str(DEMO_IMAGE),
        str(DEMO_MASK),
        str(DEMO_GLB),
        str(DEMO_STL),
        str(DEMO_REPORT),
        *(str(path) for path in PRINT_PHOTOS),
        str(RUNS_DIR),
    ]
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            # 显示具体异常，避免所有输出只出现无法判断原因的红色 Error 框。
            show_error=True,
            allowed_paths=allowed_paths,
            blocked_paths=[
                str(PROJECT_ROOT / "models"),
                str(PROJECT_ROOT / ".codex"),
            ],
            max_file_size="20mb",
            footer_links=[],
            run_history=False,
            theme="soft",
            css=APP_CSS,
            ssr_mode=False,
        )
    except OSError as exc:
        raise SystemExit(
            f"界面启动失败：{exc}\n"
            f"端口 {args.port} 可能已被占用，请改用例如 --port 17861。"
        ) from exc


if __name__ == "__main__":
    main()
