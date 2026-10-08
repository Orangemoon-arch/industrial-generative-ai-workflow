"""用 Hunyuan3D ShapeVAE 对 GLB 做版本化重建。

这不是图片条件的 Hunyuan Shape 生成，而是：
GLB -> 表面采样 -> ShapeVAE encode/decode -> 隐式表面提取 -> 新 GLB。

原始文件永远只读；输出必须使用新的路径。该脚本只负责生成重建候选，
不把重建结果自动标记为可打印或替换原模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader


AREA_EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    parts = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces": int((np.asarray(mesh.area_faces) < AREA_EPSILON).sum()),
        "area": float(mesh.area),
        "volume": float(mesh.volume),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "extents": np.asarray(mesh.extents).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Hunyuan3D ShapeVAE 将 GLB 重建为新的 GLB 候选"
    )
    parser.add_argument("--input", required=True, type=Path, help="原始 GLB")
    parser.add_argument("--output", required=True, type=Path, help="新 GLB 输出路径")
    parser.add_argument("--report", required=True, type=Path, help="JSON 报告路径")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])) / "models/hunyuan3d-2.1-shape-ms",
    )
    parser.add_argument("--vae-subfolder", default="hunyuan3d-vae-v2-1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--uniform-points", type=int, default=81920)
    parser.add_argument("--sharp-points", type=int, default=0)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=20000)
    parser.add_argument("--bounds", type=float, default=1.01)
    parser.add_argument("--mc-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def require_new_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = require_new_path(args.output)
    report_path = require_new_path(args.report)
    model_root = args.model_root.resolve()
    vae_folder = model_root / args.vae_subfolder

    if not input_path.is_file() or input_path.suffix.lower() != ".glb":
        raise ValueError("输入必须是存在的 GLB 文件。")
    if not vae_folder.is_dir():
        raise FileNotFoundError(f"ShapeVAE 子目录不存在：{vae_folder}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；模型运行前请先检查 nvidia-smi。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    source = trimesh.load(input_path, force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError("输入 GLB 未能加载为单一 Trimesh。")
    before = metrics(source)
    started = time.time()

    loader = SharpEdgeSurfaceLoader(
        num_uniform_points=args.uniform_points,
        num_sharp_points=args.sharp_points,
    )
    surface = loader(source).to(device=device, dtype=dtype)

    vae = ShapeVAE.from_pretrained(
        str(model_root),
        subfolder=args.vae_subfolder,
        use_safetensors=False,
        device=device,
    )
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()

    with torch.inference_mode():
        latents = vae.encode(surface, sample_posterior=False)
        decoded = vae.decode(latents)
        outputs = vae.latents2mesh(
            decoded,
            output_type="trimesh",
            bounds=args.bounds,
            mc_level=args.mc_level,
            num_chunks=args.num_chunks,
            octree_resolution=args.octree_resolution,
            mc_algo="mc",
            enable_pbar=True,
        )
    rebuilt = export_to_trimesh(outputs)[0]
    rebuilt.export(output_path, file_type="glb")
    reloaded = trimesh.load(output_path, force="mesh", process=False)
    if not isinstance(reloaded, trimesh.Trimesh):
        raise RuntimeError("重建 GLB 导出后无法重新加载为 Trimesh。")

    report = {
        "schema_version": "1.0",
        "operation": "hunyuan_shape_vae_glb_reconstruction",
        "status": "RECONSTRUCTION_CANDIDATE_NOT_PRINT_APPROVAL",
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "model_root": str(model_root),
        "vae_subfolder": args.vae_subfolder,
        "device": str(device),
        "dtype": str(dtype),
        "parameters": {
            "uniform_points": args.uniform_points,
            "sharp_points": args.sharp_points,
            "octree_resolution": args.octree_resolution,
            "num_chunks": args.num_chunks,
            "bounds": args.bounds,
            "mc_level": args.mc_level,
            "seed": args.seed,
        },
        "before": before,
        "after": metrics(reloaded),
        "elapsed_seconds": round(time.time() - started, 3),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "limitations": [
            "这是 ShapeVAE 重建候选，不是原网格的无损修补。",
            "必须额外比较齿数、孔径、厚度、外径和表面偏差后才能接受。",
            "原始 GLB 不得被覆盖，也不得仅凭水密性通过就批准打印。",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
