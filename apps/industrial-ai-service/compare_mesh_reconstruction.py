"""CPU-only comparison between a source GLB and a reconstructed GLB candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".glb":
        raise ValueError(f"必须提供存在的 GLB：{resolved}")
    mesh = trimesh.load(resolved, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"无法加载为单一 Trimesh：{resolved}")
    return mesh


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    parts = sorted(
        mesh.split(only_watertight=False), key=lambda part: part.area, reverse=True
    )
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces": int((np.asarray(mesh.area_faces) < 1e-12).sum()),
        "area": float(mesh.area),
        "volume": float(mesh.volume),
        "extents": np.asarray(mesh.extents).tolist(),
        "main_component_area_share": (
            float(parts[0].area / mesh.area) if parts and mesh.area else 0.0
        ),
        "secondary_components": [
            {
                "rank": rank,
                "faces": int(len(part.faces)),
                "area": float(part.area),
                "absolute_volume": float(abs(part.volume)),
            }
            for rank, part in enumerate(parts[1:], start=2)
        ],
    }


def directional_stats(distances: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(distances.mean()),
        "p95": float(np.quantile(distances, 0.95)),
        "p99": float(np.quantile(distances, 0.99)),
        "max_sampled": float(distances.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较原始 GLB 和重建候选")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    candidate_path = args.candidate.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"拒绝覆盖已有报告：{report_path}")
    if args.samples < 1000:
        raise ValueError("表面采样数不得少于 1000。")

    source = load_mesh(source_path)
    candidate = load_mesh(candidate_path)
    np.random.seed(args.seed)
    source_points, _ = trimesh.sample.sample_surface(source, args.samples)
    candidate_points, _ = trimesh.sample.sample_surface(candidate, args.samples)
    source_to_candidate = cKDTree(candidate_points).query(
        source_points, k=1, workers=-1
    )[0]
    candidate_to_source = cKDTree(source_points).query(
        candidate_points, k=1, workers=-1
    )[0]

    source_summary = mesh_summary(source)
    candidate_summary = mesh_summary(candidate)
    source_extents = np.asarray(source.extents, dtype=float)
    candidate_extents = np.asarray(candidate.extents, dtype=float)
    extent_change = np.divide(
        candidate_extents - source_extents,
        source_extents,
        out=np.zeros_like(source_extents),
        where=source_extents != 0,
    )
    report = {
        "schema_version": "1.0",
        "operation": "source_vs_reconstruction_mesh_comparison",
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            **source_summary,
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            **candidate_summary,
        },
        "comparison": {
            "samples_per_mesh": args.samples,
            "seed": args.seed,
            "source_to_candidate": directional_stats(source_to_candidate),
            "candidate_to_source": directional_stats(candidate_to_source),
            "symmetric_chamfer_l1": float(
                (source_to_candidate.mean() + candidate_to_source.mean()) / 2.0
            ),
            "sampled_hausdorff": float(
                max(source_to_candidate.max(), candidate_to_source.max())
            ),
            "extent_relative_change": extent_change.tolist(),
            "area_relative_change": float(
                candidate.area / source.area - 1.0 if source.area else 0.0
            ),
            "volume_relative_change": float(
                candidate.volume / source.volume - 1.0 if source.volume else 0.0
            ),
        },
        "automatic_gate": {
            "passed": bool(
                candidate.is_watertight
                and candidate_summary["components"] == 1
                and candidate_summary["nonmanifold_edges"] == 0
                and candidate_summary["degenerate_faces"] == 0
            ),
            "note": "该门禁只检查基础拓扑；通过也不代表机械齿形和尺寸正确。",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
