"""Create one real versioned run and verify the ERNIE service adapter."""

from __future__ import annotations

import json

import app


def main() -> None:
    run_id, prompt, _, _ = app.create_run(
        "涵道风扇",
        "服务接入验证",
        (
            "七片宽而厚的叶片围绕短圆柱轮毂等角度排列，"
            "叶片与轮毂和浅圆筒涵道连续连接，内部孔洞清晰，"
            "无长轴、无支架、无格栅"
        ),
        7,
        60,
        20260835,
        (
            "多余叶片、螺丝、螺栓、轴孔、细杆、长轴、格栅、第二层叶片、"
            "悬浮碎片、文字、底座、支架"
        ),
    )
    app.confirm_prompt(run_id, prompt)
    status, manifest, gallery = app.generate_ernie_candidates(run_id, 1)
    summary = {
        "run_id": run_id,
        "status": manifest["status"],
        "message": status,
        "manifest": str(app.manifest_path(run_id)),
        "gallery": [item[0] for item in gallery],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if manifest["status"] != "ERNIE_COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
