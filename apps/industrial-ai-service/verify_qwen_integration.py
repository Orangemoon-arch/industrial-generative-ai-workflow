"""Run one real Qwen structured review on the rejected ERNIE candidate."""

from __future__ import annotations

import json

import app


RUN_ID = "RUN-20260826-185232-20ef3b"


def main() -> None:
    status, manifest, review = app.review_candidate_with_qwen(RUN_ID, 1)
    summary = {
        "run_id": RUN_ID,
        "status": manifest["status"],
        "message": status,
        "manifest": str(app.manifest_path(RUN_ID)),
        "review": review,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if manifest["models"]["qwen"]["status"] != "COMPLETED":
        raise SystemExit(1)
    if review.get("decision") not in {"PASS", "REVIEW", "REJECT"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
