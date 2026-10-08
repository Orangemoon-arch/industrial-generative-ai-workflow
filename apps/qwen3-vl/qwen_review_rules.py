"""Pure deterministic rules for Qwen candidate review output."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_fastener_hole_only(element: str) -> bool:
    return "孔" in element and not any(
        marker in element
        for marker in ("装有", "安装", "插入", "带有螺栓", "带有螺丝")
    )


def match_forbidden_elements(
    visible_elements: list[str],
    forbidden_text: str,
) -> list[str]:
    terms = [
        term.strip()
        for term in re.split(r"[，,、;；]", forbidden_text)
        if term.strip()
    ]
    alias_groups = {
        "螺丝": ("螺丝", "紧固件", "铆钉"),
        "螺栓": ("螺栓", "紧固件", "铆钉"),
        "轴孔": ("轴孔", "中心孔", "通孔"),
        "细杆": ("细杆", "杆状", "连接杆"),
        "长轴": ("长轴", "轴杆", "突出轴"),
        "格栅": ("格栅", "网格", "护网"),
        "文字": ("文字", "字符", "标识"),
        "底座": ("底座", "基座"),
        "支架": ("支架", "撑杆"),
    }
    matched: list[str] = []
    for term in terms:
        aliases = alias_groups.get(term, (term,))
        found = False
        for element in visible_elements:
            if term in {"螺丝", "螺栓"} and _is_fastener_hole_only(element):
                continue
            if any(alias in element for alias in aliases):
                found = True
                break
        if found:
            matched.append(term)
    return matched


def normalize_review(
    parsed: dict[str, Any] | None,
    expected_count: int | None,
    forbidden_text: str,
) -> dict[str, Any]:
    if parsed is None:
        return {
            "decision": "REVIEW",
            "observation_mode": "BLIND_INVALID_JSON",
            "observed_repeat_count": None,
            "count_matches": None,
            "subject_complete": None,
            "background_clean": None,
            "view_suitable_for_3d": None,
            "structure_continuous": None,
            "forbidden_elements_found": [],
            "visible_auxiliary_elements": [],
            "structural_issues": ["模型未返回可解析的 JSON。"],
            "reason": "需要人工复核模型原始输出。",
            "recommendation": "人工检查后重新运行 Qwen 或拒绝该候选。",
        }

    observed = parsed.get("observed_repeat_count")
    if isinstance(observed, bool):
        observed = None
    elif isinstance(observed, (int, float)):
        observed = int(observed)
    else:
        observed = None
    count_matches = (
        observed == expected_count
        if expected_count is not None and observed is not None
        else None
    )

    subject_complete = optional_bool(parsed.get("subject_complete"))
    background_clean = optional_bool(parsed.get("background_clean"))
    view_suitable = optional_bool(parsed.get("view_suitable_for_3d"))
    structure_continuous = optional_bool(parsed.get("structure_continuous"))
    visible_elements = string_list(parsed.get("visible_auxiliary_elements"))
    forbidden = match_forbidden_elements(visible_elements, forbidden_text)
    issues = string_list(parsed.get("structural_issues"))

    hard_fail = (
        (expected_count is not None and count_matches is False)
        or subject_complete is False
        or view_suitable is False
        or structure_continuous is False
        or bool(forbidden)
    )
    required_observations = [
        subject_complete,
        background_clean,
        view_suitable,
        structure_continuous,
    ]
    if expected_count is not None:
        required_observations.append(count_matches)
    unknown = any(value is None for value in required_observations)
    if hard_fail:
        decision = "REJECT"
    elif unknown or issues:
        decision = "REVIEW"
    else:
        decision = "PASS"

    return {
        "decision": decision,
        "observation_mode": "BLIND_IMAGE_OBSERVATION",
        "observed_repeat_count": observed,
        "count_matches": count_matches,
        "subject_complete": subject_complete,
        "background_clean": background_clean,
        "view_suitable_for_3d": view_suitable,
        "structure_continuous": structure_continuous,
        "forbidden_elements_found": forbidden,
        "visible_auxiliary_elements": visible_elements,
        "structural_issues": issues,
        "reason": str(parsed.get("observation_reason", "")).strip(),
        "recommendation": (
            "可以进入人工确认。"
            if decision == "PASS"
            else "不得进入三维；请人工复核或重新生成候选。"
        ),
    }


def build_instruction(request: dict[str, Any]) -> str:
    counted_feature = str(request.get("counted_feature", "主要重复结构")).strip()
    visual_checks = string_list(request.get("visual_checks"))
    focus = "、".join(visual_checks) if visual_checks else "主体完整性和连接连续性"
    return f"""你是工业零部件图片盲检员。请只描述图片中实际看到的内容，不要猜测设计目标。

目标零部件：{request['part_type']}

不要假设典型零部件应该有几个重复结构，也不要迎合任何预期数量。请从图片12点方向开始顺时针逐个计数“{counted_feature}”；不存在明确重复结构或看不清时使用 null。检查重点：{focus}。请列出实际可见的附加结构，例如中心孔、通孔、螺丝、紧固件、细杆、长轴、格栅、第二层结构、底座或支架。判断主体完整性、背景、视角和连接连续性。

只输出一个 JSON 对象，不要使用 Markdown 或代码围栏。字段必须为：
{{
  "observed_repeat_count": "整数或 null",
  "subject_complete": "true、false 或 null",
  "background_clean": "true、false 或 null",
  "view_suitable_for_3d": "true、false 或 null",
  "structure_continuous": "true、false 或 null",
  "visible_auxiliary_elements": ["图片中实际可见的附加结构"],
  "structural_issues": ["仅根据图片观察到的结构问题"],
  "observation_reason": "简洁中文观察结论"
}}"""
