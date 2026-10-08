"""Part-type profiles for prompts, UI defaults, and visual review metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass(frozen=True)
class PartProfile:
    key: str
    label: str
    aliases: tuple[str, ...]
    default_requirement: str
    default_forbidden: str
    default_count: int
    count_feature: str
    count_required: bool
    minimum_count: int
    maximum_count: int
    view_instruction: str
    visual_checks: tuple[str, ...]
    assembly_considerations: str


FAN = PartProfile(
    key="ducted_fan",
    label="涵道风扇",
    aliases=("涵道风扇", "涡扇", "风扇", "叶轮", "转子"),
    default_requirement=(
        "七片宽而有实体厚度的叶片，叶片与短而封闭的中心轮毂和浅圆筒涵道"
        "连续连接，叶片间风道贯通，轮毂无轴孔"
    ),
    default_forbidden=(
        "多余叶片、螺丝、螺栓、轴孔、细杆、长轴、格栅、第二层叶片、"
        "悬浮碎片、文字、底座、支架"
    ),
    default_count=7,
    count_feature="叶片",
    count_required=True,
    minimum_count=3,
    maximum_count=20,
    view_instruction="三分之四视角，正面七个风道间隙和浅轴向纵深均清晰可见",
    visual_checks=("叶片与轮毂连续", "叶片与外环连接", "风道孔洞贯通", "没有第二层叶片"),
    assembly_considerations=(
        "外环提供可识别的非叶片抓取区域，轮毂或安装接口需要与配合件明确对应，"
        "装配路径不得穿过叶片"
    ),
)

GEAR = PartProfile(
    key="spur_gear",
    label="直齿圆柱齿轮",
    aliases=("直齿圆柱齿轮", "齿轮", "直齿轮", "圆柱齿轮"),
    default_requirement=(
        "单个完整直齿圆柱齿轮，轮齿等间距且齿形一致，中心圆孔贯通，"
        "轮毂、轮缘和齿根连续"
    ),
    default_forbidden=(
        "缺齿、多余齿、斜齿、第二个齿轮、轴、键、螺钉、底座、文字、"
        "尺寸标注、复杂阴影"
    ),
    default_count=12,
    count_feature="轮齿",
    count_required=True,
    minimum_count=6,
    maximum_count=100,
    view_instruction="略带轴向厚度的正面三分之四视角，全部轮齿和中心孔均可计数",
    visual_checks=("轮齿均匀完整", "中心孔可见且贯通", "轮毂与轮缘连续", "没有第二个齿轮"),
    assembly_considerations=(
        "中心孔作为与轴配合的候选特征，端面提供抓取区域，装配接近路径不得碰撞轮齿"
    ),
)

FLANGE = PartProfile(
    key="round_flange",
    label="圆形法兰",
    aliases=("圆形法兰", "法兰", "法兰盘", "连接法兰"),
    default_requirement=(
        "单个圆形法兰盘，中心通孔清晰，螺栓孔沿同一分度圆等角度均布，"
        "盘体厚度一致且所有孔洞贯通"
    ),
    default_forbidden=(
        "缺少螺栓孔、多余螺栓孔、螺栓、螺母、管道、第二个法兰、底座、"
        "文字、尺寸标注、复杂阴影"
    ),
    default_count=6,
    count_feature="均布螺栓孔",
    count_required=True,
    minimum_count=2,
    maximum_count=24,
    view_instruction="略带厚度的正面三分之四视角，中心孔和全部均布螺栓孔清晰可数",
    visual_checks=("中心孔可见且贯通", "螺栓孔均匀分布", "盘体连续", "没有装配螺栓或管道"),
    assembly_considerations=(
        "法兰端面、中心孔和螺栓孔阵列需要清晰可识别，保留端面抓取和同轴对准空间"
    ),
)

CUSTOM = PartProfile(
    key="custom",
    label="自定义零件",
    aliases=("自定义零件", "自定义零部件", "夹具"),
    default_requirement="单个完整工业零件，主体结构清晰，各实体部分连接连续，适合后续三维重建",
    default_forbidden="多余零件、悬浮碎片、文字、尺寸标注、底座、支架、复杂阴影",
    default_count=0,
    count_feature="主要重复结构",
    count_required=False,
    minimum_count=0,
    maximum_count=100,
    view_instruction="能同时看清主要结构和轴向厚度的三分之四视角",
    visual_checks=("主体完整", "连接连续", "无遮挡", "没有无关附属物"),
    assembly_considerations=(
        "明确标注候选抓取面、配合特征和装配接近方向，实际尺寸与公差由工程资料确认"
    ),
)

PART_PROFILES = (FAN, GEAR, FLANGE)
PROFILE_CHOICES = [profile.label for profile in (*PART_PROFILES, CUSTOM)]

_CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _parse_count_token(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token == "百":
        return 100
    if "百" in token:
        left, right = token.split("百", 1)
        hundreds = _CHINESE_DIGITS.get(left, 1)
        remainder = _parse_count_token(right) if right else 0
        return None if remainder is None else hundreds * 100 + remainder
    if "十" in token:
        left, right = token.split("十", 1)
        tens = _CHINESE_DIGITS.get(left, 1) if left else 1
        units = _CHINESE_DIGITS.get(right, 0) if right else 0
        return tens * 10 + units
    if len(token) == 1:
        return _CHINESE_DIGITS.get(token)
    return None


def declared_feature_counts(requirement: str, profile: PartProfile) -> list[int]:
    if profile.key == "custom":
        return []
    token = r"([0-9]{1,3}|[零一二两三四五六七八九十百]{1,4})"
    suffixes = {
        "ducted_fan": r"\s*(?:片|个)?\s*叶片",
        "spur_gear": r"\s*(?:个)?\s*(?:轮)?齿",
        "round_flange": r"\s*(?:个)?\s*(?:均布)?\s*(?:螺栓)?孔",
    }
    values: list[int] = []
    for match in re.finditer(token + suffixes[profile.key], str(requirement or "")):
        value = _parse_count_token(match.group(1))
        if value is not None:
            values.append(value)
    return values


def validate_requirement_count(
    requirement: str,
    profile: PartProfile,
    count: int,
) -> None:
    declared = declared_feature_counts(requirement, profile)
    conflicts = sorted({value for value in declared if value != count})
    if conflicts:
        found = "、".join(str(value) for value in conflicts)
        raise ValueError(
            f"结构要求中写出的{profile.count_feature}数量为 {found}，"
            f"但数量字段为 {count}；请统一后再创建任务。"
        )


def resolve_part_profile(part_type: str) -> PartProfile:
    normalized = str(part_type or "").strip().lower()
    if not normalized:
        raise ValueError("零件类型不能为空。")
    for profile in PART_PROFILES:
        if any(alias.lower() in normalized for alias in profile.aliases):
            return profile
    return CUSTOM


def normalized_count(profile: PartProfile, value: float | int | None) -> int:
    try:
        count = int(float(value or 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("主要重复结构数量必须是整数。") from exc
    if not profile.minimum_count <= count <= profile.maximum_count:
        if profile.count_required:
            raise ValueError(
                f"{profile.label}的{profile.count_feature}数量必须在 "
                f"{profile.minimum_count}–{profile.maximum_count} 之间。"
            )
        raise ValueError("自定义零件的重复结构数量必须在 0–100 之间。")
    return count


def profile_manifest(profile: PartProfile) -> dict[str, Any]:
    value = asdict(profile)
    value["aliases"] = list(profile.aliases)
    value["visual_checks"] = list(profile.visual_checks)
    value["prompt_profile_version"] = "part-profile-v1"
    return value


def profile_defaults(part_type: str) -> tuple[str, str, int, str]:
    profile = resolve_part_profile(part_type)
    count_note = (
        f"必须检查：{profile.count_feature}数量（允许范围 "
        f"{profile.minimum_count}–{profile.maximum_count}）。"
        if profile.count_required
        else "数量填 0 表示不设置重复结构硬约束；大于 0 时才进行数量检查。"
    )
    guidance = (
        f"**当前配置：{profile.label}**  · 计数对象：{profile.count_feature}  · "
        f"{count_note}\n\n自动检查重点：" + "、".join(profile.visual_checks)
    )
    return (
        profile.default_requirement,
        profile.default_forbidden,
        profile.default_count,
        guidance,
    )


def build_profile_prompt(
    part_type: str,
    purpose: str,
    requirement: str,
    structure_count: float | int | None,
    forbidden_elements: str = "",
) -> tuple[str, PartProfile, int]:
    part_name = str(part_type or "").strip()
    profile = resolve_part_profile(part_name)
    count = normalized_count(profile, structure_count)
    purpose_text = str(purpose or "").strip() or "工业生成式 AI 流程验证"
    requirement_text = str(requirement or "").strip() or profile.default_requirement
    validate_requirement_count(requirement_text, profile, count)
    forbidden = str(forbidden_elements or "").strip() or profile.default_forbidden

    if profile.count_required or count > 0:
        count_constraint = (
            f"数量硬约束：严格且仅有 {count} 个{profile.count_feature}，不多不少；"
            f"{profile.count_feature}形状和尺寸一致，等角度均匀排列。"
        )
    else:
        count_constraint = (
            "本任务没有指定重复结构数量，不强制旋转对称；严格按照结构描述，"
            "不要自行添加阵列、孔洞或重复零件。"
        )
    assembly_text = (
        f"装配表达约束：{profile.assembly_considerations}。"
        "图片仅用于装配流程和视觉数据原型，尺寸、公差与受力必须由工程资料确认。"
        if "装配" in purpose_text
        else ""
    )
    prompt = (
        f"单个{part_name}，用于{purpose_text}。结构约束：{requirement_text}。"
        f"{count_constraint}{assembly_text}采用工业 CAD 产品渲染风格，{profile.view_instruction}，"
        "主体完整居中，纯白背景，无复杂阴影、无遮挡。"
        f"禁止出现：{forbidden}。"
    )
    return prompt, profile, count
