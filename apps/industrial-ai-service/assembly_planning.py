"""Safe, device-agnostic planning records for future assembly experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from part_profiles import resolve_part_profile


ASSEMBLY_DIRECTIONS = [
    "沿零件轴线",
    "垂直工作台向下",
    "平行工作台侧向",
    "待现场与工程资料确认",
]


@dataclass(frozen=True)
class AssemblyTemplate:
    key: str
    label: str
    counterpart: str
    action: str
    grasp_feature: str
    mating_feature: str
    approach_direction: str
    assembly_direction: str
    success_criteria: str


TEMPLATES = {
    "ducted_fan": AssemblyTemplate(
        key="ducted_fan_mounting",
        label="风扇/转子安装预规划",
        counterpart="安装座或轴系（当前模型未定义工程安装接口）",
        action="抓取外环—搬运—安装接口对准—放置—视觉复核",
        grasp_feature="外环上避开叶片和薄弱连接的对称区域",
        mating_feature="轮毂或安装接口；必须由工程图确认，不能根据生成外形臆造",
        approach_direction="沿零件轴线",
        assembly_direction="沿零件轴线",
        success_criteria="外环姿态正确、叶片无碰撞、安装接口到位且零件保持完整",
    ),
    "spur_gear": AssemblyTemplate(
        key="gear_on_shaft",
        label="齿轮与轴插装预规划",
        counterpart="轴或轴套",
        action="抓取端面/轮毂—中心孔与轴对准—轴向插装—视觉复核",
        grasp_feature="轮毂或端面区域，避开轮齿",
        mating_feature="齿轮中心孔与轴的圆柱配合面",
        approach_direction="沿零件轴线",
        assembly_direction="沿零件轴线",
        success_criteria="中心孔与轴同轴、插装深度满足工程要求、轮齿无碰撞和缺损",
    ),
    "round_flange": AssemblyTemplate(
        key="flange_alignment",
        label="法兰对准装配预规划",
        counterpart="配对法兰或定位工装",
        action="抓取法兰外缘—端面同轴对准—孔阵列对准—贴合—视觉复核",
        grasp_feature="法兰外圆或端面非孔区域",
        mating_feature="法兰端面、中心孔和螺栓孔阵列",
        approach_direction="沿零件轴线",
        assembly_direction="沿零件轴线",
        success_criteria="端面贴合、中心同轴、螺栓孔阵列对齐且无异物夹入",
    ),
    "custom": AssemblyTemplate(
        key="custom_assembly",
        label="自定义零件装配预规划",
        counterpart="待定义配合件",
        action="识别—抓取—搬运—对准—装配—视觉复核—分拣",
        grasp_feature="待工程人员指定稳定抓取面",
        mating_feature="待工程人员指定配合特征",
        approach_direction="待现场与工程资料确认",
        assembly_direction="待现场与工程资料确认",
        success_criteria="由工程人员提供可测量的到位、姿态、间隙和完整性判据",
    ),
}

DEVICE_READINESS_ITEMS = [
    "机械臂品牌、型号、自由度、额定负载和控制器",
    "灵巧手品牌、型号、抓取能力和控制接口",
    "相机型号、安装方式、内外参与手眼标定结果",
    "机器人基座、工具中心点、相机和工作台坐标系",
    "取料区、装配区、成品区、异常区的现场边界与障碍物",
    "零件质量、质心、材料、抓取允许区域和配合公差",
    "速度、加速度、接触力/力矩和碰撞停止限制",
    "安全围栏、急停、人工使能、现场负责人和异常恢复规程",
]


def assembly_defaults(part_type: str) -> tuple[str, ...]:
    profile = resolve_part_profile(part_type)
    template = TEMPLATES[profile.key]
    guidance = (
        f"**{template.label}**：当前只生成装配工艺草案和多模态采集计划。"
        "设备资料未齐，禁止输出或执行关节角、笛卡尔位姿、轨迹、速度、力或夹爪命令。"
    )
    return (
        template.counterpart,
        template.action,
        "取料区",
        "装配区",
        "成品区",
        "异常区",
        template.grasp_feature,
        template.approach_direction,
        template.mating_feature,
        template.assembly_direction,
        template.success_criteria,
        guidance,
    )


def _required_text(value: Any, label: str, maximum: int = 500) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}不能为空。")
    if len(text) > maximum:
        raise ValueError(f"{label}过长，请控制在 {maximum} 个字符以内。")
    return text


def build_assembly_plan(
    *,
    run_id: str,
    part_type: str,
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
) -> dict[str, Any]:
    profile = resolve_part_profile(part_type)
    template = TEMPLATES[profile.key]
    zones = {
        "pickup": _required_text(pickup_zone, "取料区名称", 80),
        "assembly": _required_text(assembly_zone, "装配区名称", 80),
        "finished": _required_text(finished_zone, "成品区名称", 80),
        "reject": _required_text(reject_zone, "异常区名称", 80),
    }
    if len(set(zones.values())) != len(zones):
        raise ValueError("取料区、装配区、成品区和异常区必须使用不同名称。")
    approach = _required_text(approach_direction, "抓取接近方向", 80)
    insertion = _required_text(assembly_direction, "装配方向", 80)
    if approach not in ASSEMBLY_DIRECTIONS or insertion not in ASSEMBLY_DIRECTIONS:
        raise ValueError("方向必须从受控选项中选择，不能填写设备运动代码。")

    return {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": "PLANNING_ONLY_NO_ROBOT_CONTROL",
        "execution_readiness": "BLOCKED_DEVICE_DOCUMENTATION_AND_SIMULATION",
        "part": {
            "part_type": part_type,
            "part_profile": profile.key,
            "counterpart": _required_text(counterpart, "配合件"),
        },
        "process": {
            "template": template.key,
            "action": _required_text(action, "装配动作流程"),
            "zones": zones,
            "grasp_feature": _required_text(grasp_feature, "候选抓取特征"),
            "approach_direction": approach,
            "mating_feature": _required_text(mating_feature, "配合特征"),
            "assembly_direction": insertion,
            "success_criteria": _required_text(success_criteria, "成功判据"),
        },
        "state_machine": [
            "IDLE",
            "CHECK_WORKCELL",
            "ACQUIRE_BEFORE_PICK",
            "DETECT_PARTS",
            "HUMAN_CONFIRM_PARTS",
            "PLAN_GRASP",
            "SIMULATION_REQUIRED",
            "FUTURE_PICK",
            "FUTURE_MOVE_TO_ASSEMBLY",
            "FUTURE_ASSEMBLE",
            "ACQUIRE_POST_ASSEMBLY",
            "VERIFY_RESULT",
            "FUTURE_MOVE_FINISHED_OR_REJECT",
        ],
        "multimodal_capture": [
            {"checkpoint": "before_pick", "required": ["RGB图像", "时间戳", "区域标签", "零件ID"]},
            {"checkpoint": "after_grasp", "required": ["RGB图像或视频帧", "时间戳", "抓取状态"]},
            {"checkpoint": "before_assembly", "required": ["RGB图像或视频帧", "时间戳", "对准状态"]},
            {"checkpoint": "after_assembly", "required": ["RGB图像或视频帧", "时间戳", "装配状态"]},
            {"checkpoint": "final_sort", "required": ["RGB图像", "时间戳", "合格/异常标签"]},
        ],
        "qwen_role": (
            "Qwen3-VL 只提供零件、区域、抓取状态、对准状态和装配结果的观察建议；"
            "其输出不得直接转换为机器人控制命令。"
        ),
        "missing_device_readiness": list(DEVICE_READINESS_ITEMS),
        "prohibited_outputs": [
            "关节角命令",
            "笛卡尔位姿命令",
            "运动轨迹",
            "速度或加速度设定",
            "接触力或力矩设定",
            "灵巧手或夹爪执行命令",
        ],
        "next_gate": "补齐设备资料 → 现场坐标标定 → 离线仿真 → 空载验证 → 人工授权低速真机联调",
    }


def assembly_plan_summary(plan: dict[str, Any] | None) -> str:
    if not isinstance(plan, dict) or not plan:
        return "### 装配方案摘要\n\n尚未为当前任务保存装配方案。"
    part = plan.get("part", {})
    process = plan.get("process", {})
    zones = process.get("zones", {})
    missing = plan.get("missing_device_readiness", [])
    return (
        "### 装配方案摘要\n\n"
        f"- 主零件：{part.get('part_type', '—')}\n"
        f"- 配合件：{part.get('counterpart', '—')}\n"
        f"- 工艺草案：{process.get('action', '—')}\n"
        f"- 区域流转：{zones.get('pickup', '—')} → {zones.get('assembly', '—')} → "
        f"{zones.get('finished', '—')} / {zones.get('reject', '—')}\n"
        f"- 抓取特征：{process.get('grasp_feature', '—')}\n"
        f"- 配合特征：{process.get('mating_feature', '—')}\n"
        f"- 成功判据：{process.get('success_criteria', '—')}\n"
        f"- 真机就绪：**否**（仍缺 {len(missing)} 类设备与安全资料）\n"
        f"- 下一门禁：{plan.get('next_gate', '—')}"
    )
