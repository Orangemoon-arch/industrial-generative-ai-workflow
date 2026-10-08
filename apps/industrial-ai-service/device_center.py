"""Safe device registry and simulator for the workcell integration prototype.

This module never opens a socket, serial port, vendor SDK, or printer session.
It provides explicit simulated/read-only states so the Web workflow and audit
format can be implemented before hardware documentation is complete.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


DEVICE_SCENARIOS = [
    "当前真实边界（全部未连接）",
    "只读在线演示（模拟）",
    "相机采集演示（模拟）",
    "安全故障拦截演示（模拟）",
]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _base_devices() -> list[dict[str, Any]]:
    return [
        {
            "key": "bambu_a1",
            "category": "printer",
            "label": "Bambu Lab A1",
            "identity": "用户已确认实验打印机型号",
            "interface_readiness": "BLOCKED_SUPPORTED_INTERFACE_NOT_CONFIRMED",
            "known_capabilities": ["Windows Bambu Studio 人工切片与发送"],
            "missing": ["受支持的设备接口资料", "局域网/账号模式", "设备序列号与访问授权"],
        },
        {
            "key": "rm65_b",
            "category": "robot",
            "label": "睿尔曼 RM65-B",
            "identity": "现场照片铭牌已确认",
            "interface_readiness": "BLOCKED_OFFICIAL_PROTOCOL_AND_VERSION",
            "known_capabilities": ["6 自由度", "额定负载 5 kg", "工作半径 610 mm"],
            "missing": ["控制器/Web 版本", "匹配版本的官方 JSON/API/ROS 文档", "现场网络与安全配置"],
        },
        {
            "key": "o7_hand",
            "category": "dexterous_hand",
            "label": "O7 五指灵巧手（身份待完整确认）",
            "identity": "外形和 Windows 485 上位机资料部分对应",
            "interface_readiness": "BLOCKED_RS485_PROTOCOL_AND_ELECTRICAL_DATA",
            "known_capabilities": ["Windows USB-RS485 上位机手动操作资料"],
            "missing": ["完整型号/固件", "RS485 帧/寄存器协议", "电压电流与安装接口", "Linux/SDK 支持"],
        },
        {
            "key": "workcell_camera",
            "category": "camera",
            "label": "工作单元相机",
            "identity": "UNKNOWN",
            "interface_readiness": "BLOCKED_IDENTITY_SDK_AND_CALIBRATION",
            "known_capabilities": [],
            "missing": ["品牌型号", "SDK", "安装方式", "内参/畸变参数", "手眼标定结果"],
        },
    ]


def build_device_snapshot(scenario: str) -> dict[str, Any]:
    """Return an auditable snapshot without performing any hardware I/O."""
    selected = str(scenario or "").strip()
    if selected not in DEVICE_SCENARIOS:
        raise ValueError("设备演示场景无效。")

    devices = _base_devices()
    for device in devices:
        device.update(
            {
                "connection": "DISCONNECTED",
                "telemetry": {},
                "simulated": selected != DEVICE_SCENARIOS[0],
                "control_allowed": False,
            }
        )

    if selected == DEVICE_SCENARIOS[1]:
        for device in devices:
            device["connection"] = "READ_ONLY_ONLINE_SIMULATED"
            device["telemetry"] = {"heartbeat": "OK_SIMULATED", "fault": False}
    elif selected == DEVICE_SCENARIOS[2]:
        for device in devices:
            device["connection"] = "READ_ONLY_ONLINE_SIMULATED"
            device["telemetry"] = {"heartbeat": "OK_SIMULATED", "fault": False}
        camera = next(item for item in devices if item["category"] == "camera")
        camera["connection"] = "STREAMING_SIMULATED"
        camera["telemetry"] = {
            "frame_state": "AVAILABLE_SIMULATED",
            "calibration": "UNAVAILABLE",
            "fault": False,
        }
    elif selected == DEVICE_SCENARIOS[3]:
        for device in devices:
            device["connection"] = "READ_ONLY_ONLINE_SIMULATED"
            device["telemetry"] = {"heartbeat": "OK_SIMULATED", "fault": False}
        robot = next(item for item in devices if item["category"] == "robot")
        robot["connection"] = "SAFETY_FAULT_SIMULATED"
        robot["telemetry"] = {
            "fault": True,
            "fault_code": "SIMULATED_INTERLOCK_OPEN",
            "motion_permitted": False,
        }

    return {
        "schema_version": "1.0",
        "created_at": _now_iso(),
        "source": (
            "PROJECT_KNOWN_FACTS_NO_HARDWARE_PROBE"
            if selected == DEVICE_SCENARIOS[0]
            else "LOCAL_DEVICE_SIMULATOR"
        ),
        "scenario": selected,
        "mode": "READ_ONLY_NO_DEVICE_COMMANDS",
        "devices": devices,
        "safety": {
            "real_hardware_contacted": False,
            "network_probe_performed": False,
            "serial_port_opened": False,
            "vendor_sdk_loaded": False,
            "control_commands_available": False,
            "qwen_may_control_devices": False,
        },
        "next_gate": "设备资料 → 只读通信 → 标定 → 离线仿真 → 空载低速 → 人工授权受监护执行",
    }


def device_snapshot_summary(snapshot: dict[str, Any] | None) -> str:
    if not isinstance(snapshot, dict) or not snapshot:
        return "### 设备中心\n\n尚未刷新或加载设备状态。"
    rows = []
    for device in snapshot.get("devices", []):
        if not isinstance(device, dict):
            continue
        connection = str(device.get("connection", "UNKNOWN"))
        rows.append(
            f"| {device.get('label', '未知设备')} | {connection} | "
            f"{'否' if not device.get('control_allowed') else '是'} |"
        )
    source = snapshot.get("source", "UNKNOWN")
    warning = (
        "当前显示的是模拟状态，不能解释为真机已连接。"
        if source == "LOCAL_DEVICE_SIMULATOR"
        else "当前只展示项目已确认事实，没有探测任何设备。"
    )
    return (
        "### 设备中心\n\n"
        f"- 数据来源：**{source}**\n"
        f"- 工作模式：**{snapshot.get('mode', 'UNKNOWN')}**\n"
        f"- 说明：{warning}\n\n"
        "| 设备 | 状态 | 允许控制 |\n"
        "|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n所有设备控制均保持关闭，Qwen 不得直接控制设备。"
    )

