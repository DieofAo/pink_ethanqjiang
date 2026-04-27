#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
"""
基于 Pink 的 IK 示例：3DoF 移动底盘 (base_x / base_y / base_yaw) + 7DoF JAKA 机械臂。

URDF: left_jaka_mobile.urdf
总自由度: 10 (底盘 3 + 机械臂 7)
末端: LINK_7

说明:
- URDF 中底盘已用 prismatic+prismatic+revolute 显式建模, 因此 Pinocchio
  加载时 root_joint 留空(默认 Fixed), 10 个关节都被当作普通关节;
- 各 joint 自带 <limit lower/upper/velocity>, Pink 的 ConfigurationLimit
  与 VelocityLimit 会自动生效;
- 本脚本提供两种入口: 单次求解 solve_once() 与循环演示 run_demo()。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask


# ------------------------------------------------------------
# 常量 / 配置
# ------------------------------------------------------------
URDF_PATH = os.path.join(os.path.dirname(__file__), "left_jaka_mobile.urdf")
MESH_DIR = os.path.dirname(__file__)  # meshes/ 与 urdf 同目录

END_EFFECTOR_FRAME = "LINK_7"  # 末端
BASE_JOINT_NAMES = ["base_x", "base_y", "base_yaw"]
ARM_JOINT_NAMES = [f"J_{i}" for i in range(1, 8)]  # J_1 ... J_7


# ------------------------------------------------------------
# 模型加载
# ------------------------------------------------------------
def load_model(urdf_path: str = URDF_PATH):
    """加载 URDF, 返回 (model, data, q0)。

    root_joint 不指定, 因为 URDF 里已经手写好 3 个底盘关节。
    """
    model = pin.buildModelFromUrdf(urdf_path)  # root = Fixed (world)
    data = model.createData()
    q0 = pin.neutral(model)  # 中立位姿 (全 0)

    print(f"[model] nq = {model.nq}, nv = {model.nv}")
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        name = model.names[jid]
        print(f"  joint[{jid}] {name:10s} "
              f"idx_q={j.idx_q} idx_v={j.idx_v} nq={j.nq} nv={j.nv}")
    return model, data, q0


# ------------------------------------------------------------
# 构建 tasks
# ------------------------------------------------------------
def build_tasks(
    configuration: pink.Configuration,
    ee_frame: str = END_EFFECTOR_FRAME,
) -> Tuple[FrameTask, PostureTask]:
    """构建末端跟踪 + 姿态正则两个 task。"""
    # 主任务: 末端到目标位姿
    ee_task = FrameTask(
        ee_frame,
        position_cost=1.0,      # [cost]/[m]
        orientation_cost=0.5,   # [cost]/[rad]
    )
    ee_task.set_target_from_configuration(configuration)

    # 次任务: 冗余自由度收敛到当前(或参考)姿态, 防止底盘/关节乱漂
    posture_task = PostureTask(cost=1e-3)
    posture_task.set_target_from_configuration(configuration)

    return ee_task, posture_task


# ------------------------------------------------------------
# 单次 IK: 给定目标位姿 -> 解出关节角
# ------------------------------------------------------------
def solve_once(
    target_translation: np.ndarray,
    target_rotation: Optional[np.ndarray] = None,
    q_init: Optional[np.ndarray] = None,
    max_iters: int = 200,
    dt: float = 0.01,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, bool, float]:
    """数值迭代法求单个目标位姿的 IK。

    Args:
        target_translation: (3,) 末端目标位置, 世界系。
        target_rotation:    (3,3) 末端目标旋转, 默认单位阵。
        q_init:             初始构型 (10,), 默认中立姿态。
        max_iters:          最大迭代次数。
        dt:                 每步积分步长 [s]。
        tol:                末端误差收敛阈值 (线性部分 [m])。

    Returns:
        q_final:  (10,) 求得的关节构型
        success:  是否在 tol 内收敛
        err_norm: 末端位置误差范数 [m]
    """
    model, data, q0 = load_model()
    if q_init is not None:
        q0 = np.asarray(q_init, dtype=float).copy()

    configuration = pink.Configuration(model, data, q0)
    ee_task, posture_task = build_tasks(configuration)

    # 设定末端目标
    R = np.eye(3) if target_rotation is None else np.asarray(target_rotation)
    T_target = pin.SE3(rotation=R, translation=np.asarray(target_translation))
    ee_task.set_target(T_target)

    solver = "quadprog" if "quadprog" in qpsolvers.available_solvers \
        else qpsolvers.available_solvers[0]

    err_norm = np.inf
    for it in range(max_iters):
        # 误差 = 当前末端相对目标的 SE3 log
        err6 = ee_task.compute_error(configuration)
        err_norm = float(np.linalg.norm(err6[:3]))
        if err_norm < tol:
            return configuration.q.copy(), True, err_norm

        v = solve_ik(configuration, [ee_task, posture_task], dt, solver=solver)
        configuration.integrate_inplace(v, dt)

    return configuration.q.copy(), False, err_norm


# ------------------------------------------------------------
# 循环演示: 末端画圆, 打印底盘/机械臂分解
# ------------------------------------------------------------
def run_demo(
    steps: int = 500,
    dt: float = 0.01,
    circle_radius: float = 0.2,
) -> None:
    """让末端绕初始位置画一个 XY 平面的圆, 打印 10DoF 解。"""
    model, data, q0 = load_model()
    configuration = pink.Configuration(model, data, q0)
    ee_task, posture_task = build_tasks(configuration)

    # 以初始末端位置为圆心
    T0 = configuration.get_transform_frame_to_world(END_EFFECTOR_FRAME).copy()
    center = T0.translation.copy()
    R0 = T0.rotation.copy()

    solver = "quadprog" if "quadprog" in qpsolvers.available_solvers \
        else qpsolvers.available_solvers[0]

    # 取 joint 在 q/v 中的索引, 方便拆解打印
    base_q_idx = [model.joints[model.getJointId(n)].idx_q
                  for n in BASE_JOINT_NAMES]
    arm_q_idx = [model.joints[model.getJointId(n)].idx_q
                 for n in ARM_JOINT_NAMES]

    t = 0.0
    for step in range(steps):
        # 目标: 绕 z 轴以 0.5 Hz 画圆
        omega = 2.0 * np.pi * 0.5
        tgt = center + circle_radius * np.array(
            [np.cos(omega * t), np.sin(omega * t), 0.0]
        )
        ee_task.set_target(pin.SE3(rotation=R0, translation=tgt))

        v = solve_ik(configuration, [ee_task, posture_task], dt, solver=solver)
        configuration.integrate_inplace(v, dt)

        if step % 50 == 0:
            q = configuration.q
            base_q = [q[i] for i in base_q_idx]
            arm_q = [q[i] for i in arm_q_idx]
            err = float(np.linalg.norm(ee_task.compute_error(configuration)[:3]))
            print(f"[{step:03d}] err={err:.4f} m | "
                  f"base=(x={base_q[0]:+.3f}, y={base_q[1]:+.3f}, "
                  f"yaw={base_q[2]:+.3f}) | "
                  f"arm=[" + ", ".join(f"{a:+.2f}" for a in arm_q) + "]")
        t += dt


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== 1) 单次 IK 求解 ===")
    # 取一个合理的目标: 在机器人前方 0.5m, 高度 0.8m
    q_sol, ok, err = solve_once(
        target_translation=np.array([0.5, 0.0, 0.8]),
        target_rotation=np.eye(3),
    )
    print(f"converged={ok}, final_err={err:.5f} m")
    print("q =", np.array2string(q_sol, precision=4, suppress_small=True))

    print("\n=== 2) 循环演示: 末端画圆 ===")
    run_demo(steps=500, dt=0.01, circle_radius=0.15)
