#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
"""
批量 pink IK：给定一条末端位姿序列（世界系），用 10-DoF (底盘3 + 机械臂7)
重新求 IK，并用 FK 回验末端位姿，输出收敛结果。

输入文件每行 17 列（空格分隔）：
    ts arm status  x y z qx qy qz qw  j1 j2 j3 j4 j5 j6 j7
其中四元数顺序是 xyzw；status ∈ {success, solver_failed, large_joint_change}，
只在 success 时 7 维关节角可信。

输出文件每行（空格分隔）：
    ts arm pink_status  x y z qx qy qz qw
    base_x base_y base_yaw j1..j7  pos_err ori_err iters
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import DampingTask, FrameTask, PostureTask

from ik_left_jaka_mobile import (
    ARM_JOINT_NAMES,
    BASE_JOINT_NAMES,
    END_EFFECTOR_FRAME,
    URDF_PATH,
    load_model,
)

# ------------------------------------------------------------
# 常量
# ------------------------------------------------------------
POS_TOL = 1e-3      # [m]
ORI_TOL = 0.1       # [rad]
DT = 0.01           # pink 内部积分步长
MAX_ITERS = 500     # 单帧最大迭代
POSITION_COST = 1.0
ORIENTATION_COST = 1.0
POSTURE_COST = 1e-3  # 冗余自由度正则（小一点，允许底盘动）
DAMPING_COST_BASE = 20.0  # 底盘速度惩罚（越大底盘越不想动）
DAMPING_COST_ARM = 0.0    # 机械臂速度不惩罚


# ------------------------------------------------------------
# 数据解析
# ------------------------------------------------------------
def parse_line(line: str) -> Optional[dict]:
    """解析一行，失败返回 None。"""
    parts = line.strip().split()
    if len(parts) != 17:
        return None
    ts, arm, status = parts[0], parts[1], parts[2]
    vals = [float(x) for x in parts[3:]]
    xyz = np.array(vals[0:3])
    # 文件里是 xyzw
    qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
    quat_xyzw = np.array([qx, qy, qz, qw])
    q_arm7 = np.array(vals[7:14])

    # pinocchio 的 Quaternion 构造也是 (x, y, z, w)，直接用
    quat = pin.Quaternion(qw, qx, qy, qz)  # (w, x, y, z) signature
    quat.normalize()
    T = pin.SE3(quat.toRotationMatrix(), xyz)
    return {
        "ts": ts,
        "arm": arm,
        "status": status,
        "xyz": xyz,
        "quat_xyzw": quat_xyzw,
        "T": T,
        "q_arm7": q_arm7,
    }


def read_traj(path: str) -> List[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = parse_line(line)
            if item is not None:
                out.append(item)
    return out


# ------------------------------------------------------------
# 单帧 IK（内部多次迭代直到收敛）
# ------------------------------------------------------------
def se3_error(T_cur: pin.SE3, T_tgt: pin.SE3) -> Tuple[float, float]:
    """返回 (位置误差范数 [m], 姿态误差范数 [rad])。"""
    dT = T_tgt.actInv(T_cur)  # current relative to target
    err6 = pin.log6(dT).vector  # [v(3); w(3)]
    return float(np.linalg.norm(err6[:3])), float(np.linalg.norm(err6[3:]))


def solve_frame(
    configuration: pink.Configuration,
    ee_task: FrameTask,
    posture_task: PostureTask,
    damping_task: DampingTask,
    T_target: pin.SE3,
    ee_frame: str,
    solver: str,
    pos_tol: float = POS_TOL,
    ori_tol: float = ORI_TOL,
    dt: float = DT,
    max_iters: int = MAX_ITERS,
) -> Tuple[bool, float, float, int]:
    """在当前 configuration 上迭代收敛到 T_target。

    Returns: (ok, pos_err, ori_err, iters)
    """
    ee_task.set_target(T_target)
    # 用当前构型作为 posture 基准（每帧刷新，避免被旧帧拉偏）
    posture_task.set_target_from_configuration(configuration)

    pos_err = ori_err = np.inf
    for it in range(1, max_iters + 1):
        T_cur = configuration.get_transform_frame_to_world(ee_frame)
        pos_err, ori_err = se3_error(T_cur, T_target)
        if pos_err < pos_tol and ori_err < ori_tol:
            return True, pos_err, ori_err, it

        v = solve_ik(
            configuration,
            [ee_task, posture_task, damping_task],
            dt,
            solver=solver,
        )
        configuration.integrate_inplace(v, dt)

    # 最后再算一次 FK 误差（迭代完的 q）
    T_cur = configuration.get_transform_frame_to_world(ee_frame)
    pos_err, ori_err = se3_error(T_cur, T_target)
    return (pos_err < pos_tol and ori_err < ori_tol), pos_err, ori_err, max_iters


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def run(
    input_path: str,
    output_path: str,
    use_tracik_warmstart: bool = True,
) -> None:
    frames = read_traj(input_path)
    print(f"[io] loaded {len(frames)} frames from {input_path}")

    model, data, q_neutral = load_model()
    configuration = pink.Configuration(model, data, q_neutral.copy())

    # 构建任务（复用 ik_left_jaka_mobile.build_tasks 会每帧重建，这里直接建一次）
    ee_task = FrameTask(
        END_EFFECTOR_FRAME,
        position_cost=POSITION_COST,
        orientation_cost=ORIENTATION_COST,
    )
    posture_task = PostureTask(cost=POSTURE_COST)
    posture_task.set_target_from_configuration(configuration)

    # DampingTask：按关节维度惩罚速度（底盘大、机械臂为 0）
    base_v_idx = [
        model.joints[model.getJointId(n)].idx_v for n in BASE_JOINT_NAMES
    ]
    damping_cost_vec = np.full(model.nv, DAMPING_COST_ARM, dtype=float)
    for i in base_v_idx:
        damping_cost_vec[i] = DAMPING_COST_BASE
    damping_task = DampingTask(cost=DAMPING_COST_BASE)  # 占位
    damping_task.cost = damping_cost_vec                # 覆写为向量 cost
    print(f"[damping] cost vec = {damping_cost_vec}")

    solver = (
        "quadprog"
        if "quadprog" in qpsolvers.available_solvers
        else qpsolvers.available_solvers[0]
    )
    print(f"[solver] using {solver}")

    base_q_idx = [
        model.joints[model.getJointId(n)].idx_q for n in BASE_JOINT_NAMES
    ]
    arm_q_idx = [
        model.joints[model.getJointId(n)].idx_q for n in ARM_JOINT_NAMES
    ]

    # 首帧 warm-start
    first = frames[0]
    q_init = q_neutral.copy()
    if use_tracik_warmstart and first["status"] == "success":
        for idx, val in zip(arm_q_idx, first["q_arm7"]):
            q_init[idx] = val
    configuration.q = q_init  # pink.Configuration 内部会自动同步 data

    # 输出
    n_success = 0
    n_rescued = 0       # 原 tracik 失败但 pink 成功
    n_broken = 0        # 原 tracik 成功但 pink 失败
    t0 = time.time()

    with open(output_path, "w") as fout:
        fout.write(
            "# ts arm pink_status x y z qx qy qz qw "
            "base_x base_y base_yaw j1 j2 j3 j4 j5 j6 j7 "
            "pos_err ori_err iters\n"
        )
        for i, frame in enumerate(frames):
            ok, pe, oe, it = solve_frame(
                configuration, ee_task, posture_task, damping_task,
                frame["T"], END_EFFECTOR_FRAME, solver,
            )
            q = configuration.q
            base_q = [q[k] for k in base_q_idx]
            arm_q = [q[k] for k in arm_q_idx]
            status = "success" if ok else "failed"
            orig = frame["status"]
            if ok:
                n_success += 1
                if orig != "success":
                    n_rescued += 1
            else:
                if orig == "success":
                    n_broken += 1

            xyz = frame["xyz"]
            qx, qy, qz, qw = frame["quat_xyzw"]
            fout.write(
                f"{frame['ts']} {frame['arm']} {status} "
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} "
                f"{base_q[0]:+.6f} {base_q[1]:+.6f} {base_q[2]:+.6f} "
                + " ".join(f"{a:+.6f}" for a in arm_q)
                + f" {pe:.6e} {oe:.6e} {it}\n"
            )

            if i % 100 == 0:
                print(
                    f"[{i:04d}/{len(frames)}] orig={orig:18s} "
                    f"pink={status:7s} pos_err={pe:.2e} ori_err={oe:.2e} "
                    f"iters={it:3d} | base=({base_q[0]:+.3f},{base_q[1]:+.3f},{base_q[2]:+.3f})"
                )

    dt_total = time.time() - t0
    print(
        f"\n=== summary ===\n"
        f"  frames      : {len(frames)}\n"
        f"  pink success: {n_success} ({100.0*n_success/len(frames):.1f}%)\n"
        f"  rescued     : {n_rescued} (原 tracik 失败 -> pink 成功)\n"
        f"  broken      : {n_broken} (原 tracik 成功 -> pink 失败)\n"
        f"  elapsed     : {dt_total:.2f}s ({dt_total/len(frames)*1e3:.1f} ms/frame)\n"
        f"  output      : {output_path}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", "-i",
        default=os.path.join(os.path.dirname(__file__),
                             "xiyiye/left_umi_20260417_150221.txt"),
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="默认在输入同目录，加 .pink.txt 后缀",
    )
    p.add_argument(
        "--no-warmstart", action="store_true",
        help="不使用 tracik 7 维解作为首帧 warm-start",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = args.output or args.input.replace(".txt", ".pink.txt")
    run(args.input, out, use_tracik_warmstart=not args.no_warmstart)
