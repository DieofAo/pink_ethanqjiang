#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回放 pink 批量 IK 的结果到 ROS `/joint_states`，供 RViz 可视化。

输入文件格式（由 ik_batch_pink.py 产生）每行：
    ts arm pink_status  x y z qx qy qz qw
    base_x base_y base_yaw  j1..j7  pos_err ori_err iters

发布内容：
    话题  : /joint_states (sensor_msgs/JointState)
    关节  : base_x, base_y, base_yaw, J_1, J_2, J_3, J_4, J_5, J_6, J_7

RViz 侧建议启动（另开终端）：
    roscore
    rosparam set robot_description -t $(pwd)/left_jaka_mobile.urdf
    rosrun robot_state_publisher robot_state_publisher
    rviz  # Fixed Frame 设成 world，加 RobotModel + TF
"""
from __future__ import annotations

import argparse
import os
import sys

import rospy
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "base_x", "base_y", "base_yaw",
    "J_1", "J_2", "J_3", "J_4", "J_5", "J_6", "J_7",
]


def load_rows(path: str):
    """返回 list of 10-float (base_x, base_y, base_yaw, j1..j7)."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 期望 23 列；兼容其他列数时取 pink_status 之后的 10 个 joint
            # 列布局: 0 ts, 1 arm, 2 status, 3..9 pose7, 10..19 joint10, 20..22 err
            if len(parts) < 20:
                continue
            try:
                q10 = [float(x) for x in parts[10:20]]
            except ValueError:
                continue
            rows.append(q10)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", "-i",
        default=os.path.join(
            os.path.dirname(__file__),
            # "xiyiye/left_umi_20260417_150221.pink.txt",
            "xiyiye/batch_damp20.txt",
        ),
    )
    ap.add_argument("--hz", type=float, default=30.0, help="播放频率 Hz")
    ap.add_argument("--topic", default="/joint_states")
    ap.add_argument("--loop", action="store_true", help="循环播放")
    ap.add_argument("--start", type=int, default=0, help="起始帧")
    ap.add_argument("--end", type=int, default=-1, help="结束帧(含)，-1=末尾")
    args = ap.parse_args()

    rows = load_rows(args.input)
    if not rows:
        print(f"[err] no rows loaded from {args.input}", file=sys.stderr)
        sys.exit(1)
    end = len(rows) if args.end < 0 else min(args.end + 1, len(rows))
    rows = rows[args.start:end]
    print(f"[io] {len(rows)} frames from {args.input}")

    rospy.init_node("pink_joint_publisher", anonymous=True)
    pub = rospy.Publisher(args.topic, JointState, queue_size=10)
    rate = rospy.Rate(args.hz)

    # 等一个订阅者，避免上电瞬间第一帧被丢
    t0 = rospy.Time.now()
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        if (rospy.Time.now() - t0).to_sec() > 3.0:
            rospy.logwarn("no subscriber on %s, publishing anyway", args.topic)
            break
        rospy.sleep(0.1)

    rospy.loginfo("publish %s @ %.1f Hz, joints=%s",
                  args.topic, args.hz, JOINT_NAMES)

    idx = 0
    total = len(rows)
    while not rospy.is_shutdown():
        q = rows[idx]
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q
        pub.publish(msg)

        if idx % 100 == 0:
            rospy.loginfo(
                "[%04d/%04d] base=(%+.3f, %+.3f, %+.3f)  J=%s",
                idx, total, q[0], q[1], q[2],
                "[" + ", ".join(f"{v:+.3f}" for v in q[3:]) + "]",
            )

        idx += 1
        if idx >= total:
            if args.loop:
                idx = 0
                rospy.loginfo("--- loop restart ---")
            else:
                rospy.loginfo("done.")
                break
        rate.sleep()


if __name__ == "__main__":
    main()
