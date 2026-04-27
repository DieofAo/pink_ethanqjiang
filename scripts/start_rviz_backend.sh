#!/usr/bin/env bash
# 启动 RViz 可视化所需的后端：roscore + robot_state_publisher
# 使用：
#   终端 A:  bash scripts/start_rviz_backend.sh
#   终端 B:  rviz   # Fixed Frame = world, 加 RobotModel / TF
#   终端 C:  /home/ethanqjiang/.conda/envs/pink/bin/python publish_pink_joints.py --hz 30

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URDF="$(cd "$SCRIPT_DIR/.." && pwd)/left_jaka_mobile.urdf"

if [[ ! -f "$URDF" ]]; then
  echo "URDF not found: $URDF" >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash

# 检查 roscore
if ! pgrep -x rosmaster >/dev/null; then
  echo "[info] starting roscore ..."
  roscore &
  sleep 2
fi

echo "[info] loading URDF -> /robot_description"
rosparam set /robot_description -t "$URDF"

echo "[info] launch robot_state_publisher"
rosrun robot_state_publisher robot_state_publisher __name:=rsp_pink
