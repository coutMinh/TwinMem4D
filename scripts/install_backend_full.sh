#!/usr/bin/env bash
# Build DAAAM's full backend (ROS 2 Jazzy + Hydra/Khronos workspace) — plan task B-build.
# Requires: Ubuntu 24.04, root/sudo, apt, CUDA 12.x. Run unattended; logs go to $LOG.
#
# Usage: WS=/work/ros2_ws LOG=/work/results/runs/backend_build/build.log bash install_backend_full.sh
set -euo pipefail

WS="${WS:-$HOME/ros2_ws}"
LOG="${LOG:-$WS/build.log}"
DAAAM_URL="${DAAAM_URL:-https://github.com/MIT-SPARK/DAAAM.git}"
mkdir -p "$WS/src" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) start; WS=$WS"
. /etc/os-release
if [ "${VERSION_ID:-}" != "24.04" ]; then
  echo "BLOCKER: Ubuntu $VERSION_ID, DAAAM needs 24.04 (ROS 2 Jazzy)"; exit 2
fi
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo -n"

# packages.yaml uses SSH URLs; rewrite to HTTPS for machines without a GitHub key
git config --global url."https://github.com/".insteadOf git@github.com:

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "--- installing ROS 2 Jazzy"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends curl gnupg lsb-release software-properties-common
  $SUDO add-apt-repository -y universe
  $SUDO curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
    | $SUDO tee /etc/apt/sources.list.d/ros2.list >/dev/null
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends ros-jazzy-ros-base python3-rosdep python3-colcon-common-extensions
  $SUDO rosdep init 2>/dev/null || true
fi
set +u; source /opt/ros/jazzy/setup.bash; set -u
export ROS_DISTRO=jazzy

[ -d "$WS/src/daaam" ] || git clone "$DAAAM_URL" "$WS/src/daaam"
echo "--- upstream installer (clones 17 repos, rosdep, colcon build, pip install)"
bash "$WS/src/daaam/install/install.sh"
echo "=== $(date -u) done"
