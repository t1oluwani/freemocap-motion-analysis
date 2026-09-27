import argparse
import pandas as pd
from pathlib import Path

# The headers in the bone and joint data files follow a pattern of <prefix>_<joint_name>.
# I have defined the prefixes in the HEADER_PREFIXES list.
HEADER_PREFIXES = sorted(
    ["upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
     "thigh.L", "thigh.R", "shin.L", "shin.R"]
)

# The headers in the bone and joint data files follow a pattern of <bone_name>_<suffix>. 
# I have defined the suffixes in the HEADER_SUFFIXES list. 
HEADER_SUFFIXES = sorted(
    ["_head_center_world_x", "_head_center_world_y", "_head_center_world_z",
     "_tail_center_world_x", "_tail_center_world_y", "_tail_center_world_z",
     "_rotation_quaternion_x", "_rotation_quaternion_y", "_rotation_quaternion_z", "_rotation_quaternion_w",
     "_rotation_euler_x", "_rotation_euler_y", "_rotation_euler_z", "_rotation_euler_order"]
    + [f"_matrix_{i}_{j}" for i in range(4) for j in range(4)],
)

JOINT_TO_BONE = {
    "left_shoulder":  "upper_arm.L",
    "right_shoulder": "upper_arm.R",
    "left_elbow":     "forearm.L",
    "right_elbow":    "forearm.R",
    "left_hip":       "thigh.L",
    "right_hip":      "thigh.R",
    "left_knee":      "shin.L",
    "right_knee":     "shin.R",
}