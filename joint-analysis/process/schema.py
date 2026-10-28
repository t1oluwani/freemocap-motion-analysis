""" Schema definitions for bone and joint data. """

# The headers in the bone and joint data files follow a pattern of <prefix>_<joint_name>.
# I have defined the prefixes as the HEADER_PREFIXES list.
HEADER_PREFIXES = [
    # Lower body
    "pelvis.R",
    "thigh.R",
    "shin.R",
    "foot.R",
    "heel.02.R",
    "pelvis.L",
    "thigh.L",
    "shin.L",
    "foot.L",
    "heel.02.L",
    # Spine
    "spine",
    "spine.001",
    "neck",
    # Right arm + hand
    "shoulder.R",
    "upper_arm.R",
    "forearm.R",
    "hand.R",
    "thumb.carpal.R",
    "thumb.01.R",
    "thumb.02.R",
    "thumb.03.R",
    "palm.01.R",
    "f_index.01.R",
    "f_index.02.R",
    "f_index.03.R",
    "palm.02.R",
    "f_middle.01.R",
    "f_middle.02.R",
    "f_middle.03.R",
    "palm.03.R",
    "f_ring.01.R",
    "f_ring.02.R",
    "f_ring.03.R",
    "palm.04.R",
    "f_pinky.01.R",
    "f_pinky.02.R",
    "f_pinky.03.R",
    # Left arm + hand
    "shoulder.L",
    "upper_arm.L",
    "forearm.L",
    "hand.L",
    "thumb.carpal.L",
    "thumb.01.L",
    "thumb.02.L",
    "thumb.03.L",
    "palm.01.L",
    "f_index.01.L",
    "f_index.02.L",
    "f_index.03.L",
    "palm.02.L",
    "f_middle.01.L",
    "f_middle.02.L",
    "f_middle.03.L",
    "palm.03.L",
    "f_ring.01.L",
    "f_ring.02.L",
    "f_ring.03.L",
    "palm.04.L",
    "f_pinky.01.L",
    "f_pinky.02.L",
    "f_pinky.03.L",
]

# The headers in the bone and joint data files follow a pattern of <bone_name>_<suffix>.
# I have defined the suffixes as the HEADER_SUFFIXES list.
# (Flattened: the original version wrapped this in an extra outer list.)
HEADER_SUFFIXES = [
    # Starting point
    "_head_center_world_x",
    "_head_center_world_y",
    "_head_center_world_z",
    # Ending point
    "_tail_center_world_x",
    "_tail_center_world_y",
    "_tail_center_world_z",
    # Rotation (quaternion)
    "_rotation_quaternion_x",
    "_rotation_quaternion_y",
    "_rotation_quaternion_z",
    "_rotation_quaternion_w",
    # Rotation (euler)
    "_rotation_euler_x",
    "_rotation_euler_y",
    "_rotation_euler_z",
    "_rotation_euler_order",
] + [
    # Transformation matrix (4x4)
    f"_matrix_{i}_{j}"
    for i in range(4)
    for j in range(4)
]

# The relevant prefixes are the ones that correspond to the joints we are interested in analyzing.
RELEVANT_PREFIXES = [
    "upper_arm.L",
    "upper_arm.R",
    "forearm.L",
    "forearm.R",
    "thigh.L",
    "thigh.R",
    "shin.L",
    "shin.R",
]

# Bones needed only to build the trunk reference frame (up direction).
REFERENCE_PREFIXES = ["spine", "neck"]

# The JOINT_TO_BONE dictionary maps joint names to their corresponding bone names.
# A bone's rotation is taken to represent the joint at its head (proximal) end.
# These mappings were decided by me arbitrarily, but should line up with correct anatomically relevant locations.
JOINT_TO_BONE = {
    "left_shoulder": "upper_arm.L",
    "right_shoulder": "upper_arm.R",
    "left_elbow": "forearm.L",
    "right_elbow": "forearm.R",
    "left_hip": "thigh.L",
    "right_hip": "thigh.R",
    "left_knee": "shin.L",
    "right_knee": "shin.R",
}

# How each joint's angle is computed (see module docstring).
JOINT_SPECS = {
    "right_shoulder": {"kind": "ball", "bone": "upper_arm.R", "side": "right"},
    "right_elbow": {"kind": "hinge", "proximal": "upper_arm.R", "distal": "forearm.R"},
    "right_hip": {"kind": "ball", "bone": "thigh.R", "side": "right"},
    "right_knee": {"kind": "hinge", "proximal": "thigh.R", "distal": "shin.R"},
    "left_shoulder": {"kind": "ball", "bone": "upper_arm.L", "side": "left"},
    "left_elbow": {"kind": "hinge", "proximal": "upper_arm.L", "distal": "forearm.L"},
    "left_hip": {"kind": "ball", "bone": "thigh.L", "side": "left"},
    "left_knee": {"kind": "hinge", "proximal": "thigh.L", "distal": "shin.L"},
}
JOINT_ORDER = list(JOINT_SPECS)

# Which CSV columns feed each property in the frame dict.
PROPERTY_COLUMNS = {
    "head": ("_head_center_world_x", "_head_center_world_y", "_head_center_world_z"),
    "tail": ("_tail_center_world_x", "_tail_center_world_y", "_tail_center_world_z"),
    "quaternion": (
        "_rotation_quaternion_x",
        "_rotation_quaternion_y",
        "_rotation_quaternion_z",
        "_rotation_quaternion_w",
    ),
    "euler": ("_rotation_euler_x", "_rotation_euler_y", "_rotation_euler_z"),
}
EULER_ORDER_SUFFIX = "_rotation_euler_order"

# Below this, a joint is described as "straight" / "neutral".
NEUTRAL_THRESHOLD_DEG = 5.0