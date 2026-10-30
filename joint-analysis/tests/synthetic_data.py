"""
synthetic_data.py

Builds fake FreeMoCap bone_and_joint data with KNOWN joint angles, so the tests
can check the code against ground truth instead of a real recording.

World convention used for the fake body: it faces +Y, its right side is +X and
up is +Z. A pose is a dict such as

    {"right_elbow": 50.0,            # hinge joints (elbow, knee): flexion in degrees
     "right_hip": (70.0, 0.0),       # ball joints (shoulder, hip): (flexion, abduction)
     "left_shoulder": (40.0, 15.0)}

Joints not listed are left in the neutral standing position (limbs hanging straight
down, elbows and knees straight). Angles for ball joints must be between -90 and 90
degrees (the projection angles used by the code are only defined there).
"""

import math

import numpy as np
import pandas as pd

from bone_and_joint import (
    HEADER_PREFIXES,
    HEADER_SUFFIXES,
    REFERENCE_PREFIXES,
    RELEVANT_PREFIXES,
)

RIGHT = np.array([1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])

THIGH_LENGTH = SHIN_LENGTH = 0.45
UPPER_ARM_LENGTH, FOREARM_LENGTH = 0.30, 0.25

# Three frames used by most tests: standing, a mixed pose, and another mixed pose.
STANDARD_POSES = [
    {},
    {
        "right_elbow": 50.0,
        "left_knee": 85.0,
        "right_hip": (70.0, 0.0),
        "left_shoulder": (40.0, 15.0),
    },
    {
        "right_hip": (-20.0, 0.0),
        "left_hip": (0.0, 25.0),
        "right_shoulder": (0.0, 80.0),
        "left_elbow": 120.0,
    },
]
# FreeMoCap-style rotation_euler for frame 1's right forearm: 50 degrees about x,
# written in RADIANS the way Blender stores it.
STANDARD_EULER_RAD = {(1, "forearm.R"): (math.radians(50.0), 0.0, 0.0)}


def _unit(v):
    return v / np.linalg.norm(v)


def ball_direction(flexion_deg, abduction_deg, outward):
    """Unit direction whose flexion/abduction projection angles are exactly the given ones."""
    d = (
        math.tan(math.radians(abduction_deg)) * outward
        + math.tan(math.radians(flexion_deg)) * FORWARD
        - UP
    )
    return _unit(d)


def hinge_direction(parent_direction, flexion_deg, bend_toward):
    """Unit direction exactly flexion_deg away from parent_direction."""
    n = _unit(bend_toward - np.dot(bend_toward, parent_direction) * parent_direction)
    a = math.radians(flexion_deg)
    return math.cos(a) * parent_direction + math.sin(a) * n


def make_bones(pose=None):
    """Return {bone_name: (head_xyz, tail_xyz)} for the 8 limb bones plus spine and neck."""
    pose = pose or {}
    bones = {}
    for side, sign, word in (("R", 1.0, "right"), ("L", -1.0, "left")):
        outward = sign * RIGHT

        hip = np.array([0.1 * sign, 0.0, 1.0])
        thigh_dir = ball_direction(*pose.get(f"{word}_hip", (0.0, 0.0)), outward)
        knee = hip + THIGH_LENGTH * thigh_dir
        shin_dir = hinge_direction(thigh_dir, pose.get(f"{word}_knee", 0.0), -FORWARD)
        bones[f"thigh.{side}"] = (hip, knee)
        bones[f"shin.{side}"] = (knee, knee + SHIN_LENGTH * shin_dir)

        shoulder = np.array([0.2 * sign, 0.0, 1.45])
        upper_dir = ball_direction(*pose.get(f"{word}_shoulder", (0.0, 0.0)), outward)
        elbow = shoulder + UPPER_ARM_LENGTH * upper_dir
        fore_dir = hinge_direction(upper_dir, pose.get(f"{word}_elbow", 0.0), FORWARD)
        bones[f"upper_arm.{side}"] = (shoulder, elbow)
        bones[f"forearm.{side}"] = (elbow, elbow + FOREARM_LENGTH * fore_dir)

    bones["spine"] = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.2]))
    bones["neck"] = (np.array([0.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.65]))
    return bones


def transform_bones(bones, yaw_deg=0.0, lean_deg=0.0, shift=(0.0, 0.0, 0.0)):
    """Rigidly rotate (about Z by yaw, then forward lean about X) and translate every bone."""
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cl, sl = math.cos(math.radians(lean_deg)), math.sin(math.radians(lean_deg))
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rot_x = np.array([[1, 0, 0], [0, cl, -sl], [0, sl, cl]])
    m = rot_z @ rot_x
    shift = np.asarray(shift, dtype=float)
    return {k: (m @ h + shift, m @ t + shift) for k, (h, t) in bones.items()}


def bones_to_frame(bones, euler_deg=None, euler_order="XYZ"):
    """
    One frame dict in the same format load_relevant_frames() produces, built
    directly from bones (no CSV involved).
    """
    euler_deg = euler_deg or {}
    frame = {}
    for bone in RELEVANT_PREFIXES + REFERENCE_PREFIXES:
        head, tail = bones[bone]
        frame[bone] = {
            "head": tuple(float(x) for x in head),
            "tail": tuple(float(x) for x in tail),
            "quaternion": (0.0, 0.0, 0.0, 1.0),
            "euler_deg": tuple(euler_deg.get(bone, (0.0, 0.0, 0.0))),
            "euler_order": euler_order,
        }
    return frame


def make_frame(pose=None, euler_deg=None, euler_order="XYZ", **transform):
    """Shortcut: pose -> bones -> (optional rigid transform) -> frame dict."""
    bones = make_bones(pose)
    if transform:
        bones = transform_bones(bones, **transform)
    return bones_to_frame(bones, euler_deg, euler_order)


def write_bone_csv(
    path, skeletons, euler=None, euler_order="XYZ", drop_bones=(), extra_columns=None
):
    """
    Write a CSV with the real FreeMoCap layout: all 61 bones x 30 columns = 1,830
    columns, one row per skeleton. Bones with no entry in a skeleton get zeros.

    euler:         {(frame_index, bone_name): (x, y, z)}, written as-is (units are the
                   caller's business: pass radians to mimic Blender)
    drop_bones:    bone names whose columns are left out of the file entirely
    extra_columns: {column_name: value} extra constant columns to append
    """
    euler = euler or {}
    columns = {
        prefix + suffix: []
        for prefix in HEADER_PREFIXES
        if prefix not in drop_bones
        for suffix in HEADER_SUFFIXES
    }
    for i, bones in enumerate(skeletons):
        for prefix in HEADER_PREFIXES:
            if prefix in drop_bones:
                continue
            values = {s: 0.0 for s in HEADER_SUFFIXES}
            values["_rotation_quaternion_w"] = 1.0
            values["_rotation_euler_order"] = euler_order
            if prefix in bones:
                head, tail = bones[prefix]
                for axis, v in zip("xyz", head):
                    values[f"_head_center_world_{axis}"] = v
                for axis, v in zip("xyz", tail):
                    values[f"_tail_center_world_{axis}"] = v
            for axis, v in zip("xyz", euler.get((i, prefix), (0.0, 0.0, 0.0))):
                values[f"_rotation_euler_{axis}"] = v
            for suffix in HEADER_SUFFIXES:
                columns[prefix + suffix].append(values[suffix])
    for name, value in (extra_columns or {}).items():
        columns[name] = [value] * len(skeletons)
    pd.DataFrame(columns).to_csv(path, index=False)
    return path