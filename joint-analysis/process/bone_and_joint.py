"""
bone_and_joint.py

Library for turning FreeMoCap's Blender-exported bone_and_joint CSV into:

  1. Done via 'columns.py':
        a frame-by-frame dict containing ONLY the bones we care about:
         frames = {
             0: {"forearm.R": {"head": (x, y, z), "tail": (x, y, z),
                               "quaternion": (x, y, z, w),
                               "euler_deg": (x, y, z), "euler_order": "XYZ"},
                 "thigh.L": {...}, ...},
             1: {...},
         }

  2. Done via 'geometry.py':
  joint angles computed from the world-space bone positions, and
  
  3. Done via 'joint_and_bone.py':
  readable text such as "right elbow bent 50 degrees".





"""

import numpy as np
import pandas as pd

from geometry import compute_joint_angles

from schema import (
    RELEVANT_PREFIXES,
    JOINT_SPECS,
    JOINT_ORDER,
    NEUTRAL_THRESHOLD_DEG,
)


def describe_joint(joint_name: str, angles: dict | None) -> str:
    """e.g. 'right elbow bent 50 degrees', 'left hip flexed 70 degrees, abducted 10 degrees'."""
    label = joint_name.replace("_", " ")
    if angles is None:
        return f"{label}: no data"
    flex = angles["flexion_deg"]
    abd = angles.get("abduction_deg")
    if np.isnan(flex) or (abd is not None and np.isnan(abd)):
        return f"{label}: no data"

    t = NEUTRAL_THRESHOLD_DEG
    part = joint_name.split("_", 1)[1]

    if part in ("elbow", "knee"):
        return f"{label} straight" if flex < t else f"{label} bent {flex:.0f} degrees"

    parts = []
    if part == "hip":
        if flex >= t:
            parts.append(f"flexed {flex:.0f} degrees")
        elif flex <= -t:
            parts.append(f"extended {-flex:.0f} degrees")
        if abd >= t:
            parts.append(f"abducted {abd:.0f} degrees")
        elif abd <= -t:
            parts.append(f"adducted {-abd:.0f} degrees")
    else:  # shoulder
        if flex >= t:
            parts.append(f"raised forward {flex:.0f} degrees")
        elif flex <= -t:
            parts.append(f"raised backward {-flex:.0f} degrees")
        if abd >= t:
            parts.append(f"raised out to the side {abd:.0f} degrees")
        elif abd <= -t:
            parts.append(f"pulled across the body {-abd:.0f} degrees")
    return f"{label} " + (", ".join(parts) if parts else "neutral")


def describe_joint_euler(joint_name: str, angles: dict | None) -> str:
    """
    Readable line for FreeMoCap's own pose-bone rotation for a joint.

    FreeMoCap exposes rotation_euler of the matching bone. Its axes are the
    bone's local axes, not anatomical flexion/abduction, so we report the raw
    XYZ (in the recorded euler_order) rather than pretending it is a joint
    angle. A brief "straight/neutral" note is added when the rotation is small,
    so the combined view reads naturally.
    """
    label = joint_name.replace("_", " ")
    if angles is None:
        return f"{label}: no data"
    euler = angles.get("freemocap_euler_deg")
    order = angles.get("freemocap_euler_order", "XYZ")
    if euler is None or any(np.isnan(v) for v in euler):
        return f"{label}: no data"

    t = NEUTRAL_THRESHOLD_DEG
    if max(abs(v) for v in euler) < t:
        return f"{label}: rotation ~0 degrees"

    x, y, z = euler
    return f"{label}: rotation {order} " f"x={x:.0f} y={y:.0f} z={z:.0f} degrees"


def describe_frames(frame: dict) -> tuple[list[str], list[str]]:
    """Return (computed_lines, freemocap_lines) for one frame."""
    angles = compute_joint_angles(frame)
    computed = [describe_joint(j, angles[j]) for j in JOINT_ORDER]
    freemocap = [describe_joint_euler(j, angles[j]) for j in JOINT_ORDER]
    return computed, freemocap


def interleave(computed: list[str], freemocap: list[str]) -> list[str]:
    """Pair each computed line with its FreeMoCap line, joint by joint."""
    out = []
    for c, f in zip(computed, freemocap):
        out.append(f"  computed:   {c}")
        out.append(f"  FreeMoCap:  {f}")
    return out


def frames_to_angle_table(frames):
    return pd.concat(
        [
            frames_to_computed_angle_table(frames),
            frames_to_freemocap_angle_table(frames).drop(columns="frame"),
        ],
        axis=1,
    )


def frames_to_computed_angle_table(frames: dict[int, dict]) -> pd.DataFrame:
    """
    One row per frame: ONLY our geometric angles (flexion for all joints,
    abduction for ball joints). No FreeMoCap Euler columns.
    """
    rows = []
    for i, frame in frames.items():
        angles = compute_joint_angles(frame)
        row = {"frame": i}
        for joint in JOINT_ORDER:
            a = angles[joint]
            row[f"{joint}_flexion_deg"] = a["flexion_deg"] if a else np.nan
            if JOINT_SPECS[joint]["kind"] == "ball":
                row[f"{joint}_abduction_deg"] = a["abduction_deg"] if a else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def frames_to_freemocap_angle_table(frames: dict[int, dict]) -> pd.DataFrame:
    """
    One row per frame: ONLY FreeMoCap's own Euler rotations (degrees) for the
    bone mapped to each joint. No geometric angles.
    """
    rows = []
    for i, frame in frames.items():
        angles = compute_joint_angles(frame)
        row = {"frame": i}
        for joint in JOINT_ORDER:
            a = angles[joint]
            fm = a["freemocap_euler_deg"] if a else (np.nan,) * 3
            order = a["freemocap_euler_order"] if a else ""
            for axis, v in zip("xyz", fm):
                row[f"{joint}_freemocap_euler_{axis}_deg"] = v
            row[f"{joint}_freemocap_euler_order"] = order
        rows.append(row)
    return pd.DataFrame(rows)


def check_euler_columns(frames: dict[int, dict]) -> None:
    """Print the range of FreeMoCap's Euler values per relevant bone (sanity check)."""
    print("FreeMoCap rotation_euler range per bone (degrees):")
    all_zero = True
    for bone in RELEVANT_PREFIXES:
        vals = np.array([f[bone]["euler_deg"] for f in frames.values() if bone in f])
        if vals.size == 0 or np.isnan(vals).all():
            continue
        lo, hi = np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)
        print(
            f"  {bone:<12} x[{lo[0]:8.1f}, {hi[0]:8.1f}]  "
            f"y[{lo[1]:8.1f}, {hi[1]:8.1f}]  z[{lo[2]:8.1f}, {hi[2]:8.1f}]"
        )
        if np.nanmax(np.abs(vals)) > 1e-3:
            all_zero = False
    if all_zero:
        print(
            "  NOTE: every Euler value is ~0. If the rig is driven by constraints, the raw\n"
            "  rotation_euler properties may not reflect the motion; the head/tail-based\n"
            "  angles above are then the reliable ones."
        )
