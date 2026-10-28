"""
tables.py

Converts the frame-by-frame dict into per-frame pandas tables, one row per
frame, ready for saving as CSV or plotting.

  - frames_to_computed_angle_table():   our geometric angles only
  - frames_to_freemocap_angle_table():  FreeMoCap's Euler rotations only
  - frames_to_angle_table():            the two side by side

Depends on geometry.compute_joint_angles() for the angles and schema.py for
joint definitions and order. All angles in degrees.
"""

import numpy as np
import pandas as pd

from process.geometry import compute_joint_angles
from process.schema import JOINT_SPECS, JOINT_ORDER


def frames_to_angle_table(frames):
    """Return a single DataFrame with one row per frame, containing both our computed angles and FreeMoCap's Euler angles."""
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
