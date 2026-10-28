"""
describe.py

Turns computed joint angles into readable English, so the report can be read
without decoding numbers. Two descriptions per joint:

  - ours:        describe_joint()        e.g. "right elbow bent 50 degrees"
  - FreeMoCap's: describe_joint_euler()  e.g. "right elbow: rotation XYZ
                                               x=3 y=-52 z=0 degrees"

describe_frames() produces both for every joint in a frame; interleave()
pairs them so they can be compared joint by joint in the printed report.

check_euler_columns() prints a sanity-check range of FreeMoCap's Euler values
per bone (currently captured by the caller via redirect_stdout).

Depends on geometry.compute_joint_angles() for the angles and schema.py for
joint definitions and the neutral threshold. All angles in degrees.
"""

import numpy as np

from process.geometry import compute_joint_angles
from process.schema import JOINT_ORDER, NEUTRAL_THRESHOLD_DEG, RELEVANT_PREFIXES


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
        out.append(f"  Computed:   {c}")
        out.append(f"  FreeMoCap:  {f}")
    return out


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
