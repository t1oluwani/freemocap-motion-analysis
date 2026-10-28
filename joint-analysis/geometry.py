""" 
Geometric calculations for joint analysis. 

Each joint is reported two ways so they can be compared (as you (Prof. Payandeh) requested):
  - OUR angle: geometric, from the bone head/tail world positions.
  - FreeMoCap's own: the Blender pose-bone rotation_euler of the matching bone.

JOINT DEFINITIONS (ours)
  Elbow / knee (hinge joints): flexion = angle between the proximal bone
      direction and the distal bone direction (upper_arm->forearm,
      thigh->shin).  0 = straight, larger = more bent.
  Shoulder / hip (ball joints): measured against a trunk frame built from the
      bones themselves:
          up      = spine head -> neck head
          right   = left thigh head -> right thigh head (made perpendicular to up)
          forward = up x right
      The limb direction is projected onto the sagittal plane (up/forward) and
      the frontal plane (up/right):
          flexion   (+ = limb forward of the trunk, - = behind)
          abduction (+ = limb away from the midline, - = across the body)
      0/0 means the limb hangs straight down relative to the trunk.
      These are projection angles, not a full 3-axis joint decomposition, and
      they get less meaningful once a limb is raised near/above horizontal.
      """


import numpy as np

from schema import (
    JOINT_TO_BONE,
    JOINT_SPECS,
    JOINT_ORDER,
)

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.full(3, np.nan)


def _angle_between(u, v) -> float:
    """
    Angle in degrees between two vectors (NaN if either is degenerate).
    Formula: cos^-1((u . v) / (||u|| ||v||)) = cos^-1(u . v)
    Precondition: u and v are unit vectors.
    """
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0))))


def _signed_angle(d, axis, down) -> float:
    """
    Angle of direction d away from `down`, toward `axis`, in the plane they span.
    Formula: tan^-1((d . axis) / (d . down)) = tan^-1((d . axis) / (d . down))
    Precondition: d, axis, down are unit vectors and not collinear.
    """
    return float(np.degrees(np.arctan2(np.dot(d, axis), np.dot(d, down))))


def bone_direction(frame: dict, bone: str) -> np.ndarray:
    """
    Unit vector from the bone's head to its tail, in world coordinates.
    Formula: (tail - head) / ||tail - head||.  Returns NaN if the bone is missing or degenerate.
    Precondition: The bone exists in the frame.
    """

    return _unit(np.array(frame[bone]["tail"]) - np.array(frame[bone]["head"]))


def trunk_frame(frame: dict) -> dict:
    """
    Trunk axes (world coordinates) from the spine/neck/thigh bones:
    up, right, forward (right-handed: forward = up x right).
    """
    up = _unit(np.array(frame["neck"]["head"]) - np.array(frame["spine"]["head"]))
    right = np.array(frame["thigh.R"]["head"]) - np.array(frame["thigh.L"]["head"])
    right = _unit(right - np.dot(right, up) * up)
    return {"up": up, "right": right, "forward": np.cross(up, right)}


def compute_joint_angles(frame: dict) -> dict:
    """
    Joint angles for one frame (a {bone: properties} dict).

    Returns {joint_name: entry}, where entry is None if a needed bone is
    missing, otherwise:
        hinge joints (elbow, knee):  {"flexion_deg"}
        ball joints (shoulder, hip): {"flexion_deg", "abduction_deg"}
    plus FreeMoCap's own rotation for the matching bone:
        {"freemocap_euler_deg": (x, y, z), "freemocap_euler_order": "XYZ"}
    Values are NaN where the tracking data for that frame was missing.
    """
    try:
        trunk = trunk_frame(frame)
    except KeyError:
        trunk = None

    result = {}
    for joint in JOINT_ORDER:
        spec = JOINT_SPECS[joint]
        try:
            if spec["kind"] == "hinge":
                entry = {
                    "flexion_deg": _angle_between(
                        bone_direction(frame, spec["proximal"]),
                        bone_direction(frame, spec["distal"]),
                    )
                }
            else:
                if trunk is None:
                    raise KeyError("trunk reference bones")
                d = bone_direction(frame, spec["bone"])
                down = -trunk["up"]
                outward = trunk["right"] if spec["side"] == "right" else -trunk["right"]
                entry = {
                    "flexion_deg": _signed_angle(d, trunk["forward"], down),
                    "abduction_deg": _signed_angle(d, outward, down),
                }
            fm_bone = frame[JOINT_TO_BONE[joint]]
            entry["freemocap_euler_deg"] = fm_bone["euler_deg"]
            entry["freemocap_euler_order"] = fm_bone["euler_order"]
            result[joint] = entry
        except KeyError:
            result[joint] = None
    return result