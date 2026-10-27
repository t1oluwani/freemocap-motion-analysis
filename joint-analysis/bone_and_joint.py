"""
joint_analysis.py

Turns FreeMoCap's Blender-exported bone_and_joint CSV into:

  1. a frame-by-frame dict containing ONLY the bones we care about:
         frames = {
             0: {"forearm.R": {"head": (x, y, z), "tail": (x, y, z),
                               "quaternion": (x, y, z, w),
                               "euler_deg": (x, y, z), "euler_order": "XYZ"},
                 "thigh.L": {...}, ...},
             1: {...},
         }
  2. joint angles computed from the world-space bone positions, and
  3. readable text such as "right elbow bent 50 degrees".

Each joint is reported two ways so they can be compared (as Prof. Payandeh
asked):
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

UNITS: Blender's Python API stores rotation_euler in RADIANS. This script
assumes radians by default (--euler_unit degrees to override) and warns if the
values look like degrees.

Usage:
    python joint_analysis.py --csv path/to/..._bone_and_joint_data.csv
    python joint_analysis.py --selftest
"""

import argparse
import math
import os
import tempfile

import numpy as np
import pandas as pd

DATA_TYPE = "bone_and_joint_data.csv"
RECORDING_NUMBER = "recording_17_33_15_gmt-7"
VIDEO_FOLDER_PATH = "freemocap/video1/saved_data"
BONE_AND_JOINT_CSV_FILE = f"{VIDEO_FOLDER_PATH}/{RECORDING_NUMBER}_{DATA_TYPE}"

# The headers in the bone and joint data files follow a pattern of <prefix>_<joint_name>.
# I have defined the prefixes as the HEADER_PREFIXES list.
HEADER_PREFIXES = [
    # Lower body
    "pelvis.R", "thigh.R", "shin.R", "foot.R", "heel.02.R",
    "pelvis.L", "thigh.L", "shin.L", "foot.L", "heel.02.L",

    # Spine
    "spine", "spine.001", "neck",

    # Right arm + hand
    "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
    "thumb.carpal.R", "thumb.01.R", "thumb.02.R", "thumb.03.R",
    "palm.01.R", "f_index.01.R", "f_index.02.R", "f_index.03.R",
    "palm.02.R", "f_middle.01.R", "f_middle.02.R", "f_middle.03.R",
    "palm.03.R", "f_ring.01.R", "f_ring.02.R", "f_ring.03.R",
    "palm.04.R", "f_pinky.01.R", "f_pinky.02.R", "f_pinky.03.R",

    # Left arm + hand
    "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
    "thumb.carpal.L", "thumb.01.L", "thumb.02.L", "thumb.03.L",
    "palm.01.L", "f_index.01.L", "f_index.02.L", "f_index.03.L",
    "palm.02.L", "f_middle.01.L", "f_middle.02.L", "f_middle.03.L",
    "palm.03.L", "f_ring.01.L", "f_ring.02.L", "f_ring.03.L",
    "palm.04.L", "f_pinky.01.L", "f_pinky.02.L", "f_pinky.03.L",
]

# The headers in the bone and joint data files follow a pattern of <bone_name>_<suffix>.
# I have defined the suffixes as the HEADER_SUFFIXES list.
# (Flattened: the original version wrapped this in an extra outer list.)
HEADER_SUFFIXES = [
    # Starting point
    "_head_center_world_x", "_head_center_world_y", "_head_center_world_z",

    # Ending point
    "_tail_center_world_x", "_tail_center_world_y", "_tail_center_world_z",

    # Rotation (quaternion)
    "_rotation_quaternion_x", "_rotation_quaternion_y",
    "_rotation_quaternion_z", "_rotation_quaternion_w",

    # Rotation (euler)
    "_rotation_euler_x", "_rotation_euler_y", "_rotation_euler_z",
    "_rotation_euler_order",
] + [
    # Transformation matrix (4x4)
    f"_matrix_{i}_{j}" for i in range(4) for j in range(4)
]

# The relevant prefixes are the ones that correspond to the joints we are interested in analyzing.
RELEVANT_PREFIXES = [
    "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
    "thigh.L", "thigh.R", "shin.L", "shin.R"
]

# Bones needed only to build the trunk reference frame (up direction).
REFERENCE_PREFIXES = ["spine", "neck"]

# The JOINT_TO_BONE dictionary maps joint names to their corresponding bone names.
# A bone's rotation is taken to represent the joint at its head (proximal) end.
# These mappings were decided by me arbitrarily, but should line up with correct anatomically relevant locations.
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

# How each joint's angle is computed (see module docstring).
JOINT_SPECS = {
    "right_shoulder": {"kind": "ball",  "bone": "upper_arm.R", "side": "right"},
    "right_elbow":    {"kind": "hinge", "proximal": "upper_arm.R", "distal": "forearm.R"},
    "right_hip":      {"kind": "ball",  "bone": "thigh.R", "side": "right"},
    "right_knee":     {"kind": "hinge", "proximal": "thigh.R", "distal": "shin.R"},
    "left_shoulder":  {"kind": "ball",  "bone": "upper_arm.L", "side": "left"},
    "left_elbow":     {"kind": "hinge", "proximal": "upper_arm.L", "distal": "forearm.L"},
    "left_hip":       {"kind": "ball",  "bone": "thigh.L", "side": "left"},
    "left_knee":      {"kind": "hinge", "proximal": "thigh.L", "distal": "shin.L"},
}
JOINT_ORDER = list(JOINT_SPECS)

# Which CSV columns feed each property in the frame dict.
PROPERTY_COLUMNS = {
    "head": ("_head_center_world_x", "_head_center_world_y", "_head_center_world_z"),
    "tail": ("_tail_center_world_x", "_tail_center_world_y", "_tail_center_world_z"),
    "quaternion": ("_rotation_quaternion_x", "_rotation_quaternion_y",
                   "_rotation_quaternion_z", "_rotation_quaternion_w"),
    "euler": ("_rotation_euler_x", "_rotation_euler_y", "_rotation_euler_z"),
}
EULER_ORDER_SUFFIX = "_rotation_euler_order"

# Below this, a joint is described as "straight" / "neutral".
NEUTRAL_THRESHOLD_DEG = 5.0

def get_header_prefixes() -> list[str]:
    """Every bone-name prefix present in the CSV."""
    return HEADER_PREFIXES

def get_header_suffixes() -> list[str]:
    """Every per-bone property suffix present in the CSV."""
    return HEADER_SUFFIXES

def get_joint_to_bone_mapping() -> dict[str, str]:
    """Mapping of joint names to their corresponding bone names."""
    return JOINT_TO_BONE

def get_relevant_prefixes() -> list[str]:
    """Bone-name prefixes for the joints we analyze (removing the ones we don't care for)."""
    return RELEVANT_PREFIXES


# ---------------------------------------------------------------------------
# 1. Column selection + frame-by-frame dictionary
# ---------------------------------------------------------------------------

def build_relevant_columns_from_csv(csv_file_path: str, include_matrix: bool = False) -> list[str]:
    """
    Read just the header row of the CSV and return the names of the columns we
    need: every property (minus the 16 matrix values unless include_matrix)
    of the relevant bones plus the trunk reference bones.

    Matching is exact (<prefix> + <known suffix>)
    """
    wanted_suffixes = {s for s in HEADER_SUFFIXES if include_matrix or "_matrix_" not in s}
    headers = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
    prefixes = RELEVANT_PREFIXES + REFERENCE_PREFIXES

    relevant_columns = []
    for header in headers:
        for prefix in prefixes:
            if header.startswith(prefix + "_") and header[len(prefix):] in wanted_suffixes:
                relevant_columns.append(header)
                break
    return relevant_columns


def load_relevant_frames(csv_file_path: str, euler_unit: str = "radians") -> dict[int, dict]:
    """
    Load only the relevant columns and return a frame-by-frame dict:

        {frame_index: {bone_name: {"head": (x, y, z), "tail": (x, y, z),
                                   "quaternion": (x, y, z, w),
                                   "euler_deg": (x, y, z),
                                   "euler_order": "XYZ"}}}

    euler_unit says what unit the CSV's rotation_euler columns are in
    ("radians" is what Blender's API produces); "euler_deg" is always degrees.
    """
    if euler_unit not in ("radians", "degrees"):
        raise ValueError("euler_unit must be 'radians' or 'degrees'")

    columns = build_relevant_columns_from_csv(csv_file_path)
    df = pd.read_csv(csv_file_path, usecols=columns)

    bones = [p for p in RELEVANT_PREFIXES + REFERENCE_PREFIXES
             if all(p + s in df.columns for s in PROPERTY_COLUMNS["head"])]
    missing = [p for p in RELEVANT_PREFIXES + REFERENCE_PREFIXES if p not in bones]
    if missing:
        print(f"Warning: bones not found in this CSV: {missing}")

    arrays = {}
    for bone in bones:
        arrays[bone] = {key: df[[bone + s for s in suffixes]].to_numpy(dtype=float)
                        for key, suffixes in PROPERTY_COLUMNS.items()}
        order_col = bone + EULER_ORDER_SUFFIX
        arrays[bone]["euler_order"] = (df[order_col].astype(str).to_numpy()
                                       if order_col in df.columns
                                       else np.full(len(df), ""))

    if euler_unit == "radians":
        biggest = max((np.nanmax(np.abs(a["euler"])) for a in arrays.values()
                       if a["euler"].size and not np.isnan(a["euler"]).all()), default=0.0)
        if biggest > 2 * math.pi + 0.5:
            print(f"Warning: euler values reach {biggest:.1f}, too large for radians - "
                  f"the CSV may already be in degrees (use --euler_unit degrees).")
    scale = 180.0 / math.pi if euler_unit == "radians" else 1.0

    frames = {}
    for i in range(len(df)):
        frames[i] = {
            bone: {
                "head": tuple(a["head"][i].tolist()),
                "tail": tuple(a["tail"][i].tolist()),
                "quaternion": tuple(a["quaternion"][i].tolist()),
                "euler_deg": tuple((a["euler"][i] * scale).tolist()),
                "euler_order": str(a["euler_order"][i]),
            }
            for bone, a in arrays.items()
        }
    return frames


# ---------------------------------------------------------------------------
# 2. Geometry + joint angles
# ---------------------------------------------------------------------------

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
                entry = {"flexion_deg": _angle_between(
                    bone_direction(frame, spec["proximal"]),
                    bone_direction(frame, spec["distal"]))}
            else:
                if trunk is None:
                    raise KeyError("trunk reference bones")
                d = bone_direction(frame, spec["bone"])
                down = -trunk["up"]
                outward = trunk["right"] if spec["side"] == "right" else -trunk["right"]
                entry = {"flexion_deg": _signed_angle(d, trunk["forward"], down),
                         "abduction_deg": _signed_angle(d, outward, down)}
            fm_bone = frame[JOINT_TO_BONE[joint]]
            entry["freemocap_euler_deg"] = fm_bone["euler_deg"]
            entry["freemocap_euler_order"] = fm_bone["euler_order"]
            result[joint] = entry
        except KeyError:
            result[joint] = None
    return result


# ---------------------------------------------------------------------------
# 3. Readable text
# ---------------------------------------------------------------------------

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


def describe_frame(frame: dict) -> list[str]:
    """One readable line per joint for a single frame."""
    angles = compute_joint_angles(frame)
    return [describe_joint(j, angles[j]) for j in JOINT_ORDER]


def describe_frames(frames: dict[int, dict]) -> dict[int, list[str]]:
    """{frame_index: [readable line per joint]} for every frame."""
    return {i: describe_frame(f) for i, f in frames.items()}


def frames_to_angle_table(frames: dict[int, dict]) -> pd.DataFrame:
    """
    One row per frame: our flexion/abduction angles plus FreeMoCap's own Euler
    rotation (degrees) for the matching bone, ready for plotting/comparison.
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
            fm = a["freemocap_euler_deg"] if a else (np.nan,) * 3
            for axis, v in zip("xyz", fm):
                row[f"{joint}_freemocap_euler_{axis}_deg"] = v
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
        print(f"  {bone:<12} x[{lo[0]:8.1f}, {hi[0]:8.1f}]  "
              f"y[{lo[1]:8.1f}, {hi[1]:8.1f}]  z[{lo[2]:8.1f}, {hi[2]:8.1f}]")
        if np.nanmax(np.abs(vals)) > 1e-3:
            all_zero = False
    if all_zero:
        print("  NOTE: every Euler value is ~0. If the rig is driven by constraints, the raw\n"
              "  rotation_euler properties may not reflect the motion; the head/tail-based\n"
              "  angles above are then the reliable ones.")


def save_to_analysis_output_dir(data: pd.DataFrame, filename: str) -> None:
    """Save the given DataFrame to the analysis output directory with the specified filename."""
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=BONE_AND_JOINT_CSV_FILE, help="bone_and_joint CSV")
    ap.add_argument("--euler_unit", choices=["radians", "degrees"], default="radians")
    ap.add_argument("--print_frames", type=int, default=3, help="how many frames to print")
    ap.add_argument("--save", action="store_true", help="save per-frame angle table to analysis_output/")
    args = ap.parse_args()

    frames = load_relevant_frames(args.csv, euler_unit=args.euler_unit)
    print(f"Loaded {len(frames)} frames, {len(next(iter(frames.values())))} bones per frame.\n")
    check_euler_columns(frames)
    print()
    for i in list(frames)[:args.print_frames]:
        print(f"Frame {i}:")
        for line in describe_frame(frames[i]):
            print(f"  {line}")
    if args.save:
        save_to_analysis_output_dir(frames_to_angle_table(frames), f"{RECORDING_NUMBER}_joint_angles.csv")


if __name__ == "__main__":
    main()