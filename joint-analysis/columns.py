"""Functions for reading CSVs and extracting the relevant columns and frames."""

import math
import numpy as np
import pandas as pd

from schema import (
    HEADER_SUFFIXES,
    RELEVANT_PREFIXES,
    REFERENCE_PREFIXES,
    PROPERTY_COLUMNS,
    EULER_ORDER_SUFFIX,
)

def build_relevant_columns_from_csv(
    csv_file_path: str, include_matrix: bool = False
) -> list[str]:
    """
    Read just the header row of the CSV and return the names of the columns we
    need: every property (minus the 16 matrix values unless include_matrix)
    of the relevant bones plus the trunk reference bones.

    Matching is exact (<prefix> + <known suffix>)
    """
    wanted_suffixes = {
        s for s in HEADER_SUFFIXES if include_matrix or "_matrix_" not in s
    }
    headers = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
    prefixes = RELEVANT_PREFIXES + REFERENCE_PREFIXES

    relevant_columns = []
    for header in headers:
        for prefix in prefixes:
            if (
                header.startswith(prefix + "_")
                and header[len(prefix) :] in wanted_suffixes
            ):
                relevant_columns.append(header)
                break
    return relevant_columns


def load_relevant_frames(
    csv_file_path: str, euler_unit: str = "radians"
) -> dict[int, dict]:
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

    bones = [
        p
        for p in RELEVANT_PREFIXES + REFERENCE_PREFIXES
        if all(p + s in df.columns for s in PROPERTY_COLUMNS["head"])
    ]
    missing = [p for p in RELEVANT_PREFIXES + REFERENCE_PREFIXES if p not in bones]
    if missing:
        print(f"Warning: bones not found in this CSV: {missing}")

    arrays = {}
    for bone in bones:
        arrays[bone] = {
            key: df[[bone + s for s in suffixes]].to_numpy(dtype=float)
            for key, suffixes in PROPERTY_COLUMNS.items()
        }
        order_col = bone + EULER_ORDER_SUFFIX
        arrays[bone]["euler_order"] = (
            df[order_col].astype(str).to_numpy()
            if order_col in df.columns
            else np.full(len(df), "")
        )

    if euler_unit == "radians":
        biggest = max(
            (
                np.nanmax(np.abs(a["euler"]))
                for a in arrays.values()
                if a["euler"].size and not np.isnan(a["euler"]).all()
            ),
            default=0.0,
        )
        if biggest > 2 * math.pi + 0.5:
            print(
                f"Warning: euler values reach {biggest:.1f}, too large for radians - "
                f"the CSV may already be in degrees (use --euler_unit degrees)."
            )
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
