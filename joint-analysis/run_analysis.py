"""
run_analysis.py

Entry point (the "main"): loads a FreeMoCap bone_and_joint CSV through
bone_and_joint.py, prints the readable joint descriptions for the first few
frames, and optionally saves per-frame angle tables and the printed text to
analysis_output/.

bone_and_joint.py must sit in the same folder as this file.

Usage:
    python run_analysis.py --csv path/to/..._bone_and_joint_data.csv --save
    python run_analysis.py --combine --save      # one combined CSV/text instead of two
"""

import argparse
import io

from contextlib import redirect_stdout

from bone_and_joint import (
    check_euler_columns,
    describe_frames,
    frames_to_angle_table,
    frames_to_computed_angle_table,
    frames_to_freemocap_angle_table,
    interleave,
    load_relevant_frames,
)
from utils.output_writer import save_csv_to_output_dir, save_print_to_output_dir
from configs.path_configs import BONE_AND_JOINT_CSV_FILE, RECORDING_NUMBER

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=BONE_AND_JOINT_CSV_FILE, help="bone_and_joint CSV")
    ap.add_argument("--euler_unit", choices=["radians", "degrees"], default="radians")
    ap.add_argument(
        "--print_frames", type=int, default=3, help="how many frames to print"
    )
    ap.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="save per-frame angle table to analysis_output/",
    )
    ap.add_argument(
        "--combine",
        action="store_true",
        default=False,
        help="combine freemocap angle and computed angles as one file",
    )
    args = ap.parse_args()

    frames = load_relevant_frames(args.csv, euler_unit=args.euler_unit)
    full_lines: list[str] = []
    computed_lines: list[str] = []
    freemocap_lines: list[str] = []

    header = (
        f"Loaded {len(frames)} frames, "
        f"{len(next(iter(frames.values())))} bones per frame.\n"
    )
    full_lines.append(header)

    buf = io.StringIO()
    with redirect_stdout(buf):
        check_euler_columns(frames)
    full_lines.append(buf.getvalue())

    for i in list(frames)[: args.print_frames]:
        computed, freemocap = describe_frames(frames[i])

        if args.combine:
            full_lines.append(f"Frame {i} (computed vs FreeMoCap, interleaved):")
            full_lines.extend(interleave(computed, freemocap))
        else:
            full_lines.append(f"Frame {i} - FreeMoCap:")
            full_lines.extend(f"  {l}" for l in freemocap)
            full_lines.append(f"Frame {i} - computed:")
            full_lines.extend(f"  {l}" for l in computed)

            computed_lines.append(f"Frame {i} - computed:")
            computed_lines.extend(f"  {l}" for l in computed)
            freemocap_lines.append(f"Frame {i} - FreeMoCap:")
            freemocap_lines.extend(f"  {l}" for l in freemocap)

        full_lines.append("")

    text_full = "\n".join(full_lines)
    print(text_full)

    if args.save:
        if args.combine:
            save_csv_to_output_dir(
                frames_to_angle_table(frames),
                f"{RECORDING_NUMBER}_joint_angles_combined.csv",
            )
            save_print_to_output_dir(
                text_full,
                f"{RECORDING_NUMBER}_joint_analysis_combined.txt",
            )
        else:
            save_csv_to_output_dir(
                frames_to_computed_angle_table(frames),
                f"{RECORDING_NUMBER}_computed_joint_angles.csv",
            )
            save_print_to_output_dir(
                "\n".join(computed_lines),
                f"{RECORDING_NUMBER}_computed_joint_analysis.txt",
            )
            save_csv_to_output_dir(
                frames_to_freemocap_angle_table(frames),
                f"{RECORDING_NUMBER}_freemocap_joint_angles.csv",
            )
            save_print_to_output_dir(
                "\n".join(freemocap_lines),
                f"{RECORDING_NUMBER}_freemocap_joint_analysis.txt",
            )


if __name__ == "__main__":
    main()
