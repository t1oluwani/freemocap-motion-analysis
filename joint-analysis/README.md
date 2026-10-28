# run_analysis.py

Analyzes FreeMoCap's Blender-exported `*_bone_and_joint_data.csv` and reports joint angles **two ways per joint**:

- **Computed (ours):** geometric angles from bone head/tail world positions.
  - Elbow / knee → **flexion** (0° = straight).
  - Shoulder / hip → **flexion** + **abduction** relative to a trunk frame built from `spine`, `neck`, and the two thigh heads.
- **FreeMoCap:** the matching bone's raw `rotation_euler` (converted to degrees).

Also produces readable descriptions (e.g. `right elbow bent 50 degrees`) and can save CSVs and text reports.

---

## Files

| File | Role |
|------|------|
| `run_analysis.py` | Entry point. Command-line flags, run settings (`RECORDING_NUMBER`, default CSV path), console printing, and saving to `analysis_output/`. |
| `bone_and_joint.py` | Library. Column selection, frame loading, angle computation, readable descriptions, and the angle tables. No command line and no file output. |

Keep both files in the same folder. You only ever run `run_analysis.py`; it imports `bone_and_joint.py`.

## Requirements

```bash
pip install -r requirements.txt
```

## Run

Default CSV is `freemocap/video1/saved_data/recording_17_33_15_gmt-7_bone_and_joint_data.csv` (relative to your working directory).

```bash
python run_analysis.py --save
```

Run from the folder that contains `freemocap/` (with `run_analysis.py` and `bone_and_joint.py` alongside), or point at a file explicitly with `--csv`.

## Command-line options

| Flag | Default | What it does |
|------|---------|--------------|
| `--csv PATH` | `freemocap/video1/saved_data/recording_17_33_15_gmt-7_bone_and_joint_data.csv` | Path to the `*_bone_and_joint_data.csv` file. |
| `--euler_unit {radians,degrees}` | `radians` | Unit of the CSV's `rotation_euler` columns. Blender writes radians; the script warns if values look like degrees. |
| `--print_frames N` | `3` | How many leading frames to describe on the console. |
| `--save` | off | Write output files to `analysis_output/` (see below). |
| `--combine` | off | Only meaningful with `--save`. Produce a single combined CSV + TXT instead of separate Computed/FreeMoCap files. |

### Examples

```bash
# Print 3 frames using the default CSV
python run_analysis.py

# Print 20 frames, save split Computed + FreeMoCap outputs
python run_analysis.py --print_frames 20 --save

# Save one combined CSV + TXT with interleaved per-joint lines
python run_analysis.py --save --combine

# Analyze a different recording
python run_analysis.py --csv path/to/other_recording_bone_and_joint_data.csv --save

# CSV already in degrees (not radians)
python run_analysis.py --euler_unit degrees
```

## Console output

Each printed frame shows both views.

**Without `--combine`** — FreeMoCap block, then Computed block:

```text
Frame 1 - FreeMoCap:
  right elbow: rotation XYZ x=50 y=0 z=0 degrees
  left knee: rotation ~0 degrees
  ...
Frame 1 - Computed:
  right elbow bent 50 degrees
  left knee straight
  ...
```

**With `--combine`** — interleaved, one joint at a time:

```text
Frame 1 (Computed vs FreeMoCap, interleaved):
  Computed:   right elbow bent 50 degrees
  FreeMoCap:  right elbow: rotation XYZ x=50 y=0 z=0 degrees
  Computed:   left knee straight
  FreeMoCap:  left knee: rotation ~0 degrees
  ...
```

A FreeMoCap `rotation_euler` range per bone sanity block is printed once at the top; if every Euler value is ~0, a note explains that constraints may be driving the rig and the head/tail-based angles are then the reliable ones.

## Saved outputs (`--save`)

All files land under `analysis_output/`.

### Without `--combine` (split files)

| File | Contents |
|------|----------|
| `analysis_output/csv/<recording>_computed_joint_angles.csv` | One row per frame; our geometric angles only: `<joint>_flexion_deg`, and `<joint>_abduction_deg` for shoulders/hips. |
| `analysis_output/csv/<recording>_freemocap_joint_angles.csv` | One row per frame; FreeMoCap's Euler values only: `<joint>_freemocap_euler_x/y/z_deg` and `<joint>_freemocap_euler_order`. |
| `analysis_output/print/<recording>_computed_joint_analysis.txt` | Human-readable Computed descriptions per printed frame. |
| `analysis_output/print/<recording>_freemocap_joint_analysis.txt` | Human-readable FreeMoCap descriptions per printed frame. |

### With `--combine` (single file)

| File | Contents |
|------|----------|
| `analysis_output/csv/<recording>_joint_angles_combined.csv` | Both Computed and FreeMoCap angle columns side by side, one row per frame. |
| `analysis_output/print/<recording>_joint_analysis_combined.txt` | Console text exactly as printed (header, Euler sanity block, interleaved per-joint lines). |

CSV column names encode the joint and quantity, e.g. `right_elbow_flexion_deg`, `left_hip_abduction_deg`, `right_shoulder_freemocap_euler_x_deg` — so you can plot or diff our angle against FreeMoCap's directly.

## Notes and caveats

- **Angle conventions.** Elbow/knee flexion is the angle between the proximal and distal bone directions (upper_arm→forearm, thigh→shin); 0° = straight. Shoulder/hip flexion/abduction are projection angles onto the trunk's sagittal and frontal planes; they degrade once a limb is near or above horizontal.
- **Missing data.** If a needed bone (or the trunk reference `spine`/`neck`) is missing in a frame, affected joints report *no data* / `NaN`; hinges that don't need the trunk still work.
- **FreeMoCap Euler is not a joint angle.** It's the matching pose bone's local rotation. The script reports it honestly as a rotation tuple rather than relabeling it as flexion/abduction.
- **Threshold for "neutral".** Angles below `NEUTRAL_THRESHOLD_DEG = 5.0` (defined in `bone_and_joint.py`) are described as *straight* / *neutral* / *rotation ~0 degrees*.
- **Printed vs. saved frames.** Only the first `--print_frames` frames are described on the console and in the `.txt` reports. The saved CSVs, by contrast, include every frame in the file, regardless of `--print_frames`.

## Output directory layout

```text
analysis_output/
├── csv/
│   ├── <recording>_computed_joint_angles.csv       # --save, no --combine
│   ├── <recording>_freemocap_joint_angles.csv      # --save, no --combine
│   └── <recording>_joint_angles_combined.csv       # --save --combine
└── print/
    ├── <recording>_computed_joint_analysis.txt     # --save, no --combine
    ├── <recording>_freemocap_joint_analysis.txt    # --save, no --combine
    └── <recording>_joint_analysis_combined.txt     # --save --combine
```

`<recording>` is the `RECORDING_NUMBER` constant at the top of `run_analysis.py` (`recording_17_33_15_gmt-7` by default).