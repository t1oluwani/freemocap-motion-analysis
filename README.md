# freemocap-motion-analysis
 
Relevant Code for **SFU Project 15004: Experimental Study with Virtual Humanoid/Robot Model Development**.
 
The project's goals are to:
 
- Develop simple virtual models, such as joint-link systems for humanoid and robot applications.
- Explore open-source software design and development environments through applied tutorials.
- Design and evaluate user interaction methods with the developed models.
This repository holds the motion-capture analysis tooling built around [FreeMoCap](https://github.com/freemocap/freemocap) recordings and their Blender exports.
 
---
 
## Contents
 
### Joint analysis
 
Analyzes FreeMoCap's Blender-exported `*_bone_and_joint_data.csv` and reports joint angles two ways per joint: geometric angles computed from bone head/tail positions, and FreeMoCap's own `rotation_euler` values. It prints readable descriptions (e.g. `right elbow bent 50 degrees`) and can save per-frame CSVs and text reports so the two sets of angles can be compared directly.
 
**Setup and run instructions:** [joint-analysis/README.md](joint-analysis/README.md)
 
---
 
## Repository layout
 
```text
freemocap-motion-analysis/
├── README.md                          # this file
└── joint-analysis/
    ├── README.md                      # how to run the joint analysis
    ├── run_analysis.py                # entry point: argparse + orchestration
    ├── configs/
    │   ├── __init__.py
    │   └── path_configs.py            # recording number, input/output paths
    ├── process/
    │   ├── __init__.py
    │   ├── schema.py                  # CSV format + joint definitions
    │   ├── geometry.py                # vector math + compute_joint_angles
    │   ├── describe.py                # readable joint text
    │   └── tables.py                  # per-frame pandas tables
    ├── utils/
    │   ├── __init__.py
    │   ├── csv_parser.py              # CSV -> frame dict
    │   └── output_writer.py           # save CSV/text to output dirs
    ├── input/
    │   └── freemocap/
    │       └── video1/
    │           └── saved_data/
    │               └── recording_XX_XX_XX_gmt-X_bone_and_joint_data.csv
    └── output/
        ├── csv/                       # per-frame angle tables (*.csv)
        └── print/                     # readable joint reports (*.txt)
```