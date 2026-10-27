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
├── README.md                # this file
└── joint-analysis/
    ├── README.md            # how to run the joint analysis
    └── joint_analysis.py
```