"""End-to-end: CSV -> frames -> angles -> sentences (crosses modules)."""

import math

import numpy as np
import pandas as pd
import pytest

from utils import csv_parser
from process import describe
from process import tables
from synthetic_data import STANDARD_POSES, make_bones, make_frame

HINGE_JOINTS = ["right_elbow", "right_knee", "left_elbow", "left_knee"]
BALL_JOINTS = ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]
ALL_JOINTS = [
    "right_shoulder", "right_elbow", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "left_hip", "left_knee",
]  # fmt: skip

class TestEndToEnd:
    def test_csv_to_readable_sentences(self, standard_csv):
        frames = csv_parser.load_relevant_frames(standard_csv)
        computed0, _ = describe.describe_frames(frames[0])
        computed1, freemocap1 = describe.describe_frames(frames[1])
        computed2, _ = describe.describe_frames(frames[2])

        assert all(("straight" in c or "neutral" in c) for c in computed0)
        assert "right elbow bent 50 degrees" in computed1
        assert "left knee bent 85 degrees" in computed1
        assert "right hip flexed 70 degrees" in computed1
        assert "right elbow: rotation XYZ x=50 y=0 z=0 degrees" in freemocap1
        assert "right hip extended 20 degrees" in computed2
        assert "left hip abducted 25 degrees" in computed2
        assert "right shoulder raised out to the side 80 degrees" in computed2
        assert "left elbow bent 120 degrees" in computed2

    def test_a_whole_recording_with_a_moving_body(self, csv_factory):
        # 30 frames: elbow opens from 0 to 145 degrees while the body walks and turns
        angles = np.linspace(0, 145, 30)
        skeletons = [
            _moved(make_bones({"right_elbow": float(a)}), yaw=3.0 * i, shift=(0.05 * i, 0.02 * i, 0))
            for i, a in enumerate(angles)
        ]
        frames = csv_parser.load_relevant_frames(csv_factory(skeletons=skeletons, euler={}))
        table = tables.frames_to_computed_angle_table(frames)
        assert len(table) == 30
        assert np.allclose(table["right_elbow_flexion_deg"], angles, atol=1e-6)
        assert table["right_knee_flexion_deg"].abs().max() == pytest.approx(0.0, abs=1e-6)


def _moved(bones, yaw, shift):
    from synthetic_data import transform_bones

    return transform_bones(bones, yaw_deg=yaw, shift=shift)
