"""Tests for process/describe.py (readable text)."""

import math

import numpy as np
import pandas as pd
import pytest

from utils import csv_parser
from process import describe
from process import schema
from synthetic_data import STANDARD_POSES, make_bones, make_frame

HINGE_JOINTS = ["right_elbow", "right_knee", "left_elbow", "left_knee"]
BALL_JOINTS = ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]
ALL_JOINTS = [
    "right_shoulder", "right_elbow", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "left_hip", "left_knee",
]  # fmt: skip

class TestDescribeJoint:
    @pytest.mark.parametrize(
        "joint, angles, expected",
        [
            # hinge joints
            ("right_elbow", {"flexion_deg": 50.0}, "right elbow bent 50 degrees"),
            ("left_knee", {"flexion_deg": 85.0}, "left knee bent 85 degrees"),
            ("right_elbow", {"flexion_deg": 0.0}, "right elbow straight"),
            ("right_elbow", {"flexion_deg": 4.99}, "right elbow straight"),
            ("right_elbow", {"flexion_deg": 5.0}, "right elbow bent 5 degrees"),
            ("right_elbow", {"flexion_deg": 49.6}, "right elbow bent 50 degrees"),
            ("left_elbow", {"flexion_deg": 179.0}, "left elbow bent 179 degrees"),
            # hips
            ("right_hip", {"flexion_deg": 70.0, "abduction_deg": 0.0}, "right hip flexed 70 degrees"),
            ("right_hip", {"flexion_deg": -20.0, "abduction_deg": 0.0}, "right hip extended 20 degrees"),
            ("left_hip", {"flexion_deg": 0.0, "abduction_deg": 25.0}, "left hip abducted 25 degrees"),
            ("left_hip", {"flexion_deg": 0.0, "abduction_deg": -25.0}, "left hip adducted 25 degrees"),
            ("left_hip", {"flexion_deg": 0.0, "abduction_deg": 0.0}, "left hip neutral"),
            ("left_hip", {"flexion_deg": 4.99, "abduction_deg": -4.99}, "left hip neutral"),
            ("left_hip", {"flexion_deg": 5.0, "abduction_deg": 0.0}, "left hip flexed 5 degrees"),
            ("left_hip", {"flexion_deg": -5.0, "abduction_deg": 0.0}, "left hip extended 5 degrees"),
            (
                "right_hip",
                {"flexion_deg": 70.0, "abduction_deg": 10.0},
                "right hip flexed 70 degrees, abducted 10 degrees",
            ),
            # shoulders
            ("right_shoulder", {"flexion_deg": 40.0, "abduction_deg": 0.0}, "right shoulder raised forward 40 degrees"),
            ("right_shoulder", {"flexion_deg": -40.0, "abduction_deg": 0.0}, "right shoulder raised backward 40 degrees"),
            ("right_shoulder", {"flexion_deg": 0.0, "abduction_deg": 80.0}, "right shoulder raised out to the side 80 degrees"),
            ("left_shoulder", {"flexion_deg": 0.0, "abduction_deg": -10.0}, "left shoulder pulled across the body 10 degrees"),
            ("right_shoulder", {"flexion_deg": 0.0, "abduction_deg": 0.0}, "right shoulder neutral"),
            (
                "left_shoulder",
                {"flexion_deg": 40.0, "abduction_deg": 15.0},
                "left shoulder raised forward 40 degrees, raised out to the side 15 degrees",
            ),
        ],
    )  # fmt: skip
    def test_wording(self, joint, angles, expected):
        assert describe.describe_joint(joint, angles) == expected

    def test_none_means_no_data(self):
        assert describe.describe_joint("right_elbow", None) == "right elbow: no data"

    @pytest.mark.parametrize(
        "joint, angles",
        [
            ("right_elbow", {"flexion_deg": math.nan}),
            ("right_hip", {"flexion_deg": math.nan, "abduction_deg": 10.0}),
            ("right_hip", {"flexion_deg": 10.0, "abduction_deg": math.nan}),
            ("left_shoulder", {"flexion_deg": math.nan, "abduction_deg": math.nan}),
        ],
    )
    def test_nan_means_no_data(self, joint, angles):
        assert describe.describe_joint(joint, angles).endswith(": no data")

    def test_neutral_threshold_constant_is_what_the_tests_assume(self):
        assert schema.NEUTRAL_THRESHOLD_DEG == 5.0

class TestDescribeJointEuler:
    def test_none_means_no_data(self):
        assert describe.describe_joint_euler("right_elbow", None) == "right elbow: no data"

    def test_nan_means_no_data(self):
        a = {"freemocap_euler_deg": (math.nan, 0.0, 0.0), "freemocap_euler_order": "XYZ"}
        assert describe.describe_joint_euler("right_elbow", a) == "right elbow: no data"

    def test_missing_euler_key_means_no_data(self):
        assert describe.describe_joint_euler("right_elbow", {"flexion_deg": 1.0}) == "right elbow: no data"

    def test_small_rotation_is_reported_as_about_zero(self):
        a = {"freemocap_euler_deg": (4.9, -4.9, 0.0), "freemocap_euler_order": "XYZ"}
        assert describe.describe_joint_euler("left_knee", a) == "left knee: rotation ~0 degrees"

    def test_larger_rotation_is_reported_with_axes_and_order(self):
        a = {"freemocap_euler_deg": (50.0, 0.0, 0.0), "freemocap_euler_order": "XYZ"}
        assert describe.describe_joint_euler("right_elbow", a) == "right elbow: rotation XYZ x=50 y=0 z=0 degrees"

    def test_order_and_negative_values_are_shown(self):
        a = {"freemocap_euler_deg": (-12.0, 30.0, -90.0), "freemocap_euler_order": "ZYX"}
        assert describe.describe_joint_euler("left_hip", a) == "left hip: rotation ZYX x=-12 y=30 z=-90 degrees"

    def test_threshold_boundary_uses_the_largest_axis(self):
        near = {"freemocap_euler_deg": (0.0, 0.0, 4.99), "freemocap_euler_order": "XYZ"}
        edge = {"freemocap_euler_deg": (0.0, 0.0, 5.0), "freemocap_euler_order": "XYZ"}
        assert "~0" in describe.describe_joint_euler("left_hip", near)
        assert "~0" not in describe.describe_joint_euler("left_hip", edge)

class TestFrameDescriptions:
    def test_describe_frames_returns_two_lists_in_joint_order(self):
        computed, freemocap = describe.describe_frames(make_frame())
        assert len(computed) == len(freemocap) == 8
        for joint, c, f in zip(schema.JOINT_ORDER, computed, freemocap):
            label = joint.replace("_", " ")
            assert c.startswith(label)
            assert f.startswith(label)

    def test_known_pose_produces_the_expected_sentences(self):
        computed, _ = describe.describe_frames(make_frame(STANDARD_POSES[1]))
        assert "right elbow bent 50 degrees" in computed
        assert "left knee bent 85 degrees" in computed
        assert "right hip flexed 70 degrees" in computed
        assert "left shoulder raised forward 40 degrees, raised out to the side 15 degrees" in computed

    def test_standing_pose_reads_straight_or_neutral_everywhere(self):
        computed, freemocap = describe.describe_frames(make_frame())
        assert all(("straight" in c or "neutral" in c) for c in computed)
        assert all("~0" in f for f in freemocap)

    def test_missing_data_is_reported_per_joint_without_stopping_the_rest(self):
        frame = make_frame({"left_knee": 60.0})
        del frame["forearm.R"]
        computed, _ = describe.describe_frames(frame)
        assert "right elbow: no data" in computed
        assert "left knee bent 60 degrees" in computed

    def test_interleave_alternates_computed_and_freemocap_lines(self):
        out = describe.interleave(["a1", "a2"], ["b1", "b2"])
        assert out == ["  Computed:   a1", "  FreeMoCap:  b1", "  Computed:   a2", "  FreeMoCap:  b2"]

    def test_interleave_of_empty_lists_is_empty(self):
        assert describe.interleave([], []) == []

class TestCheckEulerColumns:
    def test_all_zero_rotations_trigger_the_note(self, capsys):
        describe.check_euler_columns({0: make_frame(), 1: make_frame()})
        out = capsys.readouterr().out
        assert "NOTE: every Euler value is ~0" in out

    def test_real_rotations_print_the_range_and_no_note(self, standard_csv, capsys):
        describe.check_euler_columns(csv_parser.load_relevant_frames(standard_csv))
        out = capsys.readouterr().out
        assert "NOTE" not in out
        forearm_r = next(line for line in out.splitlines() if "forearm.R" in line)
        assert "50.0" in forearm_r
        assert out.count("\n") >= 9  # a header plus one line per relevant bone

    def test_tiny_but_nonzero_rotation_still_counts_as_real(self, capsys):
        describe.check_euler_columns({0: make_frame(euler_deg={"shin.L": (0.01, 0.0, 0.0)})})
        assert "NOTE" not in capsys.readouterr().out

    def test_all_nan_or_missing_bones_do_not_crash(self, capsys):
        frame = make_frame(euler_deg={"shin.L": (math.nan, math.nan, math.nan)})
        del frame["thigh.R"]
        describe.check_euler_columns({0: frame})
        capsys.readouterr()  # just must not raise
