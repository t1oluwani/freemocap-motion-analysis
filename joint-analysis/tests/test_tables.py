"""Tests for process/tables.py (per-frame tables)."""

import math

import numpy as np
import pandas as pd
import pytest

from process import tables
from utils import csv_parser
from synthetic_data import STANDARD_POSES, make_bones, make_frame

HINGE_JOINTS = ["right_elbow", "right_knee", "left_elbow", "left_knee"]
BALL_JOINTS = ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]
ALL_JOINTS = [
    "right_shoulder", "right_elbow", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "left_hip", "left_knee",
]  # fmt: skip

COMPUTED_COLUMNS = [
    "frame",
    "right_shoulder_flexion_deg", "right_shoulder_abduction_deg",
    "right_elbow_flexion_deg",
    "right_hip_flexion_deg", "right_hip_abduction_deg",
    "right_knee_flexion_deg",
    "left_shoulder_flexion_deg", "left_shoulder_abduction_deg",
    "left_elbow_flexion_deg",
    "left_hip_flexion_deg", "left_hip_abduction_deg",
    "left_knee_flexion_deg",
]  # fmt: skip


class TestTables:
    @pytest.fixture
    def frames(self, standard_csv):
        return csv_parser.load_relevant_frames(standard_csv)

    def test_computed_table_columns_and_rows(self, frames):
        table = tables.frames_to_computed_angle_table(frames)
        assert list(table.columns) == COMPUTED_COLUMNS
        assert len(table) == 3
        assert list(table["frame"]) == [0, 1, 2]

    def test_computed_table_values_match_the_known_poses(self, frames):
        t = tables.frames_to_computed_angle_table(frames).set_index("frame")
        assert t.loc[0].abs().max() == pytest.approx(0.0, abs=1e-6)
        assert t.loc[1, "right_elbow_flexion_deg"] == pytest.approx(50.0)
        assert t.loc[1, "left_knee_flexion_deg"] == pytest.approx(85.0)
        assert t.loc[1, "right_hip_flexion_deg"] == pytest.approx(70.0)
        assert t.loc[1, "left_shoulder_abduction_deg"] == pytest.approx(15.0)
        assert t.loc[2, "right_hip_flexion_deg"] == pytest.approx(-20.0)
        assert t.loc[2, "right_shoulder_abduction_deg"] == pytest.approx(80.0)
        assert t.loc[2, "left_elbow_flexion_deg"] == pytest.approx(120.0)

    def test_hinge_joints_have_no_abduction_column(self, frames):
        cols = tables.frames_to_computed_angle_table(frames).columns
        assert not any("elbow_abduction" in c or "knee_abduction" in c for c in cols)

    def test_freemocap_table_columns(self, frames):
        table = tables.frames_to_freemocap_angle_table(frames)
        expected = ["frame"]
        for joint in ALL_JOINTS:
            expected += [f"{joint}_freemocap_euler_{a}_deg" for a in "xyz"]
            expected.append(f"{joint}_freemocap_euler_order")
        assert list(table.columns) == expected
        assert len(table.columns) == 33

    def test_freemocap_table_values_are_degrees_with_order(self, frames):
        t = tables.frames_to_freemocap_angle_table(frames).set_index("frame")
        assert t.loc[1, "right_elbow_freemocap_euler_x_deg"] == pytest.approx(50.0)
        assert t.loc[0, "right_elbow_freemocap_euler_x_deg"] == pytest.approx(0.0)
        assert (t["right_elbow_freemocap_euler_order"] == "XYZ").all()

    def test_freemocap_table_does_not_contain_computed_columns(self, frames):
        cols = tables.frames_to_freemocap_angle_table(frames).columns
        assert not any("flexion" in c or "abduction" in c for c in cols)

    def test_combined_table_is_computed_plus_freemocap_with_one_frame_column(self, frames):
        combined = tables.frames_to_angle_table(frames)
        assert len(combined.columns) == 13 + 32 == 45
        assert list(combined.columns).count("frame") == 1
        assert list(combined.columns)[0] == "frame"
        assert not combined.columns.duplicated().any()
        pd.testing.assert_frame_equal(
            combined[COMPUTED_COLUMNS], tables.frames_to_computed_angle_table(frames)
        )
        fm = tables.frames_to_freemocap_angle_table(frames).drop(columns="frame")
        assert list(combined.columns)[13:] == list(fm.columns)  # order preserved
        pd.testing.assert_frame_equal(combined[list(fm.columns)], fm)

    def test_missing_joint_becomes_nan_and_blank_order(self, frames):
        del frames[1]["forearm.R"]
        computed = tables.frames_to_computed_angle_table(frames)
        freemocap = tables.frames_to_freemocap_angle_table(frames)
        assert math.isnan(computed.loc[1, "right_elbow_flexion_deg"])
        assert math.isnan(freemocap.loc[1, "right_elbow_freemocap_euler_x_deg"])
        assert freemocap.loc[1, "right_elbow_freemocap_euler_order"] == ""
        assert computed.loc[0, "right_elbow_flexion_deg"] == pytest.approx(0.0, abs=1e-6)  # other frames fine

    def test_frame_numbers_come_from_the_dict_keys(self, frames):
        sparse = {5: frames[0], 9: frames[1]}
        assert list(tables.frames_to_computed_angle_table(sparse)["frame"]) == [5, 9]

    def test_empty_input_gives_empty_tables(self):
        assert len(tables.frames_to_computed_angle_table({})) == 0
        assert len(tables.frames_to_freemocap_angle_table({})) == 0
