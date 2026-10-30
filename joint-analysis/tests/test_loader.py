"""Tests for process/csv_parser.py (column selection, CSV -> frames)."""

import math

import numpy as np
import pandas as pd
import pytest

from utils import csv_parser
from process import schema
from synthetic_data import STANDARD_POSES, make_bones, make_frame

HINGE_JOINTS = ["right_elbow", "right_knee", "left_elbow", "left_knee"]
BALL_JOINTS = ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]
ALL_JOINTS = [
    "right_shoulder", "right_elbow", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "left_hip", "left_knee",
]  # fmt: skip

class TestColumnSelection:
    def test_default_selects_10_bones_x_14_columns(self, standard_csv):
        cols = csv_parser.build_relevant_columns_from_csv(standard_csv)
        assert len(cols) == 140  # 8 limb bones + spine + neck, matrix excluded
        assert not any("_matrix_" in c for c in cols)

    def test_include_matrix_gives_30_columns_per_bone(self, standard_csv):
        cols = csv_parser.build_relevant_columns_from_csv(standard_csv, include_matrix=True)
        assert len(cols) == 300
        assert sum("_matrix_" in c for c in cols) == 160

    def test_every_selected_bone_has_all_14_properties(self, standard_csv):
        cols = set(csv_parser.build_relevant_columns_from_csv(standard_csv))
        no_matrix = [s for s in schema.HEADER_SUFFIXES if "_matrix_" not in s]
        for bone in schema.RELEVANT_PREFIXES + schema.REFERENCE_PREFIXES:
            assert {bone + s for s in no_matrix} <= cols

    def test_hands_fingers_feet_and_other_bones_are_excluded(self, standard_csv):
        cols = csv_parser.build_relevant_columns_from_csv(standard_csv)
        for unwanted in ("hand", "f_index", "thumb", "palm", "foot", "heel", "pelvis", "shoulder"):
            assert not any(c.startswith(unwanted) for c in cols), unwanted

    def test_spine_does_not_pull_in_spine_001(self, standard_csv):
        cols = csv_parser.build_relevant_columns_from_csv(standard_csv)
        assert not any(c.startswith("spine.001") for c in cols)
        assert any(c.startswith("spine_") for c in cols)

    def test_selected_columns_keep_the_csv_order(self, standard_csv):
        header = list(pd.read_csv(standard_csv, nrows=0).columns)
        cols = csv_parser.build_relevant_columns_from_csv(standard_csv)
        positions = [header.index(c) for c in cols]
        assert positions == sorted(positions)

    def test_unknown_suffix_and_unrelated_columns_are_ignored(self, tmp_path):
        path = tmp_path / "tiny.csv"
        pd.DataFrame(
            {
                "upper_arm.L_head_center_world_x": [1.0],
                "upper_arm.L_bogus": [2.0],
                "unrelated": [3.0],
                "upper_arm.L.001_head_center_world_x": [4.0],
            }
        ).to_csv(path, index=False)
        assert csv_parser.build_relevant_columns_from_csv(str(path)) == ["upper_arm.L_head_center_world_x"]

    def test_missing_bone_columns_are_tolerated(self, csv_factory):
        path = csv_factory(drop_bones=("thigh.L",))
        assert len(csv_parser.build_relevant_columns_from_csv(path)) == 126  # 9 bones x 14

class TestLoadRelevantFrames:
    def test_one_entry_per_row_indexed_from_zero(self, standard_csv):
        frames = csv_parser.load_relevant_frames(standard_csv)
        assert list(frames) == [0, 1, 2]

    def test_each_frame_has_exactly_the_10_bones(self, standard_csv):
        frames = csv_parser.load_relevant_frames(standard_csv)
        expected = set(schema.RELEVANT_PREFIXES + schema.REFERENCE_PREFIXES)
        assert all(set(f) == expected for f in frames.values())

    def test_property_structure_and_tuple_lengths(self, standard_csv):
        bone = csv_parser.load_relevant_frames(standard_csv)[0]["forearm.R"]
        assert set(bone) == {"head", "tail", "quaternion", "euler_deg", "euler_order"}
        assert len(bone["head"]) == len(bone["tail"]) == len(bone["euler_deg"]) == 3
        assert len(bone["quaternion"]) == 4
        assert isinstance(bone["euler_order"], str)

    def test_positions_round_trip_through_the_csv(self, standard_csv):
        frames = csv_parser.load_relevant_frames(standard_csv)
        for i, pose in enumerate(STANDARD_POSES):
            for name, (head, tail) in make_bones(pose).items():
                assert np.allclose(frames[i][name]["head"], head)
                assert np.allclose(frames[i][name]["tail"], tail)

    def test_quaternion_is_read_in_xyzw_order(self, standard_csv):
        assert csv_parser.load_relevant_frames(standard_csv)[0]["thigh.L"]["quaternion"] == (0.0, 0.0, 0.0, 1.0)

    def test_euler_is_converted_from_radians_to_degrees_by_default(self, standard_csv):
        frames = csv_parser.load_relevant_frames(standard_csv)
        assert np.allclose(frames[1]["forearm.R"]["euler_deg"], (50.0, 0.0, 0.0))
        assert np.allclose(frames[0]["forearm.R"]["euler_deg"], (0.0, 0.0, 0.0))

    def test_euler_unit_degrees_is_passed_through_unchanged(self, csv_factory):
        path = csv_factory(euler={(0, "shin.L"): (12.0, -34.0, 56.0)})
        frames = csv_parser.load_relevant_frames(path, euler_unit="degrees")
        assert frames[0]["shin.L"]["euler_deg"] == (12.0, -34.0, 56.0)

    def test_invalid_euler_unit_raises(self, standard_csv):
        with pytest.raises(ValueError, match="euler_unit"):
            csv_parser.load_relevant_frames(standard_csv, euler_unit="gradians")

    def test_euler_order_is_read_from_the_csv(self, csv_factory):
        path = csv_factory(euler_order="ZYX")
        assert csv_parser.load_relevant_frames(path)[0]["thigh.R"]["euler_order"] == "ZYX"

    def test_warns_when_radians_values_look_like_degrees(self, csv_factory, capsys):
        path = csv_factory(euler={(0, "forearm.R"): (90.0, 0.0, 0.0)})  # 90 "radians" is absurd
        csv_parser.load_relevant_frames(path)  # default euler_unit="radians"
        assert "Warning: euler values reach" in capsys.readouterr().out

    def test_no_warning_for_sane_radians(self, standard_csv, capsys):
        csv_parser.load_relevant_frames(standard_csv)
        assert "Warning" not in capsys.readouterr().out

    def test_no_warning_when_unit_is_degrees(self, csv_factory, capsys):
        path = csv_factory(euler={(0, "forearm.R"): (90.0, 0.0, 0.0)})
        csv_parser.load_relevant_frames(path, euler_unit="degrees")
        assert "Warning" not in capsys.readouterr().out

    def test_missing_bone_is_reported_and_left_out(self, csv_factory, capsys):
        frames = csv_parser.load_relevant_frames(csv_factory(drop_bones=("thigh.L",)))
        assert "thigh.L" in capsys.readouterr().out
        assert "thigh.L" not in frames[0]
        assert len(frames[0]) == 9

    def test_nan_values_survive_loading(self, csv_factory):
        bones = make_bones()
        bones["neck"] = (np.full(3, np.nan), bones["neck"][1])
        frames = csv_parser.load_relevant_frames(csv_factory(skeletons=[bones]))
        assert all(math.isnan(v) for v in frames[0]["neck"]["head"])
