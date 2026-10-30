"""Tests for process/schema.py (constants)."""

import math

import numpy as np
import pandas as pd
import pytest

from process import schema
from synthetic_data import STANDARD_POSES, make_bones, make_frame

HINGE_JOINTS = ["right_elbow", "right_knee", "left_elbow", "left_knee"]
BALL_JOINTS = ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]
ALL_JOINTS = [
    "right_shoulder", "right_elbow", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "left_hip", "left_knee",
]  # fmt: skip

class TestConstants:
    def test_header_suffixes_is_a_flat_list_of_30_strings(self):
        # Regression: this used to be [[...]] (a list holding one list).
        assert len(schema.HEADER_SUFFIXES) == 30
        assert all(isinstance(s, str) for s in schema.HEADER_SUFFIXES)

    def test_header_suffixes_unique_and_underscore_prefixed(self):
        assert len(set(schema.HEADER_SUFFIXES)) == 30
        assert all(s.startswith("_") for s in schema.HEADER_SUFFIXES)

    def test_suffix_groups(self):
        s = schema.HEADER_SUFFIXES
        assert sum("_head_center_world_" in x for x in s) == 3
        assert sum("_tail_center_world_" in x for x in s) == 3
        assert sum("_rotation_quaternion_" in x for x in s) == 4
        assert sum("_rotation_euler_" in x for x in s) == 4
        assert sum("_matrix_" in x for x in s) == 16

    def test_61_unique_bones_giving_1830_columns(self):
        assert len(schema.HEADER_PREFIXES) == 61
        assert len(set(schema.HEADER_PREFIXES)) == 61
        assert len(schema.HEADER_PREFIXES) * len(schema.HEADER_SUFFIXES) == 1830

    def test_relevant_and_reference_bones_exist_in_the_skeleton(self):
        assert set(schema.RELEVANT_PREFIXES) <= set(schema.HEADER_PREFIXES)
        assert set(schema.REFERENCE_PREFIXES) <= set(schema.HEADER_PREFIXES)
        assert len(schema.RELEVANT_PREFIXES) == 8

    def test_joint_tables_agree_with_each_other(self):
        assert set(schema.JOINT_TO_BONE) == set(schema.JOINT_SPECS) == set(schema.JOINT_ORDER)
        assert sorted(schema.JOINT_ORDER) == sorted(ALL_JOINTS)
        # each joint maps to its own distinct, relevant bone
        assert sorted(schema.JOINT_TO_BONE.values()) == sorted(schema.RELEVANT_PREFIXES)

    def test_sides_are_consistent(self):
        for joint, bone in schema.JOINT_TO_BONE.items():
            assert bone.endswith(".R") == joint.startswith("right_")
            assert bone.endswith(".L") == joint.startswith("left_")

    def test_joint_specs_use_the_bone_that_joint_to_bone_names(self):
        for joint, spec in schema.JOINT_SPECS.items():
            if spec["kind"] == "hinge":
                # elbow -> forearm, knee -> shin: the distal bone carries the joint
                assert spec["distal"] == schema.JOINT_TO_BONE[joint]
                assert spec["proximal"] in schema.RELEVANT_PREFIXES
            else:
                assert spec["bone"] == schema.JOINT_TO_BONE[joint]

    def test_hinge_and_ball_assignment(self):
        for joint in HINGE_JOINTS:
            assert schema.JOINT_SPECS[joint]["kind"] == "hinge"
        for joint in BALL_JOINTS:
            assert schema.JOINT_SPECS[joint]["kind"] == "ball"

    def test_getters_return_the_constants(self):
        assert schema.get_header_prefixes() is schema.HEADER_PREFIXES
        assert schema.get_header_suffixes() is schema.HEADER_SUFFIXES
        assert schema.get_joint_to_bone_mapping() is schema.JOINT_TO_BONE
        assert schema.get_relevant_prefixes() is schema.RELEVANT_PREFIXES
