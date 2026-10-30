"""Tests for process/geometry.py (vectors, trunk frame, joint angles)."""

import math

import numpy as np
import pandas as pd
import pytest

from process import geometry
from process import schema
from synthetic_data import STANDARD_POSES, make_bones, make_frame

HINGE_JOINTS = ["right_elbow", "right_knee", "left_elbow", "left_knee"]
BALL_JOINTS = ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]
ALL_JOINTS = [
    "right_shoulder", "right_elbow", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "left_hip", "left_knee",
]  # fmt: skip

class TestVectorHelpers:
    def test_unit_normalizes(self):
        assert np.allclose(geometry._unit([3, 0, 4]), [0.6, 0, 0.8])
        assert np.isclose(np.linalg.norm(geometry._unit([1, 2, 3])), 1.0)

    def test_unit_of_zero_vector_is_nan_not_a_crash(self):
        assert np.isnan(geometry._unit([0, 0, 0])).all()

    @pytest.mark.parametrize(
        "u, v, expected",
        [
            ([1, 0, 0], [1, 0, 0], 0.0),
            ([1, 0, 0], [0, 1, 0], 90.0),
            ([1, 0, 0], [-1, 0, 0], 180.0),
            ([1, 0, 0], [1, 1, 0], 45.0),
            ([2, 0, 0], [0, 5, 0], 90.0),  # lengths must not matter
            ([0, 0, 1], [0, 1, 1], 45.0),
        ],
    )
    def test_angle_between(self, u, v, expected):
        assert geometry._angle_between(u, v) == pytest.approx(expected)

    def test_angle_between_degenerate_vector_is_nan(self):
        assert math.isnan(geometry._angle_between([0, 0, 0], [1, 0, 0]))

    def test_angle_between_identical_vectors_never_nan_from_rounding(self):
        v = [0.123456789, 0.987654321, 0.555555555]
        assert geometry._angle_between(v, v) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        "d, expected",
        [
            ([0, 0, -1], 0.0),  # straight down
            ([1, 0, 0], 90.0),  # fully toward the axis
            ([-1, 0, 0], -90.0),  # fully away
            ([1, 0, -1], 45.0),
            ([-1, 0, -1], -45.0),
        ],
    )
    def test_signed_angle(self, d, expected):
        axis, down = np.array([1.0, 0, 0]), np.array([0, 0, -1.0])
        assert geometry._signed_angle(np.array(d, float), axis, down) == pytest.approx(expected)

    def test_bone_direction(self):
        frame = {"b": {"head": (1.0, 1.0, 1.0), "tail": (1.0, 1.0, 3.0)}}
        assert np.allclose(geometry.bone_direction(frame, "b"), [0, 0, 1])

    def test_bone_direction_is_unit_length(self):
        frame = {"b": {"head": (0.0, 0.0, 0.0), "tail": (3.0, 4.0, 12.0)}}
        assert np.linalg.norm(geometry.bone_direction(frame, "b")) == pytest.approx(1.0)

    def test_zero_length_bone_gives_nan_direction(self):
        frame = {"b": {"head": (1.0, 2.0, 3.0), "tail": (1.0, 2.0, 3.0)}}
        assert np.isnan(geometry.bone_direction(frame, "b")).all()

    def test_missing_bone_raises_keyerror(self):
        with pytest.raises(KeyError):
            geometry.bone_direction({}, "forearm.R")

class TestTrunkFrame:
    def test_standing_body_axes(self):
        t = geometry.trunk_frame(make_frame())
        assert np.allclose(t["up"], [0, 0, 1])
        assert np.allclose(t["right"], [1, 0, 0])  # left thigh head -> right thigh head
        assert np.allclose(t["forward"], [0, 1, 0])

    def test_axes_are_orthonormal_and_right_handed(self):
        t = geometry.trunk_frame(make_frame(yaw_deg=33, lean_deg=21, shift=(1, 2, 3)))
        m = np.stack([t["right"], t["forward"], t["up"]])
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(m) == pytest.approx(1.0)
        assert np.allclose(np.cross(t["right"], t["forward"]), t["up"])

    def test_right_axis_is_made_perpendicular_to_up_when_hips_are_uneven(self):
        frame = make_frame()
        # raise the right hip so the hip line is no longer level
        head = list(frame["thigh.R"]["head"])
        head[2] += 0.1
        frame["thigh.R"]["head"] = tuple(head)
        t = geometry.trunk_frame(frame)
        assert np.dot(t["right"], t["up"]) == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(t["right"]) == pytest.approx(1.0)

    def test_follows_the_body_when_it_turns(self):
        t = geometry.trunk_frame(make_frame(yaw_deg=90))
        assert np.allclose(t["forward"], [-1, 0, 0], atol=1e-9)  # facing -X after +90 yaw

    def test_missing_trunk_bone_raises_keyerror(self):
        frame = make_frame()
        del frame["neck"]
        with pytest.raises(KeyError):
            geometry.trunk_frame(frame)

class TestComputeJointAngles:
    def test_returns_every_joint_in_order(self):
        assert list(geometry.compute_joint_angles(make_frame())) == schema.JOINT_ORDER

    def test_standing_pose_is_all_zero(self):
        angles = geometry.compute_joint_angles(make_frame())
        for joint in ALL_JOINTS:
            assert angles[joint]["flexion_deg"] == pytest.approx(0.0, abs=1e-6)
            if joint in BALL_JOINTS:
                assert angles[joint]["abduction_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_hinge_joints_have_no_abduction_entry(self):
        angles = geometry.compute_joint_angles(make_frame())
        for joint in HINGE_JOINTS:
            assert "abduction_deg" not in angles[joint]
        for joint in BALL_JOINTS:
            assert "abduction_deg" in angles[joint]

    @pytest.mark.parametrize("joint", HINGE_JOINTS)
    @pytest.mark.parametrize("flexion", [0.0, 5.0, 30.0, 90.0, 150.0, 179.0])
    def test_hinge_flexion_matches_known_angle(self, joint, flexion):
        angles = geometry.compute_joint_angles(make_frame({joint: flexion}))
        assert angles[joint]["flexion_deg"] == pytest.approx(flexion, abs=1e-6)

    @pytest.mark.parametrize("joint", BALL_JOINTS)
    @pytest.mark.parametrize(
        "flexion, abduction",
        [(0, 0), (70, 0), (-20, 0), (0, 25), (0, 80), (40, 15), (-30, -20), (60, 45)],
    )
    def test_ball_joint_flexion_and_abduction_match_known_angles(self, joint, flexion, abduction):
        a = geometry.compute_joint_angles(make_frame({joint: (flexion, abduction)}))[joint]
        assert a["flexion_deg"] == pytest.approx(flexion, abs=1e-6)
        assert a["abduction_deg"] == pytest.approx(abduction, abs=1e-6)

    def test_forward_is_positive_flexion_and_backward_is_negative(self):
        angles = geometry.compute_joint_angles(make_frame({"right_hip": (30, 0), "left_hip": (-30, 0)}))
        assert angles["right_hip"]["flexion_deg"] > 0 > angles["left_hip"]["flexion_deg"]

    def test_outward_is_positive_abduction_on_both_sides(self):
        # Mirror-image motion: both limbs moved AWAY from the midline must read positive.
        angles = geometry.compute_joint_angles(
            make_frame({"right_shoulder": (0, 40), "left_shoulder": (0, 40)})
        )
        assert angles["right_shoulder"]["abduction_deg"] == pytest.approx(40.0)
        assert angles["left_shoulder"]["abduction_deg"] == pytest.approx(40.0)

    def test_across_the_body_is_negative_abduction(self):
        a = geometry.compute_joint_angles(make_frame({"right_hip": (0, -20)}))["right_hip"]
        assert a["abduction_deg"] == pytest.approx(-20.0)

    def test_moving_one_joint_does_not_change_the_others(self):
        angles = geometry.compute_joint_angles(make_frame({"right_elbow": 50.0}))
        for joint in ALL_JOINTS:
            if joint != "right_elbow":
                assert angles[joint]["flexion_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_left_and_right_sides_give_identical_readings_for_the_same_motion(self):
        pose = {
            "right_elbow": 70.0, "left_elbow": 70.0,
            "right_hip": (35.0, 12.0), "left_hip": (35.0, 12.0),
        }  # fmt: skip
        angles = geometry.compute_joint_angles(make_frame(pose))
        for part in ("elbow", "hip"):
            assert angles[f"right_{part}"]["flexion_deg"] == pytest.approx(
                angles[f"left_{part}"]["flexion_deg"]
            )
        assert angles["right_hip"]["abduction_deg"] == pytest.approx(angles["left_hip"]["abduction_deg"])

    @pytest.mark.parametrize("yaw", [0, 40, 90, -135, 180])
    @pytest.mark.parametrize("lean", [0, 30])
    def test_angles_do_not_change_when_the_whole_body_moves(self, yaw, lean):
        pose = STANDARD_POSES[1]
        still = geometry.compute_joint_angles(make_frame(pose))
        moved = geometry.compute_joint_angles(make_frame(pose, yaw_deg=yaw, lean_deg=lean, shift=(2.0, -1.0, 0.3)))
        for joint in ALL_JOINTS:
            assert moved[joint]["flexion_deg"] == pytest.approx(still[joint]["flexion_deg"], abs=1e-6)
            if joint in BALL_JOINTS:
                assert moved[joint]["abduction_deg"] == pytest.approx(still[joint]["abduction_deg"], abs=1e-6)

    def test_freemocap_rotation_is_passed_through(self):
        frame = make_frame(euler_deg={"forearm.R": (50.0, -10.0, 5.0)}, euler_order="ZYX")
        a = geometry.compute_joint_angles(frame)["right_elbow"]
        assert a["freemocap_euler_deg"] == (50.0, -10.0, 5.0)
        assert a["freemocap_euler_order"] == "ZYX"

    def test_each_joint_reads_the_euler_of_its_own_bone(self):
        euler = {bone: (float(i), 0.0, 0.0) for i, bone in enumerate(schema.RELEVANT_PREFIXES)}
        angles = geometry.compute_joint_angles(make_frame(euler_deg=euler))
        for joint, bone in schema.JOINT_TO_BONE.items():
            assert angles[joint]["freemocap_euler_deg"] == euler[bone]

    def test_missing_trunk_bone_gives_none_for_ball_joints_but_hinges_still_work(self):
        frame = make_frame({"right_elbow": 50.0, "right_hip": (30, 0)})
        del frame["neck"]
        angles = geometry.compute_joint_angles(frame)
        for joint in BALL_JOINTS:
            assert angles[joint] is None
        assert angles["right_elbow"]["flexion_deg"] == pytest.approx(50.0)

    def test_missing_limb_bone_only_affects_the_joints_that_need_it(self):
        frame = make_frame({"left_knee": 40.0})
        del frame["forearm.R"]
        angles = geometry.compute_joint_angles(frame)
        assert angles["right_elbow"] is None
        assert angles["left_knee"]["flexion_deg"] == pytest.approx(40.0)
        assert angles["right_hip"] is not None

    def test_a_missing_thigh_breaks_both_its_own_joint_and_the_trunk_frame(self):
        # thigh heads define the trunk's right axis, so losing one removes ALL ball joints
        frame = make_frame()
        del frame["thigh.L"]
        angles = geometry.compute_joint_angles(frame)
        assert all(angles[j] is None for j in BALL_JOINTS)
        assert angles["left_knee"] is None

    def test_nan_trunk_data_gives_nan_for_ball_joints_but_not_hinges(self):
        frame = make_frame({"right_elbow": 50.0})
        frame["neck"]["head"] = (math.nan, math.nan, math.nan)
        angles = geometry.compute_joint_angles(frame)
        for joint in BALL_JOINTS:
            assert math.isnan(angles[joint]["flexion_deg"])
            assert math.isnan(angles[joint]["abduction_deg"])
        assert angles["right_elbow"]["flexion_deg"] == pytest.approx(50.0)

    def test_zero_length_bone_gives_nan_instead_of_a_crash(self):
        frame = make_frame({"right_elbow": 50.0})
        frame["forearm.R"]["tail"] = frame["forearm.R"]["head"]
        assert math.isnan(geometry.compute_joint_angles(frame)["right_elbow"]["flexion_deg"])
