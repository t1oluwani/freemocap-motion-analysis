import pytest

from synthetic_data import (
    STANDARD_EULER_RAD,
    STANDARD_POSES,
    make_bones,
    write_bone_csv,
)


@pytest.fixture
def csv_factory(tmp_path):
    """
    Returns make(poses=..., euler=..., ...) -> path of a synthetic bone_and_joint CSV
    written inside the test's temporary folder.
    """

    def make(
        poses=None,
        euler=STANDARD_EULER_RAD,
        euler_order="XYZ",
        drop_bones=(),
        skeletons=None,
        name="synthetic_bone_and_joint_data.csv",
        extra_columns=None,
    ):
        if skeletons is None:
            skeletons = [make_bones(p) for p in (STANDARD_POSES if poses is None else poses)]
        path = tmp_path / name
        write_bone_csv(path, skeletons, euler, euler_order, drop_bones, extra_columns)
        return str(path)

    return make


@pytest.fixture
def standard_csv(csv_factory):
    """The standard 3-frame synthetic CSV (standing, mixed pose A, mixed pose B)."""
    return csv_factory()