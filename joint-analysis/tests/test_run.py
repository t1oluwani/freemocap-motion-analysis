"""
Tests for run_analysis.py (the command-line entry point).

main() reads sys.argv and writes to analysis_output/ relative to the current
folder, so every test runs inside a temporary folder (monkeypatch.chdir).
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import run_analysis as ra

REC = ra.RECORDING_NUMBER
HERE = Path(__file__).resolve().parent


@pytest.fixture
def run(monkeypatch, tmp_path, capsys):
    """run(*args) -> stdout text. Runs main() inside tmp_path with the given CLI args."""
    monkeypatch.chdir(tmp_path)

    def _run(*args):
        monkeypatch.setattr(sys, "argv", ["run_analysis.py", *map(str, args)])
        ra.main()
        return capsys.readouterr().out

    return _run


def out_path(tmp_path, kind, name):
    return tmp_path / "analysis_output" / kind / name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_csv_path_is_built_from_its_parts(self):
        assert ra.BONE_AND_JOINT_CSV_FILE == f"{ra.VIDEO_FOLDER_PATH}/{ra.RECORDING_NUMBER}_{ra.DATA_TYPE}"

    def test_data_type_matches_freemocap_file_suffix(self):
        assert ra.DATA_TYPE.endswith("bone_and_joint_data.csv")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


class TestPrinting:
    def test_header_reports_frames_and_bones(self, run, standard_csv):
        out = run("--csv", standard_csv)
        assert "Loaded 3 frames, 10 bones per frame." in out

    def test_default_prints_three_frames_with_both_views(self, run, standard_csv):
        out = run("--csv", standard_csv)
        for i in range(3):
            assert f"Frame {i} - FreeMoCap:" in out
            assert f"Frame {i} - computed:" in out

    def test_print_frames_limits_the_output(self, run, standard_csv):
        out = run("--csv", standard_csv, "--print_frames", 1)
        assert "Frame 0 - computed:" in out
        assert "Frame 1 - computed:" not in out

    def test_print_frames_zero_prints_only_the_header(self, run, standard_csv):
        out = run("--csv", standard_csv, "--print_frames", 0)
        assert "Loaded 3 frames" in out
        assert "Frame 0" not in out

    def test_print_frames_larger_than_recording_is_fine(self, run, standard_csv):
        out = run("--csv", standard_csv, "--print_frames", 99)
        assert "Frame 2 - computed:" in out
        assert "Frame 3" not in out

    def test_known_angles_appear_as_readable_sentences(self, run, standard_csv):
        out = run("--csv", standard_csv)
        assert "right elbow bent 50 degrees" in out
        assert "left knee bent 85 degrees" in out
        assert "right hip flexed 70 degrees" in out
        assert "left hip abducted 25 degrees" in out
        assert "right elbow: rotation XYZ x=50 y=0 z=0 degrees" in out

    def test_combine_interleaves_instead_of_two_blocks(self, run, standard_csv):
        out = run("--csv", standard_csv, "--combine", "--print_frames", 1)
        assert "Frame 0 (computed vs FreeMoCap, interleaved):" in out
        assert "  computed:   right shoulder" in out
        assert "  FreeMoCap:  right shoulder" in out
        assert "Frame 0 - computed:" not in out

    def test_euler_sanity_check_is_included_in_the_output(self, run, standard_csv):
        assert "forearm.R" in run("--csv", standard_csv)

    def test_all_zero_euler_data_prints_the_note(self, run, csv_factory):
        out = run("--csv", csv_factory(euler={}))
        assert "NOTE: every Euler value is ~0" in out

    def test_euler_unit_degrees_changes_the_reading(self, run, csv_factory):
        path = csv_factory(euler={(0, "forearm.R"): (50.0, 0.0, 0.0)})
        as_radians = run("--csv", path, "--print_frames", 1)
        as_degrees = run("--csv", path, "--print_frames", 1, "--euler_unit", "degrees")
        assert "right elbow: rotation XYZ x=50 y=0 z=0 degrees" in as_degrees
        assert "x=50 y=0 z=0" not in as_radians  # 50 "radians" is converted (and warned about)
        assert "Warning: euler values reach" in as_radians

    def test_invalid_euler_unit_exits_with_usage_error(self, run, standard_csv):
        with pytest.raises(SystemExit) as e:
            run("--csv", standard_csv, "--euler_unit", "gradians")
        assert e.value.code == 2

    def test_nothing_is_written_without_save(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv)
        assert not (tmp_path / "analysis_output").exists()

    def test_missing_csv_raises(self, run, tmp_path):
        with pytest.raises(FileNotFoundError):
            run("--csv", tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


class TestSaveSeparate:
    def test_four_files_with_the_documented_names(self, run, standard_csv, tmp_path):
        out = run("--csv", standard_csv, "--save")
        files = {
            ("csv", f"{REC}_computed_joint_angles.csv"),
            ("csv", f"{REC}_freemocap_joint_angles.csv"),
            ("print", f"{REC}_computed_joint_analysis.txt"),
            ("print", f"{REC}_freemocap_joint_analysis.txt"),
        }
        for kind, name in files:
            assert out_path(tmp_path, kind, name).is_file(), name
        assert sorted(p.name for p in (tmp_path / "analysis_output" / "csv").iterdir()) == sorted(
            n for k, n in files if k == "csv"
        )
        assert sorted(p.name for p in (tmp_path / "analysis_output" / "print").iterdir()) == sorted(
            n for k, n in files if k == "print"
        )
        assert "Data saved to analysis_output/csv" in out.replace("\\", "/")
        assert "Print output saved to analysis_output/print" in out.replace("\\", "/")

    def test_computed_csv_holds_every_frame_with_the_right_angles(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save", "--print_frames", 1)  # printing limit must not limit saving
        df = pd.read_csv(out_path(tmp_path, "csv", f"{REC}_computed_joint_angles.csv"))
        assert len(df) == 3
        assert df.shape[1] == 13
        assert df.loc[1, "right_elbow_flexion_deg"] == pytest.approx(50.0)
        assert df.loc[1, "left_knee_flexion_deg"] == pytest.approx(85.0)
        assert df.loc[2, "right_shoulder_abduction_deg"] == pytest.approx(80.0)

    def test_freemocap_csv_has_euler_columns_only(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save")
        df = pd.read_csv(out_path(tmp_path, "csv", f"{REC}_freemocap_joint_angles.csv"), keep_default_na=False)
        assert len(df) == 3
        assert df.shape[1] == 33
        assert df.loc[1, "right_elbow_freemocap_euler_x_deg"] == pytest.approx(50.0)
        assert not any("flexion" in c for c in df.columns)

    def test_csv_has_no_index_column(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save")
        first = out_path(tmp_path, "csv", f"{REC}_computed_joint_angles.csv").read_text().splitlines()[0]
        assert first.startswith("frame,")

    def test_text_files_contain_only_their_own_view(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save")
        computed = out_path(tmp_path, "print", f"{REC}_computed_joint_analysis.txt").read_text()
        freemocap = out_path(tmp_path, "print", f"{REC}_freemocap_joint_analysis.txt").read_text()
        assert "Frame 0 - computed:" in computed and "FreeMoCap" not in computed
        assert "right elbow bent 50 degrees" in computed
        assert "Frame 0 - FreeMoCap:" in freemocap and "computed" not in freemocap
        assert "rotation XYZ" in freemocap

    def test_text_files_honour_print_frames(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save", "--print_frames", 2)
        text = out_path(tmp_path, "print", f"{REC}_computed_joint_analysis.txt").read_text()
        assert "Frame 1 - computed:" in text
        assert "Frame 2 - computed:" not in text

    def test_rerunning_overwrites_instead_of_failing_or_appending(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save")
        run("--csv", standard_csv, "--save")
        df = pd.read_csv(out_path(tmp_path, "csv", f"{REC}_computed_joint_angles.csv"))
        assert len(df) == 3


class TestSaveCombined:
    def test_two_files_with_the_documented_names(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save", "--combine")
        csv_names = [p.name for p in (tmp_path / "analysis_output" / "csv").iterdir()]
        txt_names = [p.name for p in (tmp_path / "analysis_output" / "print").iterdir()]
        assert csv_names == [f"{REC}_joint_angles_combined.csv"]
        assert txt_names == [f"{REC}_joint_analysis_combined.txt"]

    def test_combined_csv_has_computed_and_freemocap_columns(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save", "--combine")
        df = pd.read_csv(out_path(tmp_path, "csv", f"{REC}_joint_angles_combined.csv"))
        assert len(df) == 3
        assert df.shape[1] == 45
        assert df.loc[1, "right_elbow_flexion_deg"] == pytest.approx(50.0)
        assert df.loc[1, "right_elbow_freemocap_euler_x_deg"] == pytest.approx(50.0)

    def test_combined_text_is_the_interleaved_report(self, run, standard_csv, tmp_path):
        run("--csv", standard_csv, "--save", "--combine")
        text = out_path(tmp_path, "print", f"{REC}_joint_analysis_combined.txt").read_text()
        assert text.startswith("Loaded 3 frames")
        assert "interleaved" in text
        assert "  computed:   right elbow bent 50 degrees" in text


class TestSaveHelpers:
    def test_save_csv_creates_missing_folders(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        ra.save_csv_to_output_dir(pd.DataFrame({"a": [1, 2]}), "x.csv")
        assert pd.read_csv(tmp_path / "analysis_output" / "csv" / "x.csv")["a"].tolist() == [1, 2]
        assert "x.csv" in capsys.readouterr().out

    def test_save_print_writes_text_verbatim(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        ra.save_print_to_output_dir("line1\nline2", "x.txt")
        assert (tmp_path / "analysis_output" / "print" / "x.txt").read_text() == "line1\nline2"
        assert "x.txt" in capsys.readouterr().out

    def test_existing_output_folder_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("analysis_output/csv")
        ra.save_csv_to_output_dir(pd.DataFrame({"a": [1]}), "x.csv")


# ---------------------------------------------------------------------------
# Real process: the way the user actually runs it
# ---------------------------------------------------------------------------


class TestSubprocess:
    def test_runs_as_a_script_from_another_folder(self, standard_csv, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        env = {**os.environ, "PYTHONPATH": ""}  # prove the sibling import works without help
        result = subprocess.run(
            [sys.executable, str(HERE / "run_analysis.py"), "--csv", standard_csv, "--save", "--combine"],
            cwd=work, capture_output=True, text=True, env=env,
        )  # fmt: skip
        assert result.returncode == 0, result.stderr
        assert "right elbow bent 50 degrees" in result.stdout
        assert (work / "analysis_output" / "csv" / f"{REC}_joint_angles_combined.csv").is_file()

    def test_help_lists_every_flag(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "run_analysis.py"), "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0
        for flag in ("--csv", "--euler_unit", "--print_frames", "--save", "--combine"):
            assert flag in result.stdout

    def test_missing_default_csv_fails_loudly(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(HERE / "run_analysis.py")], cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr