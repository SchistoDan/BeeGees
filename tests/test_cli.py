"""Tests for the beegees CLI (beegees/__main__.py)."""
import subprocess
import sys

import pytest

from beegees import __version__
from beegees.__main__ import build_parser


class TestParserStructure:
    def setup_method(self):
        self.parser = build_parser()

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            self.parser.parse_args(["--version"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args([])

    def test_run_requires_config(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["run"])

    def test_run_config_parsed(self):
        args = self.parser.parse_args(["run", "--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"
        assert args.dryrun is False
        assert args.cores is None
        assert args.profile is None

    def test_run_dryrun_flag(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "--dryrun"])
        assert args.dryrun is True

    def test_run_dryrun_short_flag(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "-n"])
        assert args.dryrun is True

    def test_run_cores(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "--cores", "16"])
        assert args.cores == 16

    def test_run_profile_local(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "--profile", "local"])
        assert args.profile == "local"

    def test_run_profile_slurm(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "--profile", "slurm"])
        assert args.profile == "slurm"

    def test_run_extra_snakemake_args(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "--", "--forceall"])
        assert "--forceall" in args.snakemake_args

    def test_run_log_file(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml", "--log-file", "run.log"])
        assert args.log_file == "run.log"

    def test_run_log_file_default_none(self):
        args = self.parser.parse_args(["run", "--config", "c.yaml"])
        assert args.log_file is None

    def test_init_default_no_force(self):
        args = self.parser.parse_args(["init"])
        assert args.force is False

    def test_init_force_flag(self):
        args = self.parser.parse_args(["init", "--force"])
        assert args.force is True


class TestInitCommand:
    def test_init_copies_config_and_profiles(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from beegees.__main__ import cmd_init

        class FakeArgs:
            force = False

        result = cmd_init(FakeArgs())
        assert result == 0
        assert (tmp_path / "config" / "config.yaml").exists()
        assert (tmp_path / "profiles" / "local" / "config.yaml").exists()
        assert (tmp_path / "profiles" / "slurm" / "config.yaml").exists()
        assert (tmp_path / "samples_template.csv").exists()

    def test_init_creates_samples_template(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from beegees.__main__ import cmd_init

        class FakeArgs:
            force = False

        cmd_init(FakeArgs())
        content = (tmp_path / "samples_template.csv").read_text()
        assert content.startswith("ID,forward,reverse,phylum,class,order,family,genus,species")

    def test_init_refuses_overwrite_without_force(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        from beegees.__main__ import cmd_init

        class FakeArgs:
            force = False

        result = cmd_init(FakeArgs())
        assert result == 1

    def test_init_overwrites_with_force(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        (tmp_path / "profiles").mkdir()
        from beegees.__main__ import cmd_init

        class FakeArgs:
            force = True

        result = cmd_init(FakeArgs())
        assert result == 0


class TestRunCommand:
    def test_run_missing_config_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from beegees.__main__ import cmd_run

        class FakeArgs:
            config = "nonexistent_config.yaml"
            cores = None
            profile = None
            dryrun = False
            snakemake_args = []

        result = cmd_run(FakeArgs())
        assert result == 1


class TestEntryPoint:
    def test_beegees_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "beegees", "--help"],
            capture_output=True,
        )
        assert result.returncode == 0
        assert b"BeeGees" in result.stdout

    def test_beegees_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "beegees", "--version"],
            capture_output=True,
        )
        assert result.returncode == 0
        assert __version__.encode() in result.stdout
