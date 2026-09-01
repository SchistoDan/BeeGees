"""Tests for beegees.utils.configs and beegees.utils.snakemake_args."""
from pathlib import Path

import pytest

from beegees.utils.configs import (
    get_bundled_config,
    get_bundled_profile,
    get_package_dir,
    get_snakefile,
)
from beegees.utils.snakemake_args import build_snakemake_cmd, _resolve_profile


class TestConfigs:
    def test_package_dir_exists(self):
        assert get_package_dir().is_dir()

    def test_snakefile_exists(self):
        sf = get_snakefile()
        assert sf.exists()
        assert sf.name == "Snakefile"

    def test_bundled_config_exists(self):
        assert get_bundled_config().exists()

    def test_bundled_local_profile_exists(self):
        assert get_bundled_profile("local").is_dir()

    def test_bundled_slurm_profile_exists(self):
        assert get_bundled_profile("slurm").is_dir()

    def test_bundled_hmm_files_present(self):
        pkg = get_package_dir()
        assert (pkg / "resources" / "hmm" / "COI-5P.hmm").exists()
        assert (pkg / "resources" / "hmm" / "rbcL.hmm").exists()

    def test_bundled_contaminants_present(self):
        pkg = get_package_dir()
        assert (pkg / "resources" / "contaminants" / "human_mitogenome.fasta").exists()

    def test_all_scripts_bundled(self):
        scripts_dir = get_package_dir() / "workflow" / "scripts"
        expected = [
            "01_human_cox1_filter.py",
            "02_at_content_filter.py",
            "03_statistical_outlier_filter.py",
            "04_reference_filter.py",
            "05_consensus_generator.py",
            "06_aggregate_filter_metrics.py",
            "structural_validation.py",
            "barcoding_outcome.py",
            "multiqc_plots.R",
        ]
        for name in expected:
            assert (scripts_dir / name).exists(), f"Missing bundled script: {name}"


class TestSnakemakeArgs:
    def _cfg(self, path="config.yaml"):
        return Path(path)

    def test_snakefile_always_included(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, False, False, [])
        assert "--snakefile" in cmd
        snakefile_idx = cmd.index("--snakefile")
        assert cmd[snakefile_idx + 1].endswith("Snakefile")

    def test_configfile_always_included(self):
        cmd = build_snakemake_cmd(self._cfg("my.yaml"), None, None, False, False, [])
        assert "--configfile" in cmd
        idx = cmd.index("--configfile")
        assert cmd[idx + 1] == "my.yaml"

    def test_default_profile_is_local(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, False, False, [])
        assert "--profile" in cmd
        idx = cmd.index("--profile")
        assert "local" in cmd[idx + 1]

    def test_slurm_profile_resolves_to_bundled(self):
        cmd = build_snakemake_cmd(self._cfg(), None, "slurm", False, False, [])
        idx = cmd.index("--profile")
        assert "slurm" in cmd[idx + 1]

    def test_custom_profile_path_passed_through(self):
        cmd = build_snakemake_cmd(self._cfg(), None, "/my/custom/profile", False, False, [])
        idx = cmd.index("--profile")
        assert cmd[idx + 1] == "/my/custom/profile"

    def test_cores_included_when_set(self):
        cmd = build_snakemake_cmd(self._cfg(), 8, None, False, False, [])
        assert "--cores" in cmd
        assert cmd[cmd.index("--cores") + 1] == "8"

    def test_cores_omitted_when_none(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, False, False, [])
        assert "--cores" not in cmd

    def test_dryrun_flag(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, True, False, [])
        assert "--dryrun" in cmd

    def test_unlock_flag(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, False, True, [])
        assert "--unlock" in cmd
        assert "--rerun-incomplete" not in cmd

    def test_rerun_incomplete_added_on_normal_run(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, False, False, [])
        assert "--rerun-incomplete" in cmd

    def test_rerun_incomplete_added_on_dryrun(self):
        # build_snakemake_cmd adds --rerun-incomplete whenever unlock=False,
        # regardless of dryrun flag
        cmd = build_snakemake_cmd(self._cfg(), None, None, True, False, [])
        assert "--rerun-incomplete" in cmd

    def test_extra_args_appended(self):
        cmd = build_snakemake_cmd(self._cfg(), None, None, False, False, ["--forceall", "--quiet"])
        assert "--forceall" in cmd
        assert "--quiet" in cmd
        assert cmd[-2] == "--forceall"
        assert cmd[-1] == "--quiet"


class TestResolveProfile:
    def test_none_resolves_to_bundled_local(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_profile(None)
        assert "local" in result
        assert "profiles" in result

    def test_name_falls_back_to_bundled_when_no_cwd_copy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_profile("slurm")
        assert "slurm" in result
        # Must point into the installed package, not the tmp dir
        assert str(tmp_path) not in result

    def test_cwd_profile_preferred_over_bundled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cwd_slurm = tmp_path / "profiles" / "slurm"
        cwd_slurm.mkdir(parents=True)
        result = _resolve_profile("slurm")
        assert result == str(cwd_slurm)

    def test_absolute_path_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        absolute = str(tmp_path / "my-profile")
        result = _resolve_profile(absolute)
        assert result == absolute

    def test_relative_path_with_slash_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_profile("./profiles/slurm")
        assert result == "./profiles/slurm"

    def test_cwd_local_preferred_over_bundled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cwd_local = tmp_path / "profiles" / "local"
        cwd_local.mkdir(parents=True)
        result = _resolve_profile(None)
        assert result == str(cwd_local)
