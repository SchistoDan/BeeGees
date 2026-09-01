"""Tests for the config validation functions defined at the top of workflow/Snakefile.

The validate_* functions are plain Python and can be imported directly.
We stub out snakemake and pandas (not needed for validation logic) before
loading the Snakefile so this test runs without a full BeeGees conda env.
Each validate_* function calls sys.exit(1) on bad input, caught via
pytest.raises(SystemExit).
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

# ---------------------------------------------------------------------------
# Stub snakemake and pandas before loading the Snakefile.
# ---------------------------------------------------------------------------

def _install_stubs():
    """Install minimal stubs for snakemake and pandas, returning which were absent."""
    absent = []
    for name in ("snakemake", "snakemake.io"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.expand = lambda *a, **kw: []
            sys.modules[name] = mod
            absent.append(name)
    if "pandas" not in sys.modules:
        sys.modules["pandas"] = types.ModuleType("pandas")
        absent.append("pandas")
    return absent

_stubs_installed = _install_stubs()

_snakefile = get_package_dir() / "workflow" / "Snakefile"


def _extract_python_preamble(path: Path) -> str:
    """Return only the pure-Python lines before the first Snakemake rule block."""
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("rule ") or line.startswith("checkpoint "):
            break
        lines.append(line)
    return "\n".join(lines)


# Exec the Python preamble (imports + validate_* functions) into a fresh namespace.
# sys.exit is temporarily replaced with a no-op so the module-level
# validate_config(config) call (which uses an empty config dict) does not abort
# collection.
_v = types.ModuleType("_snakefile_validators")
# Inject a dummy config so the module-level validate_config(config) call does
# not raise NameError; sys.exit is no-op'd so the validation failure is silently swallowed.
_v.__dict__["config"] = {}
_real_exit = sys.exit
sys.exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))
try:
    exec(compile(_extract_python_preamble(_snakefile), str(_snakefile), "exec"), _v.__dict__)
except SystemExit:
    pass  # expected: module-level validate calls fail against empty config
finally:
    sys.exit = _real_exit
    # Remove any stubs we installed so downstream test files get an accurate
    # picture of what is really installed (affects pytest.importorskip guards).
    for name in _stubs_installed:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_config(tmp_path):
    """Minimal valid config dict with a real samples_file on disk."""
    samples = tmp_path / "samples.csv"
    samples.write_text("ID,forward\nSAMPLE01,/data/r1.fastq.gz\n")
    return {
        "run_name": "TEST",
        "samples_file": str(samples),
        "output_dir": str(tmp_path / "out"),
        "barcode_recovery": {"r": [1, 1.3], "s": [50, 100], "n": 0, "C": 5, "t": 0.5},
        "run_gene_fetch": False,
        "sequence_reference_file": str(samples),
    }


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_config_does_not_exit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: "/usr/bin/fake")
        _v.validate_config(_base_config(tmp_path))

    def test_missing_run_name_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: "/usr/bin/fake")
        cfg = _base_config(tmp_path)
        del cfg["run_name"]
        with pytest.raises(SystemExit):
            _v.validate_config(cfg)

    def test_missing_samples_file_key_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: "/usr/bin/fake")
        cfg = _base_config(tmp_path)
        del cfg["samples_file"]
        with pytest.raises(SystemExit):
            _v.validate_config(cfg)

    def test_nonexistent_samples_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: "/usr/bin/fake")
        cfg = _base_config(tmp_path)
        cfg["samples_file"] = "/nonexistent/samples.csv"
        with pytest.raises(SystemExit):
            _v.validate_config(cfg)

    def test_r_not_a_list_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: "/usr/bin/fake")
        cfg = _base_config(tmp_path)
        cfg["barcode_recovery"]["r"] = 1
        with pytest.raises(SystemExit):
            _v.validate_config(cfg)

    def test_s_empty_list_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: "/usr/bin/fake")
        cfg = _base_config(tmp_path)
        cfg["barcode_recovery"]["s"] = []
        with pytest.raises(SystemExit):
            _v.validate_config(cfg)

    def test_mge_not_on_path_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules["shutil"], "which", lambda x: None)
        with pytest.raises(SystemExit):
            _v.validate_config(_base_config(tmp_path))


# ---------------------------------------------------------------------------
# validate_fastp_config
# ---------------------------------------------------------------------------

class TestValidateFastpConfig:
    def test_empty_fastp_section_is_valid(self):
        _v.validate_fastp_config({"fastp": {}})

    def test_valid_paired_adapters_pass(self):
        _v.validate_fastp_config({"fastp": {"adapter_r1": "AGATCGGAA", "adapter_r2": "AGATCGGAA"}})

    def test_adapter_r1_non_string_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_fastp_config({"fastp": {"adapter_r1": 123, "adapter_r2": ""}})

    def test_only_r1_provided_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_fastp_config({"fastp": {"adapter_r1": "AGATCGGAA", "adapter_r2": ""}})

    def test_extra_fastp_args_non_string_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_fastp_config({"fastp": {"extra_fastp_args": ["--thread", "4"]}})

    def test_no_fastp_key_is_valid(self):
        _v.validate_fastp_config({})


# ---------------------------------------------------------------------------
# validate_gene_fetch_config
# ---------------------------------------------------------------------------

class TestValidateGeneFetchConfig:
    def test_run_gene_fetch_false_with_valid_ref_file(self, tmp_path):
        ref = tmp_path / "ref.csv"
        ref.write_text("dummy")
        _v.validate_gene_fetch_config({
            "run_gene_fetch": False,
            "sequence_reference_file": str(ref),
        })

    def test_run_gene_fetch_not_bool_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_gene_fetch_config({"run_gene_fetch": "yes"})

    def test_run_gene_fetch_false_missing_ref_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_gene_fetch_config({
                "run_gene_fetch": False,
                "sequence_reference_file": "/nonexistent/ref.csv",
            })

    def test_run_gene_fetch_true_invalid_input_type_exits(self, tmp_path):
        samples = tmp_path / "samples.csv"
        samples.write_text("ID,forward\nS01,r1.fq\n")
        with pytest.raises(SystemExit):
            _v.validate_gene_fetch_config({
                "run_gene_fetch": True,
                "samples_file": str(samples),
                "gene_fetch": {
                    "email": "a@b.com",
                    "api_key": "key123",
                    "gene": "coi",
                    "input_type": "badvalue",
                    "minimum_length": "500",
                },
            })

    def test_run_gene_fetch_true_missing_email_exits(self, tmp_path):
        samples = tmp_path / "samples.csv"
        samples.write_text("ID,forward\nS01,r1.fq\n")
        with pytest.raises(SystemExit):
            _v.validate_gene_fetch_config({
                "run_gene_fetch": True,
                "samples_file": str(samples),
                "gene_fetch": {
                    "api_key": "key123",
                    "gene": "coi",
                    "input_type": "taxid",
                    "minimum_length": "500",
                },
            })


# ---------------------------------------------------------------------------
# validate_structural_validation_config
# ---------------------------------------------------------------------------

class TestValidateStructuralValidationConfig:
    def test_valid_target_passes(self):
        _v.validate_structural_validation_config({"structural_validation": {"target": "cox1"}})

    def test_missing_target_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_structural_validation_config({"structural_validation": {}})

    def test_missing_section_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_structural_validation_config({})

    def test_rbcl_target_passes(self):
        _v.validate_structural_validation_config({"structural_validation": {"target": "rbcl"}})


# ---------------------------------------------------------------------------
# validate_taxonomic_validation_config
# ---------------------------------------------------------------------------

class TestValidateTaxonomicValidationConfig:
    def test_missing_blast_db_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_taxonomic_validation_config({
                "taxonomic_validation": {"blast_db": ""},
            })

    def test_nonexistent_blast_db_exits(self):
        with pytest.raises(SystemExit):
            _v.validate_taxonomic_validation_config({
                "taxonomic_validation": {
                    "blast_db": "/nonexistent/db/",
                    "db_taxonomy": "",
                    "expected_taxonomy": "",
                },
            })

    def test_missing_db_taxonomy_exits(self, tmp_path):
        blast_db = tmp_path / "blastdb"
        blast_db.mkdir()
        with pytest.raises(SystemExit):
            _v.validate_taxonomic_validation_config({
                "taxonomic_validation": {
                    "blast_db": str(blast_db),
                    "db_taxonomy": "",
                    "expected_taxonomy": "",
                },
            })

    def test_nonexistent_db_taxonomy_exits(self, tmp_path):
        blast_db = tmp_path / "blastdb"
        blast_db.mkdir()
        with pytest.raises(SystemExit):
            _v.validate_taxonomic_validation_config({
                "taxonomic_validation": {
                    "blast_db": str(blast_db),
                    "db_taxonomy": "/nonexistent/taxonomy.tsv",
                    "expected_taxonomy": "",
                },
            })

    def test_missing_expected_taxonomy_exits(self, tmp_path):
        blast_db = tmp_path / "blastdb"
        blast_db.mkdir()
        taxonomy = tmp_path / "taxonomy.tsv"
        taxonomy.write_text("dummy")
        with pytest.raises(SystemExit):
            _v.validate_taxonomic_validation_config({
                "taxonomic_validation": {
                    "blast_db": str(blast_db),
                    "db_taxonomy": str(taxonomy),
                    "expected_taxonomy": "",
                },
            })

    def test_nonexistent_expected_taxonomy_exits(self, tmp_path):
        blast_db = tmp_path / "blastdb"
        blast_db.mkdir()
        taxonomy = tmp_path / "taxonomy.tsv"
        taxonomy.write_text("dummy")
        with pytest.raises(SystemExit):
            _v.validate_taxonomic_validation_config({
                "taxonomic_validation": {
                    "blast_db": str(blast_db),
                    "db_taxonomy": str(taxonomy),
                    "expected_taxonomy": "/nonexistent/expected.tsv",
                },
            })

    def test_all_valid_paths_pass(self, tmp_path):
        blast_db = tmp_path / "blastdb"
        blast_db.mkdir()
        taxonomy = tmp_path / "taxonomy.tsv"
        taxonomy.write_text("dummy")
        expected = tmp_path / "expected.tsv"
        expected.write_text("dummy")
        _v.validate_taxonomic_validation_config({
            "taxonomic_validation": {
                "blast_db": str(blast_db),
                "db_taxonomy": str(taxonomy),
                "expected_taxonomy": str(expected),
            },
        })
