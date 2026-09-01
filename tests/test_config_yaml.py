"""Validate that config/config.yaml loads correctly and contains required keys/types.

This test loads the bundled config.yaml with PyYAML and checks structure —
it does not invoke Snakemake and requires no external tools.
"""
import re
from pathlib import Path

import pytest
import yaml
from beegees.utils.configs import get_package_dir


CONFIG_PATH = get_package_dir() / "config" / "config.yaml"


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG_PATH.read_text())


class TestConfigYamlStructure:
    def test_file_exists(self):
        assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"

    def test_parses_without_error(self, config):
        assert isinstance(config, dict)

    # Top-level required keys
    def test_has_run_name(self, config):
        assert "run_name" in config

    def test_has_samples_file(self, config):
        assert "samples_file" in config

    def test_has_output_dir(self, config):
        assert "output_dir" in config

    def test_has_barcode_recovery_section(self, config):
        assert "barcode_recovery" in config
        assert isinstance(config["barcode_recovery"], dict)

    def test_barcode_recovery_r_list(self, config):
        br = config["barcode_recovery"]
        assert "r" in br
        assert isinstance(br["r"], list)
        assert len(br["r"]) > 0

    def test_barcode_recovery_s_list(self, config):
        br = config["barcode_recovery"]
        assert "s" in br
        assert isinstance(br["s"], list)
        assert len(br["s"]) > 0

    def test_has_run_gene_fetch_bool(self, config):
        assert "run_gene_fetch" in config
        assert isinstance(config["run_gene_fetch"], bool)

    # fastp section
    def test_fastp_section_present(self, config):
        assert "fastp" in config
        assert isinstance(config["fastp"], dict)

    def test_fastp_adapter_r1_is_string(self, config):
        val = config["fastp"].get("adapter_r1", "")
        assert isinstance(val, str)

    def test_fastp_adapter_r2_is_string(self, config):
        val = config["fastp"].get("adapter_r2", "")
        assert isinstance(val, str)

    # fasta_cleaner section
    def test_fasta_cleaner_section_present(self, config):
        assert "fasta_cleaner" in config
        assert isinstance(config["fasta_cleaner"], dict)

    def test_fasta_cleaner_consensus_threshold_in_range(self, config):
        val = config["fasta_cleaner"]["consensus_threshold"]
        assert 0.0 <= float(val) <= 1.0

    def test_fasta_cleaner_human_threshold_in_range(self, config):
        val = config["fasta_cleaner"]["human_threshold"]
        assert 0.0 <= float(val) <= 1.0

    def test_fasta_cleaner_at_difference_in_range(self, config):
        val = config["fasta_cleaner"]["at_difference"]
        assert 0.0 <= float(val) <= 1.0

    def test_fasta_cleaner_at_mode_valid(self, config):
        assert config["fasta_cleaner"]["at_mode"].lower() in {"absolute", "higher", "lower"}

    def test_fasta_cleaner_outlier_percentile_in_range(self, config):
        val = config["fasta_cleaner"]["outlier_percentile"]
        assert 0.0 <= float(val) <= 100.0

    # structural_validation section
    def test_structural_validation_target_valid(self, config):
        target = config.get("structural_validation", {}).get("target", "")
        assert target.lower() in {"cox1", "coi", "rbcl"}

    def test_structural_validation_no_genetic_code_key(self, config):
        # genetic_code was removed from structural_validation; it is now read from barcode_recovery.C
        assert "genetic_code" not in config.get("structural_validation", {})

    # taxonomic_validation section
    def test_taxonomic_validation_taxval_rank_valid(self, config):
        rank = config.get("taxonomic_validation", {}).get("taxval_rank", "")
        valid_ranks = {"phylum", "class", "order", "family", "genus", "species"}
        assert rank.lower() in valid_ranks

    # rules section
    def test_rules_section_present(self, config):
        assert "rules" in config
        assert isinstance(config["rules"], dict)

    def test_rules_have_mem_mb(self, config):
        for rule_name, rule_cfg in config["rules"].items():
            assert "mem_mb" in rule_cfg, f"Rule '{rule_name}' missing mem_mb"

    def test_rules_threads_valid_when_present(self, config):
        """threads is optional - it is only declared for rules whose command
        actually consumes it. Where present it must be a positive int."""
        for rule_name, rule_cfg in config["rules"].items():
            if "threads" not in rule_cfg:
                continue
            threads = rule_cfg["threads"]
            assert isinstance(threads, int) and threads >= 1, (
                f"Rule '{rule_name}' has invalid threads: {threads!r}"
            )

    def test_snakefile_resource_reads_resolve(self, config):
        """Every rule_resources["<rule>"]["<key>"] in the Snakefile must resolve.

        Snakemake evaluates these at parse time, so a read with no matching config
        entry is a KeyError that kills the whole workflow before any job runs -
        including for rules the current run mode would never use.

        The rule-name character class must accept uppercase: MitoGeneExtractor is
        the one rule that has it, and a lowercase-only pattern silently skips it.
        """
        snakefile = (get_package_dir() / "workflow" / "Snakefile").read_text()
        rules = config["rules"]
        unresolved = []
        for lineno, line in enumerate(snakefile.splitlines(), 1):
            for rule, key in re.findall(
                    r'rule_resources\["([A-Za-z_]+)"\]\["([a-z_]+)"\]', line):
                if rule not in rules:
                    unresolved.append(
                        f"Snakefile:{lineno} rule_resources['{rule}']['{key}'] "
                        f"- no '{rule}' entry in config rules:")
                elif key not in rules[rule]:
                    unresolved.append(
                        f"Snakefile:{lineno} rule_resources['{rule}']['{key}'] "
                        f"- '{key}' not set for '{rule}' in config")
        assert not unresolved, (
            "Unresolved resource reads (KeyError at DAG build):\n  "
            + "\n  ".join(unresolved)
        )

    # barcode_recovery parameters
    def test_barcode_recovery_n_is_numeric(self, config):
        val = config["barcode_recovery"]["n"]
        assert isinstance(val, (int, float))

    def test_barcode_recovery_C_is_int(self, config):
        assert isinstance(config["barcode_recovery"]["C"], int)

    def test_barcode_recovery_t_is_float_in_range(self, config):
        val = config["barcode_recovery"]["t"]
        assert 0.0 <= float(val) <= 1.0

    # taxonomic_validation key names
    def test_taxonomic_validation_has_blast_db(self, config):
        tv = config.get("taxonomic_validation", {})
        assert "blast_db" in tv, "'blast_db' key missing from taxonomic_validation"
        assert "database" not in tv, "old 'database' key should not be present"

    def test_taxonomic_validation_has_db_taxonomy(self, config):
        tv = config.get("taxonomic_validation", {})
        assert "db_taxonomy" in tv, "'db_taxonomy' key missing from taxonomic_validation"
        assert "database_taxonomy" not in tv, "old 'database_taxonomy' key should not be present"
