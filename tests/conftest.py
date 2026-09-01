"""Shared fixtures and path setup for BeeGees tests."""
import functools
import importlib.util
import sys
from pathlib import Path

import pytest

from beegees.utils.configs import get_package_dir

# Resolve scripts from the installed beegees package (single source of truth)
# so tests validate exactly what ships, not a copy in the source tree.
SCRIPTS_DIR = get_package_dir() / "workflow" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@functools.lru_cache(maxsize=None)
def load_script_module(name):
    """Load a bundled workflow script as a module, once per session.

    The scripts are shipped as package data and run via `python <script>`, so they
    are not importable by name; load them from their file path instead. The result
    is cached so every caller shares one module object - otherwise exception
    classes defined in the script would not compare equal across loads.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def blast_runner(tmp_path):
    """A BLASTRunner with __init__ bypassed.

    BLASTRunner.__init__ shells out to `blastn -version` and validates a real
    BLAST database, neither of which exists in CI, so construct the object
    directly and set only the attributes the chunking and demux paths touch.
    """
    tv_local_blast = load_script_module("tv_local_blast")
    runner = object.__new__(tv_local_blast.BLASTRunner)
    runner.output_dir = tmp_path / "out"
    runner.output_dir.mkdir(parents=True, exist_ok=True)
    runner.db_path = tmp_path / "db" / "fakedb"
    runner.db_name = "fakedb"
    runner.blast_options = "-evalue 1e-5 -max_target_seqs 100 -num_threads 1"
    runner.num_processes = 4
    runner.output_csv = None
    runner.allow_partial_failures = False
    runner.processed_sequences = {}
    runner.qseqid_to_stem = {}
    runner.failed_sequences = []
    return runner
