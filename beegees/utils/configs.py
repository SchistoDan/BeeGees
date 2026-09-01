"""Locate bundled workflow files within the installed beegees package."""
import importlib.resources
from pathlib import Path


def get_package_dir() -> Path:
    return Path(importlib.resources.files("beegees"))


def get_snakefile() -> Path:
    return get_package_dir() / "workflow" / "Snakefile"


def get_bundled_config() -> Path:
    return get_package_dir() / "config" / "config.yaml"


def get_bundled_profile(name: str) -> Path:
    return get_package_dir() / "profiles" / name
