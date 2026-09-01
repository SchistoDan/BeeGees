"""Build the snakemake subprocess command for the BeeGees pipeline."""
from pathlib import Path

from beegees.utils.configs import get_snakefile, get_bundled_profile


def _resolve_profile(profile: str | None) -> str:
    """Resolve a profile name or path to a concrete directory path.

    Resolution order for simple names (no slash, not absolute):
      1. ./profiles/<name> in the current working directory (user-editable copy from init)
      2. Bundled profile shipped with the package
      3. Raw value passed straight to Snakemake (unknown name, last resort)

    Anything containing a slash or an absolute path is passed through directly.
    """
    name = profile or "local"
    if "/" in name or Path(name).is_absolute():
        return name
    cwd_profile = Path.cwd() / "profiles" / name
    if cwd_profile.is_dir():
        return str(cwd_profile)
    bundled = get_bundled_profile(name)
    return str(bundled) if bundled.is_dir() else name


def build_snakemake_cmd(
    configfile: Path,
    cores: int | None,
    profile: str | None,
    dryrun: bool,
    unlock: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = ["snakemake", "--snakefile", str(get_snakefile())]

    cmd += ["--configfile", str(configfile)]

    cmd += ["--profile", _resolve_profile(profile)]

    if cores:
        cmd += ["--cores", str(cores)]
    if dryrun:
        cmd += ["--dryrun"]
    if unlock:
        cmd += ["--unlock"]
    else:
        cmd += ["--rerun-incomplete"]

    return cmd + extra_args
