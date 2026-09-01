"""BeeGees: DNA barcoding from genome skims."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("beegees")
except PackageNotFoundError:
    __version__ = "unknown"
