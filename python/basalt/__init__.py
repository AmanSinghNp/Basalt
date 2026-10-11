"""Seismic f-k mute filtering and SEG-Y file I/O.

Thin pybind11 layer over the basalt C++ core. Contract: docs/design/FK_PIPELINE.md, section 7.
"""

from ._basalt import fk_filter, fk_mute_weight, read_segy, write_segy

__version__ = "0.2.0"

__all__ = ["fk_filter", "fk_mute_weight", "read_segy", "write_segy", "__version__"]
