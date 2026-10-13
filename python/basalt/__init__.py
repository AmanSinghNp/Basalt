"""Seismic f-k mute filtering and SEG-Y file I/O.

Thin pybind11 layer over the basalt C++ core. Contract: docs/design/FK_PIPELINE.md, section 7.

Requires an x86-64 CPU with AVX2 and FMA. The core is built with the project-wide flags
(-mavx2 -mfma, see BASALT_COMMON_FLAGS in CMakeLists.txt), so the whole extension needs them.

Errors: invalid values (out-of-range parameters, NaN or inf samples, float32 overflow) and bad
str, count or array arguments raise ValueError. Non-numeric float arguments and non-path objects
raise TypeError.
"""

from ._basalt import fk_filter, fk_mute_weight, read_segy, read_trace_headers, write_segy

__version__ = "0.2.0"

__all__ = ["fk_filter", "fk_mute_weight", "read_segy", "read_trace_headers", "write_segy",
           "__version__"]
