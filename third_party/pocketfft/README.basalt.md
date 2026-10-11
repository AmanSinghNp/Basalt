# Vendored pocketfft

This directory holds a copy of pocketfft, the C++ FFT library used by Basalt's f-k pipeline
(`src/fk/fk_pipeline.cpp`). Basalt compiles it as a header-only dependency.

| Field      | Value                                                                  |
| :--------- | :--------------------------------------------------------------------- |
| Source     | https://github.com/mreineck/pocketfft (`cpp` branch)                    |
| Commit     | `c90e55b3d529f8efa40ed01a20de22405f45fc65` (recorded in `VERSION`)     |
| License    | BSD-3-Clause (`LICENSE.md`, copyright Max-Planck-Society and contributors) |
| Files      | `pocketfft_hdronly.h`, `LICENSE.md`, `VERSION`                          |
| Modified   | No. Both `pocketfft_hdronly.h` and `LICENSE.md` are byte-identical to the files at the commit above. |

Do not edit these files in place. To update pocketfft, replace the three files with the
upstream versions from a newer commit, update `VERSION` and this table, and run the full test
suite, since the f-k pipeline's numerical tests depend on the transform.
