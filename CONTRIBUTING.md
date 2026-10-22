# Contributing to Basalt

Thanks for helping improve Basalt. This guide covers the build, the tests,
formatting, commit messages and benchmarks.

## Requirements

- C++17 compiler: GCC or Clang on Linux (see the README for other platforms)
- CMake 3.15 or newer (3.21 or newer to use `cmake --preset`)
- Ninja (recommended)
- x86_64 CPU (AVX2 and FMA are used when present, scalar fallbacks otherwise)
- clang-format 18

## Build

The repository provides CMake presets in `CMakePresets.json`:

```bash
cmake --preset gcc-release
cmake --build --preset gcc-release
```

Other presets: `gcc-debug`, `gcc-asan`, `gcc-tsan`, `clang-release`, `clang-debug` and
`msvc` (build presets `msvc-debug` and `msvc-release`). Build output goes to
`build/<preset>/`.

The build options are:

- `BASALT_BUILD_TESTS`: unit and integration tests
- `BASALT_BUILD_BENCHMARKS`: benchmark executables
- `BASALT_BUILD_CLI`: the `basalt` diagnostic CLI

All three default to ON in a top-level build.

## Test

```bash
cmake --build --preset gcc-debug
ctest --test-dir build/gcc-debug --output-on-failure
```

Run the sanitizer presets before submitting changes that touch memory,
threading or SIMD code. Once they are available:

```bash
cmake --preset gcc-asan && cmake --build --preset gcc-asan
ctest --test-dir build/gcc-asan --output-on-failure
```

Every bug fix should come with a regression test. Put tests in the existing
files under `tests/integrity/` or `tests/integration/`.

## Formatting

Format changed C++ files with clang-format 18 before committing. The
repository's `.clang-format` is authoritative. `.editorconfig` covers
whitespace, line endings and encoding for all file types.

```bash
clang-format --version   # must report 18.x
clang-format -i path/to/file.cpp
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): short imperative summary
```

Common types are `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `build`,
`ci` and `chore`. Examples from this repository: `fix(filter): ...`,
`test(bench): ...`, `docs: ...`.

## Pull Requests

- Describe the change and how you tested it. The PR template lists the
  expected checklist.
- Add a user-visible entry under `Unreleased` in `CHANGELOG.md`.
- Keep commits focused. One logical change per commit is preferred.

## Benchmarks

Build the benchmark executables (`BASALT_BUILD_BENCHMARKS=ON`, the default in
a top-level build), then run the matrix sweep tool:

```bash
./build/gcc-release/basalt_bench --sizes 256,512,1024 --threads 1,2,4 \
    --simd auto,avx2 --numa off,auto \
    --csv-out benchmark.csv --json-out benchmark.json
```

Run `./build/gcc-release/basalt_bench --help` for all options. The
Stage 1 profiling workflow is `tools/profiling/run_stage1.sh`. See
[PERFORMANCE.md](./PERFORMANCE.md) for methodology and the reporting
template.

GitHub-hosted runners are used for correctness CI only. Do not publish
performance numbers measured there as representative results.
