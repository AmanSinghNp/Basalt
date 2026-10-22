---
name: Bug report
about: Report incorrect results, a crash, or a build failure
title: "[Bug] "
labels: bug
---

## Summary

<!-- What went wrong, and what did you expect to happen? -->

## Reproduction

<!-- Minimal steps or a small program that shows the problem. -->

1.
2.
3.

## Environment

- **CPU model** (`lscpu | grep 'Model name'`):
- **CPU flags** relevant to Basalt (`grep -o -w -E 'avx2|fma|avx512f' /proc/cpuinfo | sort -u`):
- **Compiler and version** (`g++ --version` or `clang++ --version`):
- **CMake version** (`cmake --version`):
- **OS and kernel** (`uname -a`):
- **Basalt commit or version**:
- **Build type and CMake preset** (for example Release, `gcc-release`):

## Logs and output

<!-- Paste relevant console output, test failures or sanitizer reports. -->

```text
```
