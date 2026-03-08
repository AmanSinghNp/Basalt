# Basalt Stage 1 Profiling Dossier Template

Use this template for every Stage 1 baseline collection run.

## Metadata
- Run ID:
- Date (UTC):
- Commit SHA:
- Runner / Host:
- CPU model:
- Kernel version:
- perf binary path:
- perf available:
- perf list available:

## Workload Configuration
- Grid size:
- Timed shots:
- Warmup shots:
- Seed:
- Benchmark binary:

## Tooling Matrix
| Tool | Status | Notes |
|---|---|---|
| `perf stat` |  |  |
| `perf record` / `perf annotate` |  |  |
| Google benchmark harness (`basalt_bench`) |  |  |
| VTune (optional) |  |  |
| LIKWID (optional) |  |  |
| Cachegrind (optional) |  |  |

## Selected Event Groups
| Group | Source | Selected Events |
|---|---|---|
| IPC | canonical/fallback/empty |  |
| Cache | canonical/fallback/empty |  |
| DRAM | canonical/fallback/empty |  |
| SIMD | canonical/fallback/empty |  |
| Branch | canonical/fallback/empty |  |
| Faults | canonical/fallback/empty |  |

## Benchmark Summary
| Metric | Primary | Repeat |
|---|---:|---:|
| mean_ms |  |  |
| p50_ms |  |  |
| p95_ms |  |  |
| p99_ms |  |  |
| throughput_gb_s |  |  |
| checksum_real |  |  |
| checksum_imag |  |  |

## Threshold Evaluation
| Metric | Value | Target | Status |
|---|---:|---:|---|
| IPC |  | >= 1.5 |  |
| L1 miss rate |  | <= 5% |  |
| L2 miss rate |  | <= 15% |  |
| L3 miss rate |  | <= 20% |  |
| Branch miss rate |  | <= 1% |  |
| Major faults |  | == 0 |  |
| SIMD 256b ratio |  | >= 90% |  |
| p99 latency (ms) |  | <= 2.0 |  |
| Determinism drift |  | <= 3% |  |
| Checksum match |  | == true |  |

## Unsupported / Skipped Counters
- 

## Notes
- Record constraints (permissions, unavailable PMU events, or runner limitations).
- Keep this stage non-gating unless explicitly promoted later.
