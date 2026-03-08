#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVE_SCRIPT="$SCRIPT_DIR/resolve_events.sh"
COLLECT_SCRIPT="$SCRIPT_DIR/collect_artifacts.sh"

BENCH_BIN="./build/basalt_bench"
OUT_ROOT="$SCRIPT_DIR/output"
SIZE=1024
SHOTS=100
WARMUP=5
SEED=1337
PERF_REPEATS=3

usage() {
  cat <<EOF
Usage: run_stage1.sh [options]
  --bench-bin <path>      Path to basalt_bench binary (default: ./build/basalt_bench)
  --out-root <path>       Output root directory (default: tools/profiling/output)
  --size <N>              Grid size N (NxN, power of two), default: 1024
  --shots <N>             Timed shots, default: 100
  --warmup <N>            Warmup shots, default: 5
  --seed <N>              RNG seed, default: 1337
  --perf-repeats <N>      perf stat repeat count, default: 3
  --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-bin) BENCH_BIN="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --perf-repeats) PERF_REPEATS="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Benchmark binary not found or not executable: $BENCH_BIN" >&2
  exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUT_ROOT/stage1_${RUN_ID}"
RAW_DIR="$RUN_DIR/raw"
mkdir -p "$RAW_DIR"

bash "$RESOLVE_SCRIPT" > "$RUN_DIR/resolved_events.env"
source "$RUN_DIR/resolved_events.env"

echo "Running Stage 1 profiling in: $RUN_DIR"

run_bench() {
  local label="$1"
  "$BENCH_BIN" \
    --size "$SIZE" \
    --shots "$SHOTS" \
    --warmup "$WARMUP" \
    --seed "$SEED" \
    --json-out "$RUN_DIR/${label}.json" \
    > "$RAW_DIR/${label}.stdout.log" \
    2> "$RAW_DIR/${label}.stderr.log"
}

run_bench "benchmark_primary"
run_bench "benchmark_repeat"

run_perf_group() {
  local group="$1"
  local events="$2"
  local status_file="$RAW_DIR/perf_${group}.status"
  local csv_file="$RAW_DIR/perf_${group}.csv"
  local stdout_file="$RAW_DIR/perf_${group}.stdout.log"
  local error_file="$RAW_DIR/perf_${group}.error.log"

  if [[ "${PERF_AVAILABLE:-0}" -ne 1 ]]; then
    echo "skipped: perf unavailable" > "$status_file"
    return 0
  fi
  if [[ -z "$events" ]]; then
    echo "skipped: no supported counters resolved" > "$status_file"
    return 0
  fi

  set +e
  perf stat -x, -r "$PERF_REPEATS" -e "$events" -- \
    "$BENCH_BIN" \
      --size "$SIZE" \
      --shots "$SHOTS" \
      --warmup "$WARMUP" \
      --seed "$SEED" \
      --json-out "$RAW_DIR/perf_${group}_bench.json" \
      > "$stdout_file" \
      2> "$csv_file"
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "failed: perf exited with code $rc" > "$status_file"
    {
      echo "perf command failed for group: $group"
      echo "events: $events"
      echo "exit_code: $rc"
    } > "$error_file"
  else
    echo "ok" > "$status_file"
  fi
}

run_perf_group "ipc" "$GROUP_IPC"
run_perf_group "cache" "$GROUP_CACHE"
run_perf_group "dram" "$GROUP_DRAM"
run_perf_group "simd" "$GROUP_SIMD"
run_perf_group "branch" "$GROUP_BRANCH"
run_perf_group "faults" "$GROUP_FAULTS"

bash "$COLLECT_SCRIPT" "$RUN_DIR"

echo "STAGE1_RUN_DIR=$RUN_DIR"
