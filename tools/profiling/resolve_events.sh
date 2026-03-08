#!/usr/bin/env bash
set -euo pipefail

join_csv() {
  local out=""
  local item
  for item in "$@"; do
    if [[ -z "$item" ]]; then
      continue
    fi
    if [[ -z "$out" ]]; then
      out="$item"
    else
      out="$out,$item"
    fi
  done
  printf "%s" "$out"
}

declare -a UNSUPPORTED=()
declare -A RESOLVED=()
PERF_AVAILABLE=1

if ! command -v perf >/dev/null 2>&1; then
  PERF_AVAILABLE=0
  PERF_LIST_TEXT=""
else
  PERF_LIST_TEXT="$(perf list 2>/dev/null || true)"
  if [[ -z "$PERF_LIST_TEXT" ]]; then
    PERF_AVAILABLE=0
  fi
fi

has_event() {
  local candidate="$1"
  if [[ "$PERF_AVAILABLE" -ne 1 ]]; then
    return 1
  fi
  grep -Fqi "$candidate" <<<"$PERF_LIST_TEXT"
}

resolve_event() {
  local key="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if has_event "$candidate"; then
      RESOLVED["$key"]="$candidate"
      return 0
    fi
  done
  RESOLVED["$key"]=""
  UNSUPPORTED+=("$key")
  return 1
}

resolve_event "instructions" "instructions" || true
resolve_event "cycles" "cycles" || true
resolve_event "cpu_clock" "cpu-clock" || true

resolve_event "l1_hit" "mem_load_retired.l1_hit" "L1-dcache-loads" || true
resolve_event "l1_miss" "mem_load_retired.l1_miss" "L1-dcache-load-misses" || true
resolve_event "l2_hit" "mem_load_retired.l2_hit" "l2_rqsts.references" || true
resolve_event "l2_miss" "mem_load_retired.l2_miss" "l2_rqsts.miss" || true
resolve_event "l3_hit" "mem_load_retired.l3_hit" "LLC-loads" || true
resolve_event "l3_miss" "mem_load_retired.l3_miss" "LLC-load-misses" || true

resolve_event "offcore_rd" "offcore_requests.all_data_rd" || true
resolve_event "offcore_outstanding" "offcore_requests_outstanding.all_data_rd" || true

resolve_event "simd_128" "fp_arith_inst_retired.128b_packed_single" || true
resolve_event "simd_256" "fp_arith_inst_retired.256b_packed_single" || true
resolve_event "simd_scalar" "fp_arith_inst_retired.scalar_single" || true

resolve_event "branches" "branches" || true
resolve_event "branch_misses" "branch-misses" || true

resolve_event "major_faults" "major-faults" || true
resolve_event "minor_faults" "minor-faults" || true
resolve_event "page_faults" "page-faults" || true

GROUP_IPC="$(join_csv "${RESOLVED[instructions]}" "${RESOLVED[cycles]}" "${RESOLVED[cpu_clock]}")"
GROUP_CACHE="$(join_csv "${RESOLVED[l1_hit]}" "${RESOLVED[l1_miss]}" "${RESOLVED[l2_hit]}" "${RESOLVED[l2_miss]}" "${RESOLVED[l3_hit]}" "${RESOLVED[l3_miss]}")"
GROUP_DRAM="$(join_csv "${RESOLVED[l3_miss]}" "${RESOLVED[offcore_rd]}" "${RESOLVED[offcore_outstanding]}")"
GROUP_SIMD="$(join_csv "${RESOLVED[simd_128]}" "${RESOLVED[simd_256]}" "${RESOLVED[simd_scalar]}")"
GROUP_BRANCH="$(join_csv "${RESOLVED[branches]}" "${RESOLVED[branch_misses]}")"
GROUP_FAULTS="$(join_csv "${RESOLVED[major_faults]}" "${RESOLVED[minor_faults]}" "${RESOLVED[page_faults]}")"

UNSUPPORTED_EVENTS=""
if [[ "${#UNSUPPORTED[@]}" -gt 0 ]]; then
  UNSUPPORTED_EVENTS="$(IFS=';'; echo "${UNSUPPORTED[*]}")"
fi

printf "PERF_AVAILABLE='%s'\n" "$PERF_AVAILABLE"
printf "GROUP_IPC='%s'\n" "$GROUP_IPC"
printf "GROUP_CACHE='%s'\n" "$GROUP_CACHE"
printf "GROUP_DRAM='%s'\n" "$GROUP_DRAM"
printf "GROUP_SIMD='%s'\n" "$GROUP_SIMD"
printf "GROUP_BRANCH='%s'\n" "$GROUP_BRANCH"
printf "GROUP_FAULTS='%s'\n" "$GROUP_FAULTS"
printf "UNSUPPORTED_EVENTS='%s'\n" "$UNSUPPORTED_EVENTS"

for key in "${!RESOLVED[@]}"; do
  printf "EVENT_%s='%s'\n" "${key^^}" "${RESOLVED[$key]}"
done
