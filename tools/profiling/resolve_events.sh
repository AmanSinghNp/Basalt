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

discover_perf_bin() {
  local forced="${PERF_BIN:-}"
  if [[ -n "$forced" && -x "$forced" ]]; then
    printf "%s" "$forced"
    return 0
  fi

  local in_path
  in_path="$(command -v perf 2>/dev/null || true)"
  if [[ -n "$in_path" && -x "$in_path" ]]; then
    printf "%s" "$in_path"
    return 0
  fi

  local uname_r
  uname_r="$(uname -r 2>/dev/null || true)"
  local candidates=(
    "/usr/lib/linux-tools/${uname_r}/perf"
    "/usr/lib/linux-tools-${uname_r}/perf"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]]; then
      printf "%s" "$c"
      return 0
    fi
  done

  local globbed
  globbed="$(ls /usr/lib/linux-tools*/perf 2>/dev/null | head -n 1 || true)"
  if [[ -n "$globbed" && -x "$globbed" ]]; then
    printf "%s" "$globbed"
    return 0
  fi

  printf ""
  return 1
}

declare -a UNSUPPORTED=()
declare -A RESOLVED=()
PERF_AVAILABLE=0
PERF_LIST_AVAILABLE=0
PERF_BIN_RESOLVED="$(discover_perf_bin || true)"

if [[ -n "$PERF_BIN_RESOLVED" ]] && "$PERF_BIN_RESOLVED" --version >/dev/null 2>&1; then
  PERF_AVAILABLE=1
fi

PERF_LIST_TEXT=""
if [[ "$PERF_AVAILABLE" -eq 1 ]]; then
  PERF_LIST_TEXT="$("$PERF_BIN_RESOLVED" list 2>/dev/null || true)"
  if [[ -n "$PERF_LIST_TEXT" ]]; then
    PERF_LIST_AVAILABLE=1
  fi
fi

has_event() {
  local candidate="$1"
  if [[ "$PERF_AVAILABLE" -ne 1 ]]; then
    return 1
  fi
  if [[ "$PERF_LIST_AVAILABLE" -ne 1 ]]; then
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

resolve_fallback_group() {
  local out=()
  local candidate
  for candidate in "$@"; do
    if [[ "$PERF_AVAILABLE" -ne 1 ]]; then
      continue
    fi
    if [[ "$PERF_LIST_AVAILABLE" -eq 1 ]]; then
      if has_event "$candidate"; then
        out+=("$candidate")
      fi
    else
      out+=("$candidate")
    fi
  done
  join_csv "${out[@]}"
}

select_group() {
  local canonical="$1"
  local fallback="$2"
  local source_var="$3"
  local value_var="$4"

  if [[ -n "$canonical" ]]; then
    printf -v "$value_var" "%s" "$canonical"
    printf -v "$source_var" "%s" "canonical"
    return 0
  fi
  if [[ -n "$fallback" ]]; then
    printf -v "$value_var" "%s" "$fallback"
    printf -v "$source_var" "%s" "fallback"
    return 0
  fi
  printf -v "$value_var" "%s" ""
  printf -v "$source_var" "%s" "empty"
  return 0
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

CANONICAL_IPC="$(join_csv "${RESOLVED[instructions]}" "${RESOLVED[cycles]}" "${RESOLVED[cpu_clock]}")"
CANONICAL_CACHE="$(join_csv "${RESOLVED[l1_hit]}" "${RESOLVED[l1_miss]}" "${RESOLVED[l2_hit]}" "${RESOLVED[l2_miss]}" "${RESOLVED[l3_hit]}" "${RESOLVED[l3_miss]}")"
CANONICAL_DRAM="$(join_csv "${RESOLVED[l3_miss]}" "${RESOLVED[offcore_rd]}" "${RESOLVED[offcore_outstanding]}")"
CANONICAL_SIMD="$(join_csv "${RESOLVED[simd_128]}" "${RESOLVED[simd_256]}" "${RESOLVED[simd_scalar]}")"
CANONICAL_BRANCH="$(join_csv "${RESOLVED[branches]}" "${RESOLVED[branch_misses]}")"
CANONICAL_FAULTS="$(join_csv "${RESOLVED[major_faults]}" "${RESOLVED[minor_faults]}" "${RESOLVED[page_faults]}")"

FALLBACK_IPC="$(resolve_fallback_group task-clock cpu-clock)"
FALLBACK_CACHE="$(resolve_fallback_group page-faults minor-faults major-faults)"
FALLBACK_DRAM="$(resolve_fallback_group page-faults minor-faults major-faults)"
FALLBACK_SIMD="$(resolve_fallback_group task-clock cpu-clock)"
FALLBACK_BRANCH="$(resolve_fallback_group context-switches cpu-migrations)"
FALLBACK_FAULTS="$(resolve_fallback_group page-faults minor-faults major-faults)"

GROUP_IPC=""
GROUP_CACHE=""
GROUP_DRAM=""
GROUP_SIMD=""
GROUP_BRANCH=""
GROUP_FAULTS=""
GROUP_IPC_SOURCE=""
GROUP_CACHE_SOURCE=""
GROUP_DRAM_SOURCE=""
GROUP_SIMD_SOURCE=""
GROUP_BRANCH_SOURCE=""
GROUP_FAULTS_SOURCE=""

select_group "$CANONICAL_IPC" "$FALLBACK_IPC" GROUP_IPC_SOURCE GROUP_IPC
select_group "$CANONICAL_CACHE" "$FALLBACK_CACHE" GROUP_CACHE_SOURCE GROUP_CACHE
select_group "$CANONICAL_DRAM" "$FALLBACK_DRAM" GROUP_DRAM_SOURCE GROUP_DRAM
select_group "$CANONICAL_SIMD" "$FALLBACK_SIMD" GROUP_SIMD_SOURCE GROUP_SIMD
select_group "$CANONICAL_BRANCH" "$FALLBACK_BRANCH" GROUP_BRANCH_SOURCE GROUP_BRANCH
select_group "$CANONICAL_FAULTS" "$FALLBACK_FAULTS" GROUP_FAULTS_SOURCE GROUP_FAULTS

UNSUPPORTED_EVENTS=""
if [[ "${#UNSUPPORTED[@]}" -gt 0 ]]; then
  UNSUPPORTED_EVENTS="$(IFS=';'; echo "${UNSUPPORTED[*]}")"
fi

printf "PERF_BIN='%s'\n" "$PERF_BIN_RESOLVED"
printf "PERF_AVAILABLE='%s'\n" "$PERF_AVAILABLE"
printf "PERF_LIST_AVAILABLE='%s'\n" "$PERF_LIST_AVAILABLE"
printf "GROUP_IPC='%s'\n" "$GROUP_IPC"
printf "GROUP_CACHE='%s'\n" "$GROUP_CACHE"
printf "GROUP_DRAM='%s'\n" "$GROUP_DRAM"
printf "GROUP_SIMD='%s'\n" "$GROUP_SIMD"
printf "GROUP_BRANCH='%s'\n" "$GROUP_BRANCH"
printf "GROUP_FAULTS='%s'\n" "$GROUP_FAULTS"
printf "GROUP_IPC_SOURCE='%s'\n" "$GROUP_IPC_SOURCE"
printf "GROUP_CACHE_SOURCE='%s'\n" "$GROUP_CACHE_SOURCE"
printf "GROUP_DRAM_SOURCE='%s'\n" "$GROUP_DRAM_SOURCE"
printf "GROUP_SIMD_SOURCE='%s'\n" "$GROUP_SIMD_SOURCE"
printf "GROUP_BRANCH_SOURCE='%s'\n" "$GROUP_BRANCH_SOURCE"
printf "GROUP_FAULTS_SOURCE='%s'\n" "$GROUP_FAULTS_SOURCE"
printf "UNSUPPORTED_EVENTS='%s'\n" "$UNSUPPORTED_EVENTS"

for key in "${!RESOLVED[@]}"; do
  printf "EVENT_%s='%s'\n" "${key^^}" "${RESOLVED[$key]}"
done
