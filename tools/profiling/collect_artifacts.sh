#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: collect_artifacts.sh <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

python3 - "$RUN_DIR" <<'PY'
import csv
import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone

run_dir = sys.argv[1]
raw_dir = os.path.join(run_dir, "raw")
resolved_env_path = os.path.join(run_dir, "resolved_events.env")

def parse_env(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip("'").strip('"')
            out[k.strip()] = v
    return out

def safe_load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_perf_csv(path, group):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            value_s, unit, event = parts[0], parts[1], parts[2]
            if not event:
                continue

            low = value_s.lower()
            status = "counted"
            value = None
            if "not supported" in low:
                status = "not_supported"
            elif "not counted" in low:
                status = "not_counted"
            else:
                cleaned = value_s.replace(" ", "").replace("\xa0", "")
                cleaned = cleaned.replace(",", "")
                try:
                    value = float(cleaned)
                except ValueError:
                    status = "parse_error"

            records.append({
                "group": group,
                "event": event,
                "value": value,
                "unit": unit,
                "status": status,
                "source_file": os.path.basename(path),
            })
    return records

env = parse_env(resolved_env_path)

all_records = []
for perf_file in sorted(glob.glob(os.path.join(raw_dir, "perf_*.csv"))):
    base = os.path.basename(perf_file)
    group = base[len("perf_"):-len(".csv")]
    all_records.extend(parse_perf_csv(perf_file, group))

normalized_csv = os.path.join(run_dir, "normalized_metrics.csv")
with open(normalized_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["group", "event", "value", "unit", "status", "source_file"],
    )
    writer.writeheader()
    for rec in all_records:
        writer.writerow(rec)

aggregated = {}
for rec in all_records:
    if rec["status"] != "counted" or rec["value"] is None:
        continue
    aggregated.setdefault(rec["event"], []).append(rec["value"])

mean_metrics = {k: statistics.mean(v) for k, v in aggregated.items() if v}

def metric(*names):
    for n in names:
        if n in mean_metrics:
            return mean_metrics[n]
    return None

instructions = metric("instructions")
cycles = metric("cycles")
branches = metric("branches")
branch_misses = metric("branch-misses")
major_faults = metric("major-faults")

l1_hit = metric("mem_load_retired.l1_hit", "L1-dcache-loads")
l1_miss = metric("mem_load_retired.l1_miss", "L1-dcache-load-misses")
l2_hit = metric("mem_load_retired.l2_hit", "l2_rqsts.references")
l2_miss = metric("mem_load_retired.l2_miss", "l2_rqsts.miss")
l3_hit = metric("mem_load_retired.l3_hit", "LLC-loads")
l3_miss = metric("mem_load_retired.l3_miss", "LLC-load-misses")

simd_128 = metric("fp_arith_inst_retired.128b_packed_single")
simd_256 = metric("fp_arith_inst_retired.256b_packed_single")
simd_scalar = metric("fp_arith_inst_retired.scalar_single")

ipc = None
if instructions is not None and cycles and cycles > 0:
    ipc = instructions / cycles

def miss_rate(hit, miss):
    if hit is None or miss is None:
        return None
    denom = hit + miss
    if denom <= 0:
        return None
    return miss / denom

l1_miss_rate = miss_rate(l1_hit, l1_miss)
l2_miss_rate = miss_rate(l2_hit, l2_miss)
l3_miss_rate = miss_rate(l3_hit, l3_miss)

branch_miss_rate = None
if branches is not None and branch_misses is not None and branches > 0:
    branch_miss_rate = branch_misses / branches

simd_ratio = None
if simd_128 is not None and simd_256 is not None and simd_scalar is not None:
    denom = simd_128 + simd_256 + simd_scalar
    if denom > 0:
        simd_ratio = simd_256 / denom

primary = safe_load_json(os.path.join(run_dir, "benchmark_primary.json")) or {}
repeat = safe_load_json(os.path.join(run_dir, "benchmark_repeat.json")) or {}

p_summary = primary.get("summary", {})
r_summary = repeat.get("summary", {})
p_cfg = primary.get("config", {})

p99_ms = p_summary.get("p99_ms")
primary_mean = p_summary.get("mean_ms")
repeat_mean = r_summary.get("mean_ms")
primary_checksum_real = (primary.get("determinism") or {}).get("checksum_real")
repeat_checksum_real = (repeat.get("determinism") or {}).get("checksum_real")
primary_checksum_imag = (primary.get("determinism") or {}).get("checksum_imag")
repeat_checksum_imag = (repeat.get("determinism") or {}).get("checksum_imag")

determinism_drift = None
if primary_mean is not None and repeat_mean is not None and primary_mean > 0:
    determinism_drift = abs(repeat_mean - primary_mean) / primary_mean

checksum_match = None
if primary_checksum_real and repeat_checksum_real and primary_checksum_imag and repeat_checksum_imag:
    checksum_match = (
        primary_checksum_real == repeat_checksum_real
        and primary_checksum_imag == repeat_checksum_imag
    )

threshold_rows = []

def eval_threshold(name, value, op, target):
    if value is None:
        return (name, "N/A", op, target, "N/A")
    status = "FAIL"
    if op == ">=" and value >= target:
        status = "PASS"
    elif op == "<=" and value <= target:
        status = "PASS"
    elif op == "==" and value == target:
        status = "PASS"
    return (name, value, op, target, status)

threshold_rows.append(eval_threshold("IPC", ipc, ">=", 1.5))
threshold_rows.append(eval_threshold("L1 miss rate", l1_miss_rate, "<=", 0.05))
threshold_rows.append(eval_threshold("L2 miss rate", l2_miss_rate, "<=", 0.15))
threshold_rows.append(eval_threshold("L3 miss rate", l3_miss_rate, "<=", 0.20))
threshold_rows.append(eval_threshold("Branch miss rate", branch_miss_rate, "<=", 0.01))
threshold_rows.append(eval_threshold("Major faults", major_faults, "==", 0.0))
threshold_rows.append(eval_threshold("SIMD 256b ratio", simd_ratio, ">=", 0.90))
threshold_rows.append(eval_threshold("p99 latency (ms)", p99_ms, "<=", 2.0))
threshold_rows.append(eval_threshold("Determinism drift", determinism_drift, "<=", 0.03))
if checksum_match is None:
    threshold_rows.append(("Checksum match", "N/A", "==", True, "N/A"))
else:
    threshold_rows.append(("Checksum match", checksum_match, "==", True, "PASS" if checksum_match else "FAIL"))

threshold_csv = os.path.join(run_dir, "stage1_thresholds.csv")
with open(threshold_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value", "operator", "target", "status"])
    for row in threshold_rows:
        writer.writerow(row)

unsupported = env.get("UNSUPPORTED_EVENTS", "")
unsupported_list = [x for x in unsupported.split(";") if x]

event_groups = {
    "IPC": (env.get("GROUP_IPC", ""), env.get("GROUP_IPC_SOURCE", "empty")),
    "Cache": (env.get("GROUP_CACHE", ""), env.get("GROUP_CACHE_SOURCE", "empty")),
    "DRAM": (env.get("GROUP_DRAM", ""), env.get("GROUP_DRAM_SOURCE", "empty")),
    "SIMD": (env.get("GROUP_SIMD", ""), env.get("GROUP_SIMD_SOURCE", "empty")),
    "Branch": (env.get("GROUP_BRANCH", ""), env.get("GROUP_BRANCH_SOURCE", "empty")),
    "Faults": (env.get("GROUP_FAULTS", ""), env.get("GROUP_FAULTS_SOURCE", "empty")),
}

def fmt_value(v, pct=False):
    if v is None or v == "N/A":
        return "N/A"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if pct:
            return f"{100.0 * v:.3f}%"
        return f"{v:.6f}"
    return str(v)

dossier_path = os.path.join(run_dir, "stage1_dossier.md")
with open(dossier_path, "w", encoding="utf-8") as f:
    f.write("# Basalt Stage 1 Profiling Dossier\n\n")
    f.write(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n")
    f.write(f"- Run directory: `{run_dir}`\n")
    f.write(f"- perf binary: `{env.get('PERF_BIN', '') or 'N/A'}`\n")
    f.write(f"- perf available: `{env.get('PERF_AVAILABLE', '0')}`\n\n")
    f.write(f"- perf list available: `{env.get('PERF_LIST_AVAILABLE', '0')}`\n\n")

    f.write("## Workload Configuration\n\n")
    f.write(f"- size: `{p_cfg.get('size', 'N/A')}`\n")
    f.write(f"- shots: `{p_cfg.get('shots', 'N/A')}`\n")
    f.write(f"- warmup: `{p_cfg.get('warmup', 'N/A')}`\n")
    f.write(f"- seed: `{p_cfg.get('seed', 'N/A')}`\n\n")

    f.write("## Benchmark Summary\n\n")
    f.write("| Metric | Primary | Repeat |\n")
    f.write("|---|---:|---:|\n")
    f.write(f"| mean_ms | {fmt_value(primary_mean)} | {fmt_value(repeat_mean)} |\n")
    f.write(f"| p50_ms | {fmt_value(p_summary.get('p50_ms'))} | {fmt_value(r_summary.get('p50_ms'))} |\n")
    f.write(f"| p95_ms | {fmt_value(p_summary.get('p95_ms'))} | {fmt_value(r_summary.get('p95_ms'))} |\n")
    f.write(f"| p99_ms | {fmt_value(p_summary.get('p99_ms'))} | {fmt_value(r_summary.get('p99_ms'))} |\n")
    f.write(f"| throughput_gb_s | {fmt_value(p_summary.get('throughput_gb_s'))} | {fmt_value(r_summary.get('throughput_gb_s'))} |\n")
    f.write("\n")

    f.write("## Selected Counter Groups\n\n")
    f.write("| Group | Source | Selected Events |\n")
    f.write("|---|---|---|\n")
    for name, (events, source) in event_groups.items():
        f.write(f"| {name} | `{source}` | `{events if events else 'N/A'}` |\n")
    f.write("\n")

    f.write("## Threshold Evaluation\n\n")
    f.write("| Metric | Value | Target | Status |\n")
    f.write("|---|---:|---:|---|\n")
    for metric_name, value, op, target, status in threshold_rows:
        if "rate" in metric_name.lower() or "ratio" in metric_name.lower() or "drift" in metric_name.lower():
            value_fmt = fmt_value(value, pct=True if value not in ("N/A", None) else False)
            target_fmt = fmt_value(target, pct=True)
        else:
            value_fmt = fmt_value(value)
            target_fmt = fmt_value(target)
        f.write(f"| {metric_name} | {value_fmt} | {op} {target_fmt} | {status} |\n")
    f.write("\n")

    f.write("## Unsupported Canonical Events\n\n")
    if unsupported_list:
        for item in unsupported_list:
            f.write(f"- `{item}`\n")
    else:
        f.write("- None\n")
    f.write("\n")

unsupported_txt = os.path.join(run_dir, "unsupported_events.txt")
with open(unsupported_txt, "w", encoding="utf-8") as f:
    if unsupported_list:
        f.write("\n".join(unsupported_list) + "\n")
    else:
        f.write("none\n")
PY
