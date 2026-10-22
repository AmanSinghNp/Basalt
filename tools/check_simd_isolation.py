#!/usr/bin/env python3
"""Check that AVX/AVX2/FMA/AVX-512 instructions only appear in allowed functions.

Basalt's baseline is plain x86-64 (SSE2); wide-vector code lives only in
functions carrying the avx2/avx512 target attribute and is reached through a
runtime cpu_features() check. This script disassembles object files and
verifies it: any function that contains a VEX/EVEX-encoded instruction (mnemonic
starting with 'v', or a ymm/zmm/mask-register operand) must match the allowlist.

Usage: check_simd_isolation.py [--objdump PATH] [--list] FILE...
  FILE   object files (.o) or static libraries (.a) built from the Basalt sources
  --list print every offending function with the matched allowlist entry

Exit status: 0 when clean, 1 when a non-allowlisted function uses wide SIMD.
"""
import argparse
import re
import subprocess
import sys

# Demangled-name substrings of the functions that are allowed to use AVX2/FMA or
# AVX-512 (they all carry BASALT_TARGET_AVX2 / BASALT_TARGET_AVX512). Compiler
# suffixes such as .constprop.0 / .isra.0 / .part.0 and cold-section splits
# are matched by the substring.
ALLOWLIST = [
    # src/kernel/fft_avx.cpp
    "scale_avx(",
    "transpose8x8_regs(",
    "butterfly(",
    "first_three_stages_avx<",
    "fft_avx_core<",
    "fft_rows_avx2_impl(",
    # src/kernel/transpose_avx.cpp
    "transpose_8x8_avx_impl(",
    "transpose_tiled_avx2_impl(",
    # src/filter
    "apply_fk_filter_avx2(",
    "apply_fk_filter_avx512_impl(",
    # inline helpers from fast_math.hpp / fk_filter_impl.hpp / memory_hints.hpp
    "fast_exp_avx(",
    "fast_sigmoid_avx(",
    "make_band_consts(",
    "band_weight_ps(",
    "fk_weight_ps(",
    "absolute_ps(",
    "stream_store(",
]

FUNC_RE = re.compile(r"^[0-9a-fA-F]+ <(.+)>:$")
INSN_RE = re.compile(r"^\s+[0-9a-fA-F]+:\s+(\S+)(?:\s+(.*))?$")
WIDE_OPERAND_RE = re.compile(r"%[yz]mm\d+|%k[0-7]\b")


def is_wide(mnemonic: str, operands: str) -> bool:
    if mnemonic.startswith("v") and mnemonic not in ("verr", "verw"):
        return True
    if mnemonic.startswith("k") and mnemonic[1:2] in ("m", "o", "a", "n", "x", "s", "t", "u", "r"):
        # kmov*, kor*, kand*, ... (AVX-512 mask instructions)
        return True
    return bool(WIDE_OPERAND_RE.search(operands or ""))


def scan(path: str, objdump: str):
    """Yield (object_name, function, first_offending_instruction) per bad function."""
    out = subprocess.run(
        [objdump, "-d", "-C", "--no-show-raw-insn", path],
        check=True, capture_output=True, text=True,
    ).stdout
    obj = path
    func = None
    reported = set()
    for line in out.splitlines():
        m = re.match(r"^(\S+\.o):\s+file format", line)
        if m:
            obj = m.group(1)
            continue
        m = FUNC_RE.match(line)
        if m:
            func = m.group(1)
            continue
        m = INSN_RE.match(line)
        if m and func is not None:
            mnemonic, operands = m.group(1), m.group(2) or ""
            if is_wide(mnemonic, operands) and (obj, func) not in reported:
                reported.add((obj, func))
                yield obj, func, f"{mnemonic} {operands}".strip()


def allowed(func: str):
    for entry in ALLOWLIST:
        if entry in func:
            return entry
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--objdump", default="objdump")
    ap.add_argument("--list", action="store_true", help="also print allowed functions")
    args = ap.parse_args()

    bad = 0
    n_allowed = 0
    for path in args.files:
        for obj, func, insn in scan(path, args.objdump):
            entry = allowed(func)
            if entry is None:
                bad += 1
                print(f"LEAK  {obj}: {func}\n      first wide instruction: {insn}")
            else:
                n_allowed += 1
                if args.list:
                    print(f"ok    {obj}: {func}  [{entry}]")
    print(f"{n_allowed} allowlisted wide-SIMD function(s), {bad} leak(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
