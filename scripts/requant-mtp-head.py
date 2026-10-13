#!/usr/bin/env python3
"""Requant the NextN/MTP draft head (blk.<N>) of a qwen35 split GGUF from Q4_0 to Q4_K
(the eh_proj tensor to Q8_0), leaving every other tensor and all metadata byte-identical.

Why: under split decode + MTP the Mac runs blk.<N> as the draft head. After each round it
catches up on the verified rows in 2-8-row batches. Q4_0 isn't REGFED-eligible, so those batches
run through mul_mv_ext (~35 ms at 8 rows); Q4_K runs mul_mm_rf_q4_K_f32 (~7-9 ms). eh_proj is the
one tensor Q4_K fits badly (its super-block sub-scales), so it goes to Q8_0 (+~25 MiB, still rf).

Why surgery and not one llama-quantize call: touching only blk.<N> while keeping every other
tensor needs the rest pinned to their IQ2_XS types, and IQ2_XS pins require an imatrix. Targeting
Q4_K_M needs no imatrix, so this quantizes a throwaway scratch to Q4_K_M (which also re-encodes the
trunk, discarded) and splices blk.<N> back into a copy of the original. general.file_type stays
IQ2_XS, so the split connect's desc/ftype check still accepts the phone.

  requant-mtp-head.py SRC.gguf OUT.gguf [--quantize path/to/llama-quantize]

Cost: one full-file requant to a scratch (~2x the source on disk, deleted after) plus the splice.
run with a python3 that has numpy; needs a built llama-quantize for the Q4_K encoding.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def nextn_indices(r: "gguf.GGUFReader", kv) -> range:
    n_all = r.get_field(kv("block_count")).contents()
    f = r.get_field(kv("nextn_predict_layers"))
    n_nextn = f.contents() if f else 0
    if n_nextn <= 0:
        sys.exit("no NextN/MTP block (nextn_predict_layers is 0 or missing) — nothing to requant")
    return range(n_all - n_nextn, n_all)


def head_weight_pins(r, nextn: range, head_type: str, eh_proj_type: str) -> dict[str, str]:
    # Only the 2D matmul weights (attn_*, ffn_*, nextn.eh_proj). 1D norm weights are F32 and stay put.
    pins: dict[str, str] = {}
    for t in r.tensors:
        m = BLK.match(t.name)
        if m and int(m.group(1)) in nextn and t.name.endswith(".weight") and len(t.data.shape) >= 2:
            pins[t.name] = eh_proj_type if "eh_proj" in t.name else head_type
    if not pins:
        sys.exit("found no 2D blk.<NextN>.*.weight tensors to pin")
    return pins


def splice(src: str, scratch: str, out: str, arch: str, pin_names: set[str]) -> None:
    rs = gguf.GGUFReader(src)
    rq = gguf.GGUFReader(scratch)
    q_pinned = {t.name: t for t in rq.tensors if t.name in pin_names}
    missing = pin_names - q_pinned.keys()
    if missing:
        sys.exit(f"scratch is missing pinned tensors: {sorted(missing)}")

    w = gguf.GGUFWriter(out, arch)
    for f in rs.fields.values():                       # all metadata from the ORIGINAL (file_type stays)
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        val_type = f.types[0]
        sub_type = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), val_type, sub_type=sub_type)

    # original tensor order; swap ONLY the pinned matmul weights for the scratch's requantized ones
    ordered = [q_pinned.get(t.name, t) for t in rs.tensors]
    for t in ordered:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in ordered:
        w.write_tensor_data(t.data, tensor_endianess=rs.endianess)
    w.close()


def verify(src: str, out: str, pins: dict[str, str]) -> None:
    rs = gguf.GGUFReader(src)
    ro = gguf.GGUFReader(out)
    src_b = {t.name: t.data.tobytes() for t in rs.tensors}
    bad = []
    for t in ro.tensors:
        if t.name in pins:
            exp = gguf.GGMLQuantizationType[pins[t.name].upper()]
            if t.tensor_type != exp:
                bad.append(f"{t.name}: type {t.tensor_type.name} != {exp.name}")
        elif t.data.tobytes() != src_b.get(t.name):
            bad.append(f"{t.name}: bytes changed (should be untouched)")
    if [t.name for t in ro.tensors] != [t.name for t in rs.tensors]:
        bad.append("tensor order or set changed")
    fs = rs.get_field("general.file_type")
    fo = ro.get_field("general.file_type")
    fsv = fs.contents() if fs else None
    fov = fo.contents() if fo else None
    if fsv != fov:
        bad.append(f"general.file_type changed {fsv} -> {fov} (phone would refuse)")
    if bad:
        sys.exit("VERIFY FAILED:\n  " + "\n  ".join(bad))
    print(f"verify OK: {len(pins)} head weights requantized, every other tensor byte-identical, "
          f"general.file_type unchanged ({fov}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--quantize", default="llama-quantize",
                    help="path to a built llama-quantize (default: on PATH)")
    ap.add_argument("--head-type", default="q4_k", help="type for blk.<NextN> weights (default q4_k)")
    ap.add_argument("--eh-proj-type", default="q8_0", help="type for the eh_proj weight (default q8_0)")
    ap.add_argument("--scratch", help="scratch Q4_K_M file path (default OUT.scratch.gguf)")
    ap.add_argument("--keep-scratch", action="store_true", help="don't delete the scratch file")
    ap.add_argument("--dry-run", action="store_true", help="print the pins and the llama-quantize command, do nothing")
    args = ap.parse_args()

    r = gguf.GGUFReader(args.src)
    arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    kv = lambda k: f"{arch}.{k}"  # noqa: E731
    nextn = nextn_indices(r, kv)
    pins = head_weight_pins(r, nextn, args.head_type, args.eh_proj_type)

    scratch = args.scratch or (args.out + ".scratch.gguf")
    pins_path = args.out + ".pins.txt"
    with open(pins_path, "w") as fh:
        for name, typ in pins.items():
            fh.write(f"^{re.escape(name)}$={typ}\n")

    cmd = [args.quantize, "--allow-requantize", "--tensor-type-file", pins_path,
           args.src, scratch, "Q4_K_M"]
    print("pins (" + pins_path + "):")
    for name, typ in pins.items():
        print(f"  {name} -> {typ}")
    print("scratch requant:", " ".join(cmd))
    if args.dry_run:
        print("--dry-run: stopping before llama-quantize")
        return

    subprocess.run(cmd, check=True)
    splice(args.src, scratch, args.out, arch, set(pins))
    verify(args.src, args.out, pins)
    if not args.keep_scratch:
        Path(scratch).unlink(missing_ok=True)
        Path(pins_path).unlink(missing_ok=True)
    print(f"{args.out}: blk.{nextn.start} requantized; metadata and all other tensors unchanged.")


if __name__ == "__main__":
    main()
