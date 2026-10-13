#!/usr/bin/env python3
"""Two jobs for a qwen35 GGUF under split decode.

TAIL (default): write the tail for the phone/iPad — decoder layers [L, n_trunk) renumbered to
[0, n_trunk-L), plus output_norm / output / token_embd (the loader requires token_embd, and tied-output
models use it as the lm_head). The NextN/MTP block(s) are dropped by default (the tail never decodes).
The HEAD is not written: the Mac runs the full GGUF with llama_set_layer_range(ctx, 0, L), which shares
the weight pages with the decode context.

  split-gguf.py IN.gguf OUT.gguf -L 48 [--keep-mtp]

MAC REORDER (--reorder-mac): rewrite the FULL GGUF for the Mac with the tensors reordered to
output_norm, output, blk.<NextN>, blk.0..N-1, token_embd. Same tensors, same metadata, new layout.
Under split decode with MTP the Mac keeps blk.<NextN> (the draft head) and output; with mmap,
llama-model.cpp wraps each device's tensors in one Metal buffer spanning that device's first tensor
in the file to its last. In converter order output sits near the start and blk.64 at the very end, so
the Mac's buffer covers the whole file and an 8 GB Mac OOMs at startup. This order puts the head group
(output_norm, output, blk.64) contiguous at the front, so the Mac buffer is 1.52 GiB at L=4 / 3.24 GiB
at L=20 instead of the whole 8.5 GiB. No tensor bytes change (~47 s, another ~8.5 GiB of disk once).

  split-gguf.py IN.gguf OUT.gguf --reorder-mac

run with a python3 that has numpy
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def reorder_mac(r: "gguf.GGUFReader", dst: str, arch: str, n_trunk: int) -> None:
    """Write the full model with tensors in Mac-contiguous order; KV and tensor bytes unchanged."""
    w = gguf.GGUFWriter(dst, arch)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        val_type = f.types[0]
        sub_type = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), val_type, sub_type=sub_type)

    out_norm, out_w, tok_embd, other = [], [], [], []
    nextn: dict[int, list] = {}
    trunk: dict[int, list] = {}
    for t in r.tensors:
        m = BLK.match(t.name)
        if m:
            il = int(m.group(1))
            (nextn if il >= n_trunk else trunk).setdefault(il, []).append(t)
        elif t.name.startswith("output_norm"):
            out_norm.append(t)
        elif t.name == "output.weight":
            out_w.append(t)
        elif t.name == "token_embd.weight":
            tok_embd.append(t)
        else:
            other.append(t)

    ordered: list = []
    ordered += out_norm                                   # output_norm first
    ordered += out_w                                      # then output (lm_head), if present
    for il in sorted(nextn):                              # the NextN/MTP block(s): blk.64 ...
        ordered += nextn[il]
    for il in sorted(trunk):                              # the trunk: blk.0 .. blk.(n_trunk-1)
        ordered += trunk[il]
    ordered += other                                      # anything unclassified, before token_embd
    ordered += tok_embd                                   # token_embd last (keeps the head group at the front)

    if len(ordered) != len(r.tensors):
        sys.exit(f"reorder dropped/added tensors: {len(ordered)} vs {len(r.tensors)}")

    total = 0
    for t in ordered:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += t.n_bytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in ordered:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    head = ", ".join(t.name for t in (out_norm + out_w)) or "(none)"
    print(f"{dst}: reordered {len(ordered)} tensors, {total / 1e9:.2f} GB. "
          f"head group first ({head}), token_embd last.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("-L", type=int, help="first tail layer (multiple of full_attention_interval); required unless --reorder-mac")
    ap.add_argument("--reorder-mac", action="store_true",
                    help="rewrite the FULL model reordered for the Mac (output_norm, output, blk.<NextN>, blk.0..N-1, "
                         "token_embd) so an 8 GB Mac doesn't OOM under split decode + MTP. Ignores the tail flags.")
    ap.add_argument("--keep-mtp", action="store_true", help="keep the NextN/MTP block(s) (renumbered after the tail trunk)")
    ap.add_argument("--no-head", action="store_true",
                    help="drop token_embd and output (1.1 GB on the 27B): the worker gets residuals, never tokens, and the "
                         "Mac computes the logits (split prefill inside llama_decode never asks the worker for logits)")
    ap.add_argument("--no-ffn", action="store_true",
                    help="drop the trunk FFN weights (ffn_gate/up/down): the worker runs the FFN elsewhere (the ANE, "
                         "llama_set_ffn_offload) and Metal would otherwise keep ~142 MB/layer of unused weights wired")
    ap.add_argument("--no-ffn-first", type=int, default=0, metavar="N",
                    help="drop the FFN weights of only the first N tail layers (their FFN runs on the ANE: Sidecar tailane DIR L N);"
                         " the rest keep theirs on the GPU. The A18 can't wire ANE FFNs for all 12 layers of an L=52 tail")
    args = ap.parse_args()

    r = gguf.GGUFReader(args.src)
    arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    kv = lambda k: f"{arch}.{k}"  # noqa: E731

    n_all = r.get_field(kv("block_count")).contents()
    f_nextn = r.get_field(kv("nextn_predict_layers"))
    n_nextn = f_nextn.contents() if f_nextn else 0
    n_trunk = n_all - n_nextn

    if args.reorder_mac:
        if args.L is not None or args.keep_mtp or args.no_head or args.no_ffn or args.no_ffn_first:
            sys.exit("--reorder-mac writes the full model; don't combine it with -L or the tail flags")
        reorder_mac(r, args.dst, arch, n_trunk)
        return

    if args.L is None:
        sys.exit("-L is required for the tail (or pass --reorder-mac to rewrite the Mac file)")

    f_int = r.get_field(kv("full_attention_interval"))
    interval = f_int.contents() if f_int else 4
    L = args.L
    if not (0 < L < n_trunk) or L % interval != 0:
        sys.exit(f"L={L} must be a multiple of {interval} in (0, {n_trunk})")

    n_tail_trunk = n_trunk - L
    n_tail_nextn = n_nextn if args.keep_mtp else 0

    w = gguf.GGUFWriter(args.dst, arch)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        val_type = f.types[0]
        sub_type = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = f.contents()
        if f.name == kv("block_count"):
            val = n_tail_trunk + n_tail_nextn
        elif f.name == kv("nextn_predict_layers"):
            if not args.keep_mtp:
                continue
        elif f.name == gguf.Keys.General.NAME:
            val = f"{val} [tail L={L}]"
        elif isinstance(val, list) and len(val) in (n_all, n_trunk) and not f.name.startswith("tokenizer."):
            # per-layer array (e.g. attention.recurrent_layers, which newer converters write): the loader requires its length
            # to equal block_count, so keep the tail's trunk layers, then the NextN entries if those blocks are kept
            val = val[L:n_trunk] + (val[n_trunk:] if args.keep_mtp else [])
        w.add_key_value(f.name, val, val_type, sub_type=sub_type)
    w.add_uint32("split.layer_start", L)
    w.add_uint32("split.n_layer_full", n_trunk)
    if args.no_ffn or args.no_ffn_first:
        w.add_uint32("split.no_ffn", 1)
    if args.no_head:
        w.add_uint32("split.no_head", 1)

    keep = []
    for t in r.tensors:
        m = BLK.match(t.name)
        if not m:
            if args.no_head and t.name in ("token_embd.weight", "output.weight"):
                continue
            keep.append((t.name, t))
            continue
        il, rest = int(m.group(1)), m.group(2)
        if il < L:
            continue
        if (args.no_ffn or il < L + args.no_ffn_first) and il < n_trunk and rest in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            continue
        if il >= n_trunk:
            if not args.keep_mtp:
                continue
            new_il = n_tail_trunk + (il - n_trunk)
        else:
            new_il = il - L
        keep.append((f"blk.{new_il}.{rest}", t))

    total = 0
    for name, t in keep:
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += t.n_bytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, t in keep:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f"{args.dst}: tail layers [{L}, {n_trunk}) -> block_count {n_tail_trunk + n_tail_nextn}, "
          f"{len(keep)} tensors, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
