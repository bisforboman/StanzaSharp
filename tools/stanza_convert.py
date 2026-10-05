#!/usr/bin/env python3
"""
Inspect and convert Stanza .pt checkpoints into a C#-friendly form.

  inspect <file.pt>                 print the structure (tensor shapes, config) of one checkpoint
  convert <file.pt|dir> --out DIR   write <name>.safetensors (all tensors) + <name>.json (everything else)

Every tensor in the checkpoint, wherever it sits in the nested structure, goes into the
.safetensors file under a unique key. The .json file mirrors the original structure, with
each tensor replaced by {"$tensor": key, "dtype": ..., "shape": [...]}, so the C# side can
walk the JSON for config/vocab and pull weights from the safetensors file by key.

Non-JSON types are tagged rather than lost:
  {"$tuple": [...]}, {"$set": [...]}, {"$dict": [[key, value], ...]} (dicts with non-string keys),
  {"$bytes": "<base64>"}, {"$float": "nan"|"inf"|"-inf"}, {"$repr": "...", "$type": "..."} (unknown objects)

Requires: numpy, and torch for reading .pt files. No safetensors package needed.
"""
import argparse
import base64
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# Training-only state that inference never needs; skipped at the top level by default.
DEFAULT_SKIP = ("optimizer", "scheduler")

NP_TO_ST = {
    np.dtype("float64"): "F64", np.dtype("float32"): "F32", np.dtype("float16"): "F16",
    np.dtype("int64"): "I64", np.dtype("int32"): "I32", np.dtype("int16"): "I16",
    np.dtype("int8"): "I8", np.dtype("uint8"): "U8", np.dtype("bool"): "BOOL",
}


def is_torch_tensor(obj):
    return type(obj).__module__.startswith("torch") and hasattr(obj, "detach") and hasattr(obj, "numpy")


def to_numpy(obj):
    """torch.Tensor or np.ndarray -> contiguous little-endian np.ndarray."""
    if is_torch_tensor(obj):
        t = obj.detach().cpu()
        if str(t.dtype) == "torch.bfloat16":  # numpy has no bfloat16
            print("  note: bfloat16 tensor converted to float32", file=sys.stderr)
            t = t.float()
        arr = t.numpy()
    else:
        arr = obj
    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    if arr.dtype not in NP_TO_ST:
        raise TypeError(f"unsupported tensor dtype {arr.dtype}")
    return arr


def is_array(obj):
    return is_torch_tensor(obj) or isinstance(obj, np.ndarray)


class Splitter:
    """Walks a checkpoint, collecting tensors and producing a JSON-safe mirror."""

    def __init__(self):
        self.tensors = {}  # key -> np.ndarray

    def _key(self, path):
        key = ".".join(path) or "root"
        base, n = key, 1
        while key in self.tensors:
            n += 1
            key = f"{base}#{n}"
        return key

    def walk(self, obj, path=()):
        if is_array(obj):
            arr = to_numpy(obj)
            key = self._key(path)
            self.tensors[key] = arr
            return {"$tensor": key, "dtype": NP_TO_ST[arr.dtype], "shape": list(arr.shape)}
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else {"$float": str(obj)}
        if isinstance(obj, bytes):
            return {"$bytes": base64.b64encode(obj).decode("ascii")}
        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj):
                return {k: self.walk(v, path + (k,)) for k, v in obj.items()}
            return {"$dict": [[self.walk(k, path + (f"k{i}",)), self.walk(v, path + (str(k),))]
                              for i, (k, v) in enumerate(obj.items())]}
        if isinstance(obj, list):
            return [self.walk(v, path + (str(i),)) for i, v in enumerate(obj)]
        if isinstance(obj, tuple):
            return {"$tuple": [self.walk(v, path + (str(i),)) for i, v in enumerate(obj)]}
        if isinstance(obj, (set, frozenset)):
            items = sorted(obj, key=repr)  # deterministic output
            return {"$set": [self.walk(v, path + (str(i),)) for i, v in enumerate(items)]}
        print(f"  warning: unknown type {type(obj).__name__} at {'.'.join(path)}, stored as repr",
              file=sys.stderr)
        return {"$repr": repr(obj), "$type": f"{type(obj).__module__}.{type(obj).__qualname__}"}


def write_safetensors(tensors, path, metadata=None):
    """Minimal safetensors writer: u64 header length, JSON header, raw little-endian data."""
    header, offset = {}, 0
    for key, arr in tensors.items():
        header[key] = {"dtype": NP_TO_ST[arr.dtype], "shape": list(arr.shape),
                       "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hbytes += b" " * (-len(hbytes) % 8)  # pad so the data section is 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for arr in tensors.values():
            f.write(arr.tobytes(order="C"))


def load_checkpoint(path, allow_unsafe):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not allow_unsafe:
            raise SystemExit(
                f"{path}: safe load failed ({type(e).__name__}: {e}).\n"
                "The checkpoint pickles non-plain Python objects. Re-run with --allow-unsafe-pickle "
                "only if you trust the file (full unpickling can execute code).")
        print(f"  note: falling back to full unpickling for {path}", file=sys.stderr)
        return torch.load(path, map_location="cpu", weights_only=False)


def drop_skipped(ckpt, skip):
    if isinstance(ckpt, dict):
        dropped = [k for k in ckpt if k in skip]
        if dropped:
            print(f"  skipping training-only keys: {', '.join(dropped)}", file=sys.stderr)
        return {k: v for k, v in ckpt.items() if k not in skip}
    return ckpt


def convert_object(ckpt, out_base, source_name=""):
    s = Splitter()
    mirror = s.walk(ckpt)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(s.tensors, out_base.with_suffix(".safetensors"), {"source": source_name})
    with open(out_base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(mirror, f, indent=1, ensure_ascii=False)
    total = sum(a.nbytes for a in s.tensors.values())
    print(f"  {len(s.tensors)} tensors, {total / 1e6:.1f} MB -> {out_base.with_suffix('.*')}")


def print_tree(obj, indent=0, name="root", max_list=6):
    pad = "  " * indent
    if is_array(obj):
        arr = to_numpy(obj)
        print(f"{pad}{name}: tensor {NP_TO_ST[arr.dtype]} {list(arr.shape)}")
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict[{len(obj)}]")
        for k, v in obj.items():
            print_tree(v, indent + 1, repr(k) if not isinstance(k, str) else k, max_list)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        kind = type(obj).__name__
        items = list(obj)
        if items and all(not isinstance(x, (dict, list, tuple, set)) and not is_array(x) for x in items):
            preview = ", ".join(repr(x) for x in items[:max_list])
            more = f", ... (+{len(items) - max_list})" if len(items) > max_list else ""
            print(f"{pad}{name}: {kind}[{len(items)}] {preview}{more}")
        else:
            print(f"{pad}{name}: {kind}[{len(items)}]")
            for i, v in enumerate(items[:max_list]):
                print_tree(v, indent + 1, f"[{i}]", max_list)
            if len(items) > max_list:
                print(f"{pad}  ... (+{len(items) - max_list} more)")
    else:
        text = repr(obj)
        print(f"{pad}{name}: {text if len(text) <= 100 else text[:97] + '...'}")


def find_config(obj, depth=0):
    """Return the first dict found under a key named 'config' or 'args', breadth-first-ish."""
    if not isinstance(obj, dict) or depth > 4:
        return None
    for k in ("config", "args"):
        if isinstance(obj.get(k), dict):
            return k, obj[k]
    for v in obj.values():
        found = find_config(v, depth + 1)
        if found:
            return found
    return None


def cmd_inspect(args):
    ckpt = load_checkpoint(args.path, args.allow_unsafe_pickle)
    print_tree(ckpt, max_list=args.max_list)
    found = find_config(ckpt)
    if found:
        key, cfg = found
        print(f"\n===== full '{key}' =====")
        print(json.dumps(Splitter().walk(cfg), indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print("\n(no 'config' or 'args' dict found)")


def cmd_convert(args):
    src = Path(args.path)
    out = Path(args.out)
    skip = set(DEFAULT_SKIP) - set(args.keep) | set(args.skip)
    files = [src] if src.is_file() else sorted(src.rglob("*.pt"))
    if not files:
        raise SystemExit(f"no .pt files found under {src}")
    for f in files:
        rel = f.relative_to(src.parent if src.is_file() else src)
        print(f"{rel}")
        ckpt = drop_skipped(load_checkpoint(f, args.allow_unsafe_pickle), skip)
        convert_object(ckpt, out / rel.with_suffix(""), source_name=str(rel))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="print the structure and config of a checkpoint")
    pi.add_argument("path")
    pi.add_argument("--max-list", type=int, default=6, help="list items to show per level")
    pi.add_argument("--allow-unsafe-pickle", action="store_true")
    pi.set_defaults(func=cmd_inspect)

    pc = sub.add_parser("convert", help="convert a .pt file or every .pt under a directory")
    pc.add_argument("path")
    pc.add_argument("--out", required=True)
    pc.add_argument("--skip", nargs="*", default=[], help="extra top-level keys to drop")
    pc.add_argument("--keep", nargs="*", default=[], help=f"keep keys skipped by default {DEFAULT_SKIP}")
    pc.add_argument("--allow-unsafe-pickle", action="store_true")
    pc.set_defaults(func=cmd_convert)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
