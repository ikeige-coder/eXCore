#!{python}
"""Test double for llama-quantize: honours --pure / --tensor-type for the legacy formats."""
import os, re, sys
sys.path.insert(0, {root!r})
from excore.gguf import GGMLType, GGUFFile, WriteEntry, write_gguf, tensor_nbytes
from excore.precision import resolve_format
from excore.quantizers.local_rtn import LocalRTN

mode = os.environ.get("FAKE_MODE", "ok")
if mode == "fail":
    print("boom: simulated failure", file=sys.stderr)
    sys.exit(3)

args, over, pos, i = sys.argv[1:], [], [], 0
while i < len(args):
    if args[i] == "--pure":
        i += 1
    elif args[i] == "--tensor-type":
        over.append(args[i + 1].rsplit("=", 1)); i += 2
    else:
        pos.append(args[i]); i += 1
src_path, out_path, base, _threads = pos
rtn = LocalRTN()

with GGUFFile(src_path) as g:
    entries = []
    for t in g.tensors:
        target = base.upper()
        if mode != "ignore":
            for pat, ty in over:
                if re.search(pat, t.name):
                    target = ty.upper()
        if len(t.dims) == 1 or target in ("F32", "F16", "BF16"):
            gt = t.ggml_type
            produce = (lambda t=t: g.iter_tensor_bytes(t))
        else:
            gt = GGMLType[target]
            def produce(t=t, target=target):
                return [rtn.quantize_rows(g.read_rows(t, 0, t.n_rows), resolve_format(target))]
        entries.append(WriteEntry(t.name, t.dims, gt, produce))
    write_gguf(out_path, metadata_raw=g.metadata_raw, kv_count=g.kv_count,
               alignment=g.alignment, entries=entries)
