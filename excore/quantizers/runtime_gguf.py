"""Native llama.cpp quantization.

All ``runtime`` units are encoded in one ``llama-quantize`` run: the quantizer turns
the manifest into explicit ``--tensor-type`` instructions (with ``--pure`` so no
k-quant mixing heuristics change a unit behind the miner's back), runs the tool once
per build, then streams each unit's finished tensor out of its output file. Frozen
tensors are never taken from that output; the builder copies them from the source.

Whatever llama-quantize actually produced is verified against the request, because
the tool silently falls back to another type for some shapes. The audit repeats the
check on the final file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

from ..gguf import GGMLType, GGUFError, GGUFFile
from ..precision import FORMATS
from . import BuildContext, Quantizer, QuantizerError, TensorJob, register

ENV_BINARY = "EXCORE_LLAMA_QUANTIZE"
BASE_TYPE = "Q4_0"   # type for tensors without an override; their output is discarded


def find_binary(explicit: str | None = None) -> Path:
    candidates = [explicit, os.environ.get(ENV_BINARY), shutil.which("llama-quantize")]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return Path(c)
    raise QuantizerError(
        "llama-quantize not found. Run scripts/setup_bench.sh, or set "
        f"{ENV_BINARY} to the binary's path."
    )


def type_arg(fmt_name: str) -> str:
    return fmt_name.lower()


def build_command(
    binary: str | Path, source: str | Path, out: str | Path, jobs: Sequence[TensorJob], threads: int
) -> list[str]:
    cmd = [str(binary), "--pure"]
    for job in sorted(jobs, key=lambda j: j.tensor_name):
        pattern = f"^{re.escape(job.tensor_name)}$"
        cmd += ["--tensor-type", f"{pattern}={type_arg(job.fmt.name)}"]
    cmd += [str(source), str(out), BASE_TYPE, str(int(threads))]
    return cmd


class RuntimeGGUFQuantizer(Quantizer):
    name = "runtime"
    version = 1
    formats = frozenset(FORMATS)
    regenerable = False

    def prepare(self, source: GGUFFile, jobs: list[TensorJob], ctx: BuildContext) -> None:
        if not jobs:
            return
        binary = find_binary(ctx.llama_quantize)
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        out = ctx.workdir / "runtime.gguf"
        cmd = build_command(binary, source.path, out, jobs, ctx.threads)
        env = {**os.environ, "LC_ALL": "C"}
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=ctx.timeout, env=env, check=False
            )
        except subprocess.TimeoutExpired:
            raise QuantizerError(f"llama-quantize exceeded {ctx.timeout:.0f}s") from None
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-15:])
            raise QuantizerError(f"llama-quantize failed ({proc.returncode}):\n{tail}")
        try:
            produced = GGUFFile(out)
        except GGUFError as exc:
            raise QuantizerError(f"llama-quantize wrote an unreadable file: {exc}") from None
        ctx.on_close(produced.close)

        problems = []
        for job in jobs:
            t = produced.by_name.get(job.tensor_name)
            want = GGMLType[job.fmt.name]
            if t is None:
                problems.append(f"{job.tensor_name}: missing from llama-quantize output")
            elif t.ggml_type != want:
                problems.append(
                    f"{job.tensor_name}: llama-quantize produced {t.ggml_type.name}, requested {want.name}"
                )
            elif t.dims != job.info.dims:
                problems.append(f"{job.tensor_name}: shape changed to {t.dims}")
        if problems:
            raise QuantizerError("; ".join(problems))
        ctx.state[self.name] = produced

    def encode(self, job: TensorJob, source: GGUFFile, ctx: BuildContext) -> Iterator[bytes]:
        produced: GGUFFile = ctx.state[self.name]
        yield from produced.iter_tensor_bytes(produced.by_name[job.tensor_name])


register(RuntimeGGUFQuantizer())
