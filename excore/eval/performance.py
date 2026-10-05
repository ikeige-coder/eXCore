"""Speed and memory: generation tokens/s, prefill tokens/s and peak RAM, on CPU.

Throughput comes from ``llama-bench`` (JSON output, CPU only, fixed thread count).
Peak RAM is the high-water mark of the benchmark process tree, read from the kernel's
``VmHWM`` on Linux (exact, nothing can slip between samples) and sampled RSS elsewhere.

Each measurement is repeated ``limits.perf_runs`` times. The reported figures are the
conservative ones (slowest throughput, highest RAM), and the result is flagged
unstable if runs disagree by more than ``limits.perf_tolerance``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3
ENV_BENCH = "EXCORE_LLAMA_BENCH"


class PerfError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerfRun:
    prefill_tps: float
    gen_tps: float
    peak_rss_gib: float
    wall_s: float


@dataclass(frozen=True)
class PerfResult:
    prefill_tps: float
    gen_tps: float
    peak_rss_gib: float
    runs: tuple[PerfRun, ...]
    stable: bool
    spread_prefill: float
    spread_gen: float
    prompt_tokens: int
    gen_tokens: int
    threads: int

    def within_ram_limit(self, max_gib: float) -> bool:
        return self.peak_rss_gib <= max_gib

    def to_dict(self) -> dict:
        return {
            "prefill_tps": self.prefill_tps, "gen_tps": self.gen_tps,
            "peak_rss_gib": self.peak_rss_gib, "stable": self.stable,
            "spread_prefill": self.spread_prefill, "spread_gen": self.spread_gen,
            "prompt_tokens": self.prompt_tokens, "gen_tokens": self.gen_tokens,
            "threads": self.threads, "runs": len(self.runs),
        }


def find_bench(explicit: str | None = None) -> Path:
    for c in (explicit, os.environ.get(ENV_BENCH), shutil.which("llama-bench")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return Path(c)
    raise PerfError(f"llama-bench not found. Run scripts/setup_bench.sh, or set {ENV_BENCH}.")


def bench_command(
    binary: str | Path, model: str | Path, *, threads: int, prompt_tokens: int,
    gen_tokens: int, batch: int, ubatch: int,
) -> list[str]:
    return [
        str(binary), "-m", str(model), "-p", str(prompt_tokens), "-n", str(gen_tokens),
        "-t", str(threads), "-b", str(batch), "-ub", str(ubatch),
        "-ngl", "0", "-r", "1", "-o", "json",
    ]


def parse_bench_json(text: str) -> tuple[float, float]:
    """(prefill tokens/s, generation tokens/s) from llama-bench JSON output."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise PerfError("llama-bench printed no JSON array")
    try:
        rows = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PerfError(f"llama-bench JSON is malformed: {exc}") from None
    prefill = gen = None
    for r in rows:
        n_p, n_g, ts = int(r.get("n_prompt", 0)), int(r.get("n_gen", 0)), r.get("avg_ts")
        if ts is None or not float(ts) > 0:
            continue
        if n_p > 0 and n_g == 0:
            prefill = float(ts)
        elif n_g > 0 and n_p == 0:
            gen = float(ts)
    if prefill is None or gen is None:
        raise PerfError("llama-bench output lacks a prefill or generation result")
    return prefill, gen


def _hwm_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class PeakRSS(threading.Thread):
    """Tracks the peak resident memory of a process and its children."""

    def __init__(self, pid: int, interval: float = 0.02):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak = pid, interval, 0
        self._halt = threading.Event()

    def _sample(self) -> None:
        total = 0
        hwm = _hwm_bytes(self.pid)
        procs = []
        try:
            import psutil

            root = psutil.Process(self.pid)
            procs = [root] + root.children(recursive=True)
        except Exception:
            pass
        if hwm is not None:
            total = hwm
            for p in procs[1:]:
                total += _hwm_bytes(p.pid) or 0
        else:
            for p in procs:
                try:
                    total += p.memory_info().rss
                except Exception:
                    pass
        self.peak = max(self.peak, total)

    def run(self) -> None:
        while not self._halt.is_set():
            self._sample()
            self._halt.wait(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        return self.peak


def check_free_ram(min_free_gib: float) -> None:
    import psutil

    free = psutil.virtual_memory().available / GIB
    if free < min_free_gib:
        raise PerfError(f"only {free:.1f} GiB RAM available, need at least {min_free_gib:.1f} GiB")


def run_once(cmd: list[str], timeout: float) -> PerfRun:
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "LC_ALL": "C"})
    watcher = PeakRSS(proc.pid)
    watcher.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        watcher.stop()
        raise PerfError(f"llama-bench exceeded {timeout:.0f}s") from None
    peak = watcher.stop()
    if proc.returncode != 0:
        tail = "\n".join((err or out or "").splitlines()[-15:])
        raise PerfError(f"llama-bench failed ({proc.returncode}):\n{tail}")
    prefill, gen = parse_bench_json(out)
    return PerfRun(prefill, gen, peak / GIB, time.monotonic() - t0)


def _spread(values: list[float]) -> float:
    return (max(values) - min(values)) / max(values) if max(values) > 0 else 0.0


def measure(model: str | Path, cfg: dict, *, bench_binary: str | None = None) -> PerfResult:
    limits, perf = cfg["limits"], cfg["perf"]
    check_free_ram(float(limits["min_free_ram_gib"]))
    binary = find_bench(bench_binary)
    cmd = bench_command(
        binary, model, threads=int(limits["threads"]), prompt_tokens=int(perf["prompt_tokens"]),
        gen_tokens=int(perf["gen_tokens"]), batch=int(limits["batch_size"]), ubatch=int(limits["ubatch_size"]),
    )
    runs = tuple(run_once(cmd, float(limits["eval_timeout_s"])) for _ in range(int(limits["perf_runs"])))
    sp, sg = _spread([r.prefill_tps for r in runs]), _spread([r.gen_tps for r in runs])
    tol = float(limits["perf_tolerance"])
    return PerfResult(
        prefill_tps=min(r.prefill_tps for r in runs),
        gen_tps=min(r.gen_tps for r in runs),
        peak_rss_gib=max(r.peak_rss_gib for r in runs),
        runs=runs, stable=sp <= tol and sg <= tol, spread_prefill=sp, spread_gen=sg,
        prompt_tokens=int(perf["prompt_tokens"]), gen_tokens=int(perf["gen_tokens"]),
        threads=int(limits["threads"]),
    )


def host_fingerprint() -> dict:
    """What the numbers were measured on. Throughput is only comparable on the same host class."""
    info = {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version()}
    try:
        import psutil

        info["logical_cpus"] = psutil.cpu_count(logical=True)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gib"] = round(psutil.virtual_memory().total / GIB, 1)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        flags = next((l for l in text.splitlines() if l.startswith("flags")), "")
        info["isa"] = sorted(f for f in ("avx2", "avx512f", "avx512_vnni", "fma", "f16c") if f" {f}" in flags)
    except OSError:
        pass
    return info
