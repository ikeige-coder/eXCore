"""Isolated execution of build and audit steps.

Defence in depth, not a security boundary on its own. The design rule of the bot is that
*untrusted code never runs*: miner PRs may only add a YAML manifest, which is parsed as
data. The sandbox still wraps the heavy build/audit subprocesses so that a bug in a
parser, quantizer or native tool is contained:

* a scrubbed environment (no tokens or secrets leak into children),
* resource limits (file size, open files, processes, optional CPU time and memory),
* a wall-clock timeout that kills the whole process group,
* capped output,
* network isolation via ``bwrap`` or ``unshare`` when the host allows it.

``detect_isolation`` reports what the host can actually do, and ``require_isolation``
makes a missing network namespace a hard error. For production, run the validator in a
container or VM as well (see ``runbook.md``).
"""

from __future__ import annotations

import functools
import os
import resource
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_s: float = 3600.0
    cpu_seconds: int | None = None
    max_file_bytes: int | None = 96 << 30
    max_open_files: int = 1024
    max_processes: int | None = 4096
    max_address_space: int | None = None          # leave unset for builds: they mmap multi-GB files
    max_output_bytes: int = 1 << 20
    env_allow: tuple[str, ...] = ("PATH", "LANG", "EXCORE_LLAMA_QUANTIZE", "EXCORE_LLAMA_BENCH")
    network: bool = False
    require_isolation: bool = False


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    isolation: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, lines: int = 15) -> str:
        text = (self.stderr or self.stdout or "").strip()
        return "\n".join(text.splitlines()[-lines:])


def _probe(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@functools.lru_cache(maxsize=1)
def detect_isolation() -> str:
    """What network isolation this host supports: 'bwrap', 'unshare' or 'none'."""
    if shutil.which("bwrap") and _probe(["bwrap", "--unshare-net", "--ro-bind", "/", "/", "true"]):
        return "bwrap"
    if shutil.which("unshare") and _probe(["unshare", "--net", "--map-root-user", "true"]):
        return "unshare"
    return "none"


def wrap_command(cmd: Sequence[str], *, isolation: str, network: bool, cwd: Path,
                 rw_paths: Sequence[Path] = (), ro_paths: Sequence[Path] = ()) -> list[str]:
    if network or isolation == "none":
        return list(cmd)
    if isolation == "bwrap":
        w = ["bwrap", "--unshare-net", "--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/",
             "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        for p in ro_paths:
            w += ["--ro-bind", str(p), str(p)]
        for p in rw_paths:
            w += ["--bind", str(p), str(p)]
        return w + ["--chdir", str(cwd)] + list(cmd)
    if isolation == "unshare":
        return ["unshare", "--net", "--map-root-user"] + list(cmd)
    raise SandboxError(f"unknown isolation mode {isolation!r}")


def clean_env(policy: SandboxPolicy, cwd: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in policy.env_allow if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(HOME=str(cwd), TMPDIR=str(cwd / "tmp"), LC_ALL="C", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.update(extra or {})
    return env


def _limits(policy: SandboxPolicy):
    def apply() -> None:
        def lim(res: int, value: int | None) -> None:
            if value is not None:
                resource.setrlimit(res, (value, value))

        lim(resource.RLIMIT_FSIZE, policy.max_file_bytes)
        lim(resource.RLIMIT_NOFILE, policy.max_open_files)
        lim(resource.RLIMIT_NPROC, policy.max_processes)
        lim(resource.RLIMIT_CPU, policy.cpu_seconds)
        lim(resource.RLIMIT_AS, policy.max_address_space)
        lim(resource.RLIMIT_CORE, 0)

    return apply


class _Drain(threading.Thread):
    """Reads a pipe to the end, keeping at most ``cap`` bytes (so a chatty child cannot block or flood us)."""

    def __init__(self, stream, cap: int):
        super().__init__(daemon=True)
        self.stream, self.cap = stream, cap
        self.chunks: list[bytes] = []
        self.size = 0
        self.truncated = False

    def run(self) -> None:
        for block in iter(lambda: self.stream.read(65536), b""):
            room = self.cap - self.size
            if room > 0:
                self.chunks.append(block[:room])
                self.size += min(len(block), room)
            if len(block) > room:
                self.truncated = True

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", errors="replace")


class Sandbox:
    def __init__(self, policy: SandboxPolicy | None = None):
        self.policy = policy or SandboxPolicy()

    def run(self, cmd: Sequence[str], *, cwd: str | Path, env_extra: Mapping[str, str] | None = None,
            rw_paths: Sequence[str | Path] = (), ro_paths: Sequence[str | Path] = ()) -> SandboxResult:
        pol = self.policy
        isolation = detect_isolation()
        if not pol.network and pol.require_isolation and isolation == "none":
            raise SandboxError("network isolation is required but this host provides none (install bubblewrap)")
        cwd = Path(cwd)
        (cwd / "tmp").mkdir(parents=True, exist_ok=True)
        full = wrap_command(cmd, isolation=isolation, network=pol.network, cwd=cwd,
                            rw_paths=[Path(p) for p in (*rw_paths, cwd)], ro_paths=[Path(p) for p in ro_paths])
        t0 = time.monotonic()
        proc = subprocess.Popen(
            full, cwd=cwd, env=clean_env(pol, cwd, env_extra), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, preexec_fn=_limits(pol),
        )
        out, err = _Drain(proc.stdout, pol.max_output_bytes), _Drain(proc.stderr, pol.max_output_bytes)
        out.start()
        err.start()
        timed_out = False
        try:
            proc.wait(timeout=pol.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)       # nothing may outlive the call
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            out.join(5)
            err.join(5)
        return SandboxResult(proc.returncode, out.text(), err.text(), timed_out,
                             out.truncated or err.truncated, isolation, time.monotonic() - t0)
