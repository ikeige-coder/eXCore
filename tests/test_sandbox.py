import os
import sys
import time
from pathlib import Path

import pytest

import evaluator.sandbox as sbx
from evaluator.sandbox import Sandbox, SandboxError, SandboxPolicy, clean_env, wrap_command

PY = [sys.executable, "-c"]


def box(**kw):
    kw.setdefault("network", True)           # these tests are about limits, not about namespaces
    kw.setdefault("timeout_s", 20)
    return Sandbox(SandboxPolicy(**kw))


def test_runs_and_reports(tmp_path):
    r = box().run(PY + ["import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"], cwd=tmp_path)
    assert r.returncode == 3 and not r.ok and r.stdout.strip() == "out" and "err" in r.stderr
    assert box().run(PY + ["print(1)"], cwd=tmp_path).ok


def test_environment_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("EXCORE_LLAMA_QUANTIZE", "/opt/llama-quantize")
    code = "import os; print(os.environ.get('GITHUB_TOKEN'), os.environ.get('EXCORE_LLAMA_QUANTIZE'), os.environ['HOME'])"
    r = box().run(PY + [code], cwd=tmp_path)
    assert r.stdout.split() == ["None", "/opt/llama-quantize", str(tmp_path)]
    env = clean_env(SandboxPolicy(), tmp_path, {"X": "1"})
    assert env["X"] == "1" and "GITHUB_TOKEN" not in env and env["TMPDIR"] == str(tmp_path / "tmp")


def test_timeout_kills_the_whole_process_group(tmp_path):
    pidfile = tmp_path / "pid"
    code = (f"import subprocess, time; p = subprocess.Popen(['sleep', '60']); "
            f"open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(60)")
    t0 = time.monotonic()
    r = box(timeout_s=1.0).run(PY + [code], cwd=tmp_path)
    assert r.timed_out and not r.ok and time.monotonic() - t0 < 10
    pid = int(pidfile.read_text())
    time.sleep(0.3)
    try:                                   # dead, or a zombie that init has not reaped yet
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        state = "gone"
    assert state in ("Z", "gone"), f"grandchild still alive (state {state})"


def test_output_is_capped(tmp_path):
    r = box(max_output_bytes=10_000).run(PY + ["print('x' * 3_000_000)"], cwd=tmp_path)
    assert r.truncated and len(r.stdout) <= 10_000 and r.returncode == 0


def test_file_size_limit(tmp_path):
    r = box(max_file_bytes=10_000).run(PY + ["open('big', 'wb').write(b'0' * 1_000_000)"], cwd=tmp_path)
    assert not r.ok
    assert (tmp_path / "big").stat().st_size <= 10_000


def test_require_isolation_is_a_hard_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sbx, "detect_isolation", lambda: "none")
    with pytest.raises(SandboxError, match="isolation is required"):
        Sandbox(SandboxPolicy(network=False, require_isolation=True)).run(PY + ["print(1)"], cwd=tmp_path)
    r = Sandbox(SandboxPolicy(network=False)).run(PY + ["print(1)"], cwd=tmp_path)   # best effort otherwise
    assert r.ok and r.isolation == "none"


def test_wrap_command_shapes(tmp_path):
    cmd = ["python", "x.py"]
    assert wrap_command(cmd, isolation="bwrap", network=True, cwd=tmp_path) == cmd
    assert wrap_command(cmd, isolation="none", network=False, cwd=tmp_path) == cmd
    w = wrap_command(cmd, isolation="bwrap", network=False, cwd=tmp_path, rw_paths=[tmp_path / "rw"], ro_paths=[tmp_path / "ro"])
    assert w[0] == "bwrap" and "--unshare-net" in w and w[-2:] == cmd
    assert w.index("--tmpfs") < w.index("--ro-bind", w.index("--tmpfs")) < w.index("--bind")   # mounts layered after /tmp
    assert wrap_command(cmd, isolation="unshare", network=False, cwd=tmp_path)[:2] == ["unshare", "--net"]
    with pytest.raises(SandboxError):
        wrap_command(cmd, isolation="docker", network=False, cwd=tmp_path)


def test_detect_isolation_returns_known_value():
    assert sbx.detect_isolation() in {"bwrap", "unshare", "none"}
