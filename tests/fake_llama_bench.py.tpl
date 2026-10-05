#!{python}
import json, os, sys, time
if os.environ.get("FAKE_FAIL"):
    print("bench exploded", file=sys.stderr); sys.exit(2)
mb = int(os.environ.get("FAKE_ALLOC_MB", "0"))
buf = bytearray(mb << 20)
if mb:
    buf[::4096] = b"\x01" * len(range(0, len(buf), 4096))
print("log line on stderr", file=sys.stderr)
pp, tg = float(os.environ.get("FAKE_PP", "100")), float(os.environ.get("FAKE_TG", "10"))
counter = os.environ.get("FAKE_COUNTER")
if counter:
    n = int(open(counter).read()) if os.path.exists(counter) else 0
    open(counter, "w").write(str(n + 1))
    if n == 1:
        pp, tg = pp * 1.5, tg * 1.5
time.sleep(0.25)
args = sys.argv
p, n = int(args[args.index("-p") + 1]), int(args[args.index("-n") + 1])
print(json.dumps([{{"n_prompt": p, "n_gen": 0, "avg_ts": pp}}, {{"n_prompt": 0, "n_gen": n, "avg_ts": tg}}]))
