"""
runlog.py  --  Every V6 script records what it did, when, how long it took and with what settings.

  results\\run_log.jsonl            one line per run of any script (append-only, never overwritten)
  results\\<name>_<timestamp>.txt   the full printed report of that run
  results\\<name>_latest.txt        copy of the most recent report (convenience)

Use:
    with Run("score_biofors", args) as run:
        run.p("text")                    # printed AND saved to the report
        run.output("results\\x.csv")     # list of files this run wrote
        run.note(key=value)              # extra numbers for the log line
"""

import os
import sys
import json
import time
import shutil
import platform
import datetime

RESULTS = "results"


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform(), "cpus": os.cpu_count()}
    for mod in ("numpy", "cv2", "sklearn", "scipy", "joblib"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


class Run:
    def __init__(self, name, args=None, results_dir=RESULTS):
        self.name, self.args, self.dir = name, (vars(args) if args is not None else {}), results_dir
        os.makedirs(self.dir, exist_ok=True)
        self.stamp = stamp()
        self.report = os.path.join(self.dir, f"{name}_{self.stamp}.txt")
        self.outputs, self.notes = [self.report], {}

    def __enter__(self):
        self.t0, self.start = time.time(), now_iso()
        self.f = open(self.report, "w", encoding="utf-8")
        self.p(f"{self.name}   started {self.start}")
        self.p(f"settings: {json.dumps(self.args, default=str)}")
        self.p(f"versions: {json.dumps(versions())}\n")
        return self

    def p(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()

    def output(self, path):
        if path not in self.outputs:
            self.outputs.append(path)

    def note(self, **kw):
        self.notes.update(kw)

    def __exit__(self, exc_type, exc, tb):
        runtime = time.time() - self.t0
        status = "ok" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        if exc_type is KeyboardInterrupt:
            status = "interrupted (Ctrl+C)"
        self.p(f"\n{self.name}   finished {now_iso()}   runtime {runtime / 60:.1f} min   status: {status}")
        self.f.close()
        try:
            shutil.copyfile(self.report, os.path.join(self.dir, f"{self.name}_latest.txt"))
        except OSError:
            pass
        line = {"script": self.name, "start": self.start, "end": now_iso(), "runtime_s": round(runtime, 1),
                "status": status, "settings": self.args, "outputs": self.outputs, "notes": self.notes,
                "versions": versions()}
        with open(os.path.join(self.dir, "run_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        return False
