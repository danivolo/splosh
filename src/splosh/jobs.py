"""Long jobs started from the viewer (a movie, prefetching every snapshot into the map store): a compute node when one is free right now, else this machine.

    JobManager(root, slurm="-c 20 --mem=32G -t 01:00:00", local_threads=lambda: 4)
    job = mgr.submit("movie", ["movie", *files, "--out", "{dir}/movie.mp4", ...], outputs=["movie.mp4"])   # splosh CLI arguments; {dir} = the job's folder
    mgr.list(); mgr.cancel(job_id); mgr.path(job_id, name); mgr.cancel_all()

Placement: with `slurm` set, the job is `srun --immediate=WAIT <slurm args> ...`: Slurm either starts it on a node within WAIT seconds or gives up at once (no waiting in the
queue); then the job runs here instead, under `nice 10`, with `local_threads()` numba threads, one local job at a time (later ones queue). Without `slurm` every job runs
here. The server only waits for the processes and reads their output (progress = the last `k/n` printed). Every job has a folder (`root/<id>/`) with `log.txt`, `job.json`
and its outputs; the map store is shared, so maps a node renders are cache hits for the viewer afterwards (the store must be on a file system both see)."""
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

START = "SPLOSH-JOB-START"
_PROGRESS = re.compile(r"(\d+)\s*/\s*(\d+)")


class JobManager:
    def __init__(self, root, slurm=None, wait=10, local_threads=lambda: 4, env=None):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.slurm = shlex.split(slurm) if isinstance(slurm, str) else list(slurm or [])
        self.wait = int(wait)
        self.local_threads = local_threads
        self.env = dict(env or {})
        self.jobs = {}
        self._lock = threading.Lock()
        self._local = threading.Semaphore(1)
        os.makedirs(self.root, exist_ok=True)

    # ---- public
    def submit(self, kind, args, outputs=(), label="", write=None):
        """start a job: `args` are splosh CLI arguments ({dir} = the job folder), `write` = {name: text} files put in the folder first."""
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        d = os.path.join(self.root, jid)
        os.makedirs(d)
        for name, text in (write or {}).items():
            os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        args = [a.replace("{dir}", d) for a in args]
        job = {"id": jid, "kind": kind, "label": label, "args": args, "outputs": list(outputs), "dir": d, "state": "starting", "where": None, "note": "",
               "progress": None, "created": time.time(), "started": None, "ended": None, "returncode": None}
        with self._lock:
            self.jobs[jid] = job
        self._save(job)
        job["_proc"] = None
        job["_cancel"] = False
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return self.public(job)

    def list(self):
        with self._lock:
            return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)]

    def public(self, job):
        out = {k: v for k, v in job.items() if not k.startswith("_") and k != "args"}
        out["files"] = [f for f in job["outputs"] if os.path.isfile(os.path.join(job["dir"], f))]
        out["log_tail"] = self._tail(job)
        return out

    def path(self, jid, name):
        """the file `name` of job `jid` if it is one of its declared outputs (or log.txt) and exists; else LookupError."""
        job = self.jobs.get(jid)
        if job is None or (name not in job["outputs"] and name != "log.txt"):
            raise LookupError("no such job file")
        f = os.path.join(job["dir"], name)
        if not os.path.isfile(f):
            raise LookupError("not written (yet)")
        return f

    def cancel(self, jid):
        job = self.jobs.get(jid)
        if job is None:
            raise LookupError("no such job")
        job["_cancel"] = True
        p = job.get("_proc")
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)              # the job's own process group (srun forwards it to the job step)
            except ProcessLookupError:
                pass
        return self.public(job)

    def cancel_all(self):
        for jid in list(self.jobs):
            if self.jobs[jid]["state"] in ("starting", "queued", "running"):
                self.cancel(jid)

    # ---- running
    def _cmd(self, on_node, threads):
        py = [sys.executable, "-m", "splosh"]
        if on_node:
            inner = f"echo {START} $(hostname); export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-$(nproc)}}; exec " + " ".join(shlex.quote(x) for x in py)
            return ["srun", f"--immediate={self.wait}", "--job-name=splosh", *self.slurm, "bash", "-c", inner + ' "$@"', "job"]
        return ["nice", "-n", "10", "bash", "-c", f"echo {START} local; exec " + " ".join(shlex.quote(x) for x in py) + ' "$@"', "job"]

    def _attempt(self, job, on_node):
        threads = max(1, int(self.local_threads()))
        env = dict(os.environ, **self.env)
        if not on_node:
            env.update(NUMBA_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1")
        log = open(os.path.join(job["dir"], "log.txt"), "a")
        log.write(f"$ {'node' if on_node else 'local'}: {' '.join(job['args'])}\n")
        log.flush()
        try:
            p = subprocess.Popen(self._cmd(on_node, threads) + job["args"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True, bufsize=1)
        except FileNotFoundError as e:                        # no srun on this machine
            log.write(f"{e}\n")
            log.close()
            return False, None
        job["_proc"] = p
        started = False
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith(START):
                started = True
                host = line.split(None, 1)[1].strip() if " " in line else "?"
                job.update(state="running", where=host if on_node else f"this machine ({socket.gethostname()}, {threads} threads, nice 10)", started=time.time())
                self._save(job)
                continue
            m = None
            for m in _PROGRESS.finditer(line):
                pass
            if m and started and int(m.group(2)) > 0:
                k, n = int(m.group(1)), int(m.group(2))
                word = line.strip().split(None, 1)[0].lower() if line.strip() else ""
                if word == "pass":                       # a movie's two passes make ONE bar: colour limits = the first half, frames the second
                    job["two_pass"] = True
                    job["progress"], job["phase"] = [k, 2 * n], "colour limits"
                elif word == "frame" and job.get("two_pass"):
                    job["progress"], job["phase"] = [n + k, 2 * n], "frames"
                else:
                    job["progress"] = [k, n]
                    if word == "frame":
                        job["phase"] = "frames"
        p.wait()
        log.close()
        return started, p.returncode

    def _run(self, job):
        try:
            if self.slurm and not job["_cancel"]:
                job["note"] = "asking Slurm for a free node"
                started, rc = self._attempt(job, on_node=True)
                if started or job["_cancel"]:
                    return self._finish(job, rc)
                job["note"] = f"no free node within {self.wait} s: running on this machine"
            job["state"] = "queued"
            with self._local:
                if job["_cancel"]:
                    return self._finish(job, None)
                started, rc = self._attempt(job, on_node=False)
                self._finish(job, rc)
        except Exception as e:                                # pragma: no cover
            job["note"] = f"{type(e).__name__}: {e}"
            self._finish(job, -1)

    def _finish(self, job, rc):
        job["returncode"] = rc
        job["ended"] = time.time()
        job["state"] = "cancelled" if job["_cancel"] else ("done" if rc == 0 else "failed")
        if job["state"] == "done" and job["progress"]:
            job["progress"] = [job["progress"][1], job["progress"][1]]
        self._save(job)

    def _save(self, job):
        with open(os.path.join(job["dir"], "job.json"), "w") as f:
            json.dump({k: v for k, v in job.items() if not k.startswith("_")}, f, indent=1)

    @staticmethod
    def _tail(job, n=4):
        try:
            with open(os.path.join(job["dir"], "log.txt")) as f:
                return [ln.rstrip("\n")[:300] for ln in f.readlines()[-n:]]
        except OSError:
            return []
