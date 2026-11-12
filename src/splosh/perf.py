"""GIZMO's performance logs as time-series sources (series.py): any GIZMO run, read incrementally (a running simulation appends; only the new bytes are parsed).

    cpu       output/cpu.txt: per step the cumulative wall seconds of every timer (total, tree+gravity, treewalk, ..., the imbalance timers), the active count;
              derived: wall_h (cumulative hours, restarts joined), dwall (seconds of the step), h_per_code_time / h_per_P0, frac_<module> (share of the step),
              wait (the waiting timers' share of the step: treeimbal + densimbal + hydimbal + bhimbal, as step_costs.py and `astro health` count it)
    timebin   output/timebin.txt: per sync point the system step dt_sys, the active count, the lowest and highest occupied bin, the occupancy of every bin (occ_B)
    timings   output/timings.txt: per step the first tree walk's work-load balance, particle-load balance, part/sec, interactions per particle, ex-frac
    balance   output/balance.txt: per step the seconds of every module, useful (<module>) and waiting (<module>_imbal), from the step's bar characters

The parse state (byte offset, rows so far) is kept as inc_<source>.npz in the series cache folder; a log that got shorter (a new run in the same folder) is read again."""
import json
import os
import re

import numpy as np

from . import plugins
from . import series as _S
from .series import Table, output_file

WAIT_TIMERS = ("treeimbal", "densimbal", "hydimbal", "bhimbal")


def _name(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_").lower()


CHUNK = 16 << 20          # bytes read and parsed at a time: the memory of a parse stays bounded whatever the size of the log


def _arrays(rows):
    """a list of dict rows -> {column: float array} (NaN where a row lacks the column)."""
    keys = {}
    for r in rows:
        for c in r:
            keys.setdefault(c, None)
    out = {c: np.full(len(rows), np.nan) for c in keys}
    for k, r in enumerate(rows):
        for c, v in r.items():
            out[c][k] = v
    return out


def _incremental(ctx, name, path, parse):
    """the rows of `path` as {column: float array}, parsing only what was appended since the last call, in chunks of CHUNK bytes.
    parse(text) -> (list of dict rows, characters consumed: only complete records)."""
    st_path = os.path.join(_S.cache_dir(ctx), f"inc_{name}.npz")
    size = os.path.getsize(path)
    blocks, off = [], 0
    try:
        with np.load(st_path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("path") == os.path.abspath(path) and meta["offset"] <= size:
                off = meta["offset"]
                blocks.append({k: np.asarray(z["c_" + k], float) for k in meta["columns"]})
    except (OSError, KeyError, ValueError):
        pass
    off0 = off
    if off < size:
        with open(path, "rb") as f:
            f.seek(off)
            buf = ""
            while True:
                data = f.read(CHUNK).decode("latin-1")         # one character per byte: consumed characters = consumed bytes
                buf += data
                if not buf:
                    break
                rows, used = parse(buf)
                if rows:
                    blocks.append(_arrays(rows))
                off += used
                buf = buf[used:]
                if not data:                                  # end of the file: an incomplete last record stays for the next call
                    break
    keys = {}
    for b in blocks:
        for c in b:
            keys.setdefault(c, None)
    lens = [len(next(iter(b.values()))) if b else 0 for b in blocks]
    cols = {c: np.concatenate([b[c] if c in b else np.full(n, np.nan) for b, n in zip(blocks, lens)]) if blocks else np.zeros(0) for c in keys}
    if off != off0:                                           # something new was parsed: keep the state
        try:
            tmp = st_path + ".tmp.npz"
            np.savez(tmp, meta=json.dumps({"path": os.path.abspath(path), "offset": off, "columns": list(cols)}), **{"c_" + c: v for c, v in cols.items()})
            os.replace(tmp, st_path)
        except OSError:
            pass
    return cols


def _restart_order(t):
    """indices keeping, for times written twice (a restart from an earlier restart file), the later copy; in time order."""
    from .tables import stale_after_restart
    keep = ~stale_after_restart([t])[0]
    idx = np.nonzero(keep)[0]
    return idx[np.argsort(t[idx], kind="stable")]


def _increments(c):
    """per-row increments of a cumulative timer that restarts from 0 after a restart: a negative step is replaced by the value itself."""
    d = np.diff(c, prepend=0.0)
    return np.where(d < 0, c, d)


# ----------------------------------------------------------------------------- cpu.txt

_CPU_HEAD = re.compile(r"^Step (\d+), Time: ([^,]+), CPUs: (\d+)", re.M)


def _parse_cpu(text):
    rows, used = [], 0
    heads = list(_CPU_HEAD.finditer(text))
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else None
        if end is None:                                       # the last block: complete only when it ends with a blank line
            e = text.find("\n\n", m.end())
            if e < 0:
                break
            end = e + 2
        block = text[m.start():end]
        r = {"step": float(m.group(1)), "time": float(m.group(2)), "ncpu": float(m.group(3))}
        mm = re.search(r"Nactive=\s*(\d+), Imbal\(Max/Mean\)=\s*([0-9.eE+-]+)", block)
        if mm:
            r["nactive"], r["imbal"] = float(mm.group(1)), float(mm.group(2))
        for line in block.splitlines()[2:]:
            w = line.split()
            if len(w) >= 3 and w[-1].endswith("%"):
                try:
                    r["t_" + _name(" ".join(w[:-2]))] = float(w[-2])
                except ValueError:
                    pass
        rows.append(r)
        used = end
    return rows, used


def _cpu_files(ctx):
    f = output_file(ctx, "cpu.txt")
    return [f] if f else []


def _cpu(ctx):
    raw = _incremental(ctx, "cpu", _cpu_files(ctx)[0], _parse_cpu)
    o = _restart_order(raw["time"])
    t = raw["time"][o]
    tot = raw.get("t_total", np.full(len(raw["time"]), np.nan))
    dwall = _increments(np.nan_to_num(tot))                  # in file order (restarts reset the counters), then sorted
    nact = raw.get("nactive")
    cols = {"step": raw["step"][o], "ncpu": raw["ncpu"][o], "nactive": nact[o] if nact is not None else np.full(len(t), np.nan), "dwall": dwall[o]}
    cols["wall_h"] = np.cumsum(cols["dwall"]) / 3600.0
    meta = {"step": {"label": "step"}, "ncpu": {"label": "MPI tasks"}, "nactive": {"label": "active elements of the step", "group": "steps"},
            "dwall": {"label": "wall time of the step", "unit": "s", "group": "steps", "style": "dots"},
            "wall_h": {"label": "cumulative wall clock", "unit": "h", "group": "wall clock"}}
    if len(t) > 3:                                            # wall hours per unit of code time, over windows of ~1/200 of the run
        k = max(20, len(t) // 200)
        tw, hw = t[::k], cols["wall_h"][::k]
        rate = np.interp(t, 0.5 * (tw[1:] + tw[:-1]), np.diff(hw) / np.maximum(np.diff(tw), 1e-300)) if len(tw) > 2 else np.full(len(t), np.nan)
        cols["h_per_code_time"] = rate
        meta["h_per_code_time"] = {"label": "wall hours per unit of code time", "unit": "h", "group": "wall clock"}
        if ctx.get("period"):
            cols["h_per_P0"] = rate * ctx["period"]
            meta["h_per_P0"] = {"label": "wall hours per binary period P0", "unit": "h", "group": "wall clock"}
    # shares of the wall time over windows of ~1/400 of the run's steps, weighted by the steps' wall time (a share per step is mostly noise:
    # most steps are tiny); every step of a window carries the window's share
    n = len(t)
    win = np.arange(n) // max(10, n // 400) if n else np.zeros(0, int)
    wsum = np.bincount(win, weights=np.nan_to_num(cols["dwall"])) if n else np.zeros(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = lambda d: np.where(wsum > 0, np.bincount(win, weights=d, minlength=len(wsum)) / wsum, np.nan)[win]
        waits = np.zeros(n)
        for name in sorted(k[2:] for k in raw if k.startswith("t_") and k != "t_total"):
            d = np.nan_to_num(_increments(np.nan_to_num(raw["t_" + name]))[o])
            cols["frac_" + name] = share(d)
            meta["frac_" + name] = {"label": f"{name}: share of the wall time", "group": "waiting" if name in WAIT_TIMERS or "imbal" in name else ("communication" if "comm" in name else "modules")}
            if name in WAIT_TIMERS:
                waits += d
        cols["wait"] = share(waits)
    meta["wait"] = {"label": "waiting share (treeimbal + densimbal + hydimbal + bhimbal, as step_costs.py)", "group": "waiting"}
    if "imbal" in raw:
        cols["imbal_maxmean"] = raw["imbal"][o]
        meta["imbal_maxmean"] = {"label": "cpu.txt Imbal(Max/Mean) (includes the barrier waits: stays near 1, see step_costs.py)", "group": "steps"}
    return Table(t, cols, meta=meta)


# ----------------------------------------------------------------------------- timebin.txt

_SYNC = re.compile(r"^Sync-Point (\d+), Time: ([^,]+), Systemstep: (\S+)", re.M)
_BIN = re.compile(r"bin=\s*(\d+)\s+(\d+)\s+(\d+)")
_ACT = re.compile(r"Total active:\s+(\d+)\s+(\d+)")


def _parse_timebin(text):
    rows, used = [], 0
    heads = list(_SYNC.finditer(text))
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else None
        if end is None:
            e = text.find("Total active:", m.end())
            if e < 0:
                break
            e2 = text.find("\n", e)
            if e2 < 0:
                break
            end = e2 + 1
        block = text[m.start():end]
        r = {"time": float(m.group(2)), "dt_sys": float(m.group(3))}
        occ = [(int(b), int(n1) + int(n2)) for b, n1, n2 in _BIN.findall(block)]
        for b, n in occ:
            r[f"occ_{b}"] = float(n)
        if occ:
            used_b = [b for b, n in occ if n > 0]
            if used_b:
                r["bin_min"], r["bin_max"] = float(min(used_b)), float(max(used_b))
        a = _ACT.search(block)
        if a:
            r["n_active"] = float(int(a.group(1)) + int(a.group(2)))
        rows.append(r)
        used = end
    return rows, used


def _timebin_files(ctx):
    f = output_file(ctx, "timebin.txt")
    return [f] if f else []


def _timebin(ctx):
    raw = _incremental(ctx, "timebin", _timebin_files(ctx)[0], _parse_timebin)
    o = _restart_order(raw["time"])
    cols = {k: v[o] for k, v in raw.items() if k != "time"}
    tot = sum(np.nan_to_num(v) for k, v in cols.items() if k.startswith("occ_")) if any(k.startswith("occ_") for k in cols) else None
    meta = {"dt_sys": {"label": "system step dt_sys", "unit": "code time", "group": "steps", "style": "dots"},
            "n_active": {"label": "active elements", "group": "steps", "style": "dots"}, "bin_min": {"label": "lowest occupied time bin", "group": "bins"},
            "bin_max": {"label": "highest occupied time bin", "group": "bins"}}
    if tot is not None and "n_active" in cols:
        with np.errstate(invalid="ignore", divide="ignore"):
            cols["active_frac"] = np.where(tot > 0, cols["n_active"] / tot, np.nan)
        meta["active_frac"] = {"label": "active fraction", "group": "steps", "style": "dots"}
    for k in cols:
        if k.startswith("occ_"):
            meta[k] = {"label": f"elements in time bin {k[4:]}", "group": "bin occupancy"}
    return Table(raw["time"][o], cols, meta=meta)


# ----------------------------------------------------------------------------- timings.txt

_TSTEP = re.compile(r"^Step=\s*(\d+)\s+t=\s*(\S+)\s+dt=\s*(\S+)", re.M)


def _parse_timings(text):
    rows, used, last_step = [], 0, None
    heads = list(_TSTEP.finditer(text))
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else None
        if end is None:
            e = text.find("\n\n", m.end())
            if e < 0:
                break
            end = e + 2
        used = end
        step = int(m.group(1))
        if step == last_step:                                 # only the first tree walk of a step
            continue
        last_step = step
        block = text[m.start():end]
        r = {"step": float(step), "time": float(m.group(2))}
        for key, pat in (("ex_frac", r"ex-frac=\s*([0-9.eE+-]+)"), ("balance", r"work-load balance:\s*([0-9.eE+-]+)"), ("particle_balance", r"particle-load balance:\s*([0-9.eE+-]+)"),
                         ("part_per_sec", r"part/sec=\s*([0-9.eE+-]+)"), ("ia_per_part", r"ia/part=\s*([0-9.eE+-]+)")):
            mm = re.search(pat, block)
            if mm:
                r[key] = float(mm.group(1))
        rows.append(r)
    return rows, used


def _timings_files(ctx):
    f = output_file(ctx, "timings.txt")
    return [f] if f else []


def _timings(ctx):
    raw = _incremental(ctx, "timings", _timings_files(ctx)[0], _parse_timings)
    o = _restart_order(raw["time"])
    meta = {"balance": {"label": "work-load balance of the tree walk (max/mean)", "group": "balance", "style": "dots"},
            "particle_balance": {"label": "particle-load balance", "group": "balance"}, "part_per_sec": {"label": "tree walk: particles per second", "group": "speed"},
            "ia_per_part": {"label": "tree walk: interactions per particle", "group": "speed"}, "ex_frac": {"label": "tree walk: export fraction", "group": "speed"},
            "step": {"label": "step"}}
    return Table(raw["time"][o], {k: v[o] for k, v in raw.items() if k != "time"}, meta=meta)


# ----------------------------------------------------------------------------- balance.txt

_LEGEND = re.compile(r"^(.+?)\s*= '(.)' / '(.)'\s*$", re.M)
_BSTEP = re.compile(r"^Step=\s*(\d+)\s+sec=\s*(\S+)\s+Nf=\s*(\d+)\s+(\S*)\s*$", re.M)


def _balance_files(ctx):
    f = output_file(ctx, "balance.txt")
    return [f] if f else []


def _balance(ctx):
    path = _balance_files(ctx)[0]
    with open(path, "rb") as f:
        head = f.read(8192).decode("latin-1")
    legend = {}
    for name, useful, imbal in _LEGEND.findall(head):
        n = _name(name)
        legend.setdefault(useful, n)                         # a character used twice in the legend keeps its first meaning
        legend.setdefault(imbal, n + "_imbal")

    def parse(text):
        rows, used = [], 0
        for m in _BSTEP.finditer(text):
            if not text.endswith("\n") and m.end() == len(text):
                break
            sec, bar = float(m.group(2)), m.group(4)
            r = {"step": float(m.group(1)), "nf": float(m.group(3)), "sec": sec}
            if bar:
                per = sec / len(bar)
                for ch in bar:
                    k = legend.get(ch, "other")
                    r[k] = r.get(k, 0.0) + per
            rows.append(r)
            used = m.end()
        return rows, used
    raw = _incremental(ctx, "balance", path, parse)
    # balance.txt has no time column: take it from cpu.txt by step (both are written every step)
    cpu = _cpu_files(ctx)
    t = raw["step"].copy()
    if cpu:
        c = _incremental(ctx, "cpu", cpu[0], _parse_cpu)
        steps, times = c["step"], c["time"]
        last = {}
        for s, tt in zip(steps, times):
            last[s] = tt
        t = np.array([last.get(s, np.nan) for s in raw["step"]])
    keep = np.isfinite(t)
    cols = {k: np.nan_to_num(v)[keep] if k not in ("step", "nf", "sec") else v[keep] for k, v in raw.items()}
    meta = {k: {"label": k.replace("_imbal", " (waiting)").replace("_", " ") + " seconds", "unit": "s", "group": "waiting" if k.endswith("_imbal") else "useful"}
            for k in cols if k not in ("step", "nf", "sec")}
    meta.update({"sec": {"label": "seconds of the step", "unit": "s", "group": "steps", "style": "dots"}, "nf": {"label": "active elements (Nf)", "group": "steps"}})
    return Table(t[keep], cols, meta=meta, note=None if cpu else "no cpu.txt: x is the step number")


plugins.register_series("cpu", _cpu, _cpu_files, label="GIZMO cpu.txt (wall clock, modules, waiting)", defaults=["wall_h"], version=2)
plugins.register_series("timebin", _timebin, _timebin_files, label="GIZMO timebin.txt (time steps, bins)", defaults=["dt_sys"], version=1)
plugins.register_series("timings", _timings, _timings_files, label="GIZMO timings.txt (tree walk)", defaults=["balance"], version=1)
plugins.register_series("balance", _balance, _balance_files, label="GIZMO balance.txt (useful / waiting per module)", defaults=["sec"], version=1)
