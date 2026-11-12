"""What the server is doing right now, for the viewer's progress bar (GET /api/progress).

One render runs at a time (the server's render lock), so one module-level record is enough:
  begin(label)                    a render starts (label: the snapshot and field)
  stage(name, estimate=None)      a new stage: "reading", "rendering", ...; estimate = expected seconds of the stage when known
  bytes_read(n) / bytes_total(n)  reading progress: io.load_snapshot reports every block it reads
  end()                           the render is done (also on an error)
  current()                       {busy, label, stage, fraction (0..1 or None), elapsed, stage_elapsed}
Splatting is compiled code without a progress hook: its fraction is time-based, the elapsed time over the estimate from the speed of the last renders
(seconds per particle), capped below 1. The record is advisory: nothing waits on it, and a missing begin() only makes the bar show nothing.
"""
import threading
import time

_lock = threading.Lock()
_s = {"busy": False}
_rate = {}          # stage name -> seconds per unit of work (exponential average of the last renders)


def begin(label):
    with _lock:
        _s.clear()
        _s.update(busy=True, label=str(label), stage="starting", t0=time.time(), ts=time.time(), done=0, total=0, estimate=None)


def stage(name, estimate=None):
    with _lock:
        if _s.get("busy"):
            _s.update(stage=name, ts=time.time(), done=0, total=0, estimate=estimate)


def bytes_total(n):
    with _lock:
        if _s.get("busy"):
            _s["total"] = _s.get("total", 0) + int(n)


def bytes_read(n):
    with _lock:
        if _s.get("busy"):
            _s["done"] = _s.get("done", 0) + int(n)


def end():
    with _lock:
        _s.clear()
        _s["busy"] = False


def expect(name, units):
    """the expected seconds of stage `name` for `units` of work (particles), from the speed of the last renders; None before the first one."""
    r = _rate.get(name)
    return None if r is None or units <= 0 else r * units


def learn(name, units, seconds):
    """record the speed of a finished stage (exponential average, so a change of cores or field settles in a few renders)."""
    if units > 0 and seconds > 0:
        r = seconds / units
        _rate[name] = r if name not in _rate else 0.6 * _rate[name] + 0.4 * r


def current():
    with _lock:
        if not _s.get("busy"):
            return {"busy": False}
        now = time.time()
        frac = None
        if _s.get("total"):
            frac = min(1.0, _s["done"] / _s["total"])
        elif _s.get("estimate"):
            frac = min(0.97, (now - _s["ts"]) / _s["estimate"])
        return {"busy": True, "label": _s["label"], "stage": _s["stage"], "fraction": frac,
                "elapsed": round(now - _s["t0"], 2), "stage_elapsed": round(now - _s["ts"], 2),
                "estimate": _s.get("estimate"), "bytes": [_s.get("done", 0), _s.get("total", 0)]}
