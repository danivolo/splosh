"""Physical cores: the rendering kernels gain nothing from hyperthreads (cluster, bench/scaling.py 2026-11-13: one socket, 10 threads on its 10 cores
vs 20 threads on its cores and their hyperthreads: x1.05), so worker counts are planned from physical cores, not logical CPUs."""
import os


def physical_cores():
    """the number of distinct physical cores among the CPUs this process may run on (its affinity), from /sys topology; the logical count where
    the topology is not available."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:                                    # pragma: no cover (not Linux)
        return os.cpu_count() or 1
    seen = set()
    for c in cpus:
        try:
            base = f"/sys/devices/system/cpu/cpu{c}/topology/"
            with open(base + "core_id") as f1, open(base + "physical_package_id") as f2:
                seen.add((f2.read().strip(), f1.read().strip()))
        except OSError:
            seen.add(("cpu", c))
    return max(1, len(seen))


def worker_threads(requested=None):
    """threads for parallel work: the requested (or NUMBA_NUM_THREADS) count, at most the physical cores available."""
    t = int(requested or os.environ.get("NUMBA_NUM_THREADS") or (os.cpu_count() or 4))
    return max(1, min(t, physical_cores()))
