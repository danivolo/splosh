"""Worker planning counts physical cores (hyperthreads do not speed up the kernels)."""
import os

from splosh.cpus import physical_cores, worker_threads


def test_physical_cores_within_the_affinity():
    n = physical_cores()
    assert 1 <= n <= len(os.sched_getaffinity(0))


def test_worker_threads_never_exceed_the_physical_cores():
    assert worker_threads(10 ** 6) == physical_cores()
    assert worker_threads(1) == 1
