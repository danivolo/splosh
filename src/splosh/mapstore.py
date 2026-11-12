"""mapstore: an on-disk cache of rendered maps (one .npz per map: `img` float32 and `meta` JSON), so repeated requests are instant.

Key = sha1 of (snapshot signature, the View with floats rounded to 6 significant digits, schema/kernel version). The snapshot signature is a content hash of the
header attributes, the file size and 8 sampled 64 kB blocks (fast also for GB files; `full=True` hashes everything): a copied or renamed snapshot hits the same
entries, a changed one does not (a change confined to unsampled bytes of an unchanged-size file would be missed: use full=True when in doubt)."""
import hashlib
import json
import os
import tempfile
import time

import h5py
import numpy as np

from .io import load_snapshot
from .maps import View, render_map, required_gas_fields

SCHEMA = "splosh-map-1"
_sig_cache = {}


def snapshot_hash(path, full=False):
    path = os.path.abspath(path)
    st = os.stat(path)
    ck = (path, st.st_size, st.st_mtime_ns, full)
    if ck in _sig_cache:
        return _sig_cache[ck]
    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    with h5py.File(path, "r") as f:
        for k in sorted(f["Header"].attrs):
            h.update(k.encode()); h.update(np.asarray(f["Header"].attrs[k]).tobytes())
    with open(path, "rb") as fh:
        if full:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        else:
            for k in range(8):
                fh.seek(max(0, (st.st_size - 65536) * k // 7))
                h.update(fh.read(65536))
    _sig_cache[ck] = h.hexdigest()
    return _sig_cache[ck]


def _norm(v):
    return float(f"{v:.6g}") if isinstance(v, float) else v


LATER_DEFAULTS = {"tilt": "none", "kappa": 0.0}     # View fields added after maps were stored: left out of the key at their default, so old keys stay valid


def _view_dict(view, skip=()):
    return {k: _norm(v) for k, v in view.as_dict().items() if k not in skip and not (k in LATER_DEFAULTS and v == LATER_DEFAULTS[k])}


def view_key(snap_sig, view):
    d = _view_dict(view)
    return hashlib.sha1(json.dumps({"snap": snap_sig, "view": d, "schema": SCHEMA}, sort_keys=True).encode()).hexdigest()[:24]


def default_root():
    return os.environ.get("SPLOSH_STORE") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "splosh", "maps")


class MapStore:
    def __init__(self, root=None):
        self.root = root or default_root()
        os.makedirs(self.root, exist_ok=True)
        self.hits = self.misses = 0

    def _path(self, key):
        return os.path.join(self.root, key + ".npz")

    def get(self, key):
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with np.load(p) as d:
                return d["img"], json.loads(str(d["meta"]))
        except Exception:
            return None

    def put(self, key, img, meta):
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=self.root)
        os.close(fd)
        np.savez(tmp, img=img.astype(np.float32), meta=json.dumps(meta))
        os.replace(tmp, self._path(key))

    def stats(self):
        fs = [f for f in os.listdir(self.root) if f.endswith(".npz")]
        return {"root": self.root, "maps": len(fs), "bytes": sum(os.path.getsize(os.path.join(self.root, f)) for f in fs), "hits": self.hits, "misses": self.misses}

    def clear(self):
        for f in os.listdir(self.root):
            if f.endswith(".npz"):
                os.remove(os.path.join(self.root, f))

    def get_or_render(self, path, view, snap=None, full_hash=False):
        """(img, meta, hit). `snap`: an already loaded snapshot dict (skips reading the file on a miss)."""
        view.validate()
        key = view_key(snapshot_hash(path, full_hash), view)
        got = self.get(key)
        if got is not None:
            self.hits += 1
            return got[0], got[1], True
        self.misses += 1
        t0 = time.perf_counter()
        if snap is None:
            snap = load_snapshot(path, gas_fields=tuple(sorted(required_gas_fields(view))))
        t1 = time.perf_counter()
        img, meta = render_map(snap, view)
        meta["render_seconds"] = round(time.perf_counter() - t1, 4)
        meta["load_seconds"] = round(t1 - t0, 4)
        meta["snapshot"] = os.path.basename(path)
        meta["key"] = key
        self.put(key, img, meta)
        return img, meta, False


TILE_SCHEMA = "splosh-tile-1"


def tile_key(snap_sig, view, tile, tw, ox, oy, ix, iy, level=None):
    """Key of one tile: the snapshot signature, the View without its window (cx, cy, width, npix: the tile geometry replaces them), the tile size in pixels, the
    tile grid (world width `tw` of a tile, grid origin `ox`, `oy`), the tile indices and, for a power-of-two zoom level, the level. Floats of the view are rounded
    like `view_key`; the grid geometry to 12 significant digits (it is computed, not typed)."""
    d = _view_dict(view, ("cx", "cy", "width", "npix"))
    g = {"tile": int(tile), "tw": float(f"{tw:.12g}"), "ox": float(f"{ox:.12g}"), "oy": float(f"{oy:.12g}"), "ix": int(ix), "iy": int(iy), "level": None if level is None else int(level)}
    return hashlib.sha1(json.dumps({"snap": snap_sig, "view": d, "tile": g, "schema": TILE_SCHEMA}, sort_keys=True).encode()).hexdigest()[:24]
