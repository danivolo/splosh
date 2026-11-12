"""Tiled rendering: the assembled tiles equal the monolithic render_map (exact engine: float rounding; multi-resolution: the engine's own 1e-3 level error),
particles straddling tile corners and edges, particles larger than a tile and smaller than a pixel, slices and weighted fields, the tile grid and its cache keys,
the /api/tile endpoint and the CLI. A real snapshot is used when SPLOSH_TEST_SNAP names one."""
import json
import os
import struct
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from splosh import View, cli, imaging, render_map
from splosh import tiles as T
from splosh.mapstore import MapStore, tile_key
from splosh.server import make_server

TILE, NPIX, WIDTH = 64, 192, 6.0            # a 3 x 3 grid of tiles of world width 2: tile edges at -3, -1, 1, 3 (window grid), pixel 1/32
INT = (slice(T.HALO, -T.HALO),) * 2          # render_map clips (and renormalises) particles at its own window border; compare away from it


def make_snap(pos, h, n_extra=0, seed=0, mass=1e-3):
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    h = np.asarray(h, dtype=np.float64)
    if n_extra:
        pos = np.vstack([pos, rng.normal(size=(n_extra, 3)) * [1.2, 1.2, 0.3]])
        h = np.concatenate([h, 0.02 + 0.3 * rng.random(n_extra)])
    n = len(h)
    m = np.full(n, mass)
    return dict(time=0.1, header={}, Coordinates=pos, SmoothingLength=h.astype(np.float32), Masses=m.astype(np.float32),
                Density=(m / (h ** 3 * 0.1)).astype(np.float32), InternalEnergy=(2 + pos[:, 0] ** 2 + 0.3 * pos[:, 1]).astype(np.float32),
                Velocities=np.zeros((n, 3), np.float32))


def stress_snap():
    """random particles plus: five straddling tile corners / edges, one larger than a tile (H = 3.5 > 2), 200 smaller than a pixel (H = 0.004 < 1/32)."""
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0.05], [0, 1, 0]]
    s = make_snap(corners, [0.7, 0.2, 0.09, 0.45, 0.13], n_extra=3500)
    h = np.asarray(s["SmoothingLength"]).copy()
    h[5], s["Coordinates"][5] = 3.5, [0.3, 0.2, 0]
    h[6:206] = 0.004
    s["SmoothingLength"] = h
    s["Density"] = (s["Masses"] / (h ** 3 * 0.1)).astype(np.float32)
    return s


def view(**kw):
    return View(center="xyz:0,0,0", width=WIDTH, npix=NPIX, **kw)


def rel(a, b, mask=None):
    d = np.abs(a - b)[INT]
    return float(np.nanmax(d) / max(np.nanmax(np.abs(b)[INT]), 1e-30))


def weighted_close(v, snap, a, b, rtol):
    """weighted maps: compare where the weight is not negligible (ratio of two tiny sums is rounding noise elsewhere)."""
    rho, _ = render_map(snap, v.with_(field="rho"))
    ok = (rho > 1e-3 * rho.max())[INT]
    x, y = a[INT][ok], b[INT][ok]
    assert np.isfinite(x).all() == np.isfinite(y).all()
    f = np.isfinite(x) & np.isfinite(y)
    assert np.max(np.abs(x[f] - y[f]) / np.abs(y[f])) < rtol


@pytest.mark.parametrize("kw", [dict(exact=True), dict(mode="slice"), dict(mode="slice", field="u"), dict(mode="slice", field="u", weight="mass")])
def test_exact_engines_equal_monolithic(kw):
    s = stress_snap()
    v = view(**kw)
    a, _ = render_map(s, v)
    b, meta = T.render_tiled(s, v, tile=TILE, workers=1)
    assert b.shape == a.shape and b.dtype == np.float32
    if v.field == "rho":
        assert rel(b, a) < 1e-5                                # float rounding (observed 1e-7)
    else:
        weighted_close(v, s, b, a, 1e-4)
    assert meta["tiles"]["n_tiles"] == 9 and meta["view"]["npix"] == NPIX


def test_projection_weighted_and_multires_within_engine_tolerance():
    s = stress_snap()
    for kw, tol in ((dict(), 5e-3), (dict(field="u"), None), (dict(exact=True, field="u"), None)):
        v = view(**kw)
        a, _ = render_map(s, v)
        b, _ = T.render_tiled(s, v, tile=TILE)
        if tol:
            assert rel(b, a) < tol
        else:
            weighted_close(v, s, b, a, 1e-3 if not v.exact else 1e-4)


@pytest.mark.parametrize("H", [0.004, 0.02, 0.05, 0.1, 0.18, 0.5, 1.9, 2.5, 4.0])
@pytest.mark.parametrize("mode", ["proj", "slice"])
def test_single_particle_at_tile_corners(H, mode):
    """one particle on each kind of tile corner / edge, from far below a pixel to larger than a tile; exact engine, so identical to rounding."""
    s = make_snap([[-1, -1, 0], [1, -1, 0.01], [1.0 + 0.003, 1.0, 0], [0.0, 1.0, 0]], [H] * 4)
    v = view(exact=True, mode=mode)
    a, _ = render_map(s, v)
    b, _ = T.render_tiled(s, v, tile=TILE)
    assert rel(b, a) < 1e-5, (H, mode)
    if mode == "proj" and H <= 0.5:                            # whole footprints inside the window: the mass is conserved by tiles too
        dx = WIDTH / NPIX
        assert b.sum() * dx * dx == pytest.approx(4e-3, rel=2e-3)


def test_tiles_are_cropped_not_clipped_so_edge_weight_is_not_renormalised():
    """a 3-pixel particle on a tile edge keeps its exact weight in both tiles it touches (the engine renormalises over the part of the footprint inside the image;
    without the halo each tile would overcount)."""
    s = make_snap([[-1.0, 0.0, 0]], [0.1])
    v = view(exact=True)
    b, _ = T.render_tiled(s, v, tile=TILE)
    dx = WIDTH / NPIX
    assert b.sum() * dx * dx == pytest.approx(1e-3, rel=2e-3)
    left, right = b[:, :63].sum(), b[:, 64:].sum()
    assert left > 0 and right > 0


def test_workers_do_not_change_the_result_and_memmap_output(tmp_path):
    s = stress_snap()
    v = view()
    a, _ = T.render_tiled(s, v, tile=TILE, workers=1)
    b, _ = T.render_tiled(s, v, tile=TILE, workers=4)
    assert np.array_equal(a, b)
    path = str(tmp_path / "m.npy")
    c, _ = T.render_tiled(s, v, tile=TILE, out=path, workers=2)
    assert isinstance(c, np.memmap) and np.array_equal(np.load(path), a)
    buf = np.zeros((NPIX, NPIX), np.float32)
    d, _ = T.render_tiled(s, v, tile=TILE, out=buf)
    assert d is buf and np.array_equal(buf, a)


def test_npix_not_a_multiple_of_the_tile_is_cropped():
    s = stress_snap()
    v = view(exact=True).with_(npix=200)
    a, _ = render_map(s, v)
    b, meta = T.render_tiled(s, v, tile=TILE)
    assert b.shape == (200, 200) and meta["tiles"]["n_tiles"] == 16
    assert rel(b, a) < 1e-5


def test_larger_than_view_limit_image_in_bounded_memory(tmp_path):
    """npix above View's 8192: tiles of 1024 into a memory-mapped array; the mass is conserved."""
    s = make_snap([[-0.3, 0.2, 0], [0.5, 0.5, 0]], [0.4, 0.02], mass=1.0)
    v = View(center="xyz:0,0,0", width=2.0, npix=8256)
    with pytest.raises(ValueError):
        v.validate()
    img, meta = T.render_tiled(s, v, tile=1024, out=str(tmp_path / "big.npy"), workers=4)
    assert img.shape == (8256, 8256) and meta["tiles"]["n_tiles"] == 81
    dx = 2.0 / 8256
    assert float(img.sum(dtype=np.float64)) * dx * dx == pytest.approx(2.0, rel=2e-3)
    with pytest.raises(ValueError):
        T.render_tiled(s, v.with_(npix=70000), tile=1024)


# ----------------------------------------------------------------------------------------------------- global grid, levels, cache
def test_grid_covers_window_levels_are_powers_of_two():
    v = View(width=10.0, npix=512, cx=1.3, cy=-0.7)
    g = T.tile_grid(v, 256)
    assert g.base_width == 16.0 and g.level >= 0
    assert g.pixel <= v.width / v.npix < 2 * g.pixel                        # the coarsest level at least as fine as the request
    e0, e1 = g.extent(g.ix0, g.iy0), g.extent(g.ix1, g.iy1)
    assert e0[0] <= 1.3 - 5 and e1[1] >= 1.3 + 5 and e0[2] <= -0.7 - 5 and e1[3] >= -0.7 + 5
    assert g.nx * g.ny == len(g.tiles()) and g.tiles()[0] == (g.ix0, g.iy0)
    g2 = T.tile_grid(v.with_(width=5.0), 256, base_width=g.base_width)
    assert g2.level == g.level + 1 and g2.tw == pytest.approx(g.tw / 2)       # halving the width = one level up, same base
    g3 = T.tile_grid(v, 256, base_width=16.0, level=g.level + 2)
    assert g3.tw == g.tw / 4 and g3.nx >= 4 * g.nx - 3
    assert T.tile_grid(v, 512).level == g.level - 1                           # twice the pixels per tile: one level coarser
    with pytest.raises(ValueError):
        T.tile_grid(v, 100)
    w = T.window_grid(View(width=6.0, npix=200, cx=1.0, cy=2.0), 64)
    assert w.nx == w.ny == 4 and w.extent(0, 0)[0] == pytest.approx(-2.0) and w.pixel == pytest.approx(6.0 / 200)


def test_neighbouring_windows_share_tiles_and_keys():
    v = View(center="xyz:0,0,0", width=8.0, npix=512)
    g1 = T.tile_grid(v, 256, base_width=16.0)
    g2 = T.tile_grid(v.with_(cx=0.4, width=7.5), 256, base_width=16.0, level=g1.level)
    common = set(g1.tiles()) & set(g2.tiles())
    assert len(common) >= 0.4 * len(g1.tiles())
    k = lambda vv, ix, iy: tile_key("sig", vv, 256, g1.tw, 0.0, 0.0, ix, iy, g1.level)
    assert k(v, 1, 2) == k(v.with_(cx=0.4, cy=3.0, width=7.5, npix=777), 1, 2)         # the window does not enter the key
    assert k(v, 1, 2) != k(v, 2, 1) and k(v, 1, 2) != k(v.with_(field="u"), 1, 2) and k(v, 1, 2) != k(v.with_(mode="slice"), 1, 2)
    assert k(v, 1, 2) != tile_key("sig", v, 256, g1.tw / 2, 0.0, 0.0, 1, 2, g1.level + 1) and k(v, 1, 2) != tile_key("sig2", v, 256, g1.tw, 0.0, 0.0, 1, 2, g1.level)


@pytest.mark.parametrize("level,ix,iy", [(1, 0, 0), (2, -1, 1), (3, 2, -3), (-1, 0, 0)])
def test_render_tile_equals_render_map_of_its_extent(level, ix, iy):
    s = stress_snap()
    v = View(center="xyz:0,0,0", exact=True)
    img, meta = T.render_tile(s, v, level, ix, iy, tile=TILE, base_width=8.0)
    tw = 8.0 / 2.0 ** level
    cx, cy = (ix + 0.5) * tw, (iy + 0.5) * tw
    ref, _ = render_map(s, v.with_(cx=cx, cy=cy, width=tw * (TILE + 2 * T.HALO) / TILE, npix=TILE + 2 * T.HALO))
    assert img.shape == (TILE, TILE) and np.allclose(img, ref[T.HALO:-T.HALO, T.HALO:-T.HALO], rtol=1e-5, atol=1e-6 * max(ref.max(), 1e-30))
    assert meta["tile"]["extent"] == pytest.approx([ix * tw, (ix + 1) * tw, iy * tw, (iy + 1) * tw])


def test_global_grid_tiles_assemble_to_the_monolithic_image():
    """tiles of the global grid (origin 0, not aligned to any window) stitched together = a monolithic render of the same area at the same pixel size."""
    s = stress_snap()
    v = View(center="xyz:0,0,0", exact=True)
    g = T.tile_grid(v.with_(cx=0.0, width=4.0, npix=128), TILE, base_width=8.0, level=2)        # tw = 2: tiles -1..0 cover [-2, 2)^2
    assert (g.ix0, g.ix1, g.iy0, g.iy1) == (-1, 0, -1, 0)
    tiles = {(ix, iy): im for ix, iy, im, _, _ in T.iter_tiles(s, v, g)}
    big = np.block([[tiles[(-1, -1)], tiles[(0, -1)]], [tiles[(-1, 0)], tiles[(0, 0)]]])
    ref, _ = render_map(s, v.with_(cx=0.0, cy=0.0, width=4.0, npix=128))
    assert rel(big, ref) < 1e-5


def test_store_hits_and_lazy_snapshot_load(snapfile, tmp_path):
    store = MapStore(str(tmp_path / "ts"))
    v = View(center="bh", align="bh", width=4.0, npix=128, exact=True)
    img, meta = T.render_tile(snapfile, v, 1, 0, 0, tile=TILE, base_width=4.0, store=store)
    assert meta["cached"] is False and store.stats()["maps"] == 1
    img2, meta2 = T.render_tile(snapfile, v.with_(cx=1.0, width=2.0, npix=64), 1, 0, 0, tile=TILE, base_width=4.0, store=store)      # another window, same tile
    assert meta2["cached"] is True and np.array_equal(img, img2)
    # whole windows: the second assembly is served by the cache
    a, m1 = T.render_tiled(snapfile, v, tile=TILE, store=store)
    b, m2 = T.render_tiled(snapfile, v, tile=TILE, store=store)
    assert m1["tiles"]["cache_hits"] == 0 and m2["tiles"]["cache_hits"] == 4 and np.array_equal(a, b)
    ref, _ = MapStore(str(tmp_path / "ms")).get_or_render(snapfile, v)[:2]
    assert rel(a, ref) < 1e-5


def test_streaming_png_equals_png_writer(tmp_path):
    rng = np.random.default_rng(3)
    img = (10 ** rng.normal(size=(300, 220))).astype(np.float32)
    img[5:9, 5:9] = np.nan
    p1, p2 = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    lim = imaging.write_png(p1, img, cmap="magma", scale="log", vmin=0.1, vmax=10.0)
    lim2 = T.write_png_streaming(p2, img, cmap="magma", scale="log", vmin=0.1, vmax=10.0, band=64)

    def pixels(path):
        b = open(path, "rb").read()
        assert b[:8] == b"\x89PNG\r\n\x1a\n"
        k, idat, ihdr = 8, b"", None
        while k < len(b):
            n, tag = struct.unpack(">I4s", b[k:k + 8])
            data = b[k + 8:k + 8 + n]
            assert zlib.crc32(tag + data) & 0xFFFFFFFF == struct.unpack(">I", b[k + 8 + n:k + 12 + n])[0]
            if tag == b"IHDR":
                ihdr = struct.unpack(">IIBBBBB", data)
            elif tag == b"IDAT":
                idat += data
            k += 12 + n
        return ihdr, zlib.decompress(idat)
    assert pixels(p1) == pixels(p2) and lim == lim2


# ----------------------------------------------------------------------------------------------------- server
@pytest.fixture
def server(snapdir, tmp_path):
    srv, state = make_server([snapdir], token="s3cret", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", align="bh", width=6.0, npix=64))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown(); srv.server_close()


def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120)


def test_tile_endpoint(server):
    base, state = server
    u = base + "/api/tile?token=s3cret&snap=1&level=1&ix=0&iy=-1&tile=64&field=u&mode=slice&z0=0.02"
    r = get(u)
    meta = json.loads(r.headers["X-Splosh-Meta"])
    data = np.frombuffer(r.read(), dtype="<f4").reshape(64, 64)
    assert meta["n"] == 64 and meta["cached"] is False and meta["tile"]["level"] == 1 and meta["tile"]["base_width"] == 8.0 and meta["tile"]["tile_width"] == 4.0
    assert meta["tile"]["extent"] == [0.0, 4.0, -4.0, 0.0] and r.headers["Content-Type"] == "application/octet-stream" and "stats" in meta
    r2 = get(u)
    assert json.loads(r2.headers["X-Splosh-Meta"])["cached"] is True and np.array_equal(np.frombuffer(r2.read(), dtype="<f4").reshape(64, 64), data, equal_nan=True)
    # equals the library call with the same view (what the server quantises z0 to is irrelevant here: z0 = 0.02 is on the 1/4-pixel lattice of 6/64)
    v = state.view_from_query({"field": ["u"], "mode": ["slice"], "z0": ["0.02"]})
    from splosh.io import load_snapshot
    from splosh.maps import required_gas_fields
    snap = load_snapshot(state.paths[1], gas_fields=tuple(sorted(required_gas_fields(v) | {"Density"})))
    ref, _ = T.render_tile(snap, v, 1, 0, -1, tile=64, base_width=8.0)
    assert np.allclose(data, ref, rtol=1e-6, equal_nan=True)
    assert get(base + "/api/tile?token=s3cret&snap=0&level=2&ix=1&iy=1&tile=64&base=4").status == 200
    rg = get(u + "&field=rho", {"Accept-Encoding": "gzip"})                                   # same transport rules as /api/map
    assert rg.headers.get("Content-Encoding") in (None, "gzip") and rg.headers["Cache-Control"] == "no-store"


def test_tile_endpoint_token_and_bad_requests(server):
    base, state = server
    for url in (base + "/api/tile?snap=0&level=1&ix=0&iy=0", base + "/api/tile?token=bad&snap=0&level=1&ix=0&iy=0"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(url)
        assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + "/api/tile?snap=0&level=1&ix=0&iy=0", {"Host": "evil.example.com", "X-Splosh-Token": "s3cret"})
    assert e.value.code == 403
    assert get(base + "/api/tile?snap=0&level=1&ix=0&iy=0&tile=64", {"X-Splosh-Token": "s3cret"}).status == 200
    for bad, code in (("snap=0&level=1&ix=0", 400), ("snap=9&level=1&ix=0&iy=0", 400), ("snap=0&level=1&ix=0&iy=0&tile=100", 400), ("snap=0&level=99&ix=0&iy=0", 400),
                      ("snap=0&level=a&ix=0&iy=0", 400), ("snap=0&level=1&ix=0&iy=0&base=-1", 400), ("snap=0&level=1&ix=0&iy=0&field=nonexistent", 400)):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(base + "/api/tile?token=s3cret&" + bad)
        assert e.value.code == code, bad


# ----------------------------------------------------------------------------------------------------- cli
def test_cli_render_tiles_and_tiles_command(snapfile, tmp_path, capsys):
    base = ["--center", "xyz:0,0,0", "--width", "6", "--npix", "192", "--exact"]
    npy = str(tmp_path / "t.npy")
    cli.main(["render", snapfile, *base, "--tiles", "64", "--out", npy, "--store", str(tmp_path / "s1")])
    out = capsys.readouterr().out
    assert "9 tiles of 64" in out and "peak memory" in out
    ref, _ = render_map(__import__("splosh").load_snapshot(snapfile), View(center="xyz:0,0,0", width=6.0, npix=192, exact=True))
    assert rel(np.load(npy), ref) < 1e-5
    png = str(tmp_path / "t.png")
    cli.main(["render", snapfile, *base, "--tiles", "64", "--out", png, "--store", str(tmp_path / "s1")])
    assert "(9 from the cache)" in capsys.readouterr().out
    assert open(png, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
    ts = str(tmp_path / "tilestore")
    cli.main(["tiles", snapfile, *base, "--tile", "64", "--level", "1:2", "--out", ts])
    out = capsys.readouterr().out
    assert "level 1" in out and "level 2" in out
    idx = json.load(open(os.path.join(ts, "tiles_index.json")))
    assert [l["level"] for l in idx["levels"]] == [1, 2] and all(e["cached"] for l in idx["levels"] for e in l["tiles"])
    n = sum(len(l["tiles"]) for l in idx["levels"])
    assert MapStore(ts).stats()["maps"] == n
    cli.main(["tiles", snapfile, *base, "--tile", "64", "--level", "1", "--list", "--store", ts])
    assert "cached" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["tiles", snapfile, *base])
    with pytest.raises(SystemExit):
        cli.main(["tiles", snapfile, *base, "--level", "0:12", "--out", ts, "--max-tiles", "10"])


# ----------------------------------------------------------------------------------------------------- real snapshot
REAL = os.environ.get("SPLOSH_TEST_SNAP")


@pytest.mark.skipif(not REAL or not os.path.exists(REAL), reason="set SPLOSH_TEST_SNAP to a GIZMO snapshot")
@pytest.mark.parametrize("width", [10.0, 2.0])
def test_real_snapshot_tiled_equals_monolithic(width):
    import splosh
    from splosh.maps import required_gas_fields
    v = View(center="bh", align="bh", width=width, npix=512, exact=True)
    snap = splosh.load_snapshot(REAL, gas_fields=tuple(sorted(required_gas_fields(v) | required_gas_fields(v.with_(field="u")) | {"Density"})))
    for kw, tol in ((dict(), 1e-5), (dict(mode="slice"), 1e-5), (dict(field="u"), None), (dict(field="u", mode="slice"), None), (dict(exact=False), 5e-3)):
        vv = v.with_(**kw)
        a, _ = render_map(snap, vv)
        b, _ = T.render_tiled(snap, vv, tile=256)
        if tol:
            assert rel(b, a) < tol, kw
        else:
            weighted_close(vv, snap, b, a, 1e-3)
    c, _ = T.render_tiled(snap, v.with_(npix=1024), tile=512)           # 512-pixel tiles
    a, _ = render_map(snap, v.with_(npix=1024))
    assert rel(c, a) < 1e-5
