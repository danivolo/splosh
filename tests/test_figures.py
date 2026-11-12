"""shot / sheet / movie: pictures checked programmatically (size, ink, colour bar, limit policies, sidecar round trip), a fake ffmpeg on PATH for the command line."""
import json
import os
import stat
import subprocess
import sys

import numpy as np
import pytest

import splosh
from splosh import bitmapfont, cli, figures, imaging, limits, provenance
from splosh.mapstore import MapStore
from splosh.maps import View

NPIX = 128
VIEW = View(center="bh", width=8.0, npix=NPIX)
REAL = os.environ.get("SPLOSH_REAL_SNAPSHOT")


@pytest.fixture
def store(tmp_path):
    return MapStore(str(tmp_path / "store"))


def _ink(rgb, bg=figures.BG):
    return np.any(rgb[..., :3] != np.array(bg, np.uint8), axis=2)


# ----------------------------------------------------------------------------- pieces

def test_font_and_ticks():
    m = bitmapfont.text_mask("t = 0.25", 2)
    assert m.shape == bitmapfont.text_size("t = 0.25", 2) and m.any() and m.shape[0] == 18
    assert bitmapfont.text_mask("~~", 1).any() and bitmapfont.text_mask("é", 1).any()          # unknown glyph -> '?'
    assert not np.array_equal(bitmapfont.text_mask("a", 1), bitmapfont.text_mask("b", 1)[:, :5])
    assert figures.fmt_tick(1e-5) == "1e-5" and figures.fmt_tick(1e6) == "1e6" and figures.fmt_tick(0.01) == "0.01" and figures.fmt_tick(2500.0) == "2500"
    assert figures.make_ticks(1e-4, 1e2, "log") == pytest.approx([1e-4, 1e-2, 1, 1e2])          # 7 decades: thinned to at most 6 labels
    assert figures.make_ticks(1e-3, 1e0, "log") == pytest.approx([1e-3, 1e-2, 1e-1, 1])
    t = figures.make_ticks(3.0, 40.0, "log")
    assert t[0] >= 3.0 and t[-1] <= 40.0 and len(t) >= 2
    assert figures.make_ticks(0.0, 10.0, "linear") == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    a = figures.make_ticks(0.0, 100.0, "asinh")
    assert len(a) == 5 and a[0] == 0.0 and a[-1] == pytest.approx(100.0, rel=1e-2) and a == sorted(a)


def test_units_and_time_text():
    n = figures.unit_names({"UnitLength_In_CGS": 3.0857e18, "UnitMass_In_CGS": 4.0e35, "UnitVelocity_In_CGS": 1e5})
    assert n["L"] == "pc" and n["V"] == "km/s" and n["M"] == "M" and n["notes"] == ["M = 4e+35 g"]
    assert "Myr" in figures.time_text(1.0, n) or "kyr" in figures.time_text(1.0, n)
    assert figures.time_text(1.5, figures.unit_names({})) == "t = 1.5"
    assert figures.unit_names({"UnitMass_In_CGS": 1.989e33})["M"] == "Msun"


def test_limit_policies():
    rng = np.random.default_rng(0)
    maps = [10 ** rng.normal(loc=k, size=(40, 40)).astype(np.float32) for k in range(4)]
    maps[1][0, 0] = np.nan
    acc = limits.Accumulator("log")
    for m in maps:
        acc.add(m)
    allv = np.concatenate([m[np.isfinite(m)] for m in maps])
    assert acc.result() == (pytest.approx(float(allv.min())), pytest.approx(float(allv.max())))
    lo, hi = acc.result((2, 98))
    assert (lo, hi) == pytest.approx(tuple(np.percentile(allv, (2, 98))), rel=1e-5)
    assert limits.resolve("global", "log", None, 5.0, None, accumulator=acc)[0] == 5.0
    assert limits.resolve("frame", "log", (5, 95), None, None, img=maps[2]) == imaging.limits(maps[2], (5, 95), "log")
    assert limits.resolve("fixed", "log", None, 1.0, 9.0) == (1.0, 9.0)
    with pytest.raises(ValueError):
        limits.resolve("fixed", "log", None, 1.0, None)
    with pytest.raises(ValueError):
        limits.resolve("sometimes", "log", None, 1.0, 2.0)
    sub = limits.Accumulator("linear", cap=100)                      # subsampled: min / max stay exact
    sub.add(np.arange(10000, dtype=np.float32).reshape(100, 100))
    assert sub.result() == (0.0, 9999.0)
    assert limits.parse_pct("1,99.5") == (1.0, 99.5) and limits.parse_pct(None) is None
    with pytest.raises(SystemExit):
        limits.parse_pct("50,10")


# ----------------------------------------------------------------------------- shot

def test_shot_picture(snapfile, store):
    pic = splosh.shot(snapfile, VIEW, title="a title", store=store)
    assert isinstance(pic, splosh.Picture) and pic.rgba.dtype == np.uint8 and pic.rgba.shape[2] == 4 and (pic.rgba[..., 3] == 255).all()
    ny, nx = pic.rgba.shape[:2]
    assert ny > NPIX and nx > NPIX + 30                              # map + colour bar + labels + title + footer
    ink = _ink(pic.rgba)
    assert 0.3 < ink.mean() < 0.95                                   # not blank, not all ink
    assert ink[:16].any()                                            # title rows
    assert ink[-24:].any()                                           # footer text
    right = pic.rgba[:, NPIX + 15:, :3].reshape(-1, 3)
    assert len(np.unique(right, axis=0)) > 100                       # the colour bar's gradient
    assert pic.png()[:8] == b"\x89PNG\r\n\x1a\n" and np.asarray(pic).shape == pic.rgba.shape
    lo, hi = pic.limits
    assert 0 < lo < hi
    no_bar = splosh.shot(snapfile, VIEW, colorbar=False, store=store, vmin=lo, vmax=hi)
    assert no_bar.rgba.shape[1] < nx and no_bar.sidecar["limits"]["policy"] == "fixed" and no_bar.limits == (lo, hi)
    assert splosh.shot(snapfile, VIEW, store=store).sidecar["map_cache_hit"] is True


def test_shot_text_and_scalebar_regions(snapfile, store):
    with_ = splosh.shot(snapfile, VIEW, store=store, font_scale=1)
    wo = splosh.shot(snapfile, VIEW, store=store, font_scale=1, scalebar=False)
    assert with_.rgba.shape == wo.rgba.shape
    diff = np.any(with_.rgba != wo.rgba, axis=2)
    ys, xs = np.nonzero(diff)
    assert diff.any() and ys.min() > with_.rgba.shape[0] // 2 and xs.max() < NPIX      # scale bar + its label: lower left of the map only
    big = splosh.shot(snapfile, VIEW, store=store, font_scale=3)
    assert big.rgba.shape[0] > with_.rgba.shape[0]


def test_shot_stretches_and_slice(snapfile, store):
    for scale in ("linear", "asinh", "log"):
        pic = splosh.shot(snapfile, VIEW.with_(field="u", mode="slice"), scale=scale, cmap="inferno", store=store)
        assert pic.sidecar["stretch"] == scale and _ink(pic.rgba).mean() > 0.3


def test_sidecar_content_and_roundtrip(snapfile, store, tmp_path):
    out = str(tmp_path / "img.png")
    pic = splosh.shot(snapfile, VIEW.with_(align="bh", field="u", hscale=2.0), cmap="magma", scale="asinh", title="T", store=store)
    files = pic.save(out)
    assert files == [out, str(tmp_path / "img.json")]
    sc = json.load(open(files[1]))
    for k in ("schema", "splosh_version", "git_commit", "snapshots", "view", "view_summary", "limits", "colormap", "stretch", "wall_seconds", "outputs", "created"):
        assert k in sc
    assert sc["snapshots"][0]["path"] == os.path.abspath(snapfile) and sc["snapshots"][0]["time"] == pytest.approx(0.25)
    assert sc["view_summary"]["align"] == "bh" and sc["view_summary"]["field"] == "u" and sc["view_summary"]["hscale"] == 2.0 and "NOT the code" in sc["view_summary"]["kernel"]
    assert sc["colormap"] == "magma" and sc["stretch"] == "asinh" and sc["limits"]["vmin"] == pic.limits[0]
    assert sc["splosh_version"] == provenance.version()
    cli.main(["shot", "--from-sidecar", files[1], "--store", str(tmp_path / "store")])
    assert open(str(tmp_path / "img_repro.png"), "rb").read() == open(out, "rb").read()          # same picture, byte for byte
    cli.main(["shot", "--from-sidecar", files[1], "--out", str(tmp_path / "other.png"), "--no-cache"])
    assert open(str(tmp_path / "other.png"), "rb").read() == open(out, "rb").read()
    assert provenance.read_sidecar(files[1])["command"] == "shot"
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "other"}')
    with pytest.raises(SystemExit):
        provenance.read_sidecar(str(bad))


def test_sidecar_name_collision_and_git(snapfile, store, tmp_path):
    pic = splosh.shot(snapfile, VIEW, store=store)
    pic.save(str(tmp_path / "x.png"))
    (tmp_path / "x.svg").write_text("")
    assert provenance.sidecar_path(str(tmp_path / "x.svg")) == str(tmp_path / "x.svg.json")
    assert provenance.sidecar_path(str(tmp_path / "x.png")) == str(tmp_path / "x.json")
    assert provenance.git_commit() is None or isinstance(provenance.git_commit(), str)


def test_cli_shot(snapfile, tmp_path, capsys):
    out = str(tmp_path / "s.png")
    cli.main(["shot", snapfile, "--width", "6", "--npix", "96", "--field", "u", "--scale", "linear", "--vmin", "1", "--vmax", "4", "--title", "cli", "--out", out, "--store", str(tmp_path / "st")])
    assert "limits 1 .. 4" in capsys.readouterr().out and os.path.getsize(out) > 500
    sc = json.load(open(str(tmp_path / "s.json")))
    assert sc["limits"] == {"policy": "fixed", "pct": None, "vmin": 1.0, "vmax": 4.0} and sc["title"] == "cli"
    with pytest.raises(SystemExit):
        cli.main(["shot", snapfile, "--out", str(tmp_path / "bad.tif")])
    with pytest.raises(SystemExit):
        cli.main(["shot", snapfile])


def test_shot_pdf_needs_mpl_only_when_asked(snapfile, store, tmp_path):
    r = subprocess.run([sys.executable, "-c", "import splosh, splosh.figures, splosh.cli, sys; assert 'matplotlib' not in sys.modules"], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0, r.stderr
    mpl = pytest.importorskip("matplotlib")
    out = str(tmp_path / "f.pdf")
    cli.main(["shot", snapfile, "--width", "6", "--npix", "96", "--out", out, "--store", str(tmp_path / "st")])
    assert open(out, "rb").read(5) == b"%PDF-" and os.path.exists(str(tmp_path / "f.json"))
    pic = splosh.shot(snapfile, VIEW, backend="mpl", store=store)
    assert pic.rgba.shape[2] == 4 and _ink(pic.rgba, (255, 255, 255)).any()
    with pytest.raises(SystemExit):
        splosh.shot(snapfile, VIEW, store=store).save(str(tmp_path / "g.pdf"))


# ----------------------------------------------------------------------------- sheet

def _all_maps(snapdir, view, store):
    return [store.get_or_render(os.path.join(snapdir, f), view)[0] for f in sorted(os.listdir(snapdir))]


def test_sheet_global_limits_and_layout(snapdir, store, tmp_path):
    pic = splosh.contact_sheet(snapdir, VIEW, ncols=2, store=store)
    maps = _all_maps(snapdir, VIEW, store)
    pos = np.concatenate([m[np.isfinite(m) & (m > 0)] for m in maps])
    lim = pic.sidecar["limits"]
    assert lim["policy"] == "global" and lim["vmin"] == pytest.approx(float(pos.min())) and lim["vmax"] == pytest.approx(float(pos.max()))
    assert len(pic.sidecar["snapshots"]) == 3 and pic.sidecar["ncols"] == 2
    ny, nx = pic.rgba.shape[:2]
    assert nx > 2 * NPIX and ny > 2 * NPIX and _ink(pic.rgba).mean() > 0.3
    assert len(np.unique(pic.rgba[:, 2 * NPIX + 8:, :3].reshape(-1, 3), axis=0)) > 80          # colour bar
    p2 = splosh.contact_sheet(snapdir, VIEW, ncols=2, store=store, limits="global", pct=(1, 99))
    assert p2.sidecar["map_cache_hits"] == 3                                                       # reruns render nothing
    lo, hi = np.percentile(pos, (1, 99))
    assert p2.sidecar["limits"]["vmin"] == pytest.approx(float(lo), rel=1e-4) and p2.sidecar["limits"]["vmax"] == pytest.approx(float(hi), rel=1e-4)
    p3 = splosh.contact_sheet(snapdir, VIEW, store=store, limits="global", vmax=1.0)
    assert p3.sidecar["limits"]["vmax"] == 1.0 and p3.sidecar["limits"]["vmin"] == pytest.approx(float(pos.min()))


def test_sheet_frame_and_fixed_policies(snapdir, store):
    fr = splosh.contact_sheet(snapdir, VIEW, store=store, limits="frame", pct=(5, 95))
    per = fr.sidecar["limits"]["per_frame"]
    maps = _all_maps(snapdir, VIEW, store)
    for (lo, hi), m in zip(per, maps):
        assert (lo, hi) == imaging.limits(m, (5, 95), "log")
    assert len({tuple(p) for p in per}) == 3                       # different snapshots, different limits
    fx = splosh.contact_sheet(snapdir, VIEW, store=store, limits="fixed", vmin=1e-6, vmax=1e-2)
    assert fx.sidecar["limits"]["vmin"] == 1e-6 and fx.sidecar["limits"]["vmax"] == 1e-2
    with pytest.raises(ValueError):
        splosh.contact_sheet(snapdir, VIEW, store=store, limits="fixed")
    assert fr.rgba.shape[0] > NPIX and _ink(fr.rgba).mean() > 0.3


def test_selection(snapdir, tmp_path):
    names = lambda fs: [os.path.basename(f) for f in fs]
    assert names(figures.select_snapshots(snapdir)) == ["snap_000.hdf5", "snap_001.hdf5", "snap_002.hdf5"]
    assert names(figures.select_snapshots([snapdir], every=2)) == ["snap_000.hdf5", "snap_002.hdf5"]
    assert names(figures.select_snapshots(snapdir, snaps="2,0")) == ["snap_002.hdf5", "snap_000.hdf5"]
    one = os.path.join(snapdir, "snap_001.hdf5")
    assert names(figures.select_snapshots(one)) == ["snap_001.hdf5"]
    nested = tmp_path / "rundir"
    (nested).mkdir()
    os.symlink(snapdir, str(nested / "output"))
    assert len(figures.select_snapshots(str(nested))) == 3          # run directory with an output/ subdirectory
    with pytest.raises(SystemExit):
        figures.select_snapshots(snapdir, snaps="7")
    with pytest.raises(SystemExit):
        figures.select_snapshots(str(tmp_path / "nope"))
    with pytest.raises(SystemExit):
        figures.select_snapshots(snapdir, every=0)


def test_cli_sheet_and_replay(snapdir, tmp_path, capsys):
    out = str(tmp_path / "sheet.png")
    st = str(tmp_path / "st")
    cli.main(["sheet", snapdir, "--snaps", "0,2", "--npix", "96", "--width", "7", "--ncols", "2", "--limits", "global", "--pct", "1,99.5", "--title", "sheet", "--out", out, "--store", st])
    assert "sheet of 2 snapshots" in capsys.readouterr().out
    sc = json.load(open(str(tmp_path / "sheet.json")))
    assert sc["command"] == "sheet" and sc["limits"]["pct"] == [1.0, 99.5] and sc["snaps"] == ["0", "2"] and [s["name"] for s in sc["snapshots"]] == ["snap_000.hdf5", "snap_002.hdf5"]
    cli.main(["sheet", "--from-sidecar", str(tmp_path / "sheet.json"), "--store", st])
    assert open(str(tmp_path / "sheet_repro.png"), "rb").read() == open(out, "rb").read()
    cli.main(["sheet", snapdir, "--every", "2", "--npix", "96", "--limits", "frame", "--out", str(tmp_path / "fr.png"), "--store", st])
    cli.main(["sheet", "--from-sidecar", str(tmp_path / "fr.json"), "--store", st])
    assert open(str(tmp_path / "fr_repro.png"), "rb").read() == open(str(tmp_path / "fr.png"), "rb").read()
    with pytest.raises(SystemExit):
        cli.main(["sheet", snapdir, "--limits", "fixed", "--out", str(tmp_path / "x.png"), "--store", st])


# ----------------------------------------------------------------------------- movie

def _fake_ffmpeg(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "ffmpeg"
    exe.write_text('#!/bin/sh\necho "$@" > "$FAKE_FFMPEG_LOG"\nfor last; do :; done\necho video > "$last"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_FFMPEG_LOG", str(tmp_path / "ffmpeg.log"))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return str(exe)


def _no_ffmpeg(tmp_path, monkeypatch):
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


def test_movie_with_fake_ffmpeg(snapdir, tmp_path, monkeypatch, store):
    _fake_ffmpeg(tmp_path, monkeypatch)
    out = str(tmp_path / "m.mp4")
    r = splosh.movie(snapdir, out, VIEW, fps=7, store=store)
    assert r["encoded"] and os.path.exists(out) and r["frames_dir"] is None
    args = open(str(tmp_path / "ffmpeg.log")).read().split()
    assert args[:2] == ["-y", "-loglevel"] and args[args.index("-framerate") + 1] == "7" and "libx264" in args and "yuv420p" in args and args[-1] == out
    pattern = args[args.index("-i") + 1]
    assert pattern.endswith("frame_%05d.png") and os.path.dirname(pattern).startswith(str(tmp_path))          # frames next to the output, never in the system temp directory
    assert not os.path.exists(os.path.dirname(pattern))                                                       # removed after a successful encode
    sc = json.load(open(str(tmp_path / "m.json")))
    assert sc["command"] == "movie" and sc["fps"] == 7 and sc["encoded"] and len(sc["snapshots"]) == 3 and sc["ffmpeg_command"][0].endswith("ffmpeg")
    r2 = splosh.movie(snapdir, str(tmp_path / "m2.mp4"), VIEW, keep_frames=True, store=store)
    assert os.path.isdir(r2["frames_dir"]) and len(os.listdir(r2["frames_dir"])) == 3


def test_movie_without_ffmpeg_frames_and_limits(snapdir, tmp_path, monkeypatch, store, capsys):
    _no_ffmpeg(tmp_path, monkeypatch)
    out = str(tmp_path / "m.mp4")
    cli.main(["movie", snapdir, "--npix", "97", "--width", "7", "--out", out, "--store", str(tmp_path / "st"), "--fps", "5"])
    msg = capsys.readouterr().out
    assert "ffmpeg not found" in msg and "frame_00000.png" in msg and "-framerate 5" in msg and not os.path.exists(out)
    fdir = str(tmp_path / "m_frames")
    frames = sorted(os.listdir(fdir))
    assert frames == [f"frame_{k:05d}.png" for k in range(3)]

    def size(p):
        d = open(p, "rb").read()
        return int.from_bytes(d[16:20], "big"), int.from_bytes(d[20:24], "big")
    sizes = {size(os.path.join(fdir, f)) for f in frames}
    assert len(sizes) == 1 and all(v % 2 == 0 for v in next(iter(sizes)))                         # one size, even (yuv420p)
    sc = json.load(open(str(tmp_path / "m.json")))
    assert not sc["encoded"] and sc["frames_dir"] == fdir and sc["limits"]["policy"] == "global" and sc["limits"]["vmin"] < sc["limits"]["vmax"]
    first = open(os.path.join(fdir, frames[0]), "rb").read()
    cli.main(["movie", "--from-sidecar", str(tmp_path / "m.json"), "--out", str(tmp_path / "again.mp4"), "--store", str(tmp_path / "st")])
    assert open(str(tmp_path / "again_frames" / frames[0]), "rb").read() == first
    r = splosh.movie(snapdir, str(tmp_path / "f.mp4"), VIEW, limits="frame", store=store)           # per-frame limits vary but the frame size is constant
    assert len({size(os.path.join(r["frames_dir"], f)) for f in os.listdir(r["frames_dir"])}) == 1
    assert len(json.load(open(r["sidecar"]))["limits"]["per_frame"]) == 3


def test_movie_ffmpeg_failure_keeps_frames(snapdir, tmp_path, monkeypatch, store):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "ffmpeg"
    exe.write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    r = splosh.movie(snapdir, str(tmp_path / "bad.mp4"), VIEW, store=store)
    assert not r["encoded"] and "ffmpeg failed" in r["message"] and len(os.listdir(r["frames_dir"])) == 3 and r["frames_dir"].endswith("bad_frames")


# ----------------------------------------------------------------------------- real snapshot

@pytest.mark.skipif(not (REAL and os.path.exists(REAL)), reason="set SPLOSH_REAL_SNAPSHOT to a real snapshot file")
def test_real_snapshot_shot(tmp_path):
    out = str(tmp_path / "real.png")
    cli.main(["shot", REAL, "--center", "peak", "--width", "2", "--npix", "256", "--title", "real", "--out", out, "--store", str(tmp_path / "st")])
    sc = json.load(open(str(tmp_path / "real.json")))
    ink = _ink(figures_rgb(out))
    assert ink.mean() > 0.3 and sc["limits"]["vmin"] < sc["limits"]["vmax"] and sc["snapshots"][0]["time"] > 0
    cli.main(["shot", "--from-sidecar", str(tmp_path / "real.json"), "--store", str(tmp_path / "st")])
    assert open(str(tmp_path / "real_repro.png"), "rb").read() == open(out, "rb").read()


def figures_rgb(path):
    import zlib
    d = open(path, "rb").read()
    w, h = int.from_bytes(d[16:20], "big"), int.from_bytes(d[20:24], "big")
    i, idat = 8, b""
    while i < len(d):
        n = int.from_bytes(d[i:i + 4], "big")
        if d[i + 4:i + 8] == b"IDAT":
            idat += d[i + 8:i + 8 + n]
        i += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 3 * w)
    return raw[:, 1:].reshape(h, w, 3)
