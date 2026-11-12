#!/usr/bin/env python
"""Headless-browser test of `splosh serve` + viewer.html (needs Playwright, which is not a dependency of splosh).

    PLAYWRIGHT_BROWSERS_PATH=... python tools/browser_smoke.py SNAPSHOT_DIR --server-python /path/to/splosh-env/bin/python [--shots OUT_DIR] [--synthetic]

Starts the server as a subprocess (random port, random token), opens the page and checks: the canvas is not blank, markers, zoom and stepping re-request maps,
field / slice switches, blink (two stored states flip the picture, with a visible indicator), difference mode (a state minus itself is zero; with --synthetic,
where snap_001 is snap_000 with every particle mass doubled, A - B = -A and log10(A/B) = -log10 2), linked panes (same window after pan and zoom, independent
fields and steps, unlinking), the hover read-out against an independent fetch of the same map (value and unit, a slice of a derived field in astro units), the
keyboard help, a narrow window, and that the page logs no console errors. Screenshots (--shots) are written at the interesting moments. Exit code 1 on failure."""
import argparse
import json
import math
import os
import re
import secrets
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

def set_hidden(pg, sel_id, value):
    """set a select that a slider stands in for (the slider is what the user moves; the select is the page's model)"""
    pg.evaluate("([i, v]) => { const s = document.getElementById(i); s.value = v; s.dispatchEvent(new Event('change')); syncSliders(); }", [sel_id, value])


def pick_field(pg, name):
    """choose a field through the grouped picker (the button, then its group, then the field), as a user does"""
    pg.click("#fieldbtn")
    if pg.locator(f'#fieldmenu [data-top="{name}"]').count():
        pg.click(f'#fieldmenu [data-top="{name}"]')
        return
    group = pg.evaluate("(n) => fieldGroups().groups.find(([, l]) => l.includes(n))[0]", name)
    pg.click(f'#fieldmenu [data-g="{group}"]')
    pg.click(f'#fieldmenu .items [data-f="{name}"]')


CANVAS_SIG = """(sel) => { const c = document.querySelectorAll('canvas.view')[sel || 0], d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let h = 0; for (let i = 0; i < d.length; i += 4 * 53) h = (h * 31 + d[i] * 3 + d[i + 1] * 5 + d[i + 2]) % 1000000007; return h; }"""
DISTINCT = """(sel) => { const c = document.querySelectorAll('canvas.view')[sel || 0], d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const seen = new Set(); for (let i = 0; i < d.length; i += 4 * 97) seen.add(d[i] + ',' + d[i + 1] + ',' + d[i + 2]); return seen.size; }"""


def fetch_map(url, params):
    """independent copy of a map: (float32 values as a flat list, meta dict) straight from the server, no browser code involved."""
    u = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(u.query))
    q.update({k: str(v) for k, v in params.items()})
    r = urllib.request.urlopen(f"{u.scheme}://{u.netloc}/api/map?" + urllib.parse.urlencode(q), timeout=120)
    meta = json.loads(r.headers["X-Splosh-Meta"])
    raw = r.read()
    return struct.unpack(f"<{len(raw) // 4}f", raw), meta


def fetch_json(url, path, params):
    """independent GET of a JSON endpoint (/overlay, /inspect, /api/runs ...) with the page's own token; params values may be lists (repeated keys)."""
    u = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(u.query))
    q.update({k: v if isinstance(v, (list, tuple)) else str(v) for k, v in params.items()})
    r = urllib.request.urlopen(f"{u.scheme}://{u.netloc}{path}?" + urllib.parse.urlencode(q, doseq=True), timeout=120)
    return json.loads(r.read())


OVL_PIXELS = """(sel) => { const c = document.querySelectorAll('canvas.ovl')[sel || 0], d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0, h = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) { n++; h = (h * 31 + i) % 1000000007; } return [n, h]; }"""


EXPECTED_400 = []


def start_server(a):
    token = secrets.token_urlsafe(12)
    srv = subprocess.Popen([a.server_python, "-m", "splosh", "serve", a.snapdir, *a.extra_run, "--token=" + token, "--port", "0", *shlex.split(a.serve_args), *(["--layouts", a.layouts] if a.layouts else []), *(["--analysis", a.analysis] if a.analysis else [])],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env=dict(os.environ, SPLOSH_RECENT=os.path.join(tempfile.mkdtemp(prefix="splosh_smoke_"), "recent.json")))   # never the user's own list
    t0 = time.time()
    while time.time() - t0 < 300:                                   # imports and the first scan of the snapshots are slow on a loaded machine
        line = srv.stdout.readline()
        m = re.search(r"open\s+(http://\S+)", line)
        if m:
            return srv, m.group(1)
        if not line and srv.poll() is not None:
            break
    srv.kill()
    sys.exit("server did not start")


def run(pg, url, a, check):
    shot = (lambda name: pg.screenshot(path=os.path.join(a.shots, name + ".png"))) if a.shots else (lambda name: None)
    ev = pg.evaluate
    wait = lambda js, t=120000: pg.wait_for_function(js, timeout=t)
    idle = lambda: wait("window.__splosh.idle()")
    pg.goto(url)
    wait("window.__splosh && window.__splosh.panes.length > 0 && window.__splosh.map")
    idle()
    nsnap = ev("window.__splosh.snaps.length")
    check(ev(DISTINCT) > 20, f"canvas shows an image ({ev(DISTINCT)} distinct colours in the sample)")
    check(ev("window.__splosh.markers.length") >= 1, "black-hole markers received")
    shot("viewer_first")

    # ---- the original behaviour: zoom, step, field, slice
    r0, w0 = ev("window.__splosh.requests"), ev("window.__splosh.map.meta.view.width")
    box = pg.locator("canvas.view").first.bounding_box()
    pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    pg.mouse.wheel(0, -600)
    wait(f"window.__splosh.requests > {r0} && window.__splosh.map.meta.view.width < {w0} * 0.9")
    check(True, f"zoom re-requested a map: width {w0} -> {ev('window.__splosh.map.meta.view.width')}")
    pg.keyboard.press("o")
    wait(f"window.__splosh.map.meta.view.width > {w0} * 0.99")
    idle()
    if nsnap > 1:
        r1 = ev("window.__splosh.requests")
        pg.keyboard.press("ArrowRight")
        wait(f"window.__splosh.requests > {r1} && window.__splosh.idx === 1")
        wait("window.__splosh.map.meta.snapshot === window.__splosh.snaps[1].name")
        check(True, "arrow key stepped to the next snapshot")
        pick_field(pg, "u")
        wait("window.__splosh.map.meta.view.field === 'u'")
        check(True, "field switch to u rendered")
        pg.select_option("#mode", "slice")
        wait("window.__splosh.map.meta.view.mode === 'slice'")
        check(True, "slice mode rendered")
        shot("viewer_zoomed_slice")
        pg.select_option("#mode", "proj")
        pick_field(pg, "rho")
        idle()
        pg.keyboard.press("ArrowLeft")
        wait("window.__splosh.idx === 0")
        idle()
    if nsnap < 2:
        return

    # ---- blink: A = first snapshot, B = last snapshot (a different realisation)
    last = nsnap - 1
    pg.keyboard.press("Shift+A")
    check(ev("window.__splosh.A.snap") == 0, "key a stores state A")
    pg.keyboard.press("Shift+Space")
    check("blink needs" in ev("document.getElementById('status').textContent"), "blink without B gives a hint, not an error")
    ev(f"(() => {{ const s = document.getElementById('snap'); s.value = {last}; s.dispatchEvent(new Event('input')); }})()")
    wait(f"window.__splosh.idx === {last}")
    idle()
    sigB = ev(CANVAS_SIG)
    pg.keyboard.press("Shift+B")
    check(ev("window.__splosh.B.snap") == last, "key b stores state B")
    pg.keyboard.press("Shift+Space")
    wait("window.__splosh.panes[0].blink === 'A'")
    idle()
    sigA = ev(CANVAS_SIG)
    check(ev("window.__splosh.idx") == 0 and sigA != sigB, "space blinks to A: other snapshot, other picture")
    badge = pg.locator(".badge").first
    check("BLINK A" in badge.inner_text() and badge.is_visible(), "indicator shows BLINK A")
    shot("blink_A")
    pg.keyboard.press("Shift+Space")
    idle()
    check(ev("window.__splosh.idx") == last and ev(CANVAS_SIG) == sigB and "BLINK B" in badge.inner_text(), "space blinks back to B: the picture of B again, indicator BLINK B")
    shot("blink_B")
    n_net = ev("window.__splosh.netRequests")
    for _ in range(4):
        pg.keyboard.press("Shift+Space")
    idle()
    check(ev("window.__splosh.netRequests") == n_net, "repeated blinking is served by the browser cache (no new map requests)")
    ev("window.__splosh.panes[0].blink = null")

    # ---- difference of a state with itself is zero
    pg.keyboard.press("Shift+A")
    pg.keyboard.press("Shift+B")
    pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'sub'")
    idle()
    mx = ev("Math.max(...Array.from(window.__splosh.map.data).map(Math.abs))")
    check(mx == 0, f"A - A is exactly 0 (max |d| = {mx})")
    check(ev("window.__splosh.map.data.length") > 0 and "A - B" in pg.locator(".badge").first.inner_text(), "difference indicator shown")
    pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'ratio'")
    idle()
    mx = ev("Math.max(...Array.from(window.__splosh.map.data).filter(Number.isFinite).map(Math.abs))")
    check(mx == 0, f"log10(A / A) is 0 (max |d| = {mx})")
    pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'off'")
    idle()

    # ---- a known difference
    ev("(() => { const s = document.getElementById('snap'); s.value = 0; s.dispatchEvent(new Event('input')); })()")
    wait("window.__splosh.idx === 0")
    idle()
    pg.keyboard.press("Shift+A")
    ev("(() => { const s = document.getElementById('snap'); s.value = 1; s.dispatchEvent(new Event('input')); })()")
    wait("window.__splosh.idx === 1")
    idle()
    pg.keyboard.press("Shift+B")
    pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'sub'")
    idle()
    shot("diff_sub")
    sym = ev("window.__splosh.lim")
    check(abs(sym[0] + sym[1]) <= 1e-12 * abs(sym[1]) and sym[1] > 0, f"difference limits are symmetric about 0: {sym}")
    pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'ratio'")
    idle()
    shot("diff_ratio")
    if a.synthetic:
        pg.keyboard.press("~")
        idle()
        pg.keyboard.press("~")
        wait("window.__splosh.panes[0].diff === 'sub'")
        idle()
        res = ev("""() => { const S = window.__splosh, p = S.panes[0], A = p.map.dA, d = p.map.data; let mx = 0, ref = 0;
            for (let i = 0; i < d.length; i++) { mx = Math.max(mx, Math.abs(d[i] + A[i])); ref = Math.max(ref, Math.abs(A[i])); } return [mx, ref]; }""")
        check(res[1] > 0 and res[0] <= 1e-5 * res[1], f"A - B recovers -A for doubled masses (max |d + A| = {res[0]:.2e}, max A = {res[1]:.3g})")
        pg.keyboard.press("~")
        wait("window.__splosh.panes[0].diff === 'ratio'")
        idle()
        r = ev("""() => { const d = window.__splosh.map.data; let mn = 1e9, mxv = -1e9, n = 0; for (const x of d) if (Number.isFinite(x)) { mn = Math.min(mn, x); mxv = Math.max(mxv, x); n++; } return [mn, mxv, n]; }""")
        target = math.log10(0.5)
        check(r[2] > 100 and abs(r[0] - target) < 1e-4 and abs(r[1] - target) < 1e-4, f"log10(A/B) = log10(1/2) = {target:.5f} everywhere (min {r[0]:.5f}, max {r[1]:.5f}, {r[2]} pixels)")
    pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'off'")
    idle()

    # ---- hover read-out against an independent copy of the map, in the chosen unit system
    pg.select_option("#units", "astro")
    wait("window.__splosh.map.meta.unit_system === 'astro'")
    idle()

    def hover_check(label, fx=0.31, fy=0.62, pane=0):
        p = ev(f"window.__splosh.paneParams(window.__splosh.panes[{pane}])")
        vals, meta = fetch_map(url, p)
        box = pg.locator("canvas.view").nth(pane).bounding_box()
        pg.mouse.move(box["x"] + fx * box["width"], box["y"] + fy * box["height"])
        txt = ev(f"document.querySelectorAll('.readout')[{pane}].textContent")
        v = ev(f"window.__splosh.panes[{pane}].view")
        wx, wy = v["cx"] + (fx - 0.5) * v["width"], v["cy"] + (0.5 - fy) * v["width"]
        e, n = meta["extent"], meta["n"]
        i, j = int(math.floor((wx - e[0]) / (e[1] - e[0]) * n)), int(math.floor((wy - e[2]) / (e[3] - e[2]) * n))
        want = vals[j * n + i]
        unit = meta["quantity_unit"]["unit"]
        want_txt = f"{want:.5g}" if math.isfinite(want) else "no data"
        shown = re.search(r"(\S+) " + re.escape(unit) + r"$", txt) if unit not in ("", "1") else re.search(r"(\S+)$", txt)
        got = float(shown.group(1)) if shown and shown.group(1) != "data" else None
        ok = bool(unit) and (unit in txt or unit == "1") and ((got is not None and math.isfinite(want) and abs(got - want) <= 1e-4 * abs(want)) or (not math.isfinite(want) and "no data" in txt))
        check(ok, f"hover {label}: read-out '{txt}' vs independent value {want_txt} {unit}")
        return txt, unit

    txt, unit = hover_check("rho projection")
    check(unit not in ("", "code"), f"unit text comes from the map metadata ({unit!r}) and the chosen system (astro)")
    pg.select_option("#mode", "slice")
    pick_field(pg, "pressure")
    wait("window.__splosh.map.meta.view.field === 'pressure' && window.__splosh.map.meta.view.mode === 'slice'")
    idle()
    hover_check("slice of the derived field pressure", 0.52, 0.48)       # not the exact centre: that is a pixel boundary (which pixel is ambiguous)
    shot("hover_slice_pressure")
    pick_field(pg, "mach")
    wait("window.__splosh.map.meta.view.field === 'mach'")
    idle()
    hover_check("slice of the derived field mach", 0.45, 0.55)
    pg.select_option("#units", "code")
    pg.select_option("#mode", "proj")
    pick_field(pg, "rho")
    wait("window.__splosh.map.meta.view.field === 'rho' && window.__splosh.map.meta.view.mode === 'proj' && window.__splosh.map.meta.unit_system === 'code'")
    idle()

    section_overlays(pg, url, a, check, shot, wait, idle, nsnap)
    section_phase(pg, url, a, check, shot, wait, idle)
    section_gpu(pg, url, a, check, shot, wait, idle)
    section_tiles(pg, url, a, check, shot, wait, idle)

    # ---- linked panes
    pg.keyboard.press("Shift+P")
    wait("window.__splosh.panes.length === 2")
    idle()
    shot("panes_2")
    check(ev("window.__splosh.linked") and ev("document.querySelectorAll('canvas.view').length") == 2, "key p opens a second pane, linked")
    box0 = pg.locator("canvas.view").nth(0).bounding_box()
    pg.mouse.move(box0["x"] + box0["width"] / 2, box0["y"] + box0["height"] / 2)
    pg.mouse.down()
    pg.mouse.move(box0["x"] + box0["width"] / 2 + 60, box0["y"] + box0["height"] / 2 + 30, steps=5)
    pg.mouse.up()
    pg.mouse.wheel(0, -500)
    wait("window.__splosh.panes[0].view.width < " + str(w0) + " * 0.9")
    idle()
    same = ev("""() => { const [p, q] = window.__splosh.panes; return [p.view.cx === q.view.cx, p.view.cy === q.view.cy, p.view.width === q.view.width,
        JSON.stringify(p.map.extent) === JSON.stringify(q.map.extent), p.view.cx !== 0 || p.view.cy !== 0]; }""")
    check(all(same), f"linked panes keep the same window after pan and zoom (centre, width, map extents equal; moved: {same})")
    pg.mouse.move(2, 2)                                               # off the panes: no cross-hair in the other pane
    check(ev(DISTINCT, 1) > 20 and ev(CANVAS_SIG, 0) == ev(CANVAS_SIG, 1), "both panes show the same picture (same snapshot and field)")
    # pane 2 gets another field; pane 1 keeps its own
    pg.evaluate("() => selectPane(1)")
    pick_field(pg, "u")
    wait("window.__splosh.panes[1].map.meta.view.field === 'u'")
    idle()
    check(ev("window.__splosh.panes[0].map.meta.view.field") == "rho" and ev(CANVAS_SIG, 0) != ev(CANVAS_SIG, 1), "each pane has its own field")
    shot("panes_2_fields")
    while ev("window.__splosh.panes[0].snap") > 0:
        pg.keyboard.press("ArrowLeft")
        idle()
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[0].snap === 1 && window.__splosh.panes[1].snap === 1")
    idle()
    check(True, "arrow steps both panes")
    pg.keyboard.press("Shift+ArrowRight") if nsnap > 2 else pg.keyboard.press("Shift+ArrowLeft")
    wait("window.__splosh.panes[0].snap === 1 && window.__splosh.panes[1].snap !== 1")
    idle()
    check(True, "Shift+arrow steps only the selected pane")
    pg.keyboard.press("Shift+L")
    check(not ev("window.__splosh.linked") and not pg.locator("#link").is_checked(), "key l unlinks")
    w_before = ev("window.__splosh.panes[0].view.width")
    pg.mouse.move(*(lambda b: (b["x"] + b["width"] / 2, b["y"] + b["height"] / 2))(pg.locator("canvas.view").nth(1).bounding_box()))
    pg.mouse.wheel(0, -500)
    wait(f"window.__splosh.panes[1].view.width < {w_before} * 0.95")
    idle()
    check(ev("window.__splosh.panes[0].view.width") == w_before, "unlinked: zooming one pane leaves the other alone")
    # hover with two panes reports in both
    box1 = pg.locator("canvas.view").nth(1).bounding_box()
    pg.mouse.move(box1["x"] + box1["width"] * 0.4, box1["y"] + box1["height"] * 0.4)
    check("u " in ev("document.querySelectorAll('.readout')[1].textContent"), "hover read-out in the second pane (field u)")
    pg.keyboard.press("Shift+L")
    wait("window.__splosh.linked")
    idle()
    check(ev("window.__splosh.panes[0].view") == ev("window.__splosh.panes[1].view"), "relinking brings the panes back to one window")
    pg.keyboard.press("Shift+P")
    pg.keyboard.press("Shift+P")
    wait("window.__splosh.panes.length === 4")
    idle()
    shot("panes_4")
    check(ev(DISTINCT, 3) > 20, "four panes drawn")
    pg.keyboard.press("Shift+P")
    wait("window.__splosh.panes.length === 1")
    idle()

    # ---- help overlay
    pg.keyboard.press("h")
    check(pg.locator("#help").is_visible() and "blink" in pg.locator("#help").inner_text(), "? opens the keyboard help")
    shot("help")
    pg.keyboard.press("Escape")
    check(not pg.locator("#help").is_visible(), "a key closes the help")

    section_mouse(pg, check, shot, wait, idle)
    section_explore(pg, url, a, check, shot, wait, idle)

    # ---- narrow window
    pg.keyboard.press("Shift+P")
    idle()
    pg.set_viewport_size({"width": 480, "height": 820})
    pg.wait_for_timeout(300)
    sw = ev("document.documentElement.scrollWidth")
    check(sw <= 480 + 1, f"narrow window: no horizontal overflow (scrollWidth {sw})")
    shot("narrow")
    pg.set_viewport_size({"width": 1100, "height": 820})
    # ---- centre / align rules from menus
    c0 = ev("document.getElementById('center').value")
    sel_c = "#center + .row select"
    pg.select_option(sel_c, "com")
    wait("window.__splosh.map.meta.frame.center_spec === 'com'")
    idle()
    pg.select_option(sel_c, "xyz:")
    ids_box_shown = ev("getComputedStyle(document.querySelector('#center + .row input')).display !== 'none'")
    pg.fill("#center + .row input", "0.5,0,0")
    pg.dispatch_event("#center + .row input", "change")
    wait("document.getElementById('center').value === 'xyz:0.5,0,0'")
    idle()
    pg.select_option(sel_c, c0 if c0 in ("bh", "com", "peak", "mass-weighted") else "bh")
    idle()
    check(ids_box_shown and ev("document.getElementById('center').value") == (c0 if c0 in ("bh", "com", "peak", "mass-weighted") else "bh"),
          "centre rule menu: com re-requests the map, xyz: asks for the point, back to the start value")
    if a.extra_run:
        section_runs(pg, url, a, check, shot, wait, idle)
    if a.layouts:
        section_layout(pg, url, a, check)


def section_layout(pg, url, a, check):
    """the layout editor page (/layout): open a layout, drag and resize panels on the grid, edit a panel's settings, see a validation error, undo, preview, save and read the file back."""
    u = urllib.parse.urlparse(url)
    ed = pg.context.browser.new_page(viewport={"width": 1500, "height": 1000})       # a page of its own (the viewer page came from browser.new_page, whose context takes no second page)
    errs = []
    ed.on("pageerror", lambda e: errs.append(str(e)))
    ed.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    shot = (lambda name: ed.screenshot(path=os.path.join(a.shots, name + ".png"), full_page=True)) if a.shots else (lambda name: None)
    ed.goto(f"{u.scheme}://{u.netloc}/layout?{u.query}")
    ed.wait_for_function("window.__ready === true", timeout=120000)
    ev = ed.evaluate
    wait = lambda js, t=120000: ed.wait_for_function(js, timeout=t)
    check(ev("Array.from(document.querySelectorAll('#files option')).map(o => o.value).includes('seed')"), "layout editor: lists the layout files of the directory")
    token = dict(urllib.parse.parse_qsl(u.query))["token"]
    links = [ev("document.getElementById('viewerlink').href")] + ([pg.evaluate("document.getElementById('layoutlink').href")] if pg.url.startswith("http") else [])
    check(links[0].endswith("/?token=" + urllib.parse.quote(token)) and all(("token=" + urllib.parse.quote(token)) in x for x in links) and (len(links) == 1 or "/layout?" in links[1]),
          f"layout editor: viewer and editor link to each other with the token {links}")
    ed.select_option("#files", "seed")
    ed.click("#load")
    wait("L.panel.length === 2 && V")
    check(ev("document.querySelectorAll('.pbox').length") == 2, "layout editor: opening a file draws its panels on the grid")
    wait("document.getElementById('pimg').naturalWidth > 100")                 # the preview of snapshot 0
    check(ev("document.getElementById('pimg').naturalWidth") > 100, "layout editor: preview picture arrives")
    shot("layout_editor_open")

    def box(pid):
        return ed.locator(f".pbox[data-id='{pid}']").bounding_box()
    g = ev("(() => { const g = geom(); return [g.cw, g.ch]; })()")
    # ---- drag the text panel two cells to the right (the seed has 4 columns; m at col 0, t at col 1)
    b = box("t")
    ed.mouse.move(b["x"] + 20, b["y"] + 12); ed.mouse.down(); ed.mouse.move(b["x"] + 20 + g[0] * 2, b["y"] + 12, steps=6); ed.mouse.up()
    wait("L.panel[1].col === 3")
    check(ev("[L.panel[1].row, L.panel[1].col, L.panel[0].row, L.panel[0].col]") == [0, 3, 0, 0], "layout editor: dragging moves a panel and pins the others")
    # ---- a drop onto another panel is refused
    b = box("t")
    ed.mouse.move(b["x"] + 20, b["y"] + 12); ed.mouse.down(); ed.mouse.move(b["x"] + 20 - g[0] * 3, b["y"] + 12, steps=6); ed.mouse.up()
    check(ev("L.panel[1].col") == 3, "layout editor: a drop that would overlap is refused")
    # ---- resize the map panel by one column
    b = box("m")
    ed.mouse.move(b["x"] + b["width"] - 4, b["y"] + b["height"] - 4); ed.mouse.down(); ed.mouse.move(b["x"] + b["width"] - 4 + g[0], b["y"] + b["height"] - 4, steps=6); ed.mouse.up()
    wait("L.panel[0].colspan === 2 && V.panel[0].colspan === 2")
    check(ev("L.panel[0].colspan") == 2 and ev("V.panel[0].colspan") == 2, "layout editor: the corner handle resizes a panel")
    # ---- edit a setting in the inspector: a colour map, then reset it
    ed.click(".pbox[data-id='m']")
    ed.select_option("#fields .field:has(> span[title=cmap]) select", "magma")
    wait("L.panel[0].cmap === 'magma'")
    check(ev("document.querySelector('#fields .field.set > span[title=cmap]') !== null"), "layout editor: an edited setting is marked and keeps only that key")
    ed.click("#fields .field:has(> span[title=cmap]) button")
    wait("!('cmap' in L.panel[0])")
    # ---- a product name replaces the inline settings
    ed.fill("#fields .field:has(> span[title=product]) input", "dens")
    ed.dispatch_event("#fields .field:has(> span[title=product]) input", "change")
    wait("L.panel[0].product === 'dens' && !('field' in L.panel[0])")
    check(ev("Object.keys(L.panel[0]).filter(k => ['field','width','npix','center'].includes(k)).length") == 0, "layout editor: giving a product removes the inline map settings")
    ed.click("#undo")
    wait("!('product' in L.panel[0])")
    check(ev("L.panel[0].field") == "rho", "layout editor: undo restores the inline settings")
    # ---- add a time-series panel and a column
    ed.click("[data-add=timeseries]")
    wait("L.panel.length === 3 && V && V.panel.length === 3")
    check(ev("L.panel[2].type") == "timeseries" and ev("V.panel[2].row") >= 0, "layout editor: + time series adds a panel on a free cell")
    ed.fill("#fields .field:has(> span[title=columns]) input", "gas_mass, gas_mass * 2")
    ed.dispatch_event("#fields .field:has(> span[title=columns]) input", "change")
    wait("L.panel[2].columns.length === 2")
    # ---- an error is shown, and clears when fixed
    ed.fill("#fields .field:has(> span[title=id]) input", "m")
    ed.dispatch_event("#fields .field:has(> span[title=id]) input", "change")
    wait("document.getElementById('err').classList.contains('on')")
    check("unique" in ev("document.getElementById('err').textContent"), "layout editor: a duplicate id shows the validation error")
    ed.click("#undo")
    wait("!document.getElementById('err').classList.contains('on')")
    check(True, "layout editor: undo clears the error")
    # ---- preview with the table: the series panel is drawn (no 'no scalars' note)
    n0 = ev("window.__previews")
    ed.click("#refresh")
    wait(f"window.__previews > {n0}")
    check("timeseries1" not in ev("document.getElementById('notes').textContent"), "layout editor: the series panel has data from scalars.csv")
    shot("layout_editor_edited")
    # ---- save, then the file comes back as saved
    ed.fill("#file", "smoke_out")
    ed.click("#save")
    wait("document.getElementById('sstat').textContent.startsWith('saved')")
    got = fetch_json(url, "/api/layout", {"name": "smoke_out"})
    check([p["id"] for p in got["raw"]["panel"]] == [p["id"] for p in ev("L.panel")] and got["raw"]["panel"][0]["colspan"] == 2 and "cmap" not in got["raw"]["panel"][0], "layout editor: save writes the minimal layout and it reads back")
    check(not errs, f"layout editor: no page errors {errs[:3]}")
    ed.close()


def section_mouse(pg, check, shot, wait, idle):
    """mouse back / forward buttons, ctrl / shift + wheel scanning, middle click, the right-click menu, the side-panel key, a stacked time series (needs 3 snapshots, 1 pane)"""
    ev = pg.evaluate
    idle()
    ev("() => setSnap(0)")
    wait("window.__splosh.idx === 0")
    idle()
    box = pg.locator("canvas.view").first.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    btn = "(b) => document.querySelector('canvas.view').dispatchEvent(new MouseEvent('mousedown', {button: b, bubbles: true, cancelable: true}))"
    up = "(b) => [document.querySelector('canvas.view').dispatchEvent(new MouseEvent('mouseup', {button: b, bubbles: true, cancelable: true})), document.querySelector('canvas.view').dispatchEvent(new MouseEvent('auxclick', {button: b, bubbles: true, cancelable: true}))]"
    n0 = ev(btn, 4)
    ups = ev(up, 4)
    wait("window.__splosh.idx === 1")
    idle()
    ev(btn, 3); ev(up, 3)
    wait("window.__splosh.idx === 0")
    idle()
    check(n0 is False and ups == [False, False], "mouse forward / back buttons step the snapshots, the browser's navigation prevented")
    # ---- shift + wheel over the map: scan, load on release
    r0, w0 = ev("window.__splosh.requests"), ev("window.__splosh.view.width")
    pg.mouse.move(cx, cy)
    pg.keyboard.down("Shift")
    pg.mouse.wheel(0, 100)
    pg.mouse.wheel(0, 100)
    pg.mouse.wheel(0, 100)                                      # past the last snapshot: clamped
    lab = ev("document.getElementById('snaplab').textContent")
    ghost = ev("getComputedStyle(document.getElementById('snapghost')).display")
    clock = ev("document.querySelector('.clock').textContent")
    held = ev("[window.__splosh.idx, window.__splosh.requests, window.__splosh.view.width]")
    pg.keyboard.up("Shift")
    wait("window.__splosh.idx === 2")
    idle()
    check(lab.startswith("\u2192") and "3/3" in lab and ghost == "block" and "3/3" in clock and held == [0, r0, w0],
          f"shift+wheel scans the snapshots without loading or zooming (label {lab!r}, clock {clock!r}, ghost {ghost}, held {held})")
    check(ev("window.__splosh.map.meta.snapshot === window.__splosh.snaps[2].name") and ev("getComputedStyle(document.getElementById('snapghost')).display") == "none",
          "releasing shift loads the scanned snapshot")
    # ---- ctrl + wheel over the slider: no page zoom, Esc cancels
    sl = pg.locator("#snap").bounding_box()
    pg.mouse.move(sl["x"] + sl["width"] / 2, sl["y"] + sl["height"] / 2)
    z0 = ev("[window.visualViewport.scale, window.devicePixelRatio, window.innerWidth]")
    prevented = ev("() => { const e = new WheelEvent('wheel', {deltaY: -100, ctrlKey: true, bubbles: true, cancelable: true}); document.getElementById('snap').dispatchEvent(e); return e.defaultPrevented; }")
    pg.keyboard.down("Control")
    pg.mouse.wheel(0, -100)
    lab = ev("document.getElementById('snaplab').textContent")
    pg.keyboard.up("Control")
    wait("window.__splosh.idx === 0")
    idle()
    check(prevented and "1/3" in lab and ev("[window.visualViewport.scale, window.devicePixelRatio, window.innerWidth]") == z0,
          f"ctrl+wheel over the slider scans (label {lab!r}), the page does not zoom (default prevented: {prevented})")
    pg.mouse.move(cx, cy)
    pg.keyboard.down("Shift")
    pg.mouse.wheel(0, 100)
    pg.keyboard.press("Escape")
    pg.keyboard.up("Shift")
    pg.wait_for_timeout(300)
    check(ev("window.__splosh.idx") == 0 and "\u2192" not in ev("document.getElementById('snaplab').textContent"), "Esc cancels a scan")
    # ---- a plain wheel still zooms the map
    w0 = ev("window.__splosh.view.width")
    pg.mouse.wheel(0, -300)
    wait(f"window.__splosh.view.width < {w0} * 0.9")
    pg.keyboard.press("o")
    idle()
    check(True, "a plain wheel still zooms the map")
    # ---- middle click: centre here; double middle click: reset the view
    v0 = ev("({...window.__splosh.view})")
    px, py = box["x"] + box["width"] * 0.75, box["y"] + box["height"] * 0.25
    want = ev(f"(() => {{ const p = window.__splosh.panes[0], r = p.cv.getBoundingClientRect(), k = p.cv.width / r.width; return toWorld(p, ({px} - r.left) * k, ({py} - r.top) * k); }})()")
    pg.mouse.click(px, py, button="middle")
    v1 = ev("({...window.__splosh.view})")
    check(abs(v1["cx"] - want[0]) < 1e-3 * v0["width"] and abs(v1["cy"] - want[1]) < 1e-3 * v0["width"] and v1["width"] == v0["width"],
          f"middle click centres the view there ({v0['cx']:.3g}, {v0['cy']:.3g}) -> ({v1['cx']:.3g}, {v1['cy']:.3g}), wanted ({want[0]:.3g}, {want[1]:.3g})")
    pg.wait_for_timeout(600)
    ev("() => { window.__splosh.view.width *= 0.5; }")
    pg.mouse.click(cx, cy, button="middle")
    pg.mouse.click(cx, cy, button="middle")
    v2 = ev("({...window.__splosh.view})")
    d = ev("window.__splosh.def")
    check(v2["width"] == d["width"] and v2["cx"] == d["cx"] and v2["cy"] == d["cy"], f"double middle click resets the view {v2}")
    idle()
    # ---- right click: the menu; shift + right click: the browser's
    pg.mouse.click(px, py, button="right")
    vis = pg.locator("#ctx").is_visible()
    txt = pg.locator("#ctx").inner_text()
    shot("context_menu")
    w0 = ev("window.__splosh.view.width")
    pg.locator("#ctx a", has_text="Zoom in here").click()
    v3 = ev("({...window.__splosh.view})")
    check(vis and "Centre here" in txt and "Copy the coordinates" in txt and "x " in txt and abs(v3["width"] - 0.5 * w0) < 1e-9 * w0 and abs(v3["cx"] - want[0]) < 1e-3 * w0 and not pg.locator("#ctx").is_visible(),
          "right click opens the map menu at the point; Zoom in here zooms about it")
    pg.mouse.click(px, py, button="right")
    pg.keyboard.press("Escape")
    closed = not pg.locator("#ctx").is_visible()
    pg.mouse.click(px, py, button="right")
    pg.locator("#ctx a", has_text="Copy the coordinates").click()
    pg.wait_for_timeout(300)
    copied = ev("'copied: ' + (window.__splosh.copied || '')")
    sh = ev("() => { const e = new MouseEvent('contextmenu', {shiftKey: true, bubbles: true, cancelable: true, clientX: 300, clientY: 300}); document.querySelector('canvas.view').dispatchEvent(e); return e.defaultPrevented; }")
    check(closed and copied.startswith("copied: ") and sh is False and not pg.locator("#ctx").is_visible(), f"Esc closes the menu; copy ({copied!r}); shift+right click keeps the browser's menu")
    pg.keyboard.press("o")
    idle()
    # ---- q: the side panel, also in Figure mode
    wb = pg.locator("#stage").bounding_box()["width"]
    pg.mouse.move(cx, cy)
    pg.keyboard.press("q")
    pg.wait_for_timeout(400)
    hidden = ev("document.getElementById('panel').classList.contains('hid')") and not pg.locator("#grip").is_visible()
    wa = pg.locator("#stage").bounding_box()["width"]
    pg.keyboard.press("q")
    pg.wait_for_timeout(300)
    back = pg.locator("#panel").is_visible()
    pg.click("#tabs [data-tab=figure]")
    pg.keyboard.press("q")
    pg.wait_for_timeout(300)
    fig_hidden = not pg.locator("#panel").is_visible()
    pg.keyboard.press("q")
    pg.wait_for_timeout(300)
    pg.click("#tabs [data-tab=explore]")
    check(hidden and wa > wb + 100 and back and fig_hidden and pg.locator("#panel").is_visible() and "q" in pg.locator("#helptab").inner_text(),
          f"q hides and shows the side panel (stage {wb:.0f} -> {wa:.0f} px), also in Figure mode; listed in the help")
    idle()
    # ---- a stacked time series: the parts' cumulative sums (S.stackOf), drawn without errors, kept in the layout panel
    K = ev("(() => { const k = window.__splosh.stackOf([{column: 'a', t: [0, 1, 2], y: [1, 1, 1]}, {column: 'b', t: [0, 0.5, 1, 2], y: [1, 2, 3, 5]}], false);"
           " const f = window.__splosh.stackOf([{column: 'a', t: [0, 1, 2], y: [1, 1, 1]}, {column: 'b', t: [0, 1, 2], y: [1, 3, 3]}], true);"
           " return [Array.from(k.hi[1]), k.varies, Array.from(f.hi[0]), Array.from(f.hi[1])]; })()")
    check(K[0] == [2, 4, 6] and K[1] is True and K[2] == [0.5, 0.25, 0.25] and K[3] == [1, 1, 1], f"stackOf: cumulative sums on the first part's times, fractions {K}")
    ev("() => setKind(window.__splosh.panes[0], 'series')")
    pg.wait_for_timeout(1500)
    pg.select_option(".pane .sstk", "fraction")
    pg.wait_for_timeout(800)
    lay_panel = ev("panelOf(window.__splosh.panes[0], 0)")
    shot("series_fraction")
    pg.select_option(".pane .sstk", "")
    lay_off = ev("panelOf(window.__splosh.panes[0], 0)")
    check(lay_panel.get("stack") == "fraction" and "stack" not in lay_off, f"the series pane's stack option goes into its layout panel ({lay_panel.get('stack')!r})")
    ev("() => setKind(window.__splosh.panes[0], 'map')")
    idle()
    # ---- a signed field: symmetric limits, RdBu_r + asinh; an unsigned one gets its colours back
    st0 = ev("[window.__splosh.panes[0].cmap, window.__splosh.panes[0].scale]")
    pick_field(pg, "vx")
    wait("window.__splosh.map.meta.view.field === 'vx'")
    idle()
    pg.wait_for_timeout(300)
    sv = ev("[window.__splosh.panes[0].cmap, window.__splosh.panes[0].scale, window.__splosh.lim[0], window.__splosh.lim[1], document.getElementById('cmap').value]")
    pick_field(pg, "rho")
    wait("window.__splosh.map.meta.view.field === 'rho'")
    idle()
    back = ev("[window.__splosh.panes[0].cmap, window.__splosh.panes[0].scale]")
    check(sv[0] == "RdBu_r" and sv[1] == "asinh" and sv[4] == "RdBu_r" and abs(sv[2] + sv[3]) <= 1e-9 * abs(sv[3]) and sv[3] > 0 and back == st0,
          f"signed field vx: RdBu_r, asinh, limits symmetric about 0 {sv[2:4]}; rho gets {back} back")


def section_explore(pg, url, a, check, shot, wait, idle):
    """residual / gradient transforms, the asinh softening and histogram-equalisation stretches, the pixel histogram with draggable limits, the contours overlay (single pane, snapshot 0)."""
    ev = pg.evaluate
    ev("setPanes(1)")
    ev("(() => { const s = document.getElementById('snap'); s.value = 0; s.dispatchEvent(new Event('input')); })()")
    wait("window.__splosh.idx === 0")
    idle()
    # ---- the pixel histogram and its handles
    h = ev("window.__splosh.histogram()")
    check(bool(h) and len(h["cnt"]) == 96 and sum(h["cnt"]) > 1000, "pixel histogram drawn (96 bins)")
    pg.locator("#hist").scroll_into_view_if_needed()
    box = pg.locator("#hist").bounding_box()
    lim0 = ev("window.__splosh.lim")
    xs = lambda v: box["x"] + (math.log10(v) - h["a"]) / (h["z"] - h["a"]) * box["width"]
    pg.mouse.move(xs(lim0[1]), box["y"] + box["height"] / 2)
    pg.mouse.down()
    pg.mouse.move(xs(lim0[1]) - 60, box["y"] + box["height"] / 2, steps=5)
    pg.mouse.up()
    pg.wait_for_timeout(200)
    lim1 = ev("window.__splosh.lim")
    check(lim1[1] < lim0[1] * 0.9 and abs(lim1[0] - lim0[0]) < 1e-12 * max(1, abs(lim0[0])) and ev("sel().nl.auto") is False, f"dragging the upper handle lowers max: {lim0[1]:.3g} -> {lim1[1]:.3g}, min kept, auto off")
    check(abs(float(pg.locator("#vmax").input_value()) - lim1[1]) / lim1[1] < 1e-3, "the max box follows the handle")
    pg.check("#auto")
    pg.wait_for_timeout(100)
    check(abs(ev("window.__splosh.lim")[1] - lim0[1]) < 1e-9 * lim0[1], "auto limits restore the percentile range")
    # ---- asinh softening: the bar's middle label is the value drawn at t = 0.5, whatever the stretch
    pg.select_option("#scale", "asinh")
    pg.wait_for_timeout(100)
    sig1 = ev(CANVAS_SIG)
    ev("(() => { const e = document.getElementById('soft'); e.value = 0.001; e.dispatchEvent(new Event('change')); })()")
    pg.wait_for_timeout(100)
    sig2 = ev(CANVAS_SIG)
    mid = ev("valueAtT(sel(), 0.5)")
    check(sig1 != sig2, "asinh F changes the picture")
    check(abs(ev(f"sel().sfn({mid})") - 0.5) < 1e-6 and ev("sel().lim")[0] < mid < ev("sel().lim")[1], f"valueAtT inverts the stretch (t=0.5 at {mid:.3g})")
    ev("(() => { const e = document.getElementById('soft'); e.value = 0.1; e.dispatchEvent(new Event('change')); })()")
    # ---- histogram equalisation: the distribution of stretched values is flat
    pg.select_option("#scale", "histeq")
    pg.wait_for_timeout(200)
    fr = ev("(() => { const p = sel(), c = new Array(10).fill(0); let n = 0; for (const x of p.map.data) { const t = p.sfn(x); if (Number.isNaN(t)) continue; c[Math.min(9, Math.floor(t * 10))]++; n++; } return c.map(v => v / n); })()")
    check(all(0.07 < f < 0.15 for f in fr), f"histeq: ten equal bins of stretched value each hold ~10% of the pixels {[round(f, 3) for f in fr]}")
    shot("explore_histeq")
    pg.select_option("#scale", "log")
    # ---- a contrast transform
    before = ev("({cmap: sel().cmap, scale: sel().scale})")
    ev("setPost(sel(), 'resid:azim', '')")
    wait("window.__splosh.map && window.__splosh.map.meta.post && window.__splosh.idle()")
    st = ev("({cmap: sel().cmap, scale: sel().scale, lim: sel().lim, post: sel().map.meta.post, params: window.__splosh.paneParams(sel())})")
    check(st["cmap"] == "RdBu_r" and st["scale"] == "linear" and st["lim"][0] == -st["lim"][1] and st["lim"][1] > 0, f"contrast map: RdBu_r, linear, symmetric limits {st['lim']}")
    check(st["post"]["mode"] == "contrast" and st["params"].get("post") == "resid:azim", "the request carries post=resid:azim and the meta says contrast")
    vals, meta = fetch_map(url, st["params"])
    mine = ev("Array.from(sel().map.data)")
    check(len(vals) == len(mine) and max(abs(x - y) for x, y in zip(vals, mine) if x == x and y == y) < 1e-6, "browser map equals an independent /api/map request with post")
    check(ev("window.__splosh.tilePlan(sel())") is None, "no tiles while a transform is on")
    ev("(() => { const e = document.getElementById('postarg'); })()")
    pg.check("#ov_con")
    wait("window.__splosh.idle() && window.__splosh.ovIdle()")
    kinds = ev("(() => { const o = {}; for (const it of sel().ov.items) { const k = it.kind + ':' + it.overlay; o[k] = (o[k] || 0) + 1; } return o; })()")
    check(kinds.get("polyline:contours", 0) > 0 and kinds.get("legend:contours", 0) == 1, f"contours overlay items {kinds}")
    check(ev("window.__splosh.ovSpecs(sel())") == ["contours:n=8,step=0.5"], "panel builds the contours spec")
    check(ev(OVL_PIXELS)[0] > 500, "contour lines are drawn on the overlay canvas")
    shot("explore_resid_contours")
    pg.uncheck("#ov_con")
    # ---- a gradient map with a smoothing width, then back
    pg.select_option("#post", "grad")
    wait("window.__splosh.map.meta.post && window.__splosh.map.meta.post.kind === 'grad' && window.__splosh.idle()")
    check(ev("sel().cmap") == "magma" and ev("sel().scale") == "asinh", "gradient map: magma, asinh")
    pg.fill("#postarg", "0.2")
    pg.dispatch_event("#postarg", "change")
    wait("window.__splosh.map.meta.view.post === 'grad:0.2' && window.__splosh.idle()")
    check(True, "the width box changes the request to grad:0.2")
    shot("explore_grad")
    pg.select_option("#post", "none")
    wait("!window.__splosh.map.meta.post && window.__splosh.idle()")
    after = ev("({cmap: sel().cmap, scale: sel().scale})")
    check(after == before, f"turning the transform off restores the colours {after}")
    ev("document.activeElement && document.activeElement.blur()")           # the text boxes keep the focus otherwise, and the next section types keys


OFF_PIXELS = "(() => { const p = window.__splosh.panes[0], c = p.off.getContext('2d'); return [p.bakedBy, Array.from(c.getImageData(0, 0, p.off.width, p.off.height).data)]; })()"


def section_gpu(pg, url, a, check, shot, wait, idle):
    """GPU colouring (WebGL2) gives the same image as the CPU loop for the linear, log and asinh stretches (within one colour-table step)."""
    ev = pg.evaluate
    if not ev("window.__splosh.GL.available()"):
        check(ev("document.getElementById('gpu').disabled"), "no WebGL2 here: the GPU switch is disabled and the CPU colours the maps")
        return
    for scale in ("log", "asinh", "linear"):
        pg.select_option("#scale", scale)
        idle()
        pg.check("#gpu")
        idle()
        by_g, g = ev(OFF_PIXELS)
        pg.uncheck("#gpu")
        idle()
        by_c, c = ev(OFF_PIXELS)
        n = len(g) // 4
        diff = [max(abs(g[4 * i + k] - c[4 * i + k]) for k in range(3)) for i in range(n)]
        bad = sum(1 for d in diff if d > 0)
        check(by_g == "gpu" and by_c == "cpu" and bad <= 0.01 * n and max(diff) <= 24,
              f"GPU colouring equals the CPU loop for scale {scale}: {bad} of {n} pixels differ, by at most {max(diff)} (a colour-table step)")
    pg.check("#gpu")
    pg.select_option("#scale", "log")
    idle()
    pg.select_option("#mode", "volume")
    wait("window.__splosh.panes[0].map && window.__splosh.panes[0].map.meta.view.mode === 'volume'")
    idle()
    m = ev("window.__splosh.panes[0].map.meta")
    check(m["label"] == "opacity (volume)" and m["kappa"] > 0 and not ev("!!window.__splosh.panes[0].map.tiled"), f"volume mode: {m['label']}, kappa {m['kappa']:.3g}, one map (no tiles)")
    pg.select_option("#mode", "proj")
    idle()


def section_phase(pg, url, a, check, shot, wait, idle):
    """phase diagram of the window (/api/phase), a lasso drawn with the mouse, the highlight of the selected gas, the map of only the selection (/api/map?sel=)."""
    ev = pg.evaluate
    pg.select_option("#ph_x", "density")
    pg.select_option("#ph_y", "u")
    pg.click("#ph_go")
    wait("window.__splosh.panes[0].ph.grid !== null && window.__splosh.panes[0].ph.meta !== null")
    m = ev("window.__splosh.panes[0].ph.meta")
    check(m["x"]["label"] == "density" and m["y"]["log"] and m["n_in_view"] > 100, f"phase diagram of the window: density vs log u, {m['n_in_view']} particles")
    total = ev("window.__splosh.panes[0].ph.grid.reduce((a, b) => a + b, 0)")
    check(abs(total - m["total_in_range"]) <= 1e-4 * m["total_in_range"], f"the drawn histogram holds the window's gas mass ({total:.4g} vs {m['total_in_range']:.4g})")
    box = pg.locator("#phcv").bounding_box()
    x0, y0, w, h = box["x"], box["y"], box["width"], box["height"]
    pg.mouse.move(x0 + 0.05 * w, y0 + 0.05 * h)
    pg.mouse.down()
    for fx, fy in ((0.5, 0.05), (0.5, 0.95), (0.05, 0.95), (0.05, 0.1)):         # the left half of the diagram
        pg.mouse.move(x0 + fx * w, y0 + fy * h, steps=6)
    pg.mouse.up()
    wait("window.__splosh.panes[0].selItem !== null")
    crit = ev("window.__splosh.panes[0].selCrit")
    check(crit and len(crit["poly"]) >= 4 and crit["x"] == "density", f"the lasso became a selection polygon of {len(crit['poly']) if crit else 0} vertices")
    p = ev("window.__splosh.paneParams(window.__splosh.panes[0])")
    ref = fetch_json(url, "/api/selection", {**p, "sel": json.dumps(crit)})
    npts = ev("window.__splosh.panes[0].selItem.x.length")
    check(0 < npts == len(ref["points"]) < m["n_in_view"], f"highlight: {npts} selected particles drawn, equal to an independent /api/selection, fewer than the window's {m['n_in_view']}")
    check("selection:" in pg.inner_text("#phmsg"), "the panel states the selection's count and mass")
    shot("phase_lasso_highlight")
    pg.check("#ph_only")
    wait("window.__splosh.panes[0].map && window.__splosh.panes[0].map.meta.selection !== undefined")
    idle()
    vals, meta = fetch_map(url, {**ev("window.__splosh.paneParams(window.__splosh.panes[0])")})
    full, _ = fetch_map(url, {k: v for k, v in p.items() if k != "sel"})
    sel_sum, full_sum = sum(v for v in vals if v == v), sum(v for v in full if v == v)
    check(meta["selection"]["n_selected"] > 0 and 0 < sel_sum < full_sum, f"map of only the selection: {meta['selection']['n_selected']} particles, {sel_sum / full_sum:.2f} of the mass in the window")
    shot("phase_selection_only")
    pg.click("#ph_clear")
    wait("window.__splosh.panes[0].map && window.__splosh.panes[0].map.meta.selection === undefined && window.__splosh.panes[0].selItem === null")
    pg.uncheck("#ph_only")
    idle()
    check(True, "clear selection: the full map and no highlight again")


def section_overlays(pg, url, a, check, shot, wait, idle, nsnap):
    """overlays on the canvas above the map, following pan and zoom, and click-to-inspect (single pane, snapshot 0 of the run: with gas B field, stars)."""
    ev = pg.evaluate
    oidle = lambda: wait("window.__splosh.idle() && window.__splosh.ovIdle()")
    ev("(() => { const s = document.getElementById('snap'); s.value = 0; s.dispatchEvent(new Event('input')); })()")
    wait("window.__splosh.idx === 0")
    idle()
    check(ev(OVL_PIXELS)[0] == 0, "overlay canvas is empty while no overlay is selected")
    for k in ("vv", "sv", "pstar"):
        pg.check("#ov_" + k)
    oidle()
    specs = ev("window.__splosh.ovSpecs(window.__splosh.panes[0])")
    check(specs == ["vectors:v,grid=24,arrows=prop", "streamlines:v,density=28", "particles:star"], f"panel builds the spec strings {specs}")
    kinds = ev("(() => { const o = {}; for (const it of window.__splosh.panes[0].ov.items) o[it.kind] = (o[it.kind] || 0) + 1; return o; })()")
    check(kinds.get("arrow", 0) > 50 and kinds.get("polyline", 0) > 3 and kinds.get("points", 0) == 1 and kinds.get("legend", 0) == 1, f"overlay items drawn: {kinds}")
    npx, h0 = ev(OVL_PIXELS)
    check(npx > 2000, f"overlay canvas has drawn pixels ({npx})")
    p = ev("window.__splosh.paneParams(window.__splosh.panes[0])")
    ref = fetch_json(url, "/overlay", {**p, "spec": specs[:1]})
    mine = ev("window.__splosh.panes[0].ov.items.filter(i => i.overlay === 'vectors:v' && i.kind === 'arrow').length")
    check(mine == len([i for i in ref["items"] if i.get("kind") == "arrow"]) and mine > 0, f"browser arrows equal an independent /overlay request ({mine})")
    shot("overlays_arrows_streamlines_stars")
    # pan: the old overlay is transformed at once, a new one is requested after the debounce
    nreq = ev("window.__splosh.ovRequests")
    box = pg.locator("canvas.view").first.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + 90, cy + 40, steps=4)
    h1 = ev(OVL_PIXELS)[1]
    pg.mouse.up()
    check(h1 != h0, "panning moves the overlay canvas at once (old overlay transformed)")
    wait(f"window.__splosh.ovRequests > {nreq}")
    oidle()
    check(ev("window.__splosh.panes[0].ov.items.length") > 0, "after the pan a new overlay arrived for the new window")
    pg.keyboard.press("o")
    wait("window.__splosh.map.meta.view.width > 0 && window.__splosh.panes[0].view.cx === window.__splosh.def.cx")
    oidle()
    # errors become a message, the other overlays stay
    if nsnap > 2:
        pg.keyboard.press("ArrowRight"); pg.keyboard.press("ArrowRight")
        wait("window.__splosh.idx === 2")
        idle()
        pg.check("#ov_vB")
        EXPECTED_400.append("vectors:B without MagneticField")        # the browser logs a failed request as a console error: deliberate here
        wait("document.getElementById('ovmsg').textContent.includes('MagneticField')")
        oidle()
        check(ev("window.__splosh.panes[0].ov.items.filter(i => i.kind === 'arrow').length") > 0, "no B field: a short message, the velocity arrows are still drawn")
        pg.uncheck("#ov_vB")
        pg.check("#ov_pbh"); pg.check("#ov_pdm")
        oidle()
        msg = ev("document.getElementById('ovmsg').textContent")
        check(len(msg) < 400 and "star" in msg.lower() or "dm" in msg.lower() or "PartType" in msg or "no " in msg.lower(), f"missing particle types give a note, not an error: {msg!r}")
        pg.uncheck("#ov_pbh"); pg.uncheck("#ov_pdm")
        pg.keyboard.press("ArrowLeft"); pg.keyboard.press("ArrowLeft")
        wait("window.__splosh.idx === 0")
        idle()
    pg.check("#ov_vB")
    oidle()
    check(ev("window.__splosh.panes[0].ovMsg") == "" and ev("window.__splosh.panes[0].ov.items.some(i => i.overlay === 'vectors:B')"), "vectors B drawn for a snapshot with MagneticField")
    pg.uncheck("#ov_vB")
    oidle()
    # ---- click-to-inspect
    fx, fy = 0.56, 0.44
    pg.mouse.click(box["x"] + fx * box["width"], box["y"] + fy * box["height"])
    wait("window.__splosh.panes[0].insp")
    r = ev("window.__splosh.panes[0].insp")
    ip = ev("window.__splosh.paneParams(window.__splosh.panes[0])")
    ref = fetch_json(url, "/inspect", {**ip, "x": r["x"], "y": r["y"], "limit": 10})
    top = [e["id"] for e in r["res"]["particles"]]
    check(r["res"]["mode"] == "proj" and top and top == [e["id"] for e in ref["particles"]], f"click inspects: top particles {top[:3]} equal an independent /inspect request")
    c = [e["contribution"] for e in r["res"]["particles"]]
    check(all(x >= y for x, y in zip(c, c[1:])) and 0 < sum(c) <= 1.0000001, f"projection: ranked by contribution ({sum(c) * 100:.1f}% listed)")
    txt = pg.locator("#insp").inner_text()
    check(all(w in txt for w in ("id ", "H ", "rho ", " u ", "pos ", "vel ")) and "%" in txt, "panel lists id, H, rho, u, T, position, velocity and contribution")
    circ = ev("(() => { const p = window.__splosh.panes[0], e = p.insp.res.particles[0]; return [e.circle.r > 0, Math.hypot(p.insp.x - e.circle.x, p.insp.y - e.circle.y) < e.circle.r]; })()")
    check(all(circ), "smoothing circle of the selected particle contains the clicked point")
    h2 = ev(OVL_PIXELS)[1]
    pg.locator("#insp .card").nth(1).click()
    check(ev("window.__splosh.panes[0].insp.sel") == 1 and ev(OVL_PIXELS)[1] != h2, "selecting another particle in the panel moves the circle")
    shot("inspect_projection")
    pg.keyboard.press("Escape")
    check(ev("window.__splosh.panes[0].insp") is None and "click a point" in pg.locator("#insp").inner_text(), "Esc clears the inspection")
    # double click still centres and does not inspect
    cxw = ev("window.__splosh.panes[0].view.cx")
    pg.mouse.dblclick(box["x"] + 0.7 * box["width"], box["y"] + 0.5 * box["height"])
    wait(f"window.__splosh.panes[0].view.cx !== {cxw}")
    pg.wait_for_timeout(500)
    check(ev("window.__splosh.panes[0].insp") is None, "double click centres and does not inspect")
    pg.keyboard.press("o")
    idle()
    pg.select_option("#mode", "slice")
    wait("window.__splosh.map.meta.view.mode === 'slice'")
    idle()
    pg.keyboard.down("Shift")
    pg.mouse.click(box["x"] + 0.5 * box["width"], box["y"] + 0.5 * box["height"])
    pg.keyboard.up("Shift")
    wait("window.__splosh.panes[0].insp")
    r = ev("window.__splosh.panes[0].insp")
    d = ev("(() => { const p = window.__splosh.panes[0], R = p.insp.res; return R.particles.map(e => Math.hypot(e.pos[0] - R.x, e.pos[1] - R.y, e.pos[2] - R.z0)); })()")
    check(r["res"]["mode"] == "slice" and d == sorted(d) and all(x < e["H"] for x, e in zip(d, r["res"]["particles"])), f"slice: particles within H of the point, nearest first ({len(d)})")
    shot("inspect_slice")
    pg.keyboard.press("Escape")
    pg.select_option("#mode", "proj")
    wait("window.__splosh.map.meta.view.mode === 'proj'")
    idle()
    for k in ("vv", "sv", "pstar"):
        pg.uncheck("#ov_" + k)
    oidle()
    check(ev(OVL_PIXELS)[0] == 0, "switching the overlays off empties the canvas")


def section_tiles(pg, url, a, check, shot, wait, idle):
    """deep zoom through /api/tile: the composite equals the plain map of the same window, tile request counts when panning, placeholder and spinner, fallbacks."""
    ev = pg.evaluate
    state = fetch_json(url, "/api/state", {})
    base, size = state["tile"]["base"], state["tile"]["size"]
    check(base > 0 and size == 256 and math.log2(base) == int(math.log2(base)), f"/api/state gives the tile grid: base {base}, tile {size}")
    set_hidden(pg, "npix", "256")
    wait("window.__splosh.map.n === 256")
    idle()
    check(not ev("!!window.__splosh.map.tiled") and ev("window.__splosh.tilePlan(window.__splosh.panes[0])") is None, "full view: no tiles (the normal map path)")

    def goto(cx, cy, width):
        ev(f"(() => {{ const p = sel(); p.view.cx = {cx}; p.view.cy = {cy}; p.view.width = {width}; showControls(); drawAll(); requestView(30); }})()")

    goto(0.75, 0.5, 1.0)                                                      # window [0.25, 1.25] x [0, 1]: tiles x = 0, 1 at level 3 (tile width 1)
    wait("window.__splosh.map.tiled && Math.abs(window.__splosh.map.extent[0] - 0.25) < 1e-9")
    idle()
    m = ev("(() => { const m = window.__splosh.map; return {tiled: m.meta.tiled, n: m.n, extent: m.extent, nt: window.__splosh.netTiles}; })()")
    check(m["tiled"]["tiles"] == 2 and m["tiled"]["level"] == 3 and m["n"] == 256, f"window 1 / base {base} is assembled from {m['tiled']['tiles']} tiles at level {m['tiled']['level']} ({m['n']} pixels)")
    p = ev("window.__splosh.paneParams(window.__splosh.panes[0])")
    vals, meta = fetch_map(url, {**p, "cx": 0.75, "cy": 0.5, "width": 1.0, "npix": 256})
    comp = ev("Array.from(window.__splosh.map.data)")
    n = m["n"]
    check(meta["n"] == n and all(abs(x - y) < 1e-9 for x, y in zip(meta["extent"], m["extent"])), "composite and plain map cover the same window on the same pixel grid")
    mx = max(vals)
    bad = 0.0
    for j in range(16, n - 16):
        for i in range(16, n - 16):
            bad = max(bad, abs(comp[j * n + i] - vals[j * n + i]))
    check(mx > 0 and bad <= 5e-3 * mx, f"tiled composite equals the plain map within the engine tolerance (largest difference {bad / mx:.2e} of the maximum, interior pixels)")
    shot("tiles_composite")
    # colour limits: one pair for the whole composite, also across the tile seam
    seam = ev("""() => { const p = window.__splosh.panes[0], n = p.map.n, d = p.map.data, [lo, hi] = p.lim, ctx = p.off.getContext('2d'), lut = window.__splosh.luts[p.cmap]; let bad = 0, cnt = 0;
        const sx = Math.round((1.0 - p.map.extent[0]) / (p.map.extent[1] - p.map.extent[0]) * n);      // the seam between tile 0 and tile 1 (x = 1)
        for (let j = 5; j < n; j += 17) for (let i = sx - 3; i < sx + 3; i++) { const t = stretch(d[j * n + i], lo, hi, p.scale); if (Number.isNaN(t)) continue;
          const c = ctx.getImageData(i, n - 1 - j, 1, 1).data, w = lut[Math.round(t * 255)]; cnt++; if (Math.abs(c[0] - w[0]) + Math.abs(c[1] - w[1]) + Math.abs(c[2] - w[2]) > 0) bad++; } return [bad, cnt]; }""")
    check(seam[1] > 50 and seam[0] == 0, f"colours across the tile seam use the pane's one pair of limits ({seam[1]} samples)")
    # pan by one tile: one new tile
    n0 = ev("window.__splosh.netTiles")
    goto(1.75, 0.5, 1.0)
    wait("Math.abs(window.__splosh.map.extent[0] - 1.25) < 1e-9 && window.__splosh.map.tiled")
    idle()
    n1 = ev("window.__splosh.netTiles")
    check(n1 - n0 == 1, f"panning by one tile requests one new tile ({n1 - n0}); the other came from the browser cache")
    h0 = ev("window.__splosh.tileHits")
    goto(0.75, 0.5, 1.0)
    wait("Math.abs(window.__splosh.map.extent[0] - 0.25) < 1e-9")
    idle()
    check(ev("window.__splosh.netTiles") == n1 and ev("window.__splosh.tileHits") > h0, "panning back is served by the browser cache (no tile request)")
    # a window that needs new tiles: coarse placeholder and spinner meanwhile
    goto(0.75, 1.25, 1.0)      # moved up by a quarter window: two tiles are new, the old map still covers the lower part
    pg.wait_for_function("document.querySelector('.spin.on')", polling=10, timeout=20000)
    ph = ev("(() => { const p = window.__splosh.panes[0]; return [p.map.tiled, JSON.stringify(p.map.extent)]; })()")
    check(ev(DISTINCT) > 20 and ph[0], f"while tiles load the previous map stays on the canvas as the placeholder {ph[1]} with a spinner")
    wait("Math.abs(window.__splosh.map.extent[2] - 0.75) < 1e-9")
    idle()
    check(not ev("document.querySelector('.spin').classList.contains('on')"), "spinner off once the tiles arrived")
    # prefetch of the ring around the block
    goto(0.75, 5.5, 1.0)
    wait("Math.abs(window.__splosh.map.extent[2] - 5.0) < 1e-9")
    idle()
    pg.check("#prefetch")
    n0 = ev("window.__splosh.netTiles")
    goto(0.5, 5.5, 1.0)
    pg.wait_for_function(f"window.__splosh.netTiles >= {n0} + 6", timeout=60000)
    check(True, f"prefetch fetched the ring of neighbouring tiles ({ev('window.__splosh.netTiles') - n0} requests)")
    pg.uncheck("#prefetch")
    # fallbacks
    goto(0.0, 0.0, 6.0)
    wait("window.__splosh.map && !window.__splosh.map.tiled && window.__splosh.map.meta.view.width === 6")
    idle()
    check(ev("window.__splosh.tilePlan(window.__splosh.panes[0])") is None, f"window 6 > base {base} / 4: normal map path")
    goto(0.0, 0.0, 1e-9)
    check(ev("window.__splosh.tilePlan(window.__splosh.panes[0])") is None, "a tiny window (level above 24) falls back to /api/map")
    idle()
    wait("window.__splosh.map && !window.__splosh.map.tiled")
    goto(0.0, 0.0, 1.0)
    wait("window.__splosh.map.tiled")
    idle()
    pg.uncheck("#tiles")
    wait("window.__splosh.map && !window.__splosh.map.tiled && window.__splosh.map.meta.view.width === 1")
    idle()
    check(True, "the deep-zoom checkbox switches back to /api/map")
    pg.check("#tiles")
    wait("window.__splosh.map.tiled")
    idle()
    pg.keyboard.press("Shift+A"); pg.keyboard.press("Shift+B"); pg.keyboard.press("~")           # difference mode: normal path
    wait("window.__splosh.panes[0].diff === 'sub'")
    idle()
    check(not ev("!!window.__splosh.map.tiled") and ev("window.__splosh.tilePlan(window.__splosh.panes[0])") is None, "difference mode never uses tiles")
    pg.keyboard.press("~"); pg.keyboard.press("~")
    wait("window.__splosh.panes[0].diff === 'off'")
    pg.keyboard.press("o")
    set_hidden(pg, "npix", "96")
    wait("window.__splosh.map.n === 96")
    idle()


def section_runs(pg, url, a, check, shot, wait, idle):
    """two runs with different snapshot spacing in two panes: clocks, stepping all panes to the nearest snapshot in time within the tolerance, 'no snapshot at this time', offsets,
    alignment modes, the JS rule against /api/align, each pane's own run. Expects run 1 (the directory) at times 0, 1, 2 and run 2 at 0, 1.5, 3, 4.5."""
    ev = pg.evaluate
    runs = fetch_json(url, "/api/runs", {})["runs"]
    check(len(runs) == 2 and [x["time"] for x in runs[0]["snapshots"]] == [0, 1, 2] and [x["time"] for x in runs[1]["snapshots"]] == [0, 1.5, 3, 4.5], "/api/runs lists both runs with their snapshot times")
    st = lambda: ev("window.__splosh.panes.map(p => [p.run, p.snap, !!p.missing])")
    blur = lambda: ev("document.activeElement && document.activeElement.blur()")
    pg.keyboard.press("o")                                          # the full view: plain maps for the comparison with /api/map
    wait("window.__splosh.panes[0].view.width === window.__splosh.def.width")
    set_hidden(pg, "panes", "2")
    wait("window.__splosh.panes.length === 2")
    idle()
    check(ev("window.__splosh.align") == "time" and pg.input_value("#alignmode") == "time", "with two runs the panes are aligned by time by default")
    pg.evaluate("() => selectPane(1)")
    pg.select_option("#run", "1")
    wait("window.__splosh.panes[1].run === 1")
    idle()
    ev("(() => { const s = document.getElementById('snap'); s.value = 0; s.dispatchEvent(new Event('input')); })()")
    wait("window.__splosh.panes[0].snap === 0 && window.__splosh.panes[1].snap === 0")
    idle()
    cap = [ev(f"document.querySelectorAll('.cap')[{k}].textContent") for k in (0, 1)]
    check(cap[0].startswith(runs[0]["label"] + ":") and cap[1].startswith(runs[1]["label"] + ":"), f"each pane names its run in the caption: {cap}")
    clocks = lambda: [ev(f"document.querySelectorAll('.clock')[{k}].textContent") for k in (0, 1)]
    shot("runs_two_panes_t0")
    pg.evaluate("() => selectPane(0)")
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[0].snap === 1")
    idle()
    al = fetch_json(url, "/api/align", {"mode": "time", "at": 1.0})["runs"]
    check(st() == [[0, 1, False], [1, 1, False]] and al[1]["snap"] == 1, f"step all panes: run 1 to t=1, run 2 to the nearest in time (t=1.5, within its tolerance 0.75) {st()}")
    c = clocks()
    myr = runs[1]["snapshots"][1]["time_myr"]
    check("t = 1.00000" in c[0] and "t = 1.50000" in c[1] and f"{myr:.5g} Myr" in c[1], f"the panes' clocks show their own times (code and Myr): {c}")
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[0].snap === 2")
    idle()
    check(st() == [[0, 2, False], [1, 1, False]], f"run 2 stays at t=1.5 while run 1 goes to t=2 (0.5 away, tolerance 0.75) {st()}")
    pg.keyboard.press("ArrowRight")
    idle()
    check(st() == [[0, 2, False], [1, 1, False]], "stepping past the last snapshot of the selected pane changes nothing")
    pg.evaluate("() => selectPane(1)")
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[1].snap === 2")
    idle()
    check(st() == [[0, 2, True], [1, 2, False]], f"run 2 at t=3: run 1 has no snapshot within 0.5 (nearest t=2): pane 1 is marked {st()}")
    check(pg.locator(".nosnap.on").count() == 1 and "no snapshot at this time" in pg.locator(".nosnap.on").inner_text() and "no snapshot at this time" in clocks()[0], "the pane without a snapshot says 'no snapshot at this time'")
    al = fetch_json(url, "/api/align", {"mode": "time", "at": 3.0})["runs"]
    check(al[0]["snap"] is None and al[1]["snap"] == 2, "the server's /api/align agrees")
    shot("runs_no_snapshot")
    pg.keyboard.press("ArrowLeft")
    wait("window.__splosh.panes[1].snap === 1")
    idle()
    check(st() == [[0, 2, False], [1, 1, False]], f"stepping back finds the snapshot again {st()}")
    pg.fill("#tol", "0.1")
    pg.dispatch_event("#tol", "change"); blur()
    idle()
    check(st()[0][2] and not st()[1][2], f"an explicit tolerance 0.1 leaves run 1 without a snapshot at t=1.5 {st()}")
    pg.fill("#tol", "")
    pg.dispatch_event("#tol", "change"); blur()
    idle()
    check(st() == [[0, 2, False], [1, 1, False]], "empty tolerance: automatic again")
    # the JS rule equals the server's for many wanted times, per run
    bad = []
    for at in (-0.3, 0.4, 0.75, 1.25, 1.5, 2.75, 3.9, 4.9, 7.0):
        al = fetch_json(url, "/api/align", {"mode": "time", "at": at})["runs"]
        for r in (0, 1):
            js = ev(f"window.__splosh.alignedSnap({r}, {at}, null).snap")
            if js != al[r]["snap"]:
                bad.append((at, r, js, al[r]["snap"]))
    check(not bad, f"the page's nearest-snapshot rule equals /api/align {bad}")
    # offsets: run 2 shifted by 1.5 (aligned = t - 1.5): its t=1.5 snapshot is at 0
    pg.fill("#offsets", "0, 1.5")
    pg.dispatch_event("#offsets", "change"); blur()
    wait("window.__splosh.panes[0].snap === 0")
    idle()
    check(st() == [[0, 0, False], [1, 1, False]], f"offset 1.5 for run 2: its t=1.5 snapshot lines up with run 1 at t=0 {st()}")
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[1].snap === 2")
    idle()
    check(st()[1] == [1, 2, False] and st()[0][1] == 1 and not st()[0][2], f"aligned 1.5 is between run 1's t=1 and t=2: the tie goes to the side of the pane's current snapshot {st()}")
    pg.fill("#offsets", "")
    pg.dispatch_event("#offsets", "change"); blur()
    idle()
    # index and Myr modes
    pg.select_option("#alignmode", "index")
    idle()
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[1].snap === 3")
    idle()
    check(st()[0][1] == 2 and st()[0][2], f"index alignment: run 1 (3 snapshots) has no index 3 {st()}")
    pg.keyboard.press("ArrowLeft"); pg.keyboard.press("ArrowLeft")
    wait("window.__splosh.panes[1].snap === 1")
    idle()
    check(st() == [[0, 1, False], [1, 1, False]], f"index alignment: the same index in every pane {st()}")
    pg.select_option("#alignmode", "time_myr")
    idle()
    pg.keyboard.press("ArrowRight")
    wait("window.__splosh.panes[1].snap === 2")
    idle()
    rr = fetch_json(url, "/api/runs", {})["runs"]
    want = fetch_json(url, "/api/align", {"mode": "time_myr", "at": rr[1]["snapshots"][2]["time_myr"]})["runs"]
    check((want[0]["snap"] is None) == st()[0][2] and (want[0]["snap"] is None or want[0]["snap"] == st()[0][1]), f"time_myr alignment follows /api/align {st()} {[x['snap'] for x in want]}")
    pg.select_option("#alignmode", "time")
    idle()
    # Shift+arrow moves only the selected pane even when aligned
    before = st()
    pg.keyboard.press("Shift+ArrowLeft")
    idle()
    after = st()
    check(after[1][1] == before[1][1] - 1 and after[0][1] == before[0][1], f"Shift+arrow steps only the selected pane {before} -> {after}")
    # a pane takes the other run
    pg.evaluate("() => selectPane(0)")
    pg.select_option("#run", "1")
    wait("window.__splosh.panes[0].run === 1")
    idle()
    cap0 = ev("document.querySelectorAll('.cap')[0].textContent")
    check(ev("window.__splosh.panes[0].run") == 1 and runs[1]["label"] in cap0, f"a pane can show the other run: {cap0}")
    # the map of a pane is the map of its run's snapshot (independent request with the pane's own parameters)
    pane = 1
    pp = ev(f"window.__splosh.paneParams(window.__splosh.panes[{pane}])")
    vals, meta = fetch_map(url, pp)
    got = ev(f"Array.from(window.__splosh.panes[{pane}].map.data)")
    diff = max((abs(x - y) / (abs(y) + 1e-30) for x, y in zip(got, vals) if math.isfinite(x) and math.isfinite(y)), default=-1)
    check(pp.get("run") == runs[1]["label"] and len(got) == len(vals) and diff <= 1e-5, f"(max relative difference {diff:.2e}, {len(got)} / {len(vals)} pixels, tiled {ev('!!window.__splosh.panes[1].map.tiled')}) pane 2 shows run {pp.get('run')} snapshot {pp['snap']}: its map equals an independent /api/map?run= request")
    shot("runs_pane_run_b")
    pg.select_option("#run", "0")
    set_hidden(pg, "panes", "1")
    wait("window.__splosh.panes.length === 1")
    idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapdir")
    ap.add_argument("--server-python", default=sys.executable)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--serve-args", default="", help="extra arguments for `splosh serve` as one string, e.g. \"--center bh --align bh --width 10\"")
    ap.add_argument("--extra-run", action="append", default=[], help="another run directory served beside SNAPDIR (several runs in the panes; enables the run and time-alignment checks, which expect the layout of tests/test_viewer_browser.py)")
    ap.add_argument("--layouts", default=None, help="a directory with a layout `seed.toml` (4 columns: a map `m` at column 0, a text panel `t` at column 1) for the layout-editor checks; served with --layouts and --analysis")
    ap.add_argument("--analysis", default=None, help="an analysis directory (scalars.csv with a gas_mass column) for the editor's time-series panel")
    ap.add_argument("--only", choices=["layout"], default=None, help="run one section only (needs its options: layout -> --layouts, --analysis)")
    ap.add_argument("--synthetic", action="store_true", help="snap_001 is snap_000 with all particle masses doubled: also check the exact difference values")
    a = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright missing")
    srv, url = start_server(a)
    fails, errors = [], []

    def check(ok, what):
        print(("ok   " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
    try:
        with sync_playwright() as p:
            br = p.chromium.launch()
            pg = br.new_page(viewport={"width": 1100, "height": 820})
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            try:
                if a.only == "layout":
                    section_layout(pg, url, a, check)
                else:
                    run(pg, url, a, check)
            except Exception as e:                                   # a timeout or a JS error: report, keep the screenshot
                check(False, f"aborted: {type(e).__name__}: {str(e)[:300]}")
                if a.shots:
                    pg.screenshot(path=os.path.join(a.shots, "failure.png"))
            br.close()
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    errors = [e for e in errors if not ("status of 400" in e and EXPECTED_400 and (EXPECTED_400.pop() or True))]       # one allowed per deliberate bad request
    check(not errors, f"no console errors {errors[:3]}")
    print("FAILED" if fails else "PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
