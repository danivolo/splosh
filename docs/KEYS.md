# Keyboard shortcuts, aligned with SPLASH

Source for SPLASH: its in-app list (`h` in the plot window; pasted by the user 2026-11-09) and the docs
(https://splash-viz.readthedocs.io/en/latest/basic.html, settings.html). Goal: a SPLASH user finds the same keys doing the same things;
SPLOSH's own extras go on keys SPLASH leaves free. **Status (2026-11-09): implemented** as below, with the user's answers to the three open questions (space = next snapshot as in
SPLASH; digits are step counts, panes by Tab; right click is NOT bound: the browser's menu stays). Not yet: `g` (line profile). Since 2026-11-13 `shift`+drag zooms a map to a box too.
"(today)" / "moved" refer to the keys before this change.

Legend: **=** same key and meaning as SPLASH · **~** same key, SPLOSH's closest equivalent · **new** SPLOSH only · **moved** an SPLOSH key that changes.

## Time stepping
| key | action | vs SPLASH |
|---|---|---|
| `space` | next snapshot | = (today: blink, moved to `shift+space`) |
| `b` | previous snapshot | = (today: store B, moved to `B`) |
| `0`-`9` then `space` / `b` | forward / back n snapshots | = (today `1`-`4` select a pane: moved to `Tab`) |
| `→` / `←` | next / previous *(today)*; `shift`: all panes *(today)* | new |
| `Home` / `End` | first / last snapshot | new |
| `Enter` (or `ctrl+m`) | play / pause through the snapshots ("Hollywood mode") | = |
| `z` / `Z` | the next step, zoom or limit change x10 | = |

## Zoom, window, limits
| key | action | vs SPLASH |
|---|---|---|
| `+` / `-` | zoom in / out (x2); `_` out by 20 % | = |
| wheel, drag, double click | zoom, pan, centre here *(today)* | new (SPLASH: left-drag = zoom box) |
| `shift`+drag | zoom to the box drawn: its centre becomes the centre, its larger side the width (the window stays square); a shift+click (no drag) still inspects at once | ~ (SPLASH: left click = select/zoom box) |
| `a` | adapt the colour limits to the data (auto limits on) | = (today: store A, moved to `A`) |
| `l` | log / linear colour scale | = (today: link panes, moved to `L`) |
| `o` | reset the view: the default centre and width (a zoomed plot pane: its zoom) | ~ (SPLASH: re-centre on the origin; today `r`) |
| `C` | centre on the cursor | = |
| `n` | centre on the nearest black hole (sink) to the cursor | = |
| `t` | track the particle nearest the cursor (centre rule `ids:<its id>`); again: off | = |
| `r` | re-render (drop this view from the cache and request it again) | = (replot; today `r` resets the view, moved to `o`) |
| `s` | keep these colour limits for all snapshots (lock) | ~ (SPLASH: save settings for all steps) |

## Colour and rendering
| key | action | vs SPLASH |
|---|---|---|
| `m` / `M` | next / previous colour map | = |
| `i` | invert the colour map | = |
| `f` / `F` | next / previous field (in the picker's order) | = (docs) |
| `k` / `K` | volume mode: opacity kappa down / up | = |
| `x` | projection <-> slice | ~ (SPLASH: take a cross section) |
| `u` / `d` (`U` / `D`) | slice: move the plane z0 up / down by one pixel (ten) | = (today `d` cycles the difference, moved to `~`) |

## Rotation (tilt rule, applied after the alignment)
| key | action | vs SPLASH |
|---|---|---|
| `,` `.` / `<` `>` | rotate about the line of sight by +/- 15 / 30 degrees | = |
| `[` `]` / `{` `}` | rotate about the horizontal axis by +/- 15 / 30 degrees | = |
| `/` `\` / `?` `\|` | rotate about the vertical axis by +/- 15 / 30 degrees | = (today `?` opens help and `\` hides the panel: moved to `h` and `` ` ``) |

## Turning the view with the mouse
| action | effect |
|---|---|
| alt + drag on a map | turn the view (about the screen's vertical / horizontal axis, 0.5 degree per pixel); quick 128-pixel renders while dragging, the full map on release |
| View > orientation widget | a small trackball in each pane: the aligned frame's x, y, z as the current tilt shows them (dim = pointing away); drag it to turn (rendered on release), click an axis tip to look along that axis; buttons face-on, edge-on, disc (L), orbit (Lbh) |

## Overlays and particles
| key | action | vs SPLASH |
|---|---|---|
| `v` / `V` | vectors: fewer / more arrows (grid); streamlines: lower / higher density | = |
| `w` | vectors: back to the default arrow scaling and grid | = |
| `p` | inspect the particle nearest the cursor (the click-inspect card) | = (label closest particle; today `p` cycles the number of panes, moved to `P`) |
| `c` | show the smoothing circle of the particle nearest the cursor | = |
| `R` | clear the selection (phase-diagram lasso) and every range restriction | = |
| `Backspace`, `Esc` | clear inspect marks and selections *(Esc today)* | = |
| `g` | draw a line: a profile of the field along it (with its gradient) | ~ (SPLASH: line and its gradient) |

## Panes, compare, layout (SPLOSH's own)
| key | action | vs SPLASH |
|---|---|---|
| `Tab` / `shift+Tab` | select the next / previous pane | moved (today `1`-`4`) |
| `P` | cycle the number of panes 1-4 | moved (today `p`) |
| `L` | link / unlink pan and zoom | moved (today `l`) |
| `A` / `B` | store the state A / B | moved (today `a` / `b`) |
| `shift+space` | blink A / B | moved (today `space`) |
| `~` | difference mode: off, A - B, log A/B | moved (today `d`) |
| `` ` `` | maps only: hide the side panel and the menu bar (the bar returns at the top edge) | moved (today `\`) |
| `e` / `E` | set the movie range end / start to this snapshot (then File > Movie) | ~ (SPLASH: animation sequence end point) |
| `h`, `F1` | the key list | = (`?` is a rotation key in SPLASH) |

## Zoom in the plot panes: radial profile, time series, heatmap, phase diagram, time-averaged radial profile, spectrum (SPLOSH's own, 2026-11-13)
| input | action |
|---|---|
| `shift`+drag | zoom to the box drawn (a box under 6 px tall or wide zooms the other axis only) |
| drag along the x-axis strip (the tick labels under the plot) / the y-axis strip (left of it) | zoom that axis to the band drawn |
| wheel over an axis strip | zoom that axis about the cursor |
| wheel inside the plot | time series, heatmap: zoom in time (the window is fetched again); profile, phase diagram, time-averaged radial profile, spectrum: both axes |
| `alt`+wheel inside the plot | zoom y only |
| drag inside a zoomed plot / middle drag | pan (the zoomed axes); on a phase diagram a plain drag draws a selection (lasso), so it pans with the middle button |
| double click, `o`, right click > Reset the zoom | reset the zoom (the whole run, automatic y; a phase diagram's automatic ranges) |
| `R` on a phase-diagram pane | clear its selection and the highlight on the maps |

A zoomed y replaces the automatic y limits until the reset; `ctrl` / `shift`+wheel still scan the snapshots over a plot pane. The zoom is saved in layouts as the panel keys
`xlim` / `ylim` (in the panel's x and y units: a time window in t/P0, Myr or code time as the panel's `xaxis` says) and restored when the layout is opened; a phase pane keeps its `xr` / `yr`.

## Mouse and window (SPLOSH's own, 2026-11-12)
| input | action |
|---|---|
| mouse back / forward buttons | previous / next snapshot (the browser's history navigation is prevented) |
| `ctrl` or `shift` + wheel over the slider or a pane | scan the snapshot number (label, clock and a mark on the slider show the target); loads on releasing the key; `Esc` cancels |
| middle click / middle drag | centre here / pan; double middle click: reset the view |
| right click | a menu at that point (map: centre, zoom, reset, nearest black hole, track, inspect, copy coordinates / value; time series: go to that time, reset the zoom, stack mode, copy; profile / phase diagram: reset the zoom, clear the selection, copy x y); `shift`+right click: the browser's menu |
| `q` | show / hide the side panel (also in Figure mode) |

## Not taken from SPLASH
- `G` / `T` / `H` (move legend, title, vector legend), `ctrl+t` / `^` (annotations), `q` (quit): SPLOSH has fixed labels and no plot window to quit;
  annotations belong to the publication layouts (docs: the figures plan), not to the viewer.
- `0` (hide selected particles), `1`-`9` (colour selected particles), `x` / `y` (range restrictions) in SPLASH's "select area" mode: SPLOSH selects
  through the phase-diagram lasso; a map lasso (`shift`+drag) could take these later.

## Decided
1. Space = next snapshot (SPLASH); blink on `shift+space`.
2. `0`-`9` are a SPLASH-style step count; panes by `Tab`.
3. Right click was not bound at first (the browser's context menu stays); since 2026-11-12 it opens SPLOSH's menu and `shift`+right click gives the browser's.
4. `q` (SPLASH: quit) hides the side panel: there is no window to quit, and hiding the panel is harmless if pressed by habit.
