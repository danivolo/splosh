"""Dashed overlay lines ("dash": [on, off] px) in the rasteriser and in splosh.mpl; plugin overlays listed for the viewer."""
import numpy as np
import pytest

from splosh import overlays as O


def test_dashed_segments_cover_about_on_over_period():
    segs = O._dashed([((0.0, 0.0), (100.0, 0.0))], [6, 4])
    covered = sum(b[0] - a[0] for a, b in segs)
    assert 58 <= covered <= 62 and len(segs) == 10
    assert O._dashed([((0, 0), (5, 0))], None) == [((0, 0), (5, 0))]


def test_rasteriser_draws_gaps():
    rgb = np.zeros((40, 200, 3), np.uint8)
    O.draw_items(rgb, [{"kind": "polyline", "points": [[-0.9, 0.0], [0.9, 0.0]], "color": [255, 255, 255], "linewidth": 2, "dash": [8, 8]}], [-1, 1, -0.2, 0.2])
    row = rgb[20, 30:170, 0] > 100
    assert row.any() and (~row).any()                        # lit and dark runs along the line


def test_mpl_dashed_items():
    mpl = pytest.importorskip("matplotlib"); mpl.use("Agg")
    import matplotlib.pyplot as plt
    from splosh import mpl as avm
    fig, ax = plt.subplots()
    avm.draw_items(ax, [{"kind": "arrow", "x": 0, "y": 0, "dx": 0.5, "dy": 0.2, "color": "magenta", "linewidth": 1.5, "dash": [5, 3]},
                        {"kind": "polyline", "points": [[0, 0], [0.3, 0.4], [0.6, 0.1]], "color": "cyan", "dash": [2, 2]}], [-1, 1, -1, 1])
    assert any(l.get_linestyle() not in ("-", "solid") for l in ax.lines)
    plt.close(fig)


def test_line_width_follows_the_picture_size():
    """a 1 px line of the viewer is about 1/768 of the picture: on a 2304 px frame it is drawn about 3 px wide."""
    def width(n):
        rgb = np.zeros((n, n, 3), np.uint8)
        O.draw_items(rgb, [{"kind": "polyline", "points": [[-0.9, 0.0], [0.9, 0.0]], "color": [255, 255, 255], "linewidth": 1.0}], [-1, 1, -1, 1])
        return int((rgb[:, n // 2, 0] > 60).sum())
    assert width(768) <= 2 and width(2304) >= 3
