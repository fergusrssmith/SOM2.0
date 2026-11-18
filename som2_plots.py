"""
som2_plots.py: QC and explanation plots for the SOM1.0 / SOM2.0 workflow notebook.

All section plots use an arbitrary line through the wells (well_line), and all map plots use a horizon slice
(horizon_slice): the zone sample a fixed time below a horizon.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEIS_CMAP = "gray_r"


# ------------------------------------------------------------------------------------------------ geometry
def well_line(wells, n_il, n_xl, extend=60):
    """Arbitrary line through the wells (ordered along inline), extended `extend` traces at each end.
    Returns (path (n, 2) of [il_idx, xl_idx], {well: position along path})."""
    pts = sorted(((int(w[0][0]), int(w[1][0]), n) for n, w in wells.items()), key=lambda p: p[0])

    def ext(p, q, d):
        L = max(1.0, np.hypot(p[0] - q[0], p[1] - q[1]))
        return p[0] + (p[0] - q[0]) * d / L, p[1] + (p[1] - q[1]) * d / L

    nodes = [ext(pts[0], pts[1], extend)] + [(p[0], p[1]) for p in pts] + [ext(pts[-1], pts[-2], extend)]
    path = []
    for (a0, a1), (c0, c1) in zip(nodes[:-1], nodes[1:]):
        n = int(np.ceil(np.hypot(c0 - a0, c1 - a1)))
        path += [(int(round(a0 + (c0 - a0) * s / n)), int(round(a1 + (c1 - a1) * s / n))) for s in range(n)]
    path = np.clip(np.array(path), 0, [n_il - 1, n_xl - 1])
    pos = {n: int(np.argmin(np.hypot(path[:, 0] - i, path[:, 1] - x))) for i, x, n in pts}
    return path, pos


def zone_to_line(values, zone_index, path, t0, t1):
    """Zone-sample values -> (n_t, n_trace) image along the path (NaN outside the zone)."""
    zl = zone_index[path[:, 0], path[:, 1], t0:t1]
    img = np.full(zl.shape, np.nan, dtype=float)
    ok = zl >= 0
    img[ok] = values[zl[ok]]
    return img.T


def horizon_slice(values, zone_index, horizon, offset_ms, twt):
    """Zone-sample values on the surface `horizon + offset_ms` -> (n_il, n_xl) map (NaN outside the zone)."""
    k = np.clip(np.rint((horizon + offset_ms - twt[0]) / np.median(np.diff(twt))).astype(int), 0, len(twt) - 1)
    zi = np.take_along_axis(zone_index, k[..., None], axis=-1)[..., 0]
    out = np.full(zi.shape, np.nan)
    out[zi >= 0] = values[zi[zi >= 0]]
    return out


# ------------------------------------------------------------------------------------------------ drawing
def decorate_line(ax, twt_range, pos, top=None, base=None, path=None, label=True):
    if top is not None:
        ax.plot(top[path[:, 0], path[:, 1]], color="w", lw=1.2)
        ax.plot(base[path[:, 0], path[:, 1]], color="w", lw=1.2)
    for n, p in pos.items():
        ax.axvline(p, color="k", lw=1.4)
        if label:
            ax.text(p, twt_range[0] + 4, n, ha="center", va="top", fontsize=8,
                    bbox=dict(fc="w", lw=0, alpha=0.75, pad=1))
    ax.set_ylim(twt_range[1], twt_range[0])
    ax.set_ylabel("TWT (ms)")


def seismic_line(ax, cube, path, twt, t0, t1, clip=98):
    s = cube[path[:, 0], path[:, 1]][:, t0:t1].T
    v = np.percentile(np.abs(s), clip)
    ax.imshow(s, cmap=SEIS_CMAP, vmin=-v, vmax=v, aspect="auto", extent=[0, len(path), twt[t1 - 1], twt[t0]])
    return s


def overlay(ax, img, path, twt, t0, t1, cmap, vmin=None, vmax=None, alpha=0.85):
    return ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, aspect="auto", interpolation="nearest",
                     extent=[0, len(path), twt[t1 - 1], twt[t0]])


def facies_legend(fig, names, cmap, ncol=None, y=0.0):
    h = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(names))]
    fig.legend(h, names, loc="lower center", ncol=ncol or len(names), bbox_to_anchor=(0.5, y), frameon=False)


def rgb_from_scores(Z, comps=(0, 1, 2), clip=98):
    """First three columns of Z -> RGB in [0, 1] (each channel clipped at its `clip` percentile)."""
    out = np.empty((len(Z), 3))
    for c, j in enumerate(comps):
        v = Z[:, j] / (np.percentile(np.abs(Z[:, j]), clip) + 1e-9)
        out[:, c] = (np.clip(v, -1, 1) + 1) / 2
    return out


def rgb_line(values_rgb, zone_index, path, t0, t1):
    zl = zone_index[path[:, 0], path[:, 1], t0:t1]
    img = np.ones(zl.shape + (3,)) * 0.15
    ok = zl >= 0
    img[ok] = values_rgb[zl[ok]]
    return img.transpose(1, 0, 2)


# ------------------------------------------------------------------------------------------------ SOM views (hexagonal)
from matplotlib.patches import RegularPolygon  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, to_rgb  # noqa: E402


def _hex_centres(shape):
    """Neuron k = row * C + col drawn on a hexagonal lattice: odd rows shifted half a cell (same as som2.som_coords)."""
    R, C = shape
    r, c = np.divmod(np.arange(R * C), C)
    return np.stack([c + 0.5 * (r % 2), -r * np.sqrt(3) / 2], 1)


def hex_map(ax, shape, facecolors, labels=None, edgecolor="0.35", hatch=None, title="", fontsize=6, text_colors=None):
    """Draw one hexagon per neuron. facecolors: (n, 3|4) in [0, 1]; labels: per-neuron strings; hatch: bool mask."""
    xy = _hex_centres(shape)
    for k, (x, y) in enumerate(xy):
        ax.add_patch(RegularPolygon((x, y), 6, radius=0.57, orientation=0, facecolor=facecolors[k], edgecolor=edgecolor,
                                    lw=0.6, hatch="////" if hatch is not None and hatch[k] else None))
        if labels is not None and labels[k]:
            ax.text(x, y, labels[k], ha="center", va="center", fontsize=fontsize,
                    color=(text_colors[k] if text_colors is not None else "k"))
    ax.set_xlim(xy[:, 0].min() - 0.7, xy[:, 0].max() + 0.7)
    ax.set_ylim(xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def node_grid(ax, values, shape, cmap="viridis", title="", fmt=None, vmin=None, vmax=None, colors=None):
    """One value (or explicit RGB colour) per neuron, on the hexagonal map, with a colour bar for values."""
    if colors is not None:
        fc = np.asarray(colors, float)
        fc = fc / 255 if fc.max() > 1 else fc
        hex_map(ax, shape, fc, title=title)
        return
    v = np.asarray(values, float)
    norm = Normalize(vmin=np.nanmin(v) if vmin is None else vmin, vmax=np.nanmax(v) if vmax is None else vmax)
    cm = plt.get_cmap(cmap)
    fc = cm(norm(v))
    labels = [fmt(x) for x in v] if fmt else None
    tc = ["w" if (0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]) < 0.5 else "k" for f in fc]
    hex_map(ax, shape, fc, labels, title=title, text_colors=tc)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax, fraction=0.046)


def u_matrix(W, shape, topology="hex"):
    """Mean distance from each neuron to its map neighbours (high = boundary between clusters)."""
    R, C = shape
    r, c = np.divmod(np.arange(R * C), C)
    pos = np.stack([r * np.sqrt(3) / 2, c + 0.5 * (r % 2)], 1) if topology == "hex" else np.stack([r, c], 1).astype(float)
    d = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
    nb = (d > 0) & (d < 1.01)
    dist = np.sqrt(((W[:, None] - W[None]) ** 2).sum(-1))
    return np.array([dist[k, nb[k]].mean() for k in range(R * C)])


def node_facies_bars(ax, counts, shape, cmap, title=""):
    """Facies mix of the calibration samples captured by each neuron (stacked bar inside each hexagon);
    hatched = no well sample landed on that neuron."""
    xy = _hex_centres(shape)
    hex_map(ax, shape, np.ones((len(xy), 3)), hatch=counts.sum(1) == 0, title=title)
    for k, (x, y) in enumerate(xy):
        tot = counts[k].sum()
        if not tot:
            continue
        y0 = y - 0.38
        for f in range(counts.shape[1]):
            if counts[k, f]:
                h = 0.76 * counts[k, f] / tot
                ax.add_patch(plt.Rectangle((x - 0.2, y0), 0.4, h, color=cmap(f), lw=0))
                y0 += h
        ax.text(x, y + 0.45, int(tot), ha="center", va="center", fontsize=5)


def facies_topology(ax, counts, post, shape, cmap, names, title="", numbers=True):
    """Winning facies per neuron ('facies topology'): colour = most probable facies (paler = less
    certain), label = neuron number and probability; white = no well sample on that neuron (its facies is inferred
    from neighbouring neurons)."""
    win = post.argmax(1)
    pr = post.max(1)
    seen = counts.sum(1) > 0
    fc = []
    for k in range(len(win)):
        base = np.array(to_rgb(cmap(win[k])))
        a = 0.3 + 0.7 * np.clip((pr[k] - 1 / post.shape[1]) / (1 - 1 / post.shape[1]), 0, 1)
        fc.append(base * a + (1 - a) if seen[k] else np.array([1.0, 1.0, 1.0]))
    labels = [(f"{k + 1}\n" if numbers else "") + f"P={pr[k]:.2f}" for k in range(len(win))]
    hex_map(ax, shape, np.array(fc), labels, title=title, fontsize=5)


def neuron_key(ax, shape, node_rgb, title="neuron colour key", numbers=True):
    labels = [str(k + 1) for k in range(shape[0] * shape[1])] if numbers else None
    fc = np.asarray(node_rgb, float)
    fc = fc / 255 if fc.max() > 1 else fc
    tc = ["w" if (0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]) < 0.5 else "k" for f in fc]
    hex_map(ax, shape, fc, labels, title=title, text_colors=tc)


def confusion(ax, cm, names, title=""):
    """cm rows = predicted, cols = true (as returned by som2.score); shown normalised per true facies."""
    cm = np.asarray(cm, float)
    col = cm / np.maximum(cm.sum(0, keepdims=True), 1)
    ax.imshow(col, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{col[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                        color="w" if col[i, j] > 0.5 else "k")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("true (well log)"), ax.set_ylabel("predicted")
    ax.set_title(title, fontsize=10)


def facies_cmap(codes, colours):
    return ListedColormap([colours.get(c, "#cccccc") for c in codes])


def component_planes(W, labels, shape, title="", ncols=8, max_n=24):
    """Kohonen component planes on the hexagonal map: one map per input variable (red = high)."""
    n = min(len(labels), max_n)
    rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(1.9 * ncols, 1.9 * rows + 0.5))
    cm = plt.get_cmap("RdBu_r")
    for k, ax in enumerate(np.ravel(axes)):
        if k < n:
            v = W[:, k]
            hex_map(ax, shape, cm(Normalize(v.min(), v.max())(v)), edgecolor="none", title=labels[k])
            ax.title.set_fontsize(7)
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def selection_curve(ax, names, r2, target, title=""):
    """Variance of all candidates reconstructed as distinct variables are added."""
    ax.plot(range(1, len(r2) + 1), r2, "o-", color="steelblue")
    ax.axhline(target, color="k", ls="--", lw=1)
    ax.set_xticks(range(1, len(r2) + 1), names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance of all candidates reconstructed")
    ax.set_title(title, fontsize=10)


# ------------------------------------------------------------------------------------------------ wells on sections
def well_strips(ax, wells, well_pos, twt, cmap, width=4, dt=4.0):
    """Draw each well's facies log as a narrow coloured column at its position on a section (black outline)."""
    for name, (il, xl, t, f) in wells.items():
        if name not in well_pos:
            continue
        x0 = well_pos[name] - width / 2
        for tv, fv in zip(twt[t], f):
            ax.add_patch(plt.Rectangle((x0, tv - dt / 2), width, dt, color=cmap(fv), lw=0, zorder=5))
        ax.add_patch(plt.Rectangle((x0, twt[t].min() - dt / 2), width, np.ptp(twt[t]) + dt, fill=False, ec="k", lw=0.8, zorder=6))


def wiggle_traces(ax, cube, il, xl, twt, lo, hi, half_width=0, sf=0.45):
    """Variable-area wiggle display. half_width=0: just the trace at the well (positive lobes filled black);
    half_width>0: the traces around the well along the crossline, drawn with the `wiggle` package."""
    tt = (twt >= lo) & (twt <= hi)
    if half_width == 0:
        tr = cube[il, xl, tt].astype(float)
        tr = tr / (np.abs(tr).max() + 1e-9) * sf
        ax.plot(tr, twt[tt], "k", lw=0.8)
        ax.fill_betweenx(twt[tt], 0, tr, where=tr > 0, color="k", lw=0)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(hi, lo)
        return
    from wiggle import wiggle
    xs = np.arange(max(0, xl - half_width), min(cube.shape[1], xl + half_width + 1))
    data = cube[il, xs][:, tt].T
    data = data / (np.abs(data).max() + 1e-9)
    wiggle(data, tt=twt[tt], xx=np.arange(len(xs), dtype=float), ax=ax, color="k", sf=sf)
    ax.axvline(xl - xs[0], color="r", lw=0.8, alpha=0.6)
    ax.set_ylim(hi, lo)
