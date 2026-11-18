"""Generate som2_workflow.ipynb.  Usage: python tools/make_workflow_notebook.py <out.ipynb> [<cells.py>]"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# SOM2.0: seismic facies from foundation-model embeddings, calibrated to wells

This notebook compares two unsupervised seismic-facies workflows on the same seismic volume, the same analysis zone,
the same self-organising map (SOM) and the same well calibration:

| | **SOM1.0** (multi-attribute) | **SOM2.0** (embedding) |
|---|---|---|
| Input features | ~36 single-trace instantaneous + banded attributes computed from the seismic | 768-channel embedding from the NCS seismic foundation model ([NorskRegnesentral/NCS_models](https://github.com/NorskRegnesentral/NCS_models)) |
| Reduction | PCA used to **select** the dominant 8–12 attributes, then z-score | PCA to *k* components (12 and 32 tested) |
| Classification | 8 × 8 SOM, batch algorithm | identical |
| Calibration | SOM nodes → well facies: biggest winner, and probabilistic (neighbour-smoothed) | identical |
| Scoring | leave-one-well-out at the well trace | identical |

**Why SOM2.0 can help:** a seismic attribute describes one sample (or a short window) of one trace. An embedding
from a model pre-trained on many seismic surveys describes each location *in the context of its surroundings*:
reflector continuity, terminations, texture and stacking pattern. Similar geology lands close together in the
embedding space, which is exactly what a SOM needs.

Steps:
1. configuration
2. seismic and analysis zone
3. well tie
4. SOM1.0
5. SOM2.0
6. calibration and scores
7. well-log comparison
8. facies sections
9. caveats
""")

md("## 1 · Configuration")
code(r"""
from pathlib import Path

DATA = Path(r"C:\Data\Test")                                    # local data root (nothing is uploaded)
SEISMIC = DATA / "SEGY" / "1988_AVO18.sgy"                       # base seismic volume used by both workflows
NCS_FEATURES = DATA / "SEGY" / "1988_AVO18_ncs_features_dens44.zarr"   # NCS embedding of the same volume
LAS_DIR = DATA / "LAS"                                           # well facies logs in TWT (one LAS per well)
FACIES_CURVE = "elastic_lithofluid_facies_upscaled"
TOP_HORIZON = DATA / "Horizons" / "Balder.txt"                   # analysis zone: top ...
BASE_HORIZON = DATA / "Horizons" / "MAURT.txt"                   # ... and base
FACIES_NAMES = {1: "Brine sand", 2: "Soft shale", 3: "Shale", 4: "Oil sand", 7: "Tuff", 8: "Overburden",
                10: "Carbonaceous"}
FACIES_COLOURS = {1: "#4e9fd6", 2: "#9a8c6a", 3: "#7a7a7a", 4: "#e8b33a", 7: "#b07cc6", 8: "#3d3d3d",
                  10: "#5a3e2b"}
SOM_SHAPE = (8, 8)
NCS_K = (12, 32)                 # numbers of PCA components tested for SOM2.0
CAL_RADIUS = 1                   # calibration uses the (2r+1)^2 traces around each well; scoring uses the well trace
SEED = 0
""")
code(r"""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, str(Path.cwd()))
import som2
np.random.seed(SEED)
""")

md(r"""
## 2 · Seismic and analysis zone
Both workflows use the same seismic volume and the same zone, the samples between the top and base horizons.
""")
code(r"""
with som2.timer("load seismic"):
    cube, ilines, xlines, twt = som2.load_segy(SEISMIC)
dt_ms = float(np.median(np.diff(twt)))
to_ilxl, resid = som2.survey_affine(SEISMIC)
top = som2.load_horizon(TOP_HORIZON, ilines, xlines, to_ilxl)
base = som2.load_horizon(BASE_HORIZON, ilines, xlines, to_ilxl)
mask = som2.zone_from_horizons(top, base, twt)
zone_index = np.full(mask.shape, -1, np.int64); zone_index[mask] = np.arange(int(mask.sum()))
print(f"seismic {cube.shape}, dt {dt_ms} ms | zone {int(mask.sum()):,} samples, "
      f"thickness {np.percentile(base - top, 5):.0f}-{np.percentile(base - top, 95):.0f} ms (P5-P95)")
print(f"inline/xline <-> X/Y affine fit residual {resid:.2f} m")
""")

md(r"""
## 3 · Well tie
Well locations come from the LAS header (X/Y), mapped to inline/crossline with an affine fit to the SEG-Y trace
headers. Wells are treated as vertical. The facies log, in TWT, is put on the seismic sample grid by majority vote
within ± half a seismic sample. Ties go to the log sample nearest the seismic time. Code 0 (unclassified) is
dropped.

**Which facies curve:** use the *upscaled* (seismic-scale) curve for calibration and scoring. A raw curve adds beds
of a few milliseconds, one or two seismic samples thick, that no seismic method can resolve. In tests they were
never predicted and only lowered the scores. Resampling the seismic to 1 ms to match a finer log adds no
resolution either, because the band-limited data carry the same frequencies at either rate.
""")
code(r"""
wells_raw, info = som2.well_facies_on_grid(sorted(LAS_DIR.glob("*.las")), FACIES_CURVE, to_ilxl, ilines, xlines, twt)
display(pd.DataFrame(info))
codes = sorted({int(c) for (_, _, _, f) in wells_raw.values() for c in f})
code_to_k = {c: k for k, c in enumerate(codes)}
n_fac = len(codes)
fac_names = [FACIES_NAMES.get(c, str(c)) for c in codes]
fac_cmap = ListedColormap([FACIES_COLOURS.get(c, "#cccccc") for c in codes])
wells = {}
for name, (il, xl, t, f) in wells_raw.items():
    keep = mask[il, xl, t]                                   # well samples inside the zone
    wells[name] = (il[keep], xl[keep], t[keep], np.array([code_to_k[int(c)] for c in f[keep]]))
counts = pd.DataFrame({n: pd.Series(w[3]).map(dict(enumerate(fac_names))).value_counts() for n, w in wells.items()}).fillna(0).astype(int)
counts.loc["total"] = counts.sum(); display(counts)
majority = counts.drop("total").sum(1).max() / counts.loc["total"].sum()
print(f"majority-class baseline (predict the most common facies everywhere): {majority:.0%}")
""")

md(r"""
## 4 · SOM1.0: multi-attribute workflow
The attributes are single-trace instantaneous attributes (envelope, phase, frequency, sweetness, thin bed,
bandwidth, Q, attenuation, relative acoustic impedance and others) and *banded* attributes (averages between
phase breaks or envelope breaks). All are computed from the single base volume.

PCA is then used the way it usually is in multi-attribute SOM work: to **pick the attributes** that dominate the
leading eigenvectors. The SOM runs on those attributes.
""")
code(r"""
with som2.timer("SOM1.0 attributes"):
    attr_names, X = som2.som1_attributes(cube, dt_ms, mask)
pca1 = som2.pca_fit(X)
sel_idx, sel_names = som2.som1_select(pca1, attr_names)
Z1 = ((X[:, sel_idx] - X[:, sel_idx].mean(0)) / (X[:, sel_idx].std(0) + 1e-9)).astype(np.float32)
del X
print(len(attr_names), "attributes computed; PCA-selected for the SOM:", sel_names)
with som2.timer("SOM1.0 SOM"):
    W1, snaps1 = som2.train_som(Z1, SOM_SHAPE, seed=SEED)
    b1 = som2.bmu(Z1, W1)
""")

md(r"""
## 5 · SOM2.0: foundation-model embedding
The embedding comes from the NCS model (ViT encoder, masked-autoencoder pre-training on public Norwegian
Continental Shelf seismic). To create it for a new volume:

```python
som2.run_ncs(SEISMIC, "features.zarr", model="NorskRegnesentralSTI/NCS-v1-2d-base", densify=4)   # GPU recommended
```

**Registration:** the store's `inline/xline/time_depth` coordinates are 0-based sample indices of the input
volume, at the patch start + `patch_size // 2`. The true patch centre is therefore half a sample earlier. The
features are interpolated from that grid onto the seismic samples of the zone. PCA is fitted on a regular
sub-grid of embedding vectors.
""")
code(r"""
ds = som2.open_ncs(NCS_FEATURES)
som2_runs = {}
for k in NCS_K:
    with som2.timer(f"SOM2.0 k={k}"):
        pca2, _, Z2 = som2.ncs_pcs_on_zone(ds, mask, ilines, xlines, twt, k)
        W2, snaps2 = som2.train_som(Z2, SOM_SHAPE, seed=SEED)
        som2_runs[k] = dict(pca=pca2, b=som2.bmu(Z2, W2), snaps=snaps2)
    print(f"k={k}: {pca2['var_ratio'][:k].sum():.0%} of embedding variance")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
for a, (title, v, k) in zip(ax, [("SOM1.0 attributes", pca1["var_ratio"], len(sel_idx)),
                                 ("SOM2.0 embedding", som2_runs[max(NCS_K)]["pca"]["var_ratio"], max(NCS_K))]):
    n = min(40, len(v)); a.bar(range(1, n + 1), v[:n], color="steelblue")
    a2 = a.twinx(); a2.plot(range(1, n + 1), np.cumsum(v[:n]), "k-"); a2.set_ylim(0, 1)
    a.set(title=f"{title}: variance per component", xlabel="component")
plt.tight_layout(); plt.show()
""")
md(r"""
The attribute PCA is steep: a few components carry most of the variance, because many attributes measure the
same thing (envelope, energy, sweetness). The embedding spectrum is much flatter, so more components are needed
to represent it. That's why SOM2.0 is tested with both 12 and 32 components.
""")

md(r"""
## 6 · Calibration and leave-one-well-out scores
For each well in turn, the SOM nodes are calibrated with the *other* wells and the held-out well is predicted.

- **Biggest winner:** each node takes the facies most often seen in it. Nodes that no calibration well visited
  stay *unknown*.
- **Probabilistic:** P(facies | node) ∝ P(node | facies) · P(facies)^0.5. P(node | facies) comes from well counts
  smoothed over neighbouring SOM nodes, which fills unvisited nodes and gives a confidence.
""")
code(r"""
runs = {"SOM1.0": b1, **{f"SOM2.0 (k={k})": r["b"] for k, r in som2_runs.items()}}
rows, per_well = [], {}
for name, b in runs.items():
    scores, pw = som2.leave_one_well_out(b, wells, zone_index, n_fac, SOM_SHAPE, CAL_RADIUS)
    per_well[name] = pw
    for m, s in scores.items():
        rows.append({"workflow": name, "calibration": m, "accuracy": s["accuracy"], "balanced accuracy": s["balanced_accuracy"],
                     "unknown": s["unknown"], **{f"recall {n}": r for n, r in zip(fac_names, s["recall"])},
                     **{f"acc {w}": pw[w][f"acc_{m}"] for w in wells}})
scores_df = pd.DataFrame(rows)
display(scores_df.style.format(precision=2, subset=scores_df.columns[2:]).hide(axis="index"))
""")
code(r"""
prob = scores_df[scores_df.calibration == "prob"].set_index("workflow")
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
prob[[f"acc {w}" for w in wells] + ["accuracy"]].T.rename(index=lambda s: s.replace("acc ", "")).plot.bar(ax=ax[0], rot=0)
ax[0].axhline(majority, color="k", ls="--", lw=1, label="majority class"); ax[0].set(title="Accuracy per held-out well (probabilistic)", ylim=(0, 1)); ax[0].legend()
prob[[f"recall {n}" for n in fac_names]].T.rename(index=lambda s: s.replace("recall ", "")).plot.bar(ax=ax[1], rot=0)
ax[1].set(title="Recall per facies (probabilistic)", ylim=(0, 1))
plt.tight_layout(); plt.show()
""")

md("## 7 · Well-log comparison (held-out predictions)")
code(r"""
best2 = f"SOM2.0 (k={max(NCS_K)})"
fig, axes = plt.subplots(1, len(wells), figsize=(4.2 * len(wells), 6), sharey=False)
for ax, w in zip(np.atleast_1d(axes), wells):
    t = twt[per_well["SOM1.0"][w]["t_index"]]
    cols = [per_well["SOM1.0"][w]["true"], per_well["SOM1.0"][w]["prob"], per_well[best2][w]["prob"]]
    ax.imshow(np.stack(cols, 1), cmap=fac_cmap, vmin=-0.5, vmax=n_fac - 0.5, aspect="auto", interpolation="nearest",
              extent=[0, 3, t.max() + dt_ms / 2, t.min() - dt_ms / 2])
    ax.set_xticks([0.5, 1.5, 2.5], ["well log", "SOM1.0", "SOM2.0"]); ax.set_title(w); ax.set_ylabel("TWT (ms)")
handles = [plt.Rectangle((0, 0), 1, 1, color=fac_cmap(k)) for k in range(n_fac)]
fig.legend(handles, fac_names, loc="lower center", ncol=n_fac); plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()
""")

md(r"""
## 8 · Facies sections through the wells
Maps use all wells for calibration (probabilistic). The section is an arbitrary line through the wells.
""")
code(r"""
def calibrated(b):
    zi, fa = som2.calibration_samples(wells, zone_index, CAL_RADIUS)
    _, _, post = som2.calibrate(b[zi], fa, SOM_SHAPE[0] * SOM_SHAPE[1], n_fac, SOM_SHAPE)
    return post[b].argmax(1), post[b].max(1)

# arbitrary line through the wells, ordered along the inline axis and extended 60 traces each side
pts = sorted(((w[0][0], w[1][0], n) for n, w in wells.items()), key=lambda p: p[0])
ext = lambda p, q, d: (p[0] + (p[0] - q[0]) * d / max(1, np.hypot(p[0] - q[0], p[1] - q[1])), p[1] + (p[1] - q[1]) * d / max(1, np.hypot(p[0] - q[0], p[1] - q[1])))
nodes = [ext(pts[0], pts[1], 60)] + [(p[0], p[1]) for p in pts] + [ext(pts[-1], pts[-2], 60)]
path = []
for (a0, a1), (c0, c1) in zip(nodes[:-1], nodes[1:]):
    n = int(np.ceil(np.hypot(c0 - a0, c1 - a1)))
    path += [(int(round(a0 + (c0 - a0) * s / n)), int(round(a1 + (c1 - a1) * s / n))) for s in range(n)]
path = np.clip(np.array(path), 0, [len(ilines) - 1, len(xlines) - 1])
well_pos = {n: int(np.argmin(np.hypot(path[:, 0] - i, path[:, 1] - x))) for i, x, n in pts}
zi_line = zone_index[path[:, 0], path[:, 1]]                     # (n_trace, n_t)
t_lo, t_hi = np.searchsorted(twt, top[path[:, 0], path[:, 1]].min() - 60), np.searchsorted(twt, base[path[:, 0], path[:, 1]].max() + 60)

def line_img(vals):
    img = np.full(zi_line.shape, np.nan); ok = zi_line >= 0; img[ok] = vals[zi_line[ok]]; return img[:, t_lo:t_hi].T

ext_ = [0, len(path), twt[t_hi - 1], twt[t_lo]]
fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
s = cube[path[:, 0], path[:, 1]][:, t_lo:t_hi].T; v = np.percentile(np.abs(s), 98)
ax[0].imshow(s, cmap="gray_r", vmin=-v, vmax=v, aspect="auto", extent=ext_); ax[0].set_title("Seismic (arbitrary line through the wells)")
for name, b in [("SOM1.0", b1), (best2, som2_runs[max(NCS_K)]["b"])]:
    f, c = calibrated(b)
    a = ax[1] if name == "SOM1.0" else ax[2]
    a.imshow(s, cmap="gray_r", vmin=-v, vmax=v, aspect="auto", extent=ext_)
    a.imshow(line_img(f), cmap=fac_cmap, vmin=-0.5, vmax=n_fac - 0.5, aspect="auto", extent=ext_, alpha=0.8, interpolation="nearest")
    a.set_title(f"{name}: most likely facies")
for a in ax:
    a.plot(top[path[:, 0], path[:, 1]], "w-", lw=1); a.plot(base[path[:, 0], path[:, 1]], "w-", lw=1)
    for n, p in well_pos.items():
        a.axvline(p, color="k", lw=1.5); a.text(p, twt[t_lo] + 8, n, ha="center", va="top", bbox=dict(fc="w", lw=0, alpha=0.7))
    a.set_ylabel("TWT (ms)")
fig.legend(handles, fac_names, loc="lower center", ncol=n_fac); plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.show()
""")

md(r"""
## 9 · Caveats
- **Few wells.** Leave-one-well-out with three wells (about 140 samples) gives noisy per-facies recalls: each rests
  on 10–40 samples. Read the per-well bars as well as the totals.
- **Attribute set.** SOM1.0 uses the published formulas for standard single-trace and banded attributes, with
  PCA-based selection. Window lengths and implementation details will differ between software packages.
- **NCS model.** These results use the 2D model (`NCS-v1-2d-base`), which processes sections independently and
  can leave tiling seams. The 2.5D model, which combines four directions, should reduce them.
- **Vertical wells and nearest-trace tie.** Deviated wells need per-sample inline/crossline positions.
""")
code(r"""
print({k: round(v, 1) for k, v in som2.TIMINGS.items()})
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in cells:
    c["id"] = f"c{cells.index(c):02d}"
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write("\n\n".join(c["source"] for c in cells if c["cell_type"] == "code"))
