"""Generate som2_workflow.ipynb.  Usage: python tools/make_workflow_notebook.py <out.ipynb> [<cells.py>]"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


# =====================================================================================================
md(r"""
# SOM2.0: seismic facies from foundation-model embeddings, calibrated to wells

This notebook compares two unsupervised seismic-facies workflows. They share the same seismic volume, analysis
zone, self-organising map (SOM) and well calibration. **Only the features fed to the SOM differ.**

| | **SOM1.0** (multi-attribute) | **SOM2.0** (embedding) |
|---|---|---|
| Input features | ~36 single-trace instantaneous + banded attributes computed from the seismic | 768-channel embedding from the NCS seismic foundation model ([NorskRegnesentral/NCS_models](https://github.com/NorskRegnesentral/NCS_models)) |
| Reduction | PCA used to **select** the dominant 8–12 attributes, then z-score | PCA to *k* components (12 and 32 tested) |
| Classification | 8 × 8 SOM, batch algorithm | identical |
| Calibration | SOM nodes → well facies: biggest winner, and probabilistic (neighbour-smoothed) | identical |
| Scoring | leave-one-well-out at the well trace | identical |

**The idea in one sentence:** a seismic attribute describes one trace over a few samples. An embedding from a model
pre-trained on thousands of seismic surveys describes each location *in the context of its surroundings*:
reflector continuity, terminations, texture and stacking pattern. Similar geology lands close together in the
embedding space, which is exactly what a SOM needs.

**How to read this notebook:** every step ends with one or more figures and a short *How to read it* note. Run
the cells top to bottom. Section 1 holds every path and setting.
""")

md("## 1 · Configuration")
code(r"""
from pathlib import Path

DATA = Path(r"C:\Data\Test")                                    # local data root (nothing is uploaded)
SEISMIC = DATA / "SEGY" / "1988_AVO18.sgy"                       # base seismic volume used by both workflows
NCS_FEATURES = DATA / "SEGY" / "1988_AVO18_ncs_features_dens44.zarr"   # NCS embedding of the same volume
LAS_DIR = DATA / "LAS"                                           # well facies logs in TWT (one LAS per well)
FACIES_CURVE = "elastic_lithofluid_facies_upscaled"              # seismic-scale facies curve used for scoring
RAW_FACIES_CURVE = "elastic_lithofluid_facies"                   # raw curve, shown for context only
TOP_HORIZON = DATA / "Horizons" / "Balder.txt"                   # analysis zone: top ...
BASE_HORIZON = DATA / "Horizons" / "MAURT.txt"                   # ... and base
FACIES_NAMES = {1: "Brine sand", 2: "Soft shale", 3: "Shale", 4: "Oil sand", 7: "Tuff", 8: "Overburden",
                10: "Carbonaceous"}
FACIES_COLOURS = {1: "#4e9fd6", 2: "#9a8c6a", 3: "#7a7a7a", 4: "#e8b33a", 7: "#b07cc6", 8: "#3d3d3d",
                  10: "#c2410c"}
SOM_SHAPE = (8, 8)
NCS_K = (12, 32)                 # numbers of PCA components tested for SOM2.0
CAL_RADIUS = 1                   # calibration uses the (2r+1)^2 traces around each well; scoring uses the well trace
SEED = 0
""")
code(r"""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr
sys.path.insert(0, str(Path.cwd()))
import som2, som2_plots as sp
plt.rcParams.update({"figure.dpi": 90, "axes.titlesize": 11})
np.random.seed(SEED)
""")

# =====================================================================================================
md(r"""
## 2 · Seismic and analysis zone
Both workflows use the same seismic volume and the same zone: the samples between the top and base horizons.
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
print(f"seismic {cube.shape} (inline x xline x samples), dt {dt_ms:g} ms, {twt[0]:.0f}-{twt[-1]:.0f} ms")
print(f"zone: {int(mask.sum()):,} samples; thickness {np.percentile(base - top, 5):.0f}-{np.percentile(base - top, 95):.0f} ms (P5-P95)")
print(f"inline/xline <-> X/Y affine fit residual {resid:.2f} m")
""")

# =====================================================================================================
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
las = sorted(LAS_DIR.glob("*.las"))
wells_raw, info = som2.well_facies_on_grid(las, FACIES_CURVE, to_ilxl, ilines, xlines, twt)
display(pd.DataFrame(info))
codes = sorted({int(c) for (_, _, _, f) in wells_raw.values() for c in f})
code_to_k = {c: k for k, c in enumerate(codes)}
n_fac = len(codes)
fac_names = [FACIES_NAMES.get(c, str(c)) for c in codes]
fac_cmap = sp.facies_cmap(codes, FACIES_COLOURS)
wells = {}
for name, (il, xl, t, f) in wells_raw.items():
    keep = mask[il, xl, t]                                   # well samples inside the zone
    wells[name] = (il[keep], xl[keep], t[keep], np.array([code_to_k[int(c)] for c in f[keep]]))
counts = pd.DataFrame({n: pd.Series(w[3]).map(dict(enumerate(fac_names))).value_counts() for n, w in wells.items()})
counts = counts.reindex(fac_names).fillna(0).astype(int)                 # rows in facies-code order
majority = counts.sum(1).max() / counts.values.sum()
print(f"majority-class baseline (predict the most common facies everywhere): {majority:.0%}")
path, well_pos = sp.well_line(wells, len(ilines), len(xlines))
t0 = int(np.searchsorted(twt, top[path[:, 0], path[:, 1]].min() - 50))
t1 = int(np.searchsorted(twt, base[path[:, 0], path[:, 1]].max() + 50))
trange = (twt[t0], twt[t1 - 1])
""")

md("### QC: where the data are")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ext = [xlines[0], xlines[-1], ilines[-1], ilines[0]]
im = ax[0].imshow(top, cmap="viridis_r", extent=ext, aspect="auto"); plt.colorbar(im, ax=ax[0], label="TWT (ms)")
ax[0].contour(xlines, ilines, top, levels=12, colors="w", linewidths=0.4)
ax[0].plot(xlines[path[:, 1]], ilines[path[:, 0]], "w--", lw=1.2, label="section line")
for w in info:
    inside = w["inside_survey"]
    ax[0].plot(w["xline"], np.clip(w["inline"], ilines[0], ilines[-1]), "o" if inside else "x",
               color="red" if inside else "orange", ms=8, mec="k")
    ax[0].text(w["xline"] + 6, np.clip(w["inline"], ilines[0], ilines[-1]),
               w["well"] + ("" if inside else " (outside survey)"), color="w", fontsize=8)
ax[0].set(title="Top horizon time structure + wells", xlabel="crossline", ylabel="inline"); ax[0].legend(loc="lower right", fontsize=8)
im = ax[1].imshow(base - top, cmap="magma", extent=ext, aspect="auto"); plt.colorbar(im, ax=ax[1], label="ms")
ax[1].set(title="Analysis zone thickness (base - top)", xlabel="crossline", ylabel="inline")
sp.seismic_line(ax[2], cube, path, twt, t0, t1)
sp.overlay(ax[2], np.where(sp.zone_to_line(np.ones(int(mask.sum())), zone_index, path, t0, t1) > 0, 1.0, np.nan), path, twt, t0, t1,
           "autumn", alpha=0.18)
sp.decorate_line(ax[2], trange, well_pos, top, base, path); ax[2].set(title="Seismic along the line, zone shaded", xlabel="trace along line")
plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:* left, the top-horizon time structure, with the wells and the section line used throughout.
Wells marked × fall outside the 3D and can't be used. Middle, the zone thickness: every sample inside it is
classified. Right, the seismic along the line through the wells, with the zone shaded between the two horizons.
""")

md("### QC: the facies logs")
code(r"""
fig, axes = plt.subplots(1, len(wells) + 1, figsize=(4 * len(wells) + 4, 6.5), gridspec_kw={"width_ratios": [1] * len(wells) + [1.2]})
for ax, (name, (il, xl, t, f)) in zip(axes, wells.items()):
    hdr, c = som2.read_las(next(p for p in las if som2.read_las(p)[0].get("WELL") == name))
    lo, hi = twt[t].min() - 6, twt[t].max() + 6
    tr = c["DEPT"]; sel = (tr >= lo) & (tr <= hi)
    trace = cube[il[0], xl[0]]; tt = (twt >= lo) & (twt <= hi)
    ax.plot(trace[tt] / np.abs(trace[tt]).max() * 0.45 + 0.5, twt[tt], "k", lw=0.8)
    ax.fill_betweenx(twt[tt], 0.5, trace[tt] / np.abs(trace[tt]).max() * 0.45 + 0.5, where=trace[tt] > 0, color="k")
    for j, (curve, dt) in enumerate([(RAW_FACIES_CURVE, 2.0), (FACIES_CURVE, 2.0)]):
        for tv, fv in zip(tr[sel], c[curve][sel]):
            if np.isfinite(fv) and fv > 0:
                ax.add_patch(plt.Rectangle((1.1 + j, tv - dt / 2), 0.85, dt, color=FACIES_COLOURS.get(int(fv), "#ccc"), lw=0))
    for tv, fv in zip(twt[t], f):
        ax.add_patch(plt.Rectangle((3.1, tv - dt_ms / 2), 0.85, dt_ms, color=fac_cmap(fv), lw=0))
    ax.axhline(top[il[0], xl[0]], color="r", lw=1); ax.axhline(base[il[0], xl[0]], color="r", lw=1)
    ax.set_xlim(0, 4); ax.set_ylim(hi, lo); ax.set_title(name)
    ax.set_xticks([0.5, 1.55, 2.55, 3.55], ["seismic\ntrace", "raw\nlog", "upscaled\nlog", f"on seismic\n{dt_ms:g} ms"], fontsize=8)
    ax.set_ylabel("TWT (ms)")
counts.T.plot.barh(stacked=True, ax=axes[-1], color=[fac_cmap(i) for i in range(n_fac)], legend=False)
axes[-1].set(title="Calibration samples per well (zone)", xlabel="samples at the seismic rate")
all_codes = sorted(set(codes) | {7, 10})
sp.facies_legend(fig, [FACIES_NAMES[c] for c in all_codes], sp.facies_cmap(all_codes, FACIES_COLOURS), y=-0.01)
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()
""")
md(r"""
*How to read it:* for each well, the seismic trace at the well, then the raw facies log, the upscaled log, and the
facies actually used, one per seismic sample. Red lines mark the zone top and base. Thin raw-log beds (tuff,
carbonaceous) disappear at seismic scale: they are real, but below seismic resolution. Right, how many calibration
samples each well contributes per facies. Small counts mean noisy per-facies scores.
""")

# =====================================================================================================
md(r"""
## 4 · SOM1.0: multi-attribute workflow
The attributes are single-trace instantaneous attributes (envelope, phase, frequency, sweetness, thin bed,
bandwidth, Q, attenuation, relative acoustic impedance and others) and *banded* attributes (averages between
phase breaks or envelope breaks), all computed from the single base volume. PCA is then used to **pick the
attributes** that dominate the leading eigenvectors, and the SOM runs on those attributes.
""")
code(r"""
with som2.timer("SOM1.0 attributes"):
    attr_names, X = som2.som1_attributes(cube, dt_ms, mask)
pca1 = som2.pca_fit(X)
sel_idx, sel_names = som2.som1_select(pca1, attr_names)
Z1 = ((X[:, sel_idx] - X[:, sel_idx].mean(0)) / (X[:, sel_idx].std(0) + 1e-9)).astype(np.float32)
pc1_rgb = sp.rgb_from_scores(som2.pca_transform(X, pca1, 3))          # for the colour-blend QC below
print(len(attr_names), "attributes computed; PCA-selected for the SOM:", sel_names)
""")
md("### QC: how PCA picks the attributes")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={"width_ratios": [1.1, 1, 1]})
L = np.abs(pca1["comps"][:6]).T
im = ax[0].imshow(L, cmap="Greys", aspect="auto"); plt.colorbar(im, ax=ax[0], fraction=0.04, label="|loading|")
ax[0].set_yticks(range(len(attr_names)), attr_names, fontsize=7)
for i in sel_idx:
    ax[0].get_yticklabels()[i].set_color("red"); ax[0].get_yticklabels()[i].set_fontweight("bold")
ax[0].set_xticks(range(6), [f"PC{i+1}\n{v:.0%}" for i, v in enumerate(pca1["var_ratio"][:6])])
ax[0].set_title("Attribute loadings on the first 6 PCs\n(red = selected for the SOM)")
corr = np.corrcoef(Z1[np.random.default_rng(0).choice(len(Z1), 200_000, replace=False)].T)
im = ax[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar(im, ax=ax[1], fraction=0.046)
ax[1].set_xticks(range(len(sel_names)), sel_names, rotation=60, ha="right", fontsize=7)
ax[1].set_yticks(range(len(sel_names)), sel_names, fontsize=7); ax[1].set_title("Correlation between the selected attributes")
n = min(36, len(pca1["var_ratio"])); ax[2].bar(range(1, n + 1), pca1["var_ratio"][:n], color="steelblue")
a2 = ax[2].twinx(); a2.plot(range(1, n + 1), np.cumsum(pca1["var_ratio"][:n]), "k-"); a2.set_ylim(0, 1); a2.set_ylabel("cumulative")
ax[2].set(title="Variance explained per component", xlabel="principal component")
plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:* left, each row is an attribute and each column a principal component. Dark means the attribute
drives that component. The red attributes are the ones PCA selection keeps. Middle, how strongly the selected
attributes correlate with each other: large red blocks mean several of them measure the same thing (for example
envelope, energy and sweetness). That is a known weakness of selecting by variance. Right, variance per component.
""")
md("### QC: the selected attributes on the section")
code(r"""
show = sel_idx[:6]
fig, axes = plt.subplots(len(show), 1, figsize=(14, 2.3 * len(show)), sharex=True)
for ax, i in zip(axes, show):
    img = sp.zone_to_line(X[:, i], zone_index, path, t0, t1)
    v = np.nanpercentile(np.abs(img), [2, 98])
    sp.seismic_line(ax, cube, path, twt, t0, t1)
    lo, hi = np.nanpercentile(img, [2, 98])
    sp.overlay(ax, img, path, twt, t0, t1, "turbo", lo, hi, alpha=0.75)
    sp.decorate_line(ax, trange, well_pos, top, base, path, label=(i == show[0])); ax.set_title(attr_names[i], fontsize=9, loc="left")
plt.tight_layout(); plt.show()
del X
""")
md(r"""
*How to read it:* each panel is one attribute that goes into the SOM, shown inside the zone along the line through
the wells. Several look alike: they carry the same information.
""")
code(r"""
with som2.timer("SOM1.0 SOM"):
    W1, snaps1 = som2.train_som(Z1, SOM_SHAPE, seed=SEED)
    b1 = som2.bmu(Z1, W1)
""")

# =====================================================================================================
md(r"""
## 5 · SOM2.0: foundation-model embedding
The embedding comes from the NCS model: a ViT encoder with masked-autoencoder pre-training on public Norwegian
Continental Shelf seismic. To create it for a new volume:

```python
som2.run_ncs(SEISMIC, "features.zarr", model="NorskRegnesentralSTI/NCS-v1-2d-base", densify=4)   # GPU recommended
```

**Registration:** the store's `inline/xline/time_depth` coordinates are 0-based sample indices of the input
volume, at the patch start + `patch_size // 2`. The true patch centre is therefore half a sample earlier. The
features are interpolated from that grid onto the seismic samples of the zone. PCA is fitted on a regular sub-grid
of embedding vectors.
""")
code(r"""
ds = som2.open_ncs(NCS_FEATURES)
F = ds["features"]
gi = som2._coord_to_index(ds["inline"].values, ilines.astype(float))
gx = som2._coord_to_index(ds["xline"].values, xlines.astype(float))
gt = som2._coord_to_index(ds["time_depth"].values, twt)
fi = np.abs(gi[None] - path[:, :1]).argmin(1); fx = np.abs(gx[None] - path[:, 1:]).argmin(1)
ft = np.abs(gt[None] - np.arange(t0, t1)[:, None]).argmin(1)
with som2.timer("read embedding along line"):
    E = F.isel(inline=xr.DataArray(fi), xline=xr.DataArray(fx)).values            # (trace, feature-time, 768)
E = E[:, ft]                                                                       # nearest feature sample per seismic sample
En = E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)
ref_w = max(wells, key=lambda w: (wells[w][3] == code_to_k.get(4, -1)).sum())     # well with the most oil sand
ref_t = int(np.median(wells[ref_w][2][wells[ref_w][3] == code_to_k.get(4, 0)])) - t0
ref = En[well_pos[ref_w], ref_t]
print(f"embedding grid {dict(F.sizes)}; reference point: {ref_w} at {twt[ref_t + t0]:.0f} ms")
""")
code(r"""
som2_runs = {}
for k in NCS_K:
    with som2.timer(f"SOM2.0 k={k}"):
        pca2, _, Z2 = som2.ncs_pcs_on_zone(ds, mask, ilines, xlines, twt, k)
        W2, snaps2 = som2.train_som(Z2, SOM_SHAPE, seed=SEED)
        som2_runs[k] = dict(pca=pca2, W=W2, b=som2.bmu(Z2, W2), rgb=sp.rgb_from_scores(Z2))
    print(f"k={k}: {pca2['var_ratio'][:k].sum():.0%} of embedding variance")
K2 = max(NCS_K)
""")
md("### QC: what the embedding sees")
code(r"""
fig, ax = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
ch = E[..., 0].T
sp.seismic_line(ax[0], cube, path, twt, t0, t1)
sp.overlay(ax[0], ch, path, twt, t0, t1, "viridis", *np.percentile(ch, [2, 98]), alpha=0.75)
ax[0].set_title("One embedding channel (channel 0): each of the 768 channels is a learned pattern detector", loc="left", fontsize=9)
sim = (En @ ref).T
sp.seismic_line(ax[1], cube, path, twt, t0, t1)
im = sp.overlay(ax[1], sim, path, twt, t0, t1, "RdYlBu_r", -0.2, 1.0, alpha=0.75)
ax[1].plot(well_pos[ref_w], twt[ref_t + t0], "w*", ms=16, mec="k")
ax[1].set_title(f"Cosine similarity to the embedding at the star ({ref_w}, oil sand): red = 'looks the same' to the model", loc="left", fontsize=9)
ax[2].imshow(sp.rgb_line(pc1_rgb, zone_index, path, t0, t1), aspect="auto", extent=[0, len(path), twt[t1 - 1], twt[t0]])
ax[2].set_title("SOM1.0 features: first 3 attribute PCs as red/green/blue", loc="left", fontsize=9)
ax[3].imshow(sp.rgb_line(som2_runs[K2]["rgb"], zone_index, path, t0, t1), aspect="auto", extent=[0, len(path), twt[t1 - 1], twt[t0]])
ax[3].set_title("SOM2.0 features: first 3 embedding PCs as red/green/blue", loc="left", fontsize=9)
for a in ax:
    sp.decorate_line(a, trange, well_pos, top, base, path, label=(a is ax[0]))
ax[-1].set_xlabel("trace along line"); plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:*
- **Top:** one of the 768 embedding channels, a pattern the model learned during pre-training.
- **Second:** pick one point (the star, in oil sand) and colour every location by how similar its embedding is.
  The model groups the reservoir interval laterally without being told anything about it.
- **Bottom two:** the first three components of each workflow's features, shown as colours. Colour changes are
  what the SOM can separate. The attribute blend (SOM1.0) changes at the scale of individual wavelet loops. The
  embedding blend (SOM2.0) changes at the scale of geological units.
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
for a, (title, v, k) in zip(ax, [("SOM1.0 attributes", pca1["var_ratio"], len(sel_idx)), ("SOM2.0 embedding", som2_runs[K2]["pca"]["var_ratio"], K2)]):
    n = min(40, len(v)); a.bar(range(1, n + 1), v[:n], color=["gold" if i < k else "steelblue" for i in range(n)])
    a2 = a.twinx(); a2.plot(range(1, n + 1), np.cumsum(v[:n]), "k-"); a2.set_ylim(0, 1)
    a.set(title=f"{title}: variance per component", xlabel="component")
plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:* the attribute PCA is steep. A few components carry most of the variance, because many
attributes measure the same thing. The embedding spectrum is much flatter, so more components are needed to
represent it. That's why SOM2.0 is tested with both 12 and 32 components (gold bars = 32 used).
""")

# =====================================================================================================
md(r"""
## 6 · The self-organising maps
Each SOM is an 8 × 8 grid of prototype vectors, or nodes. Every zone sample is assigned to its best-matching
node. Neighbouring nodes hold similar prototypes, so the node colours below are chosen to vary smoothly across the
grid.
""")
code(r"""
nodes_rgb = som2.node_colours(SOM_SHAPE)
fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
for r, (name, W, b) in enumerate([("SOM1.0", W1, b1), (f"SOM2.0 (k={K2})", som2_runs[K2]["W"], som2_runs[K2]["b"])]):
    hits = np.bincount(b, minlength=W.shape[0])
    sp.node_grid(ax[r, 0], hits / hits.sum() * 100, SOM_SHAPE, "Greens", f"{name}: % of zone samples per node", fmt=lambda v: f"{v:.1f}")
    sp.node_grid(ax[r, 1], sp.u_matrix(W, SOM_SHAPE), SOM_SHAPE, "bone_r", f"{name}: U-matrix (dark = cluster boundary)")
    sp.node_grid(ax[r, 2], None, SOM_SHAPE, colors=nodes_rgb, title="node colour key")
plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:*
- **Left, hit map:** how many samples each node captures. A few very full nodes and many empty ones means the map
  is poorly used.
- **Middle, U-matrix:** the distance between neighbouring prototypes. Dark ridges separate natural clusters in the
  data.
- **Right:** the colour each node is drawn with in the next figure.
""")
code(r"""
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for a, (name, b) in zip(ax, [("SOM1.0", b1), (f"SOM2.0 (k={K2})", som2_runs[K2]["b"])]):
    a.imshow(sp.rgb_line(nodes_rgb[b] / 255, zone_index, path, t0, t1), aspect="auto", extent=[0, len(path), twt[t1 - 1], twt[t0]])
    sp.decorate_line(a, trange, well_pos, top, base, path); a.set_title(f"{name}: unsupervised SOM classes (64 nodes; no wells used yet)", loc="left", fontsize=9)
plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:* this is the raw, unsupervised result, before any well information. Similar colours mean similar
nodes. SOM2.0 classes follow layers and change smoothly along them. SOM1.0 classes flicker from sample to sample,
because its features change with every wavelet loop.
""")

# =====================================================================================================
md(r"""
## 7 · Calibrating the nodes to the wells
Every well sample falls into one node. Counting the facies seen in each node turns the 64 unsupervised classes into
facies.
- **Biggest winner:** each node takes its most frequent facies. Nodes that no calibration well visited stay
  *unknown*.
- **Probabilistic:** P(facies | node) ∝ P(node | facies) · P(facies)^0.5. P(node | facies) comes from the well
  counts, smoothed over neighbouring SOM nodes. This fills the unvisited nodes and gives a confidence.
""")
code(r"""
zi_all, fa_all = som2.calibration_samples(wells, zone_index, CAL_RADIUS)
calib = {}
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
for r, (name, b) in enumerate([("SOM1.0", b1), (f"SOM2.0 (k={K2})", som2_runs[K2]["b"])]):
    cnt, win, post = som2.calibrate(b[zi_all], fa_all, SOM_SHAPE[0] * SOM_SHAPE[1], n_fac, SOM_SHAPE)
    calib[name] = post
    sp.node_facies_bars(ax[r, 0], cnt, SOM_SHAPE, fac_cmap, f"{name}: well samples per node (hatched = none)")
    sp.node_grid(ax[r, 1], None, SOM_SHAPE, colors=np.where((win >= 0)[:, None], np.array([fac_cmap(max(w, 0))[:3] for w in win]) * 255, 255),
                 title=f"{name}: biggest winner (white = unknown)")
    oil = code_to_k.get(4)
    if oil is not None:
        sp.node_grid(ax[r, 2], post[:, oil], SOM_SHAPE, "YlOrBr", f"{name}: P(oil sand | node)", vmin=0, vmax=1, fmt=lambda v: f"{v:.2f}")
sp.facies_legend(fig, fac_names, fac_cmap, y=-0.01); plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.show()
""")
md(r"""
*How to read it:*
- **Left:** the facies mix of the well samples landing in each node, with the number of samples. Hatched nodes saw
  no well data.
- **Middle:** the biggest-winner facies per node. With only a few wells, many nodes stay unknown (white).
- **Right:** the probabilistic calibration's oil-sand probability per node. A good map concentrates it in a compact
  group of neighbouring nodes.
""")

# =====================================================================================================
md(r"""
## 8 · Blind test: leave one well out
For each well in turn, the nodes are calibrated with the *other* wells and the held-out well is predicted. This is
the honest test: the scored well never influenced its own prediction.
""")
code(r"""
runs = {"SOM1.0": b1, **{f"SOM2.0 (k={k})": r["b"] for k, r in som2_runs.items()}}
rows, per_well, lowo_scores = [], {}, {}
for name, b in runs.items():
    scores, pw = som2.leave_one_well_out(b, wells, zone_index, n_fac, SOM_SHAPE, CAL_RADIUS)
    per_well[name], lowo_scores[name] = pw, scores
    for m, s in scores.items():
        rows.append({"workflow": name, "calibration": m, "accuracy": s["accuracy"], "balanced accuracy": s["balanced_accuracy"],
                     "unknown": s["unknown"], **{f"recall {n}": r for n, r in zip(fac_names, s["recall"])},
                     **{f"acc {w}": pw[w][f"acc_{m}"] for w in wells}})
scores_df = pd.DataFrame(rows)
display(scores_df.style.format(precision=2, subset=scores_df.columns[2:]).hide(axis="index"))
""")
code(r"""
best2 = f"SOM2.0 (k={K2})"
prob = scores_df[scores_df.calibration == "prob"].set_index("workflow")
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6), gridspec_kw={"width_ratios": [1.1, 1.3, 1, 1]})
prob[[f"acc {w}" for w in wells] + ["accuracy"]].T.rename(index=lambda s: s.replace("acc ", "")).plot.bar(ax=ax[0], rot=0)
ax[0].axhline(majority, color="k", ls="--", lw=1, label="majority class"); ax[0].set(title="Accuracy per held-out well", ylim=(0, 1)); ax[0].legend(fontsize=8)
prob[[f"recall {n}" for n in fac_names]].T.rename(index=lambda s: s.replace("recall ", "")).plot.bar(ax=ax[1], rot=30, legend=False)
ax[1].set(title="Recall per facies", ylim=(0, 1))
sp.confusion(ax[2], lowo_scores["SOM1.0"]["prob"]["confusion"], fac_names, "SOM1.0 confusion")
sp.confusion(ax[3], lowo_scores[best2]["prob"]["confusion"], fac_names, f"{best2} confusion")
plt.tight_layout(); plt.show()
""")
md(r"""
*How to read it:*
- **Left:** accuracy at each held-out well and pooled. The dashed line is what you'd score by always predicting the
  most common facies.
- **Second:** recall, the share of each facies' samples that were predicted correctly.
- **Right two:** confusion matrices. Columns are the true facies, rows the predicted one, and each column sums to
  100%. A strong diagonal means facies are identified correctly. Off-diagonal cells show which facies get confused.
""")

# =====================================================================================================
md("## 9 · Results at the wells and away from them")
code(r"""
fig, axes = plt.subplots(1, len(wells), figsize=(4.4 * len(wells), 6.2))
for ax, w in zip(np.atleast_1d(axes), wells):
    t = twt[per_well["SOM1.0"][w]["t_index"]]
    cols = [per_well["SOM1.0"][w]["true"], per_well["SOM1.0"][w]["prob"], per_well[best2][w]["prob"]]
    ax.imshow(np.stack(cols, 1), cmap=fac_cmap, vmin=-0.5, vmax=n_fac - 0.5, aspect="auto", interpolation="nearest",
              extent=[0, 3, t.max() + dt_ms / 2, t.min() - dt_ms / 2])
    ax.set_xticks([0.5, 1.5, 2.5], ["well log", "SOM1.0", "SOM2.0"])
    ax.set_title(f"{w}   SOM1.0 {per_well['SOM1.0'][w]['acc_prob']:.0%} | SOM2.0 {per_well[best2][w]['acc_prob']:.0%}", fontsize=10)
    ax.set_ylabel("TWT (ms)")
sp.facies_legend(fig, fac_names, fac_cmap); plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()
""")
md(r"""
*How to read it:* the blind prediction at each well, made without that well's own data, next to its facies log.
""")
code(r"""
fig, ax = plt.subplots(4, 1, figsize=(14, 13.5), sharex=True)
for r, (name, b) in enumerate([("SOM1.0", b1), (best2, som2_runs[K2]["b"])]):
    post = calib[name][b]
    sp.seismic_line(ax[2 * r], cube, path, twt, t0, t1)
    sp.overlay(ax[2 * r], sp.zone_to_line(post.argmax(1).astype(float), zone_index, path, t0, t1), path, twt, t0, t1, fac_cmap, -0.5, n_fac - 0.5, 0.82)
    ax[2 * r].set_title(f"{name}: most likely facies (all wells used for calibration)", loc="left", fontsize=9)
    sp.seismic_line(ax[2 * r + 1], cube, path, twt, t0, t1)
    im = sp.overlay(ax[2 * r + 1], sp.zone_to_line(post.max(1), zone_index, path, t0, t1), path, twt, t0, t1, "cividis", 1 / n_fac, 1.0, 0.85)
    ax[2 * r + 1].set_title(f"{name}: confidence (probability of the most likely facies; dark blue = low, yellow = high)", loc="left", fontsize=9)
for a in ax:
    sp.decorate_line(a, trange, well_pos, top, base, path, label=(a is ax[0]))
sp.facies_legend(fig, fac_names, fac_cmap); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.show()
""")
md(r"""
*How to read it:* facies and confidence along the line, calibrated with all wells. Look for geological plausibility:
continuous layers, sensible stacking, and an oil-sand body above brine sand. Also check where confidence drops,
usually at facies boundaries and away from the wells.
""")
code(r"""
oil = code_to_k.get(4, 0)
t_oil = np.concatenate([twt[w[2][w[3] == oil]] - top[w[0][0], w[1][0]] for w in wells.values() if (w[3] == oil).any()])
offset = float(np.median(t_oil)) if len(t_oil) else 40.0
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
ext = [xlines[0], xlines[-1], ilines[-1], ilines[0]]
for a, (name, b) in zip(ax[:2], [("SOM1.0", b1), (best2, som2_runs[K2]["b"])]):
    m = sp.horizon_slice(calib[name][b].argmax(1).astype(float), zone_index, top, offset, twt)
    a.imshow(m, cmap=fac_cmap, vmin=-0.5, vmax=n_fac - 0.5, extent=ext, aspect="auto", interpolation="nearest")
    a.set_title(f"{name}: facies {offset:.0f} ms below the top horizon")
m = sp.horizon_slice(calib[best2][som2_runs[K2]["b"]][:, oil], zone_index, top, offset, twt)
im = ax[2].imshow(m, cmap="YlOrBr", vmin=0, vmax=1, extent=ext, aspect="auto"); plt.colorbar(im, ax=ax[2], label="P(oil sand)")
ax[2].set_title(f"{best2}: probability of oil sand")
for a in ax:
    a.plot(xlines[path[:, 1]], ilines[path[:, 0]], "w--", lw=0.8)
    for n, w in wells.items():
        a.plot(xlines[w[1][0]], ilines[w[0][0]], "o", color="red", mec="k"); a.text(xlines[w[1][0]] + 6, ilines[w[0][0]], n, fontsize=8, color="k", bbox=dict(fc="w", lw=0, alpha=0.75, pad=1))
    a.set(xlabel="crossline", ylabel="inline")
sp.facies_legend(fig, fac_names, fac_cmap); plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()
""")
md(r"""
*How to read it:* map view on a surface that follows the top horizon at the typical oil-sand level, taken from the
wells. Look for coherent bodies with plausible shapes and edges, rather than salt-and-pepper noise.
""")

# =====================================================================================================
md(r"""
## 10 · Caveats
- **Few wells.** Leave-one-well-out with three wells (about 140 samples) gives noisy per-facies recalls, each
  resting on 10–50 samples. Read the per-well bars as well as the totals.
- **Attribute set.** SOM1.0 uses the published formulas for standard single-trace and banded attributes, with
  PCA-based selection. Window lengths and implementation details will differ between software packages.
- **Thin beds.** Facies thinner than a few milliseconds (tuff, carbonaceous) are below seismic resolution for both
  workflows.
- **NCS model.** Each embedding vector summarises a 16 × 16-sample patch, so detail finer than about 64 ms vertically
  comes only from where the patches are placed. The 2D model processes sections independently and can leave
  tiling seams; the 2.5D model combines four directions and should reduce them.
- **Vertical wells and nearest-trace tie.** Deviated wells need per-sample inline/crossline positions.
""")
code(r"""
print({k: round(v, 1) for k, v in som2.TIMINGS.items()})
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write("\n\n".join(c["source"] for c in cells if c["cell_type"] == "code"))
