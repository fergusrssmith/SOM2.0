"""
som2.py: SOM1.0 vs SOM2.0 seismic facies classification, calibrated to wells.

  SOM1.0  single-trace instantaneous + banded attributes from one seismic volume
          -> PCA used to select the dominant 8-12 attributes -> z-score -> self-organising map
  SOM2.0  seismic foundation-model embedding (NCS, github.com/NorskRegnesentral/NCS_models; 768 channels)
          -> PCA (k components) -> the same self-organising map
Both maps are calibrated to well facies logs (biggest winner, or probabilistic with neighbour smoothing)
and scored leave-one-well-out.

numpy/scipy on CPU; torch (GPU if available) is used for SOM training when installed.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    DEVICE = "cpu"

TIMINGS: dict[str, float] = {}


# =====================================================================================
# I/O, horizons, zone
# =====================================================================================
class timer:
    """with timer('name'): ...  -> stores wall-clock seconds in TIMINGS"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.name}] ...", flush=True)

    def __exit__(self, *a):
        TIMINGS[self.name] = TIMINGS.get(self.name, 0) + time.time() - self.t
        print(f"[{self.name}] {TIMINGS[self.name]:.1f}s", flush=True)


def load_segy(path, iline_byte=189, xline_byte=193):
    """Return cube (n_il, n_xl, n_t) float32, inlines, xlines, twt_ms."""
    import segyio
    with segyio.open(str(path), iline=iline_byte, xline=xline_byte) as f:
        cube = segyio.tools.cube(f).astype(np.float32)
        return cube, np.asarray(f.ilines), np.asarray(f.xlines), np.asarray(f.samples, float)


def read_csv(path):
    """Minimal CSV reader -> dict of numpy arrays (string columns kept as str)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {n: data[n] for n in data.dtype.names}


def load_horizon(path, ilines, xlines, to_ilxl=None):
    """ASCII horizon -> (n_il, n_xl) TWT grid. Accepts RokDoc 5-column exports (X Y Inline Crossline TWT) or
    3 columns (inline xline twt, or x y twt when `to_ilxl` is given). Header lines are skipped; gaps are
    filled with the nearest value."""
    rows = []
    for ln in Path(path).read_text(errors="replace").splitlines():
        try:
            v = [float(q) for q in ln.replace(",", " ").split()]
        except ValueError:
            continue
        if len(v) >= 5:
            rows.append(v[2:5])
        elif len(v) >= 3:
            rows.append(v[:3])
    a = np.array(rows)
    if a[:, 0].min() > 1e5 and to_ilxl is not None:
        ilxl = np.array([to_ilxl(x, y) for x, y in a[:, :2]])
        a[:, 0], a[:, 1] = ilxl[:, 0], ilxl[:, 1]
    g = np.full((len(ilines), len(xlines)), np.nan)
    ii = np.rint(a[:, 0] - ilines[0]).astype(int)
    xx = np.rint(a[:, 1] - xlines[0]).astype(int)
    ok = (ii >= 0) & (ii < len(ilines)) & (xx >= 0) & (xx < len(xlines))
    g[ii[ok], xx[ok]] = a[ok, 2]
    if np.isnan(g).any():
        idx = ndimage.distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
        g = g[tuple(idx)]
    return g


def zone_from_horizons(top, base, twt, pad_ms=0.0):
    """Boolean (n_il, n_xl, n_t) mask of samples between two horizon grids."""
    return (twt[None, None] >= (top - pad_ms)[..., None]) & (twt[None, None] <= (base + pad_ms)[..., None])

# =====================================================================================
# Wells: LAS facies logs (TWT) tied to the seismic grid
# =====================================================================================
def read_las(path):
    txt = Path(path).read_text()
    head, data = txt.split("~A", 1)
    hdr = {m.group(1).strip(): m.group(2).strip() for m in re.finditer(r"^(\w+)\s*\.\S*\s+(.*?)\s*:", head, re.M)}
    curves = re.findall(r"^(\S+)\s+\.\S*\s+:\s*\d+", head.split("~CURVE", 1)[1], re.M)
    rows = np.array([ln.split() for ln in data.splitlines()[1:] if ln.strip()], float)
    null = float(hdr.get("NULL", -999.25))
    rows[rows == null] = np.nan
    return hdr, {c: rows[:, i] for i, c in enumerate(curves)}


def survey_affine(segy):
    """Least-squares (inline, xline) <-> (X, Y) from trace headers. Returns forward coeffs and an inverse fn."""
    import segyio
    with segyio.open(str(segy), iline=189, xline=193) as f:
        H = f.attributes
        il, xl = H(189)[:].astype(float), H(193)[:].astype(float)
        cx = H(segyio.TraceField.CDP_X)[:].astype(float)
        cy = H(segyio.TraceField.CDP_Y)[:].astype(float)
        sc = float(f.header[0][segyio.TraceField.SourceGroupScalar])
    s = (1 / abs(sc)) if sc < 0 else (sc if sc > 0 else 1)
    cx, cy = cx * s, cy * s
    A = np.c_[il, xl, np.ones_like(il)]
    cX = np.linalg.lstsq(A, cx, rcond=None)[0]
    cY = np.linalg.lstsq(A, cy, rcond=None)[0]
    resid = float(np.median(np.hypot(A @ cX - cx, A @ cY - cy)))
    M = np.array([[cX[0], cX[1]], [cY[0], cY[1]]])

    def to_ilxl(x, y):
        return np.linalg.solve(M, [x - cX[2], y - cY[2]])

    return to_ilxl, resid


def well_facies_on_grid(las_paths, curve, to_ilxl, ilines, xlines, twt, null_code=0):
    """Vertical wells: LAS facies log in TWT -> one code per seismic sample.

    If the seismic is coarser than the log, each seismic sample takes the majority facies of the log samples within
    +/- half a seismic sample; ties go to the log sample nearest the seismic time (not to the lowest code). If the
    seismic is as fine as or finer than the log, each seismic sample takes the nearest log sample (within half a
    log step). Returns (wells, info): wells = {name: (il_idx, xl_idx, t_idx, codes)} for wells inside the survey."""
    dt_seis = float(np.median(np.diff(twt)))
    wells, info = {}, []
    for p in las_paths:
        hdr, c = read_las(p)
        name = hdr.get("WELL", Path(p).stem)
        il, xl = to_ilxl(float(hdr["XWELL"]), float(hdr["YWELL"]))
        i, x = int(round(il - ilines[0])), int(round(xl - xlines[0]))
        inside = 0 <= i < len(ilines) and 0 <= x < len(xlines)
        info.append(dict(well=name, inline=round(float(il), 1), xline=round(float(xl), 1), inside_survey=inside))
        if not inside:
            continue
        t, f = c["DEPT"], c[curve]
        ok = np.isfinite(f)
        t, f = t[ok], f[ok].astype(int)
        dt_log = float(np.median(np.diff(t)))
        ks, codes = [], []
        for k, ts in enumerate(twt):
            d = np.abs(t - ts)
            if dt_seis > dt_log:                                   # coarser seismic: majority, nearest breaks ties
                w = d <= dt_seis / 2
                if not w.any():
                    continue
                vals, cnt = np.unique(f[w], return_counts=True)
                tied = vals[cnt == cnt.max()]
                code = int(tied[0]) if len(tied) == 1 else int(f[w][np.argmin(np.where(np.isin(f[w], tied), d[w], np.inf))])
            else:                                                  # finer (or equal) seismic: nearest log sample
                j = int(np.argmin(d))
                if d[j] > dt_log / 2 + 1e-6:
                    continue
                code = int(f[j])
            if code != null_code:
                ks.append(k), codes.append(code)
        n = len(ks)
        wells[name] = (np.full(n, i), np.full(n, x), np.array(ks, int), np.array(codes, int))
    return wells, info

# =====================================================================================
# SOM1.0: single-trace instantaneous + banded attributes, PCA-based attribute selection
# =====================================================================================
def _segment_mean(values, breaks):
    """Average `values` between consecutive True entries of `breaks` along the last axis (per trace)."""
    nt = values.shape[-1]
    seg = np.cumsum(breaks, axis=-1)
    trace = np.arange(int(np.prod(values.shape[:-1])))[:, None]
    gid = (trace * (nt + 1) + seg.reshape(-1, nt)).ravel()
    s = np.bincount(gid, weights=values.reshape(-1).astype(np.float64))
    c = np.bincount(gid)
    return (s / np.maximum(c, 1))[gid].reshape(values.shape).astype(np.float32)


def som1_attribute_names():
    inst = ["amplitude", "hilbert", "envelope", "inst_phase", "inst_frequency", "normalized_amplitude",
            "envelope_1st_derivative", "envelope_2nd_derivative", "acceleration_of_phase", "bandwidth",
            "smooth_frequency", "thin_bed", "inst_q", "attenuation", "sweetness", "relative_acoustic_impedance",
            "average_energy", "dominant_frequency", "envelope_modulated_phase"]
    banded = ["phase_breaks", "envelope_breaks", "envelope_bands_on_phase_breaks",
              "envelope_bands_on_envelope_breaks", "frequency_bands_on_phase_breaks"]
    for a in ("sweetness", "rai", "q", "bandwidth", "attenuation", "thin_bed"):
        banded += [f"{a}_bands_on_phase_breaks", f"{a}_bands_on_envelope_breaks"]
    return inst + banded


def som1_attributes(cube, dt_ms, mask, smooth_ms=20.0, energy_ms=44.0, break_ms=20.0, rai_smooth_ms=12.0):
    """Return (names, X): the ~36 SOM1.0 single-trace attributes, sampled on the zone.
    Window lengths are in milliseconds, so results do not depend on the sample rate (at 4 ms they are
    5, 11, 5 and 3 samples)."""
    nwin = lambda ms: max(1, int(round(ms / dt_ms))) | 1                  # noqa: E731  odd sample count
    smooth_n, energy_n, break_n, rai_n = nwin(smooth_ms), nwin(energy_ms), nwin(break_ms), nwin(rai_smooth_ms)
    names = som1_attribute_names()
    X = np.empty((int(mask.sum()), len(names)), np.float32)
    col = iter(range(len(names)))

    def put(a):
        X[:, next(col)] = np.nan_to_num(np.asarray(a, np.float32)[mask], posinf=0, neginf=0)

    a = cube.astype(np.float32) / (np.std(cube[mask]) + 1e-9)
    dt_s = dt_ms / 1000
    an = signal.hilbert(a, axis=-1)
    h = an.imag.astype(np.float32)
    env = np.abs(an).astype(np.float32)
    phw = np.angle(an).astype(np.float32)                                # wrapped, radians
    freq = np.gradient(np.unwrap(phw, axis=-1), dt_s, axis=-1) / (2 * np.pi)
    sfreq = ndimage.uniform_filter1d(freq, smooth_n, axis=-1)
    de = np.gradient(env, dt_s, axis=-1)
    d2e = np.gradient(de, dt_s, axis=-1)
    clipq = lambda q: np.clip(q, *np.percentile(q[mask], [1, 99]))       # noqa: E731  tame 1/x spikes
    bwid = np.sqrt(np.abs(de) / (env + 1e-6)) / (2 * np.pi)
    thin = freq - sfreq
    q = clipq(env / np.where(np.abs(de) < 1e-6, 1e-6, de) * freq)
    att = np.abs(de) / np.maximum(np.abs(sfreq), 1.0)
    sweet = env / np.maximum(np.abs(sfreq), 1.0)
    rai = np.cumsum(a - a.mean(-1, keepdims=True), axis=-1)
    rai = ndimage.uniform_filter1d(rai - ndimage.uniform_filter1d(rai, max(3, int(100 / dt_ms)), axis=-1), rai_n, axis=-1)
    for v in (a, h, env, phw, freq, np.cos(phw), de, d2e, np.gradient(freq, dt_s, axis=-1), bwid, sfreq, thin, q,
              att, sweet, rai):
        put(v)
    put(ndimage.uniform_filter1d(env ** 2, energy_n, axis=-1))                                    # average energy
    put(ndimage.uniform_filter1d(env * freq, energy_n, axis=-1) /
        (ndimage.uniform_filter1d(env, energy_n, axis=-1) + 1e-6))                                 # dominant (envelope-weighted) frequency
    put(env * phw / np.pi)                                                                         # envelope-modulated phase

    # banded: phase breaks = wrap-arounds of the phase; envelope breaks = envelope troughs
    pb = np.zeros(a.shape, bool)
    pb[..., 1:] = np.diff(phw, axis=-1) < -np.pi
    eb = np.zeros(a.shape, bool)
    eb[..., 1:-1] = (env[..., 1:-1] < env[..., :-2]) & (env[..., 1:-1] <= env[..., 2:])
    han = np.hanning(break_n)[None, None] / np.hanning(break_n).max()
    put(ndimage.convolve(pb.astype(np.float32), han, mode="nearest"))
    put(ndimage.convolve(eb.astype(np.float32), han, mode="nearest"))
    put(_segment_mean(env, pb)); put(_segment_mean(env, eb)); put(_segment_mean(freq, pb))
    for v in (sweet, rai, q, bwid, att, thin):
        put(_segment_mean(v, pb)); put(_segment_mean(v, eb))
    assert next(col, None) is None, "attribute count mismatch"
    return names, X


def select_distinct(X, names, r_max=0.8, var_target=0.9, max_n=None, n_sample=200_000, seed=0):
    """Pick influential AND distinct input variables (no double counting), with no fixed count.

    Walk the principal components in order; from each, take its highest-loading variable that is not correlated
    |r| >= r_max with any variable already chosen. Stop when the chosen set reconstructs `var_target` of the
    total (standardised) variance of ALL candidate variables, i.e. tr(C[:, S] C[S, S]^-1 C[S, :]) / p, where C is
    their correlation matrix, or when `max_n` is reached.
    Returns (indices, names, r2_curve) with r2_curve[j] = variance reconstructed by the first j+1 picks."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(n_sample, len(X)), replace=False)].astype(np.float64)
    p = pca_fit(S, n_fit=len(S), seed=seed)
    C = np.corrcoef(((S - S.mean(0)) / (S.std(0) + 1e-9)).T)
    chosen, r2 = [], []
    for comp in p["comps"]:
        for i in np.argsort(-np.abs(comp)):
            if i in chosen or (chosen and np.max(np.abs(C[i, chosen])) >= r_max):
                continue
            chosen.append(int(i))
            Cs = C[:, chosen]
            r2.append(float(np.trace(Cs @ np.linalg.pinv(C[np.ix_(chosen, chosen)]) @ Cs.T) / C.shape[0]))
            break
        if (r2 and r2[-1] >= var_target) or len(chosen) >= (max_n or X.shape[1]):
            break
    return chosen, [names[i] for i in chosen], r2


# =====================================================================================
# SOM2.0: NCS seismic foundation-model embedding
# =====================================================================================
def run_ncs(seismic_path, out_zarr, model="NorskRegnesentralSTI/NCS-v1-2d-base", densify=4, batch_size=32,
            dtype="float16", device=None):
    """Run the NCS foundation model (github.com/NorskRegnesentral/NCS_models) on a SEG-Y -> Zarr feature store."""
    from NCS.inference import run_inference
    return run_inference(model_path=model, input_path=str(seismic_path), output_path=str(out_zarr),
                         densify=densify, batch_size=batch_size, device=device or DEVICE, dtype=dtype)


def open_ncs(zarr_path):
    """Return the xarray Dataset holding `features` (handles the per-direction group)."""
    import xarray as xr
    import zarr
    root = zarr.open(str(zarr_path), mode="r")
    groups = [g for g in getattr(root, "group_keys", lambda: [])()]
    for g in (groups or [None]):
        ds = xr.open_zarr(str(zarr_path), group=g) if g else xr.open_zarr(str(zarr_path))
        if "features" in ds:
            print("NCS group:", g, dict(ds["features"].sizes))
            return ds
    raise KeyError("no 'features' variable found in " + str(zarr_path))


def _coord_to_index(coord, axis_values):
    """NCS coords may be sample indices or real inline/xline/twt values: pick whichever span fits."""
    coord = np.asarray(coord, float)
    axis_values = np.asarray(axis_values, float)
    span_c = coord.max() - coord.min()
    span_v = axis_values[-1] - axis_values[0]
    span_i = len(axis_values) - 1
    inside_values = axis_values.min() - 1e-6 <= coord.min() and coord.max() <= axis_values.max() + 1e-6
    if inside_values and abs(span_c - span_v) <= abs(span_c - span_i):
        return np.interp(coord, axis_values, np.arange(len(axis_values)))
    # NCS writes 0-based sample indices of the unpadded volume: start + patch_size//2 for a patch
    # covering [start, start+patch_size), whose true centre is half a sample earlier.
    return coord - 0.5


def grid_to_zone_operator(ds, mask, ilines, xlines, twt):
    """Trilinear interpolation from the NCS feature grid to every zone sample, built once: for each zone sample the
    8 surrounding grid nodes (flat indices) and their weights. Apply with `interp_rows`."""
    gi = _coord_to_index(ds["inline"].values, np.asarray(ilines, float))
    gx = _coord_to_index(ds["xline"].values, np.asarray(xlines, float))
    gt = _coord_to_index(ds["time_depth"].values, np.asarray(twt, float))
    shp = (len(gi), len(gx), len(gt))
    q = np.argwhere(mask).astype(float)

    def axis(g, v):
        j = np.clip(np.searchsorted(g, v) - 1, 0, len(g) - 2)
        return j, np.clip((v - g[j]) / (g[j + 1] - g[j]), 0, 1)

    (ji, fi), (jx, fx), (jt, ft) = axis(gi, q[:, 0]), axis(gx, q[:, 1]), axis(gt, q[:, 2])
    corners, weights = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                corners.append(((ji + a) * shp[1] + (jx + b)) * shp[2] + (jt + c))
                weights.append((fi if a else 1 - fi) * (fx if b else 1 - fx) * (ft if c else 1 - ft))
    return np.stack(corners, 1).astype(np.int64), np.stack(weights, 1).astype(np.float32)


def interp_rows(grid_values, operator, rows=None, chunk=200_000):
    """grid_values (n_grid_nodes, c) -> values at the zone samples `rows` (all if None), via the operator."""
    corners, weights = operator
    rows = np.arange(len(corners)) if rows is None else np.asarray(rows)
    out = np.empty((len(rows), grid_values.shape[1]), np.float32)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        out[s:s + chunk] = np.einsum("rk,rkc->rc", weights[r], grid_values[corners[r]])
    return out


def ncs_pcs_on_zone(ds, mask, ilines, xlines, twt, k_or_var=0.7, n_fit=120_000, seed=0):
    """PCA of the 768-channel embedding, then the first k PCs on every zone sample.
    `k_or_var`: an int = number of PCs, or a float < 1 = variance to capture (the embedding spreads its variance
    widely, so ~0.7 needs tens to ~100 PCs). Returns (pca, k, Z)."""
    with timer("ncs_load"):
        G = np.asarray(ds["features"].values, np.float32)
        G = G.reshape(-1, G.shape[-1])
    with timer("ncs_pca"):
        rng = np.random.default_rng(seed)
        p = pca_fit(G[rng.choice(len(G), min(n_fit, len(G)), replace=False)], n_fit=n_fit, seed=seed)
        k = choose_k(p["var_ratio"], k_or_var)
        P = (((G - p["mean"]) / p["std"]) @ p["comps"][:k].T).astype(np.float32)
        del G
    with timer("ncs_interp_to_seismic"):
        Z = interp_rows(P, grid_to_zone_operator(ds, mask, ilines, xlines, twt))
    return p, k, Z


# =====================================================================================
# PCA
# =====================================================================================
def pca_fit(X, n_fit=300_000, seed=0):
    """Z-score + PCA via the covariance eigendecomposition. Returns dict(mean, std, comps, var_ratio)."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(n_fit, len(X)), replace=False)].astype(np.float64)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    S = (S - mu) / sd
    w, v = np.linalg.eigh(np.cov(S, rowvar=False))
    order = np.argsort(w)[::-1]
    w, v = np.clip(w[order], 0, None), v[:, order]
    return dict(mean=mu, std=sd, comps=v.T, var_ratio=w / w.sum())


def choose_k(var_ratio, k_or_var):
    """An int is used as is; a float < 1 is the variance to capture."""
    if isinstance(k_or_var, (int, np.integer)):
        return int(k_or_var)
    return int(np.searchsorted(np.cumsum(var_ratio), float(k_or_var)) + 1)


def pca_transform(X, p, k, chunk=500_000):
    out = np.empty((len(X), k), np.float32)
    C = p["comps"][:k].T
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = ((X[i:i + chunk] - p["mean"]) / p["std"]) @ C
    return out

# =====================================================================================
# Self-organising map (batch algorithm; identical settings for SOM1.0 and SOM2.0)
# =====================================================================================
def train_som(Z, shape=(8, 8), epochs=25, n_train=150_000, sigma0=None, sigma1=0.6, seed=0, n_snap=16):
    """Batch SOM with PCA-plane initialisation. Returns (W, snapshots of W[:, :2])."""
    rng = np.random.default_rng(seed)
    S = Z[rng.choice(len(Z), min(n_train, len(Z)), replace=False)].astype(np.float32)
    R, C = shape
    gy, gx = np.mgrid[0:R, 0:C]
    grid = np.stack([gy.ravel(), gx.ravel()], 1).astype(np.float32)
    sd = S.std(0)
    W = np.zeros((R * C, S.shape[1]), np.float32)          # init on the plane of the first two dims
    W[:, 0] = (grid[:, 1] / (C - 1) - 0.5) * 2 * sd[0] * 0.2
    W[:, 1] = (grid[:, 0] / (R - 1) - 0.5) * 2 * sd[1] * 0.2
    sigma0 = sigma0 or max(R, C) / 2
    D2 = ((grid[:, None] - grid[None]) ** 2).sum(-1)
    snaps = [W[:, :2].copy()]
    use_t = torch is not None
    if use_t:
        St = torch.tensor(S, device=DEVICE)
    for e in range(epochs):
        sig = sigma0 * (sigma1 / sigma0) ** (e / max(1, epochs - 1))
        H = np.exp(-D2 / (2 * sig ** 2)).astype(np.float32)
        if use_t:
            Wt = torch.tensor(W, device=DEVICE)
            b = torch.cdist(St, Wt).argmin(1)
            onehot = torch.zeros(len(S), R * C, device=DEVICE).scatter_(1, b[:, None], 1.0)
            Ht = torch.tensor(H, device=DEVICE)
            num = Ht @ (onehot.T @ St)
            den = Ht @ onehot.sum(0)
            W = (num / den[:, None].clamp_min(1e-9)).cpu().numpy()
        else:
            b = bmu(S, W)
            cnt = np.bincount(b, minlength=R * C).astype(np.float32)
            sums = np.zeros_like(W)
            np.add.at(sums, b, S)
            W = (H @ sums) / np.maximum(H @ cnt, 1e-9)[:, None]
        if e in set(np.linspace(0, epochs - 1, n_snap - 1).astype(int)):
            snaps.append(W[:, :2].copy())
    return W.astype(np.float32), np.array(snaps)


def bmu(Z, W, chunk=200_000):
    out = np.empty(len(Z), np.int32)
    w2 = (W ** 2).sum(1)
    for i in range(0, len(Z), chunk):
        z = Z[i:i + chunk]
        out[i:i + chunk] = np.argmin(w2[None] - 2 * z @ W.T, 1)
    return out


def node_colours(shape):
    R, C = shape
    gy, gx = np.mgrid[0:R, 0:C]
    u, v = (gy.ravel() / (R - 1))[:, None], (gx.ravel() / (C - 1))[:, None]
    c = np.array([[66, 135, 245], [245, 191, 66], [235, 64, 110], [60, 200, 160]], float)
    return (1 - u) * (1 - v) * c[0] + (1 - u) * v * c[1] + u * (1 - v) * c[3] + u * v * c[2]

# =====================================================================================
# Well calibration and leave-one-well-out scoring
# =====================================================================================
def calibrate(node_of_sample, fac, n_nodes, n_fac, shape, smooth=0.9, prior_power=0.5):
    """Biggest winner, plus a Bayesian version: P(facies | node) ∝ P(node | facies) · P(facies)^prior_power.

    P(node | facies) comes from well counts smoothed over neighbouring SOM nodes. prior_power=1 is the
    plain posterior (favours common facies); 0 treats facies equally (protects rare ones such as thin
    sands); 0.5 is a compromise.
    """
    counts = np.zeros((n_nodes, n_fac))
    np.add.at(counts, (node_of_sample, fac), 1)
    winner = np.where(counts.sum(1) > 0, counts.argmax(1), -1)
    R, C = shape
    gy, gx = np.mgrid[0:R, 0:C]
    g = np.stack([gy.ravel(), gx.ravel()], 1)
    K = np.exp(-((g[:, None] - g[None]) ** 2).sum(-1) / (2 * smooth ** 2))
    sm = K @ counts + 1e-3
    lik = sm / sm.sum(0, keepdims=True)                     # P(node | facies)
    prior = (np.bincount(fac, minlength=n_fac) + 1) / (len(fac) + n_fac)
    post = lik * prior[None] ** prior_power
    post /= post.sum(1, keepdims=True)
    return counts, winner, post


def score(true, pred, n_fac):
    ok = pred >= 0
    cm = np.zeros((n_fac, n_fac), int)                     # rows = predicted, cols = true (RokDoc convention)
    np.add.at(cm, (pred[ok], true[ok]), 1)
    recall = [float(np.mean(pred[true == f] == f)) if np.any(true == f) else np.nan for f in range(n_fac)]
    return dict(accuracy=float(np.mean(pred == true)), unknown=float(np.mean(~ok)),
                balanced_accuracy=float(np.nanmean(recall)), recall=recall, confusion=cm.tolist())


def calibration_samples(wells, zone_index, radius=1):
    """Zone indices + facies for the calibration footprint ((2r+1)^2 traces) around each well.
    `wells` = {name: (il_idx, xl_idx, t_idx, facies)} with facies already mapped to 0..n_fac-1."""
    n_il, n_xl, _ = zone_index.shape
    zi, fa = [], []
    for il, xl, t, f in wells.values():
        for di in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                zi.append(zone_index[np.clip(il + di, 0, n_il - 1), np.clip(xl + dx, 0, n_xl - 1), t])
                fa.append(f)
    zi, fa = np.concatenate(zi), np.concatenate(fa)
    return zi[zi >= 0], fa[zi >= 0]


def leave_one_well_out(bmu_zone, wells, zone_index, n_fac, shape, radius=1):
    """Calibrate on all wells but one, predict the held-out well trace, rotate. Both calibration methods."""
    n_nodes = shape[0] * shape[1]
    pooled = {"winner": ([], []), "prob": ([], [])}
    per_well = {}
    for held in wells:
        zi, fa = calibration_samples({k: v for k, v in wells.items() if k != held}, zone_index, radius)
        _, win, post = calibrate(bmu_zone[zi], fa, n_nodes, n_fac, shape)
        il, xl, t, f = wells[held]
        tz = zone_index[il, xl, t]
        keep = tz >= 0
        tz, tf = tz[keep], f[keep]
        pw, pp = win[bmu_zone[tz]], post[bmu_zone[tz]].argmax(1)
        for m, p in (("winner", pw), ("prob", pp)):
            pooled[m][0].append(tf), pooled[m][1].append(p)
        per_well[held] = dict(true=tf, winner=pw, prob=pp, t_index=t[keep],
                              acc_winner=float(np.mean(pw == tf)), acc_prob=float(np.mean(pp == tf)))
    scores = {m: score(np.concatenate(a), np.concatenate(b), n_fac) for m, (a, b) in pooled.items()}
    return scores, per_well


# =====================================================================================
# Output volumes
# =====================================================================================
def write_zone_volume(path, template_segy, mask, zone_values, fill=0.0, iline_byte=189, xline_byte=193):
    """Write zone-sample values as a SEG-Y with the template's geometry and trace headers (`fill` outside zone)."""
    import segyio
    with segyio.open(str(template_segy), iline=iline_byte, xline=xline_byte) as src:
        vol = np.full((len(src.ilines), len(src.xlines), len(src.samples)), fill, np.float32)
        vol[mask] = zone_values
        spec = segyio.tools.metadata(src)
        with segyio.create(str(path), spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = src.header
            for i, il in enumerate(src.ilines):
                dst.iline[il] = vol[i]
    return Path(path)
