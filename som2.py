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
    """Vertical wells: LAS facies log in TWT -> one code per seismic sample (majority vote within +/- half a
    sample). Returns (wells, info): wells = {name: (il_idx, xl_idx, t_idx, codes)} for wells inside the survey."""
    half = float(np.median(np.diff(twt))) / 2
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
        ks, codes = [], []
        for k, ts in enumerate(twt):
            w = (np.abs(t - ts) <= half) & np.isfinite(f)
            if w.any():
                vals, cnt = np.unique(f[w].astype(int), return_counts=True)
                code = int(vals[cnt.argmax()])
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


def som1_attributes(cube, dt_ms, mask, smooth_n=5, energy_n=11):
    """Return (names, X): the ~36 SOM1.0 single-trace attributes, sampled on the zone."""
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
    rai = ndimage.uniform_filter1d(rai - ndimage.uniform_filter1d(rai, max(3, int(100 / dt_ms)), axis=-1), 3, axis=-1)
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
    han = np.hanning(5)[None, None] / np.hanning(5).max()
    put(ndimage.convolve(pb.astype(np.float32), han, mode="nearest"))
    put(ndimage.convolve(eb.astype(np.float32), han, mode="nearest"))
    put(_segment_mean(env, pb)); put(_segment_mean(env, eb)); put(_segment_mean(freq, pb))
    for v in (sweet, rai, q, bwid, att, thin):
        put(_segment_mean(v, pb)); put(_segment_mean(v, eb))
    assert next(col, None) is None, "attribute count mismatch"
    return names, X


def som1_select(pca, names, n_min=8, n_max=12, rel=0.6, per_pc=5):
    """SOM1.0 attribute selection by PCA: walk the eigenvectors in order and
    keep each one's dominant attributes (|loading| >= rel x the largest), until n_min..n_max are chosen."""
    chosen = []
    for pc, comp in enumerate(pca["comps"]):
        order = np.argsort(-np.abs(comp))
        top = [i for i in order[:per_pc] if abs(comp[i]) >= rel * abs(comp[order[0]])]
        for i in top:
            if i not in chosen and len(chosen) < n_max:
                chosen.append(int(i))
        if len(chosen) >= n_min:
            break
    return chosen, [names[i] for i in chosen]

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


def ncs_pcs_on_zone(ds, mask, ilines, xlines, twt, k_cfg, n_fit=200_000, seed=0):
    """PCA of the 768-ch NCS features, then interpolate the k PCs onto the seismic zone samples."""
    from scipy.interpolate import RegularGridInterpolator
    F = ds["features"]
    dims = list(F.dims)
    il_c, xl_c, t_c = (np.asarray(ds[d].values, float) for d in dims[:3])
    ni, nx, nt, nf = F.shape
    with timer("ncs_pca_fit"):
        step = max(1, int(np.ceil((ni * nx * nt / n_fit) ** (1 / 3))))   # regular sub-grid ~ n_fit vectors
        S = np.asarray(F[::step, ::step, ::step].values, np.float32).reshape(-1, nf)
        p = pca_fit(S, n_fit=len(S), seed=seed)
        k = choose_k(p["var_ratio"], k_cfg)
    with timer("ncs_pca_transform"):
        PC = np.empty((ni, nx, nt, k), np.float32)
        for i in range(ni):
            PC[i] = pca_transform(F[i].values.reshape(-1, nf), p, k).reshape(nx, nt, k)
    with timer("ncs_interp_to_seismic"):
        gi = _coord_to_index(il_c, ilines.astype(float))
        gx = _coord_to_index(xl_c, xlines.astype(float))
        gt = _coord_to_index(t_c, twt)
        pts = np.argwhere(mask).astype(float)
        out = np.empty((len(pts), k), np.float32)
        for c in range(k):
            f = RegularGridInterpolator((gi, gx, gt), PC[..., c], bounds_error=False, fill_value=None)
            out[:, c] = f(pts)
    return p, k, out

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


def choose_k(var_ratio, k_cfg):
    if isinstance(k_cfg, int):
        return k_cfg
    k = int(np.searchsorted(np.cumsum(var_ratio), 0.90) + 1)
    return int(np.clip(k, 8, 12))


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
