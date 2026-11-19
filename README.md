# SOM2.0: seismic facies from foundation-model embeddings

SOM2.0 classifies seismic facies by running a self-organising map (SOM) on **embeddings from a seismic foundation
model** instead of on hand-picked seismic attributes. It then calibrates the map to well facies logs.

This repo compares it with the standard multi-attribute workflow (**SOM1.0**). Both use the same seismic volume,
analysis zone, SOM and well calibration, and both are scored leave-one-well-out.

| | SOM1.0 (multi-attribute) | SOM2.0 (embedding) |
|---|---|---|
| Features | ~33 attributes in 7 families (instantaneous, amplitude, spectral, similarity, dip, curvature, texture) | 768-channel embedding from the [NCS seismic foundation model](https://github.com/NorskRegnesentral/NCS_models) |
| Input selection | continuity screen (neighbour-trace r ≥ 0.8), then distinct attributes (no pair with \|r\| ≥ 0.8) added by PCA influence until they reconstruct 90 % of all-attribute variance; z-scored | PCs of the 2.5D embedding (crop overlap 12) capturing ~70 % of its variance, whitened (unit variance per PC) |
| Classification | 8 × 8 SOM (Kohonen map, with component planes) | identical |
| Calibration | wells: biggest winner, or probabilistic | identical |
| Scoring | leave-one-well-out | identical |
| Output | facies, confidence and P(oil sand) SEG-Y volumes | identical |

## Video walkthrough

An 8-minute animated walkthrough of both workflows on the example data: attributes and their selection, the NCS
embedding, the two SOMs, well calibration and the blind test.

https://github.com/user-attachments/assets/a53bd788-4a03-4b39-8bfa-a1b82f512988

## Files

- [`som2_workflow.ipynb`](som2_workflow.ipynb): the workflow, step by step. Each step has QC plots and a short
  *How to read it* note. The committed copy is executed, so the figures from the public example dataset are
  visible on GitHub. `tools/make_workflow_notebook.py` regenerates a clean, unexecuted copy.
- [`som2.py`](som2.py): the functions the notebook uses:
  - SEG-Y, horizon and LAS input, and the well tie
  - SOM1.0 attribute library, continuity screen and distinct, variance-targeted input selection
  - NCS feature loading and registration
  - PCA, batch SOM, calibration and leave-one-well-out scoring
  - SEG-Y volume export
- [`som2_plots.py`](som2_plots.py): the QC and explanation plots: base map, sections along a line through the
  wells, horizon slices, SOM hit maps, U-matrices and component planes, node calibration views, and confusion matrices.
- [`tools/make_workflow_notebook.py`](tools/make_workflow_notebook.py): regenerates the notebook.
- [`requirements.txt`](requirements.txt): Python packages.
- [`media/`](media): the walkthrough video and its thumbnail.

## Inputs (kept locally; no data in this repo)

- **Seismic:** a SEG-Y volume (time-migrated stack or partial stack).
- **NCS embedding** of that volume, as a Zarr store. Create it with the NCS inference pipeline
  (`som2.run_ncs(...)`), ideally on a GPU.
- **Analysis zone:** top and base horizons as ASCII. RokDoc 5-column exports (X Y Inline Crossline TWT) and
  3-column files both work.
- **Wells:** LAS files with a facies curve in TWT and `XWELL`/`YWELL` in the header.

Set the paths in section 1 of the notebook, then run it top to bottom. It needs about 16 GB of RAM for a
400 × 400 × 250 volume and runs in minutes on a CPU, given an existing embedding.

## Notes

- **NCS feature coordinates** are 0-based sample indices of the input volume, at patch start + `patch_size // 2`.
  `som2.py` corrects the half-sample offset when it interpolates onto the seismic grid.
- **Code licence:** the code in this repo is yours to licence as you choose. The NCS model code and weights are
  downloaded from their own sources and are under their own licences.
