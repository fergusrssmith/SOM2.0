# SOM2.0: seismic facies from foundation-model embeddings

SOM2.0 classifies seismic facies by running a self-organising map (SOM) on **embeddings from a seismic foundation
model** instead of on hand-picked seismic attributes. It then calibrates the map to well facies logs.

This repo compares it with the standard multi-attribute workflow (**SOM1.0**). Both use the same seismic volume,
analysis zone, SOM and well calibration, and both are scored leave-one-well-out.

| | SOM1.0 (multi-attribute) | SOM2.0 (embedding) |
|---|---|---|
| Features | ~36 single-trace instantaneous + banded attributes | 768-channel embedding from the [NCS seismic foundation model](https://github.com/NorskRegnesentral/NCS_models) |
| Reduction | PCA used to select the dominant 8–12 attributes | PCA to *k* components |
| Classification | 8 × 8 SOM | identical |
| Calibration | wells: biggest winner, or probabilistic | identical |
| Scoring | leave-one-well-out | identical |

## Files

- [`som2_workflow.ipynb`](som2_workflow.ipynb): the workflow, step by step. Each step has QC plots and a short
  *How to read it* note.
- [`som2.py`](som2.py): the functions the notebook uses:
  - SEG-Y, horizon and LAS input, and the well tie
  - SOM1.0 attributes and PCA selection
  - NCS feature loading and registration
  - PCA, batch SOM, calibration and leave-one-well-out scoring
- [`som2_plots.py`](som2_plots.py): the QC and explanation plots: base map, sections along a line through the
  wells, horizon slices, SOM hit maps and U-matrices, node calibration views, and confusion matrices.
- [`tools/make_workflow_notebook.py`](tools/make_workflow_notebook.py): regenerates the notebook.
- [`requirements.txt`](requirements.txt): Python packages.

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
