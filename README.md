# Membrane curvature & 3D cell-surface reconstruction code

Custom analysis and visualization code for quantifying plasma-membrane
curvature and surface area, and for building 3D surface reconstructions, from
confocal timelapse imaging of the plasma-membrane marker Lti6B-GFP in
Arabidopsis cells. Written to accompany the manuscript's Data/Code
Availability statement.

This repository contains **code only**. The raw confocal timelapse data
(TZYX hyperstack TIFFs) are not included here because of file size; see the
manuscript's Data Availability statement for how to obtain them.

## Contents

| Folder | What it does | Figures it supports |
|---|---|---|
| `1_growth_mask_curvature_3d/` | Quantifies 2D/3D surface area and mean-curvature (Monge-patch H) over each cell's newly-expanded ("growth mask") region, across the full wound-response timelapse, for 24 wound-adjacent cells. The same core functions (segmentation, height map, curvature) were also used, applied to each cell's *entire* segmented footprint independently at t = 0 and t = last (i.e. without the growth-mask restriction), to produce the whole-cell area/curvature comparison reported in the manuscript; see that folder's README for how to run that variant. | Growth-region curvature vs. area analysis; whole-cell (t = 0 vs. t = last) area/curvature comparison |
| `2_hand_traced_2d_curvature/` | Fiji/ImageJ macro computing 2D outline curvature (circumcircle / discrete Menger curvature along a sliding window) from manually traced cell outlines, plus an area-normalized curvature index. | 2D hand-traced curvature comparisons |
| `3_representative_3d_reconstruction_wound/` | End-to-end pipeline (screening → segmentation → marching-cubes mesh → registration → rendering) that builds the representative 3D dome-shaped surface reconstructions shown for individual wound-adjacent cells (e.g. first vs. last timepoint). | Representative 3D cell-surface renderings (wound) |
| `4_3d_reconstruction_osmotic_stress/` | Same style of 3D surface reconstruction pipeline, applied to the hyperosmotic- and hypoosmotic-stress control conditions, including drift/registration correction between timepoints. | Representative 3D cell-surface renderings (hyperosmotic/hypoosmotic controls) |

## Method summary

Each cell's plasma-membrane footprint is segmented per timepoint from a
local, Otsu-thresholded region around a manually-identified interior seed
point (robust to whole-frame photobleaching/intensity drift). A per-pixel
height map is built from the z-index of peak membrane intensity (sub-voxel,
via parabolic interpolation), Gaussian-smoothed, and converted to physical
units (µm) using the acquisition's XY/Z voxel spacing. From this height map
h(x,y):

- **2D area** = segmented pixel count x (µm/pixel)²
- **3D surface area** = tilt-weighted integral, dA = sqrt(1 + h_x² + h_y²), over the segmented region
- **Mean curvature H** = the standard Monge-patch formula, H = (k1+k2)/2, computed from finite-difference gradients of h(x,y); reported as mean|H| and max|H| (raw and 3x3-median-filter denoised, to suppress single-pixel z-quantization/segmentation-edge noise spikes)

For 3D rendering, the segmented footprint is smoothed (physical-unit Gaussian
filter) and surfaced with marching cubes (scikit-image), then rendered with
matplotlib/trimesh (static figures) and plotly (interactive HTML viewers with
synchronized camera controls between paired timepoints).

## Environment

```
pip install numpy scipy scikit-image tifffile pandas matplotlib trimesh plotly pillow
```

Python 3.10+ recommended. Tested with the package versions pinned in
`requirements.txt`.

## Data you need to supply

None of the folders include the raw confocal TIFFs. Each script expects
`crop*.tif` hyperstacks (axes TZYX: timepoints x z-slices x Y x X) with
ImageJ-style metadata (`slices`, `frames`, `finterval`, `spacing`,
`XResolution`/`YResolution`), named and organized as described in each
subfolder's own README. Per-cell segmentation parameters (interior seed
point, approximate dome z-index, local search radius) were hand-tuned and
visually validated against the raw fluorescence signal for every cell/frame
used in the manuscript (see each subfolder's README for how that validation
was done); they are hard-coded in the scripts alongside the cell they apply
to.

## Provenance note

This code was developed interactively (with AI pair-programming assistance)
over the course of the project, iterating on segmentation parameters and
denoising choices while continuously validating reconstructed outlines
against the raw fluorescence signal (not just automated "success" flags).
Where a parameter or methodological choice was non-obvious, the reasoning is
kept in comments/docstrings in the relevant script rather than removed, so
that a reader (or the authors, later) can audit why a given choice was made.

## License

See `LICENSE`.

## Citation

See `CITATION.cff`.
