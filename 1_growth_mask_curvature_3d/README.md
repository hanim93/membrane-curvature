# Growth-region 3D curvature & surface-area analysis (wound)

`curvature_surface_area_analysis.py` quantifies, for each of 24 wound-adjacent
cells, the 2D area, 3D surface area, and mean-curvature statistics of the
region of the cell footprint that is newly present at the last timepoint but
was not present at t = 0 (the "growth mask" = footprint(t_last) AND NOT
footprint(t=0)), over the full observation window of each recording (literal
t = 0 to the literal last frame).

Full method, parameters, data requirements, and usage are documented in the
script's own module docstring — open the file and read the top comment block
before running it. In short:

```
pip install numpy scipy scikit-image tifffile pandas
python curvature_surface_area_analysis.py                # all crops
python curvature_surface_area_analysis.py crop1 crop5     # a subset
```

## Files

- `curvature_surface_area_analysis.py` — the analysis script (self-contained; the
  core segmentation/height-map/curvature functions are inlined so this file
  runs standalone).
- `red_mark_pixelcoords.csv` — pixel coordinates of each cell's hand-drawn
  boundary annotation, one row per point (`tag, point_index, px_x, px_y,
  image_H, image_W, n_red`); used only to derive a local crop-window
  center per cell (its per-tag mean), not as the analysis mask itself.
  (Originally two pickle files; converted to plain CSV for this release so
  nothing in this repository requires unpickling.)

## Data you must supply

Point `UPLOAD_DIR` (top of the script) at the folder containing the
`crop*.tif` raw confocal hyperstacks named in the `CELLS` dict inside the
script (e.g. `crop1_expanding_slightly_NC_center_8.tif`). These are not
included in this repository — see the top-level README / manuscript's Data
Availability statement.

## Output

`growth_mask_curvature_surfacearea.csv`, with one row per cell:
`tag, t0_used, tlast_used, n_growth_px, growth_area2d_um2,
growth_area3d_um2, growth_meanAbsH, growth_maxAbsH_raw,
growth_maxAbsH_denoised`.

Running the script also prints the Pearson/Spearman correlations between
growth area and growth curvature, reproducing the correlation statistics
reported for this analysis in the manuscript.

## Whole-cell (t = 0 vs. t = last) variant

The manuscript also reports a whole-cell comparison: 2D/3D area and
mean/max |H| curvature over each cell's *entire* segmented footprint
(not just the newly-expanded growth-mask region), computed independently
at t = 0 and at the last frame. `process_one_wholecell()` in
`curvature_surface_area_analysis.py` implements this using exactly the
same validated segmentation, height-map, and curvature functions (and
the same per-cell `CELLS` parameters) as the growth-mask analysis above
— the only difference is that the metrics are taken over the whole
footprint at each timepoint instead of the growth-mask difference
between them. Example usage:

```python
from curvature_surface_area_analysis import load_marks, process_one_wholecell
marks = load_marks()
result = process_one_wholecell('crop1', marks['crop1'])
```

Note: the original driver script used to generate the manuscript's
whole-cell figures was not preserved alongside this repository, so
`process_one_wholecell()` should be read as a faithful implementation of
the documented method (same segmentation/height-map pipeline, same
hand-validated per-cell parameters used throughout this repository)
rather than a guaranteed byte-for-byte reproduction of that script's
exact output.
