# 3D cell-surface reconstruction (hyperosmotic / hypoosmotic stress controls)

Same style of segmentation -> marching-cubes mesh -> rendering pipeline as
`../3_representative_3d_reconstruction_wound/`, applied to one representative
cell each from the hyperosmotic-stress and hypoosmotic-stress control
conditions, used as non-wounding comparison figures.

## Pipeline order

1. **`hyper_surface_inspect.py`** / **`hypo_surface_inspect.py`** — read the
   first/last timepoint of `crop1.tif`, `crop2.tif` (hyperosmo) or
   `crop16.tif`, `crop17.tif` (hypoosmo, default crop numbers; override via
   command-line args) and cache them to `{condition}_surface_output/`.
2. **`hyper_surface_build.py`** / **`hypo_surface_build.py`** — segment
   (Otsu threshold + marker-based watershed) and mesh (marching cubes) the
   first/last timepoint pair for one or two selected crops per condition;
   writes the paired height-colored comparison figure, raw-data QC montage,
   mesh `.npz` files, and `reconstruction_metadata.json`.
3. **`drift_correct_selected.py`** — for specific selected crops
   (`('hypo', 21)`, `('hyper', 2)` by default), estimates and applies a 3D
   translation between timepoints (phase cross-correlation refined by a
   held-out-block cross-validated optimization, scored on membrane signal
   *outside* the segmented cell), then re-runs the same segmentation/mesh
   build on the registered pair via a patched copy of
   `hypo_surface_build.py`'s `build()` function (condition-specific
   parameter overrides are applied as literal source-code substitutions —
   see the `code.replace(...)` calls — rather than duplicating the whole
   build function). Output goes to `drift_corrected/{condition}_crop{n}/`.
4. **`render_separate_surfaces.py`** — re-renders the (non-drift-corrected)
   saved meshes from step 2 as two separate single-panel figures (one per
   timepoint) with a shared height colormap.
5. **`render_separate_corrected.py`** — same as step 4, but for the
   drift-corrected meshes from step 3.

## Data you must supply

`crop*.tif` raw hyperstacks for each condition, in two sibling folders named
`Crop_hyperosmo/` and `Crop_hypoosmo/`. By default the inspect scripts look
for these as `../Crop_hyperosmo` / `../Crop_hypoosmo` relative to this
folder; set the `RAW_DATA_DIR` environment variable to override (see the
`SRC = ...` line near the top of `hyper_surface_inspect.py` /
`hypo_surface_inspect.py`).

## Environment

```
pip install numpy tifffile matplotlib scikit-image scipy trimesh plotly pillow
```

## Note on `drift_correct_selected.py`

This script dynamically patches and executes a modified copy of
`hypo_surface_build.py`'s source at runtime (string substitution + `exec`),
rather than a separate build function, because it was written to re-run the
*exact* same segmentation logic on registered data with only the minimal,
explicitly-listed parameter differences needed per condition/crop. This is
unusual but transparent: every substitution is a single, readable
`code.replace(old, new)` call, and the resulting `reconstruction.json`
records which condition/crop it was built for. If you adapt this pipeline,
consider refactoring this into an explicit parameterized function instead.
