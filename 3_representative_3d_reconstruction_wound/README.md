# Representative 3D cell-surface reconstruction (wound-adjacent cells)

End-to-end pipeline that screens candidate wound-adjacent cells, segments and
3D-reconstructs their membrane surface at the first and last timepoint of
each recording, estimates and corrects inter-timepoint drift, and renders the
final paired first-vs-last 3D comparison figure (static PNG/SVG/TIFF, plus a
rotatable, camera-synchronized interactive HTML viewer).

All scripts operate on 30 `crop*.tif` confocal timelapse hyperstacks (axes
TZYX), expected directly alongside this folder (i.e. in its parent directory,
`ROOT`), plus a vendored dependency folder `ROOT/.analysis_dependencies`
(created by `pip install --target`; see Environment below) that some scripts
add to `sys.path` so they can run without a separate virtual environment.

## Pipeline order

Run from inside this folder (each script resolves paths relative to its own
location):

1. **`inspect_crops.py`** — reads every `crop*.tif`'s first and last
   timepoint (z-stacks only, not the full timelapse) and metadata, saves them
   to `endpoint_data/crop{n}.npz` + `.json`, and writes overview screening
   montages to `screening/`.
2. **`explore.py`** (`zsheet(n)`) / **`screen_all.py`** — per-crop z-slice and
   XY/XZ section montages (with each candidate's hand-tuned seed point
   overlaid) used to visually pick a representative candidate and its
   segmentation parameters.
3. **`reconstruct_preview.py`** (`go(n)`, config in `CFG`) — segments the
   footprint (Otsu threshold around the seed + watershed) and builds a
   marching-cubes mesh for the first and last timepoint of a candidate crop;
   writes a QC montage to `screening/crop{n}_watershed.jpg`.
4. **`validate_candidate.py`** — additional multi-angle and orthogonal
   (XZ/XY) QC views of one candidate's reconstructed mesh against the raw
   signal, used to confirm the segmentation is trustworthy before finalizing.
5. **`registration_check.py`** — estimates the 3D rigid translation between
   the first and last timepoint from the membrane signal in the
   *surrounding* (non-cell) region via phase cross-correlation + a
   normalized-cross-correlation refinement, and re-segments/re-meshes the
   registered pair.
6. **`finalize.py`** — builds the final published comparison figure (and
   `.ply` meshes, `.npz` mesh data, interactive HTML, and
   `reconstruction_metadata.json` with full provenance) for the selected,
   registered representative crop.
7. **`render_height_color.py`** — re-renders the already-saved final meshes
   from step 6 with a shared height colormap and XYZ axes added (reads the
   saved mesh files only; does not re-run segmentation/registration).

The scripts are written for a single hard-coded representative crop
(`crop25` in `finalize.py`/`registration_check.py`/`render_height_color.py`)
selected after screening all 30 candidates (see `screen_all.py`'s `CONFIG`
and `reconstruct_preview.py`'s `CFG` for the per-cell seed points that were
tried). To reproduce the selection for a different crop, change the crop
number and per-cell parameters (seed point, baseline z-index) accordingly —
these were hand-picked per cell and visually validated, not auto-tuned.

## Method notes

- Footprint segmentation: Otsu threshold (x 0.65) on a Gaussian-smoothed
  intensity projection around a manually chosen z-range and interior seed
  point, refined by a marker-based watershed.
- Mesh: the binary mask is Gaussian-smoothed in physical units and surfaced
  with `skimage.measure.marching_cubes`; Z is interpolated 3x for display
  only (does not improve the acquired Z resolution, which is coarser than
  XY).
- Registration: a global translation (not deformation/non-rigid) is
  estimated from the membrane signal *outside* the segmented cell, since
  that tissue should ideally not move if there were no drift; this is
  reported as an estimate, not a validated fiducial measurement, since the
  surrounding tissue can itself deform.
- Color (height-coloring renders only): color encodes height above the
  shared reference plane (Z = 0), not curvature.
- These figures are an exploratory, single-cell, membrane-signal-based shape
  comparison. They do not by themselves constitute a quantitative,
  whole-cell-volume-conserving, or statistically validated curvature
  measurement — for that, see `../1_growth_mask_curvature_3d/`.

## Environment

```
pip install numpy tifffile matplotlib scikit-image scipy trimesh plotly pillow
```

(`sys.path.insert(0, ROOT/'.analysis_dependencies')` in several scripts is a
vendored-dependency pattern from the original working environment; a normal
`pip install` of the packages above into your active environment makes that
line a no-op and is the recommended way to run this code.)
