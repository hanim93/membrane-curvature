# Hand-traced 2D outline curvature (Fiji/ImageJ macro)

`Script.ijm` computes 2D membrane-outline curvature from manually traced cell
outlines in Fiji/ImageJ, using a discrete circumcircle (Menger curvature)
estimate over a sliding window along each outline, plus an area-normalized
curvature index.

## Requirements

[Fiji](https://fiji.sc/) (ImageJ with the standard bundled plugins). No
additional plugins are required.

## Input (set up before running)

1. Open the target image in Fiji/ImageJ.
2. In the ROI Manager:
   - **ROI 1**: a straight-line ROI drawn over a scale bar of known physical
     length (edit `scaleBarLength_um` at the top of the macro to match).
   - **ROI 2, ROI 3, ...**: one freehand/polygon outline per traced cell
     boundary.
3. Run the macro (`Plugins > Macros > Run...`, select `Script.ijm`).

## Parameters (top of the script)

- `scaleBarLength_um` — physical length of the scale-bar ROI, in µm.
- `step` — half-width (in interpolated points) of the 3-point window used for
  the local circumcircle curvature fit at each point along the outline.
- `interval` — spacing (px) used when the outline is resampled/smoothed
  (`Interpolate ... smooth`) before curvature is computed.

## Output

A `Results` table (and saved CSV, `curvature_summary_<image title>.csv`) with,
per traced cell:

- `Cell_ID` — ROI name
- `Curvature_1_per_um` — mean local curvature (1/µm) along the traced outline
- `Area_um2` — enclosed area
- `R_eq_um` — equivalent radius, sqrt(Area / π)
- `Normalized_Curvature` — mean curvature x equivalent radius (dimensionless;
  normalizes curvature for cell size so differently-sized cells are
  comparable)

You will be prompted to choose a folder to save the results CSV to at the end
of the run.
