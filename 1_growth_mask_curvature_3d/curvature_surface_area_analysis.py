"""
=============================================================================
 Curvature & surface-area analysis pipeline (standalone)
=============================================================================

What this reproduces
---------------------
For each of 24 wound-adjacent cells ("crop1" ... "crop28", a few tags
skipped), over the FULL timelapse (t=0 -> last frame of the recording, no
window restriction), this computes -- all on the exact same "growth_mask"
region (the part of the cell's segmented footprint present at the last
frame but NOT present at t=0, i.e. the region that newly expanded):

  1. 2D growth area   (um^2)  -- flat pixel count * XY_UM^2
  2. 3D growth surface area (um^2) -- tilt-weighted integral
        dA = sqrt(1 + (dh/dx)^2 + (dh/dy)^2)  over the growth_mask
  3. Curvature over the growth_mask, from the mean-curvature formula
     for a Monge patch (H = (k1+k2)/2), computed on a per-pixel height
     map h(x,y):
        - mean |H|                          (growth_meanAbsH)
        - max |H|, raw                      (growth_maxAbsH_raw)
        - max |H|, denoised (FINAL metric)  (growth_maxAbsH_denoised)

  growth_maxAbsH_denoised is obtained by applying a 3x3 median filter
  (scipy.ndimage.median_filter, size=3) to the FULL |H| curvature map
  BEFORE restricting to growth_mask and taking the max. This is applied
  uniformly to all 24 crops (no manual exclusion of any crop). It was
  adopted after diagnosing that the raw max was dominated by single-pixel
  noise spikes (z-quantization "ring" artifacts from the discretized
  z-argmax height map, and isolated bright spikes right at the
  growth_mask boundary) that are not real curvature signal -- see
  "Provenance" below.

Final published results (n=24, one crop excluded for zero growth pixels):
  2D area   vs 3D area            : r=0.97 (Pearson), rho=0.96 (Spearman)
  3D area   vs max|H| denoised    : r=0.69 (Pearson), rho=0.76 (Spearman)
  2D area   vs max|H| denoised    : r=0.62 (Pearson), rho=0.667 (Spearman)

Provenance / history (for context, not required to run this script)
---------------------------------------------------------------------
- "growth_mask" = footprint(t_last) AND NOT footprint(t=0), where
  footprint = the FULL segmented cell interior (not a narrow band around
  a marked line). Early versions of this analysis (not included here)
  mistakenly used a narrow dilated band around hand-marked points; that
  was corrected once it was confirmed the hand-drawn red line in the
  *_red.tif preview images represents the CELL BOUNDARY itself, and
  "inside the red line" means the actual segmented cell interior.
- Raw max curvature (growth_maxAbsH_raw) was found to disagree between
  Pearson/Spearman (r=0.51 vs rho=0.34), which flagged instability.
  Visual diagnostics on the two most extreme crops (by max curvature)
  showed both were single-pixel noise artifacts, not real dome
  curvature. Manually excluding just those 2 crops "worked" but was
  judged to be a form of cherry-picking (only the most extreme points
  were ever inspected). The uniform 3x3 median-filter denoising below
  is the principled fix that was adopted instead -- it changes nothing
  about which crops are included (still n=24, no exclusions) and only
  suppresses single-pixel spikes.
- This whole analysis covers the FULL observation window per crop
  (literal t=0 to literal last frame of the recording), not the
  manually-defined "expansion window" subset used in some earlier,
  different figures in this project.

Requirements
------------
    pip install numpy scipy scikit-image tifffile pandas

Input data this script expects (NOT included -- see "Data you must
supply" below):
  1. red_mark_pixelcoords.csv (bundled alongside this script), with one
     row per boundary-mark point: tag, point_index, px_x, px_y, image_H,
     image_W, n_red. Only the per-tag mean of px_x/px_y is used here (as
     a crop-window center) -- px_x/px_y are the pixel coordinates of a
     hand-drawn red boundary mark in the *_red.tif preview image for
     that crop.
  2. The raw "crop"-prefixed TIFF hyperstacks themselves (TZYX: T
     timepoints, Z slices, Y, X), one per crop tag, e.g.
     "crop1_expanding_slightly_NC_center_8.tif". These are the
     Lti6B-GFP plasma-membrane-marker confocal z-stack timelapses.
     Filenames are listed in CELLS below. Point UPLOAD_DIR (below) at
     the folder containing them.

Data you must supply
---------------------
Edit UPLOAD_DIR below to point at the folder containing the crop*.tif
files, and MARKS_DIR to the folder containing red_mark_pixelcoords.csv
(both default to the same folder as this script).

Usage
-----
    python curvature_surface_area_analysis.py                # all crops
    python curvature_surface_area_analysis.py crop1 crop5     # subset

Output
------
Writes growth_mask_curvature_surfacearea.csv to OUT_DIR, with columns:
    tag, t0_used, tlast_used, n_growth_px,
    growth_area2d_um2, growth_area3d_um2,
    growth_meanAbsH, growth_maxAbsH_raw, growth_maxAbsH_denoised
=============================================================================
"""
import sys, os
import numpy as np
import tifffile
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths -- EDIT THESE for your environment
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = SCRIPT_DIR   # folder containing the crop*.tif raw hyperstacks
MARKS_DIR = SCRIPT_DIR    # folder containing red_mark_pixelcoords.csv
OUT_DIR = SCRIPT_DIR      # where the output CSV is written

# ---------------------------------------------------------------------------
# Calibration constants (confocal acquisition settings)
# ---------------------------------------------------------------------------
XY_UM = 0.86317   # um / pixel, xy
Z_UM = 3.0        # um / z-step


# ---------------------------------------------------------------------------
# Core geometry / image-processing functions (inlined from pipeline.py)
# ---------------------------------------------------------------------------

def segment_footprint_local(stack, center_xy, dome_z, search_radius=55,
                             ring_offset=(1, 6), sigma=3.0, thr_factor=0.85,
                             min_size=30):
    """Segment one cell's footprint using a LOCAL threshold computed only
    within a window around center_xy (robust to whole-frame photobleaching).
    stack: single-timepoint ZYX array. center_xy: (x, y) interior seed
    point (must be inside the cell, on a dark/low-intensity pixel)."""
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    H, W = stack.shape[1], stack.shape[2]
    y0, y1 = max(0, y - search_radius), min(H, y + search_radius)
    x0, x1 = max(0, x - search_radius), min(W, x + search_radius)

    zlo = max(0, dome_z + ring_offset[0])
    zhi = min(stack.shape[0], dome_z + ring_offset[1])
    if zhi <= zlo:
        return None

    ridge_local = stack[zlo:zhi, y0:y1, x0:x1].max(axis=0)
    ridge_s = filters.gaussian(ridge_local, sigma=sigma)
    try:
        thr = filters.threshold_otsu(ridge_s)
    except ValueError:
        return None
    mask_low = ridge_s < thr * thr_factor
    mask_low = morphology.remove_small_objects(mask_low, min_size=min_size)
    lbl = measure.label(mask_low)
    ly, lx = y - y0, x - x0
    seed_label = lbl[ly, lx]
    if seed_label == 0:
        return None
    fp_local = ndi.binary_fill_holes(lbl == seed_label)
    fp_local_open = morphology.binary_opening(fp_local, morphology.disk(4))
    lbl2 = measure.label(fp_local_open)
    ly, lx = y - y0, x - x0
    seed_label2 = lbl2[ly, lx]
    if seed_label2 == 0:
        if fp_local_open.sum() == 0:
            fp_local_clean = fp_local
        else:
            sizes = np.bincount(lbl2.ravel())
            sizes[0] = 0
            seed_label2 = np.argmax(sizes)
            fp_local_clean = ndi.binary_fill_holes(lbl2 == seed_label2)
    else:
        fp_local_clean = ndi.binary_fill_holes(lbl2 == seed_label2)

    full = np.zeros(stack.shape[1:], dtype=bool)
    full[y0:y1, x0:x1] = fp_local_clean
    return full


def touches_border(mask):
    """True if the footprint mask touches the outer edge of the array --
    such a footprint is a partial/clipped cell, not the real boundary."""
    if mask is None:
        return None
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def height_map_idx(stack, zlo, zhi, sigma_z=1.0, sigma_xy=2.5):
    """Per-pixel z-of-peak-intensity height index map (sub-voxel via
    parabolic interpolation around the discrete argmax)."""
    zhi = min(zhi, stack.shape[0])
    raw = stack[zlo:zhi]
    smooth3d = ndi.gaussian_filter(raw, sigma=(sigma_z, sigma_xy, sigma_xy))
    peak_z = np.argmax(smooth3d, axis=0)
    Z = smooth3d.shape[0]
    iz = np.clip(peak_z, 1, Z - 2)
    yy, xx = np.meshgrid(np.arange(smooth3d.shape[1]), np.arange(smooth3d.shape[2]), indexing='ij')
    f0 = smooth3d[iz - 1, yy, xx]
    f1 = smooth3d[iz, yy, xx]
    f2 = smooth3d[iz + 1, yy, xx]
    denom = (f0 - 2 * f1 + f2)
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = 0.5 * (f0 - f2) / denom
    delta = np.where(np.isfinite(delta), np.clip(delta, -1, 1), 0.0)
    return iz + delta + zlo


def compute_curvature(h_um, dx, dy):
    """Standard Monge-patch mean (H) and Gaussian (K) curvature from a
    height map h_um(x,y), via finite-difference gradients. H = (k1+k2)/2,
    units of 1/length (same length unit as h_um/dx/dy, here um -> um^-1).
    Radius of curvature R = 1/H."""
    hy, hx = np.gradient(h_um, dy, dx)
    hyy, hyx = np.gradient(hy, dy, dx)
    hxy, hxx = np.gradient(hx, dy, dx)
    hxy_sym = 0.5 * (hxy + hyx)
    denom = (1 + hx ** 2 + hy ** 2)
    H = ((1 + hy ** 2) * hxx - 2 * hx * hy * hxy_sym + (1 + hx ** 2) * hyy) / (2 * denom ** 1.5)
    K = (hxx * hyy - hxy_sym ** 2) / (denom ** 2)
    return H, K


def get_footprint(arr_t, seed, dome_z, sr, min_solidity=None):
    """Segment + reject border-touching + optional solidity filter."""
    mask = segment_footprint_local(arr_t, seed, dome_z, search_radius=sr, min_size=30)
    if mask is None:
        return None
    if touches_border(mask):
        return None
    if min_solidity is not None:
        lbl = measure.label(mask)
        props = measure.regionprops(lbl)
        if not props:
            return None
        main = max(props, key=lambda p: p.area)
        if main.solidity < min_solidity:
            return None
    return mask


def footprint_near(arr, n_t, cfg, t_target, max_offset=60):
    """Try t_target first; if segmentation fails there (border-touch, low
    solidity, weak signal, etc.), search outward (t+1,-1,+2,-2,...) for
    the nearest frame where it succeeds."""
    for off in range(0, max_offset + 1):
        for t in ({t_target + off, t_target - off} if off else {t_target}):
            if t < 0 or t >= n_t:
                continue
            frame = np.asarray(arr[t]).astype(np.float32)
            mask = get_footprint(frame, cfg['seed'], cfg['dome_z'], cfg['sr'], cfg.get('min_solidity'))
            if mask is not None:
                return mask, t
    return None, None


# ---------------------------------------------------------------------------
# Per-crop segmentation parameters (hand-tuned, validated seeds/windows)
# ---------------------------------------------------------------------------
# seed = (x, y) interior (dark) seed point, FULL-FRAME pixel coords
# dome_z = z-slice index of the dome apex (approx, refined per-frame)
# sr = local search_radius (px) for segment_footprint_local
# min_solidity = optional convex-hull solidity floor (rejects leaked/
#                multi-lobed segmentations)
CELLS = {
    'crop1':  dict(fname='crop1_expanding_slightly_NC_center_8.tif',   seed=(85, 84),  dome_z=11, sr=55),
    'crop2':  dict(fname='crop2_expanding_slightly_NC_center_9.tif',   seed=(63, 51),  dome_z=7,  sr=55),
    'crop3':  dict(fname='crop3_expanding_NC_cutsite_1.tif',           seed=(75, 87),  dome_z=11, sr=45),
    'crop4':  dict(fname='crop4_expanding_slightly_NC_cutsite_1.tif',  seed=(63, 87),  dome_z=9,  sr=45),
    'crop5':  dict(fname='crop5_expanding_NC_cutsite_2.tif',           seed=(99, 87),  dome_z=18, sr=55),
    'crop6':  dict(fname='crop6_expanding_NC_cutsite_2.tif',           seed=(75, 51),  dome_z=9,  sr=65),
    'crop7':  dict(fname='crop7_expanding_NC_cutsite_3.tif',           seed=(75, 75),  dome_z=7,  sr=40),
    'crop8':  dict(fname='crop8_expanding_NC_cutsite_3.tif',           seed=(75, 123), dome_z=9,  sr=80),
    'crop9':  dict(fname='crop9_expanding_NC_center_12.tif',  seed=(63, 75),  dome_z=11, sr=45),
    'crop10': dict(fname='crop10_expanding_NC_center_12.tif', seed=(63, 99),  dome_z=9,  sr=75),
    'crop11': dict(fname='crop11_expanding_NC_center_12.tif', seed=(80, 65),  dome_z=9,  sr=65),
    'crop12': dict(fname='crop12_expanding_NC_center_12.tif', seed=(75, 99),  dome_z=7,  sr=55, min_solidity=0.9),
    'crop13': dict(fname='crop13_expanding_NC_center_14.tif', seed=(95, 90),  dome_z=6,  sr=65),
    'crop15': dict(fname='crop15_expanding_NC_center_1.tif',  seed=(130, 45), dome_z=9,  sr=95),
    'crop16': dict(fname='crop16_expanding_NC_center_2.tif',  seed=(45, 50),  dome_z=6,  sr=95),
    'crop17': dict(fname='crop17_expanding_NC_center_2.tif',  seed=(90, 80),  dome_z=6,  sr=70),
    'crop19': dict(fname='crop19_expanding_slightly_NC_center_4.tif', seed=(75, 75), dome_z=5,  sr=75),
    'crop20': dict(fname='crop20_expanding_NC_center_4.tif',  seed=(58, 55),  dome_z=4,  sr=55),
    'crop22': dict(fname='crop22_expanding_NC_center_6.tif',  seed=(80, 50),  dome_z=11, sr=95),
    'crop23': dict(fname='crop23_expanding_NC_center_7.tif',  seed=(90, 60),  dome_z=13, sr=100),
    'crop24': dict(fname='crop24_expanding_NC_3.tif',  seed=(90, 90), dome_z=12, sr=60, min_solidity=0.9),
    'crop25': dict(fname='crop25_expanding_NC_4.tif',  seed=(55, 85), dome_z=13, sr=65),
    'crop26': dict(fname='crop26.tif',                 seed=(55, 60), dome_z=9,  sr=65),
    'crop27': dict(fname='crop27_expanding_NC_5.tif',  seed=(45, 75), dome_z=13, sr=60),
    'crop28': dict(fname='crop28_expanding_NC_cutsite_3.tif', seed=(70, 75), dome_z=9, sr=85),
}


def load_marks():
    """Load the hand-marked crop-window-center coordinates from
    red_mark_pixelcoords.csv. Only the mean of px_x/px_y is used (as the
    local-crop-window center); the marks themselves come from a manual
    red-boundary annotation on each crop's *_red.tif preview image and are
    not otherwise used in this quantitative pipeline (growth_mask is
    derived purely from segmentation, not from the marks). Returns the
    same {tag: {'px_x': array, 'px_y': array, ...}} structure the rest of
    this script expects."""
    df = pd.read_csv(os.path.join(MARKS_DIR, 'red_mark_pixelcoords.csv'))
    marks = {}
    for tag, g in df.groupby('tag'):
        g = g.sort_values('point_index')
        marks[tag] = dict(px_x=g['px_x'].to_numpy(), px_y=g['px_y'].to_numpy(),
                          H=int(g['image_H'].iloc[0]), W=int(g['image_W'].iloc[0]),
                          n_red=int(g['n_red'].iloc[0]))
    return marks


# ---------------------------------------------------------------------------
# Main per-crop computation
# ---------------------------------------------------------------------------

def dome_shape(arr, t, y0, y1, x0, x1, cy_l, cx_l, win=8):
    """Height map h(x,y) in um, over the local crop window, at timepoint t."""
    sub_full = np.asarray(arr[t, :, y0:y1, x0:x1]).astype(np.float32)
    yy0, yy1 = max(0, int(cy_l) - win), int(cy_l) + win
    xx0, xx1 = max(0, int(cx_l) - win), int(cx_l) + win
    zprof = sub_full[:, yy0:yy1, xx0:xx1].mean(axis=(1, 2))
    dome_z_l = int(np.argmax(zprof))
    zlo, zhi = max(0, dome_z_l - 6), min(sub_full.shape[0], dome_z_l + 16)
    if zhi <= zlo + 2:
        return None
    h_idx = height_map_idx(sub_full, zlo, zhi)
    h_idx_s = ndi.gaussian_filter(h_idx, sigma=1.5)
    return h_idx_s * Z_UM


def process_one(tag, rm, half=45):
    """Compute 2D growth area, 3D growth surface area, and growth-region
    curvature (mean|H|, max|H| raw, max|H| denoised) for one crop, over
    the growth_mask = footprint(t_last) AND NOT footprint(t=0), using the
    FULL timelapse (literal t=0 to literal last frame)."""
    cfg = CELLS[tag]
    path = os.path.join(UPLOAD_DIR, cfg['fname'])
    arr = tifffile.memmap(path)
    n_t, n_z, H_full, W_full = arr.shape
    cy0, cx0 = float(rm['px_y'].mean()), float(rm['px_x'].mean())
    y0 = max(0, int(cy0) - half); y1 = min(H_full, int(cy0) + half)
    x0 = max(0, int(cx0) - half); x1 = min(W_full, int(cx0) + half)
    cy_l, cx_l = cy0 - y0, cx0 - x0

    fp0, t0_used = footprint_near(arr, n_t, cfg, 0)
    fpLast, tlast_used = footprint_near(arr, n_t, cfg, n_t - 1)
    if fp0 is None or fpLast is None:
        print(f"[{tag}] footprint failed"); return None

    fp0_local = fp0[y0:y1, x0:x1]
    fpLast_local = fpLast[y0:y1, x0:x1]
    growth_mask = fpLast_local & (~fp0_local)
    n_growth_px = int(growth_mask.sum())
    if n_growth_px == 0:
        print(f"[{tag}] no growth pixels (excluded)")
        return dict(tag=tag, t0_used=t0_used, tlast_used=tlast_used, n_growth_px=0,
                    growth_area2d_um2=0.0, growth_area3d_um2=0.0,
                    growth_meanAbsH=np.nan, growth_maxAbsH_raw=np.nan, growth_maxAbsH_denoised=np.nan)

    growth_area2d_um2 = n_growth_px * XY_UM ** 2

    hLast = dome_shape(arr, tlast_used, y0, y1, x0, x1, cy_l, cx_l)
    if hLast is None:
        print(f"[{tag}] height map failed at t={tlast_used}"); return None

    hy, hx_ = np.gradient(hLast, XY_UM, XY_UM)
    Hc, Kc = compute_curvature(hLast, XY_UM, XY_UM)
    absH = np.abs(Hc)

    # 3x3 median filter on the FULL |H| map, applied uniformly to every
    # crop BEFORE restricting to growth_mask -- suppresses single-pixel
    # z-quantization / segmentation-edge noise spikes while preserving
    # genuine multi-pixel curvature peaks. See module docstring.
    absH_denoised = ndi.median_filter(absH, size=3)

    hx_g, hy_g = hx_[growth_mask], hy[growth_mask]
    dA_g = np.sqrt(1 + hx_g ** 2 + hy_g ** 2)
    growth_area3d_um2 = float(np.sum(dA_g) * XY_UM ** 2)

    H_g = absH[growth_mask]
    Hd_g = absH_denoised[growth_mask]
    growth_meanAbsH = float(np.nanmean(H_g))
    growth_maxAbsH_raw = float(np.nanmax(H_g))
    growth_maxAbsH_denoised = float(np.nanmax(Hd_g))

    print(f"[{tag}] t0={t0_used} tlast={tlast_used} n_growth_px={n_growth_px} "
          f"area2d={growth_area2d_um2:.1f} area3d={growth_area3d_um2:.1f} "
          f"meanAbsH={growth_meanAbsH:.4f} maxAbsH_raw={growth_maxAbsH_raw:.4f} "
          f"maxAbsH_denoised={growth_maxAbsH_denoised:.4f}")
    return dict(tag=tag, t0_used=t0_used, tlast_used=tlast_used, n_growth_px=n_growth_px,
                growth_area2d_um2=growth_area2d_um2, growth_area3d_um2=growth_area3d_um2,
                growth_meanAbsH=growth_meanAbsH, growth_maxAbsH_raw=growth_maxAbsH_raw,
                growth_maxAbsH_denoised=growth_maxAbsH_denoised)


def process_one_wholecell(tag, rm, half=45):
    """Whole-cell (not growth-mask-restricted) variant of process_one():
    computes 2D area, 3D surface area, and curvature (mean|H|, max|H| raw,
    max|H| denoised) over each cell's ENTIRE segmented footprint,
    independently at t=0 and at the last frame, using exactly the same
    core segmentation/height-map/curvature functions and the same
    per-cell parameters (CELLS) as process_one() above -- the only
    difference is that the metrics are not restricted to growth_mask.

    This is the straightforward way to reproduce the manuscript's
    whole-cell (t=0 vs. t=last) area/curvature comparison with this
    repository's validated pipeline. Note: the exact driver script used
    to generate the published whole-cell figures was not preserved
    alongside this one, so if you are checking against a specific
    published number, treat this as a faithful implementation of the
    documented method rather than a byte-for-byte reproduction of that
    script -- the segmentation/height-map parameters here are the same
    hand-tuned, validated ones used throughout this repository."""
    cfg = CELLS[tag]
    path = os.path.join(UPLOAD_DIR, cfg['fname'])
    arr = tifffile.memmap(path)
    n_t, n_z, H_full, W_full = arr.shape
    cy0, cx0 = float(rm['px_y'].mean()), float(rm['px_x'].mean())
    y0 = max(0, int(cy0) - half); y1 = min(H_full, int(cy0) + half)
    x0 = max(0, int(cx0) - half); x1 = min(W_full, int(cx0) + half)
    cy_l, cx_l = cy0 - y0, cx0 - x0

    out = {}
    for label, t_target in (('t0', 0), ('tlast', n_t - 1)):
        fp, t_used = footprint_near(arr, n_t, cfg, t_target)
        if fp is None:
            print(f"[{tag}] {label} footprint failed")
            out[f'{label}_used'] = None
            out[f'{label}_n_px'] = out[f'{label}_area2d_um2'] = out[f'{label}_area3d_um2'] = np.nan
            out[f'{label}_meanAbsH'] = out[f'{label}_maxAbsH_raw'] = out[f'{label}_maxAbsH_denoised'] = np.nan
            continue
        fp_local = fp[y0:y1, x0:x1]
        n_px = int(fp_local.sum())
        h = dome_shape(arr, t_used, y0, y1, x0, x1, cy_l, cx_l)
        if h is None or n_px == 0:
            print(f"[{tag}] {label} height map failed"); continue
        hy, hx_ = np.gradient(h, XY_UM, XY_UM)
        Hc, _ = compute_curvature(h, XY_UM, XY_UM)
        absH = np.abs(Hc)
        absH_denoised = ndi.median_filter(absH, size=3)
        hx_f, hy_f = hx_[fp_local], hy[fp_local]
        dA_f = np.sqrt(1 + hx_f ** 2 + hy_f ** 2)
        out[f'{label}_used'] = t_used
        out[f'{label}_n_px'] = n_px
        out[f'{label}_area2d_um2'] = n_px * XY_UM ** 2
        out[f'{label}_area3d_um2'] = float(np.sum(dA_f) * XY_UM ** 2)
        out[f'{label}_meanAbsH'] = float(np.nanmean(absH[fp_local]))
        out[f'{label}_maxAbsH_raw'] = float(np.nanmax(absH[fp_local]))
        out[f'{label}_maxAbsH_denoised'] = float(np.nanmax(absH_denoised[fp_local]))
    out['tag'] = tag
    print(f"[{tag}] whole-cell  t0: n_px={out.get('t0_n_px')} area2d={out.get('t0_area2d_um2')}  "
          f"tlast: n_px={out.get('tlast_n_px')} area2d={out.get('tlast_area2d_um2')}")
    return out


if __name__ == "__main__":
    marks = load_marks()
    tags = sys.argv[1:] if len(sys.argv) > 1 else list(CELLS.keys())
    results = []
    for tag in tags:
        if tag not in marks:
            print(f"[{tag}] SKIP (no mark data)"); continue
        try:
            res = process_one(tag, marks[tag])
            if res:
                results.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{tag}] ERROR: {e}")
    if results:
        out = pd.DataFrame(results)
        out_path = os.path.join(OUT_DIR, "growth_mask_curvature_surfacearea.csv")
        out.to_csv(out_path, index=False)
        print("\nsaved", out_path)

        # quick correlation summary (excludes the zero-growth crop, if any)
        d = out[out['n_growth_px'] > 0]
        if len(d) >= 3:
            from scipy import stats
            for xcol, ycol, label in [
                ('growth_area2d_um2', 'growth_area3d_um2', '2D area vs 3D area'),
                ('growth_area3d_um2', 'growth_maxAbsH_denoised', '3D area vs max|H| denoised'),
                ('growth_area2d_um2', 'growth_maxAbsH_denoised', '2D area vs max|H| denoised'),
            ]:
                r, p = stats.pearsonr(d[xcol], d[ycol])
                rho, ps = stats.spearmanr(d[xcol], d[ycol])
                print(f"{label}: n={len(d)}  Pearson r={r:.3f} p={p:.2g}  Spearman rho={rho:.3f} p={ps:.2g}")
