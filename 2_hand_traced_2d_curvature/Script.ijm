// ===================================================================
// Fully automated: scale calibration -> spline -> curvature -> area
//                  -> R_eq -> normalized curvature -> CSV export
// Prereq: ROI1 = scale bar (straight line), ROI2.. = cell outlines
// ===================================================================

scaleBarLength_um = 100; // 스케일바 실제 길이
step = 5;                // curvature circle-fit window
interval = 2;            // interpolate(smoothing) 간격

// ---- Step 1: ROI1으로 pixel calibration ----
roiManager("select", 0);
getLine(x1, y1, x2, y2, lw);
lengthPx = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
umPerPixel = scaleBarLength_um / lengthPx;
setVoxelSize(umPerPixel, umPerPixel, 1, "um");
print("Calibration: " + umPerPixel + " um/pixel");

nROIs = roiManager("count");
run("Clear Results");
row = 0;

// ---- Steps 2~5: ROI2부터 순회 ----
for (r = 1; r < nROIs; r++) {

    // Step 2: smoothing (freehand area엔 Fit Spline 대신 Interpolate 사용)
    roiManager("select", r);
    run("Interpolate", "interval=" + interval + " smooth adjust");
    roiManager("update");

    // Step 3: local curvature 평균 (3점 circumcircle 기반)
    roiManager("select", r);
    roiName = Roi.getName();
    getSelectionCoordinates(xpoints, ypoints);
    n = xpoints.length;

    sumKappa = 0;
    validCount = 0;
    for (i = 0; i < n; i++) {
        i0 = (i - step + n) % n;
        i1 = i;
        i2 = (i + step) % n;

        px1 = xpoints[i0]; py1 = ypoints[i0];
        px2 = xpoints[i1]; py2 = ypoints[i1];
        px3 = xpoints[i2]; py3 = ypoints[i2];

        D = 2 * (px1*(py2-py3) + px2*(py3-py1) + px3*(py1-py2));
        if (D != 0) {
            ux = ((px1*px1+py1*py1)*(py2-py3) + (px2*px2+py2*py2)*(py3-py1) + (px3*px3+py3*py3)*(py1-py2)) / D;
            uy = ((px1*px1+py1*py1)*(px3-px2) + (px2*px2+py2*py2)*(px1-px3) + (px3*px3+py3*py3)*(px2-px1)) / D;
            r_px = sqrt((px2-ux)*(px2-ux) + (py2-uy)*(py2-uy));
            kappa = (1 / r_px) / umPerPixel; // -> 1/um
            sumKappa += kappa;
            validCount++;
        }
    }
    meanKappa = sumKappa / validCount;

    // Step 4: area 측정
    roiManager("select", r);
    List.setMeasurements();
    area_um2 = List.getValue("Area");

    // Step 5: R_eq, normalized curvature
    Req = sqrt(area_um2 / PI);
    normKappa = meanKappa * Req;

    setResult("Cell_ID", row, roiName);
    setResult("Curvature_1_per_um", row, meanKappa);
    setResult("Area_um2", row, area_um2);
    setResult("R_eq_um", row, Req);
    setResult("Normalized_Curvature", row, normKappa);
    row++;
}
updateResults();

// ---- Step 6: CSV로 저장 (Excel에서 바로 열림) ----
dir = getDirectory("Choose folder to save results");
saveAs("Results", dir + "curvature_summary_" + getTitle() + ".csv");

print("Done. " + row + " cells processed.");