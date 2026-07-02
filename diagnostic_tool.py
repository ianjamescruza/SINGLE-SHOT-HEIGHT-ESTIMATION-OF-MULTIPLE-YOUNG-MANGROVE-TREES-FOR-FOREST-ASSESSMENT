import cv2
import numpy as np
import glob
import os
import logging

BASE_DIR         = r"C:\Thesis\dataset"
CHECKERBOARD_DIR = os.path.join(BASE_DIR, "checkerboard_images")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration_matrix.npz")
REPORT_FILE      = os.path.join(BASE_DIR, "diagnostic_report.txt")

CHESSBOARD_COLS = 8
CHESSBOARD_ROWS = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(REPORT_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


def analyze_calibration():
    log.info("")
    log.info("=" * 65)
    log.info("CALIBRATION COMPARISON")
    log.info("=" * 65)

    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    cb_images  = []
    for ext in extensions:
        cb_images.extend(glob.glob(os.path.join(CHECKERBOARD_DIR, ext)))
    cb_images = sorted(set(os.path.normcase(os.path.abspath(p)) for p in cb_images))

    if len(cb_images) == 0:
        log.error(f"  No checkerboard images found in: {CHECKERBOARD_DIR}")
        return None

    log.info(f"  Checkerboard images found : {len(cb_images)}")
    log.info(f"  Pattern                   : {CHESSBOARD_COLS}x{CHESSBOARD_ROWS} inner corners")

    objp        = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                       cv2.CALIB_CB_FAST_CHECK       +
                       cv2.CALIB_CB_NORMALIZE_IMAGE)

    obj_std  = [];  img_std  = []
    obj_flag = [];  img_flag = []
    image_size = None
    count_std  = 0
    count_flag = 0

    for path in cb_images:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
            log.info(f"  Image size : {image_size[0]} x {image_size[1]} px")

        ret_s, corners_s = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None
        )
        if ret_s:
            c_s = cv2.cornerSubPix(gray, corners_s, (11, 11), (-1, -1), criteria)
            obj_std.append(objp);  img_std.append(c_s)
            count_std += 1

        ret_f, corners_f = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), detection_flags
        )
        if ret_f:
            c_f = cv2.cornerSubPix(gray, corners_f, (11, 11), (-1, -1), criteria)
            obj_flag.append(objp);  img_flag.append(c_f)
            count_flag += 1

    log.info(f"  Standard detection   : {count_std}/{len(cb_images)} images")
    log.info(f"  With flags detection : {count_flag}/{len(cb_images)} images")

    results = {}

    if count_std >= 15:
        ret_a, mtx_a, dist_a, _, _ = cv2.calibrateCamera(
            obj_std, img_std, image_size, None, None
        )
        results["A_standard_no_flags"] = (ret_a, mtx_a, dist_a)
        log.info(f"  Method A RMS : {ret_a:.4f} px")

        ret_b, mtx_b, dist_b, _, _ = cv2.calibrateCamera(
            obj_std, img_std, image_size, None, None,
            flags=cv2.CALIB_FIX_K3
        )
        results["B_standard_fixK3"] = (ret_b, mtx_b, dist_b)
        log.info(f"  Method B RMS : {ret_b:.4f} px  (CALIB_FIX_K3)")
    else:
        log.warning(f"  Methods A and B skipped — {count_std} images detected")

    if count_flag >= 15:
        ret_c, mtx_c, dist_c, _, _ = cv2.calibrateCamera(
            obj_flag, img_flag, image_size, None, None,
            flags=cv2.CALIB_FIX_K3
        )
        results["C_flagdetect_fixK3"] = (ret_c, mtx_c, dist_c)
        log.info(f"  Method C RMS : {ret_c:.4f} px  (detection flags with CALIB_FIX_K3)")
    else:
        log.warning(f"  Method C skipped — {count_flag} images detected")

    if not results:
        log.error("  No calibration methods produced results.")
        return None

    best_method = min(results, key=lambda k: results[k][0])
    best_rms, best_mtx, best_dist = results[best_method]

    log.info("")
    log.info("  CALIBRATION COMPARISON SUMMARY:")
    log.info(f"  {'Method':<30} {'RMS (px)':>10}  {'Grade'}")
    log.info(f"  {'-' * 55}")

    for method, (rms, _, _) in sorted(results.items(), key=lambda x: x[1][0]):
        grade  = ("EXCELLENT"  if rms < 0.5 else
                  "VERY GOOD"  if rms < 0.7 else
                  "ACCEPTABLE" if rms < 1.0 else "POOR")
        marker = "  BEST" if method == best_method else ""
        log.info(f"  {method:<30} {rms:>10.4f}  {grade}{marker}")

    log.info("")
    log.info(f"  BEST METHOD : {best_method}")
    log.info(f"  BEST RMS    : {best_rms:.4f} px")
    log.info("")
    log.info("  Camera matrix:")
    log.info(f"    fx={best_mtx[0,0]:.4f}  fy={best_mtx[1,1]:.4f}  "
             f"cx={best_mtx[0,2]:.4f}  cy={best_mtx[1,2]:.4f}")
    log.info("")
    log.info("  Distortion coefficients — k1  k2  p1  p2  k3:")
    log.info(f"    {best_dist.ravel()}")

    if os.path.exists(CALIBRATION_FILE):
        log.info("")
        log.info("  calibration_matrix.npz detected.")

    return best_method, best_rms


def final_recommendation(best_method, best_rms):
    log.info("")
    log.info("=" * 65)
    log.info("FINAL RECOMMENDATION")
    log.info("=" * 65)
    log.info("")

    if best_rms < 0.5:
        log.info(f"  RMS = {best_rms:.4f} px  [EXCELLENT]")
        log.info("  Calibration quality is excellent.")
    elif best_rms < 0.7:
        log.info(f"  RMS = {best_rms:.4f} px  [VERY GOOD]")
        log.info("  Calibration quality is very good.")
    elif best_rms < 1.0:
        log.info(f"  RMS = {best_rms:.4f} px  [ACCEPTABLE]")
    else:
        log.info(f"  RMS = {best_rms:.4f} px  [POOR]")
        log.info("  Calibration quality is insufficient.")

    log.info("")
    log.info("=" * 65)
    log.info("DIAGNOSTIC COMPLETE")
    log.info(f"  Report : {REPORT_FILE}")
    log.info("=" * 65)


def main():
    log.info("")
    log.info("=" * 65)
    log.info("  DIAGNOSTIC TOOL")
    log.info(f"  Base directory   : {BASE_DIR}")
    log.info(f"  Checkerboard dir : {CHECKERBOARD_DIR}")
    log.info("=" * 65)

    calib_result = analyze_calibration()
    if calib_result is None:
        log.error("Calibration analysis failed.")
        return
    best_method, best_rms = calib_result

    final_recommendation(best_method, best_rms)


if __name__ == "__main__":
    main()