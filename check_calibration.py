import cv2
import numpy as np
import glob
import os
import logging

# CONFIGURATION
CHECKERBOARD_DIR = r"C:\Thesis\dataset\checkerboard_images"
CHECKERBOARD     = (8, 6)
LOG_FILE         = r"C:\Thesis\dataset\check_calibration_log.txt"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_FAST_CHECK      +
                   cv2.CALIB_CB_NORMALIZE_IMAGE)

# LOGGING SETUP
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# IMAGE COLLECTION
extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
images     = []
for ext in extensions:
    images.extend(glob.glob(os.path.join(CHECKERBOARD_DIR, ext)))
images = sorted(set(os.path.normcase(os.path.abspath(p)) for p in images))

log.info("")
log.info("=" * 60)
log.info("  CHECK CALIBRATION")
log.info(f"  Checkerboard folder : {CHECKERBOARD_DIR}")
log.info(f"  Pattern             : {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners")
log.info(f"  Images found        : {len(images)}")
log.info("=" * 60)

if len(images) == 0:
    log.error(f"  No checkerboard images found in:\n  {CHECKERBOARD_DIR}")
    exit(1)

# CORNER DETECTION
objp        = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

obj_std   = [];  img_std   = []
obj_flag  = [];  img_flag  = []

image_size    = None
found_std     = 0
found_flag    = 0
not_found_any = []

log.info("")
log.info("  Scanning checkerboard images...")

for fpath in images:
    fname = os.path.basename(fpath)
    img   = cv2.imread(fpath)

    if img is None:
        log.warning(f"  SKIP — unreadable : {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])
        log.info(f"  Image size detected : {image_size[0]} x {image_size[1]} px")

    ret_s, corners_s = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret_s:
        corners_s = cv2.cornerSubPix(gray, corners_s, (11, 11), (-1, -1), criteria)
        obj_std.append(objp)
        img_std.append(corners_s)
        found_std += 1

    ret_f, corners_f = cv2.findChessboardCorners(gray, CHECKERBOARD, detection_flags)
    if ret_f:
        corners_f = cv2.cornerSubPix(gray, corners_f, (11, 11), (-1, -1), criteria)
        obj_flag.append(objp)
        img_flag.append(corners_f)
        found_flag += 1

    if not ret_s and not ret_f:
        not_found_any.append(fname)

log.info("")
log.info(f"  Standard detection   : corners found in {found_std} / {len(images)} images")
log.info(f"  With flags detection : corners found in {found_flag} / {len(images)} images")

if not_found_any:
    log.info(f"\n  Corners NOT found in {len(not_found_any)} image(s):")
    for f in not_found_any:
        log.info(f"    {f}")

if found_std < 15 and found_flag < 15:
    log.error("  Insufficient usable images for calibration.")
    exit(1)

# CALIBRATION METHOD COMPARISON
log.info("")
log.info("=" * 60)
log.info("  CALIBRATION METHOD COMPARISON")
log.info("=" * 60)

results = {}

if found_std >= 15:
    ret_a, mtx_a, dist_a, _, _ = cv2.calibrateCamera(
        obj_std, img_std, image_size, None, None
    )
    results["A — Standard with no flags        "] = (ret_a, mtx_a, dist_a)
    log.info(f"  Method A RMS : {ret_a:.4f} px")
else:
    log.info(f"  Method A     : SKIPPED — {found_std} images detected")

if found_std >= 15:
    ret_b, mtx_b, dist_b, _, _ = cv2.calibrateCamera(
        obj_std, img_std, image_size, None, None,
        flags=cv2.CALIB_FIX_K3
    )
    results["B — Standard with CALIB_FIX_K3   "] = (ret_b, mtx_b, dist_b)
    log.info(f"  Method B RMS : {ret_b:.4f} px  (CALIB_FIX_K3)")
else:
    log.info(f"  Method B     : SKIPPED — {found_std} images detected")

if found_flag >= 15:
    ret_c, mtx_c, dist_c, _, _ = cv2.calibrateCamera(
        obj_flag, img_flag, image_size, None, None,
        flags=cv2.CALIB_FIX_K3
    )
    results["C — Detect flags with CALIB_FIX_K3"] = (ret_c, mtx_c, dist_c)
    log.info(f"  Method C RMS : {ret_c:.4f} px  (detection flags with CALIB_FIX_K3)")
else:
    log.info(f"  Method C     : SKIPPED — {found_flag} images detected ")

if not results:
    log.error("  No calibration methods produced results.")
    exit(1)

# CALIBRATION RESULT
best_method = min(results, key=lambda k: results[k][0])
best_rms, best_mtx, best_dist = results[best_method]

log.info("")
log.info("=" * 60)
log.info("  CALIBRATION RESULTS SUMMARY")
log.info("=" * 60)
log.info(f"  {'Method':<40} {'RMS (px)':>10}  {'Grade'}")
log.info(f"  {'-' * 58}")

for method, (rms, _, _) in sorted(results.items(), key=lambda x: x[1][0]):
    grade  = ("EXCELLENT"  if rms < 0.5 else
              "VERY GOOD"  if rms < 0.7 else
              "ACCEPTABLE" if rms < 1.0 else
              "POOR")
    marker = "  BEST" if method == best_method else ""
    log.info(f"  {method:<40} {rms:>10.4f}  {grade}{marker}")

log.info("")
log.info(f"  Best method : {best_method.strip()}")
log.info(f"  Best RMS    : {best_rms:.4f} px")
log.info("")

if best_rms < 0.5:
    log.info("  VERDICT : EXCELLENT")
elif best_rms < 0.7:
    log.info("  VERDICT : VERY GOOD")
elif best_rms < 1.0:
    log.info("  VERDICT : ACCEPTABLE ")
else:
    log.info("  VERDICT : POOR")

log.info("")
log.info("  Camera matrix (best method):")
log.info(f"    fx={best_mtx[0,0]:.4f}  fy={best_mtx[1,1]:.4f}  "
         f"cx={best_mtx[0,2]:.4f}  cy={best_mtx[1,2]:.4f}")
log.info("")
log.info("  Distortion coefficients — k1  k2  p1  p2  k3 (best method):")
log.info(f"    {best_dist.ravel()}")
log.info("")

log.info(f"  Log saved : {LOG_FILE}")
log.info("=" * 60)