import cv2
import numpy as np
import os
import glob
import logging
import shutil
from tqdm import tqdm


# PATHS
BASE_DIR          = r"C:\Thesis\dataset"
CHECKERBOARD_DIR  = os.path.join(BASE_DIR, "checkerboard_images")
RAW_IMAGES_DIR    = os.path.join(BASE_DIR, "raw_images")
SELECTED_DIR      = os.path.join(BASE_DIR, "selected_images")
ENHANCED_DIR      = os.path.join(BASE_DIR, "enhanced_images")
CALIBRATION_FILE  = os.path.join(BASE_DIR, "calibration_matrix.npz")
LOG_FILE          = os.path.join(BASE_DIR, "preprocessing_log.txt")
SESSION_FILTER    = None

CHESSBOARD_COLS = 8
CHESSBOARD_ROWS = 6

# Image Selection
MIN_WIDTH        = 640
MIN_HEIGHT       = 480
BLUR_THRESHOLD   = 100.0

# Camera Undistortion
UNDISTORT_ALPHA        = 0
BLACK_BORDER_THRESHOLD = 10

# Color Correction — CLAHE
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_SIZE  = (8, 8)

# Noise Reduction — Gaussian Blur
GAUSSIAN_KERNEL  = (3, 3)

# Sharpness Enhancement — Unsharp Mask
UNSHARP_AMOUNT   = 1.5
UNSHARP_SIGMA    = 1


# LOGGING SETUP
os.makedirs(BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


# HELPER FUNCTIONS
def collect_all_raw_images():
    extensions = ["*.JPG", "*.JPEG", "*.PNG"]
    all_paths  = []

    if not os.path.exists(RAW_IMAGES_DIR):
        log.error(f"  raw_images folder not found: {RAW_IMAGES_DIR}")
        return []

    session_folders = sorted([
        os.path.join(RAW_IMAGES_DIR, d)
        for d in os.listdir(RAW_IMAGES_DIR)
        if os.path.isdir(os.path.join(RAW_IMAGES_DIR, d))
    ])

    if len(session_folders) == 0:
        log.warning("  No session subfolders found inside raw_images\\")
        return []

    # Apply SESSION_FILTER if set
    if SESSION_FILTER is not None:
        session_folders = [
            f for f in session_folders
            if os.path.basename(f) == SESSION_FILTER
        ]
        if len(session_folders) == 0:
            log.error(f"  SESSION_FILTER '{SESSION_FILTER}' not found in raw_images\\")
            return []
        log.info(f"  SESSION_FILTER active: processing ONLY '{SESSION_FILTER}'")
    else:
        log.info(f"  SESSION_FILTER: None — processing ALL sessions")

    log.info(f"  Auto-detected {len(session_folders)} session folder(s):")
    for folder in session_folders:
        log.info(f"    {os.path.basename(folder)}\\")

    for session_dir in session_folders:
        for ext in extensions:
            all_paths.extend(glob.glob(os.path.join(session_dir, ext)))

    return sorted(all_paths)


# IMAGE SELECTION
def image_selection(all_images):
    log.info("")
    log.info("=" * 65)
    log.info("IMAGE SELECTION")
    log.info("=" * 65)
    log.info(f"  Min resolution : {MIN_WIDTH} x {MIN_HEIGHT} px")
    log.info(f"  Blur threshold : {BLUR_THRESHOLD} (Laplacian variance)")

    os.makedirs(SELECTED_DIR, exist_ok=True)

    accepted_paths = []
    rejected_count = 0

    for path in tqdm(all_images, desc="  Checking quality", unit="img"):
        img   = cv2.imread(path)
        fname = os.path.basename(path)

        if img is None:
            log.warning(f"  REJECTED — unreadable              : {fname}")
            rejected_count += 1
            continue

        h, w = img.shape[:2]

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            log.warning(f"  REJECTED — too small ({w}x{h})       : {fname}")
            rejected_count += 1
            continue

        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < BLUR_THRESHOLD:
            log.warning(
                f"  REJECTED — blurry (score={blur_score:.1f}) : {fname}"
            )
            rejected_count += 1
            continue

        dest = os.path.join(SELECTED_DIR, fname)
        shutil.copy2(path, dest)
        accepted_paths.append(dest)

    log.info("")
    log.info(
        f"  Result : {len(accepted_paths)} accepted, "
        f"{rejected_count} rejected out of {len(all_images)} total."
    )
    log.info(f"  Output : {SELECTED_DIR}")
    return accepted_paths


# CAMERA CALIBRATION
def camera_calibration():
    log.info("")
    log.info("=" * 65)
    log.info("CAMERA CALIBRATION")
    log.info("=" * 65)

    if os.path.exists(CALIBRATION_FILE):
        log.info(f"  Saved calibration found: {CALIBRATION_FILE}")
        log.info("  Loading — skipping recomputation.")
        data          = np.load(CALIBRATION_FILE)
        camera_matrix = data["camera_matrix"]
        dist_coeffs   = data["dist_coeffs"]
        log.info("  Camera matrix loaded successfully.")
        log.info(
            f"  Undistortion alpha = {UNDISTORT_ALPHA}  "
            f"({'crop to valid region' if UNDISTORT_ALPHA == 0 else 'keep all + trim borders'})"
        )
        return camera_matrix, dist_coeffs

    extensions = ["*.JPG", "*.JPEG", "*.PNG"]
    cb_images  = []
    for ext in extensions:
        cb_images.extend(glob.glob(os.path.join(CHECKERBOARD_DIR, ext)))
    cb_images = sorted(cb_images)

    if len(cb_images) == 0:
        raise FileNotFoundError(
            f"\nNo checkerboard images found in:\n  {CHECKERBOARD_DIR}\n"
        )

    log.info(f"  Found {len(cb_images)} checkerboard calibration images.")
    log.info(f"  Detecting {CHESSBOARD_COLS}x{CHESSBOARD_ROWS} inner corners")

    objp        = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)

    obj_points  = []
    img_points  = []
    image_size  = None
    found_count = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in tqdm(cb_images, desc="  Detecting corners", unit="img"):
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            obj_points.append(objp)
            img_points.append(corners_refined)
            found_count += 1
        else:
            log.warning(f"  Corners not found: {os.path.basename(path)}")

    log.info(f"  Corners detected in {found_count} / {len(cb_images)} images.")

    if found_count < 15:
        raise ValueError(
            f"\nOnly {found_count} usable checkerboard images.\n"
            f"Verifying CHESSBOARD_COLS={CHESSBOARD_COLS} and "
            f"CHESSBOARD_ROWS={CHESSBOARD_ROWS} match the board.\n"
        )

    log.info("  Computing calibration matrix...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    log.info(f"  RMS reprojection error : {ret:.4f} px")
    log.info(f"  Camera matrix:\n{camera_matrix}")
    log.info(f"  Distortion coefficients: {dist_coeffs.ravel()}")

    np.savez(
        CALIBRATION_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size)
    )
    log.info(f"  Calibration saved: {CALIBRATION_FILE}")
    return camera_matrix, dist_coeffs


def _trim_black_borders(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    row_means  = gray.mean(axis=1)
    valid_rows = np.where(row_means > threshold)[0]

    col_means  = gray.mean(axis=0)
    valid_cols = np.where(col_means > threshold)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        log.warning("  No valid region found, returning original.")
        return img

    top    = int(valid_rows[0])
    bottom = int(valid_rows[-1]) + 1
    left   = int(valid_cols[0])
    right  = int(valid_cols[-1]) + 1

    return img[top:bottom, left:right]


def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), UNDISTORT_ALPHA, (w, h)
    )

    undistorted = cv2.undistort(
        img, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    if UNDISTORT_ALPHA == 0:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y + rh, x:x + rw]
    else:
        undistorted = _trim_black_borders(undistorted, BLACK_BORDER_THRESHOLD)

    return undistorted

# IMAGE ENHANCEMENT
def enhance_image(img):
    # Color Correction — CLAHE
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(
        clipLimit    = CLAHE_CLIP_LIMIT,
        tileGridSize = CLAHE_TILE_SIZE
    )
    l   = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Noise Reduction — Gaussian Blur
    img = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, sigmaX=0)

    # Sharpness — Unsharp Masking
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=UNSHARP_SIGMA)
    img     = cv2.addWeighted(img, UNSHARP_AMOUNT, blurred, -(UNSHARP_AMOUNT - 1), 0)

    return img


def image_enhancement(accepted_paths, camera_matrix, dist_coeffs):
    log.info("")
    log.info("=" * 65)
    log.info("IMAGE ENHANCEMENT")
    log.info("=" * 65)
    log.info("  Per-image processing order:")
    log.info(f"    0. Undistort     (alpha={UNDISTORT_ALPHA}, "
             f"border_threshold={BLACK_BORDER_THRESHOLD})")
    log.info(f"    1. CLAHE         (clipLimit={CLAHE_CLIP_LIMIT}, "
             f"tileSize={CLAHE_TILE_SIZE})")
    log.info(f"    2. GaussianBlur  (kernel={GAUSSIAN_KERNEL})")
    log.info(f"    3. UnsharpMask   (amount={UNSHARP_AMOUNT}, "
             f"sigma={UNSHARP_SIGMA})")
    log.info("  Output resolution : FULL")

    os.makedirs(ENHANCED_DIR, exist_ok=True)
    enhanced_paths = []

    for path in tqdm(accepted_paths, desc="  Enhancing", unit="img"):
        img   = cv2.imread(path)
        fname = os.path.basename(path)

        if img is None:
            log.warning(f"  Could not read: {fname}")
            continue

        img = undistort_image(img, camera_matrix, dist_coeffs)
        img = enhance_image(img)

        dest = os.path.join(ENHANCED_DIR, fname)
        cv2.imwrite(dest, img)
        enhanced_paths.append(dest)

    log.info(
        f"  {len(enhanced_paths)} enhanced images saved to: {ENHANCED_DIR}"
    )
    return enhanced_paths

# MAIN
def main():
    log.info("")
    log.info("=" * 65)
    log.info("  PREPROCESSING PIPELINE")
    log.info(f"  Base directory   : {BASE_DIR}")
    log.info(f"  Session filter   : {SESSION_FILTER if SESSION_FILTER else 'ALL sessions'}")
    log.info(f"  Checkerboard     : 8x6 inner corners")
    log.info(f"  Undistort alpha  : {UNDISTORT_ALPHA}")
    log.info(f"  Border threshold : {BLACK_BORDER_THRESHOLD}")
    log.info(f"  CLAHE clip       : {CLAHE_CLIP_LIMIT}")
    log.info(f"  Gaussian kernel  : {GAUSSIAN_KERNEL}")
    log.info(f"  Unsharp amount   : {UNSHARP_AMOUNT}  sigma={UNSHARP_SIGMA}")
    log.info("=" * 65)

    all_raw = collect_all_raw_images()
    if len(all_raw) == 0:
        log.error("No raw images found.")
        return
    log.info(f"  Total raw images found: {len(all_raw)}")

    # Image Selection
    accepted = image_selection(all_raw)
    if not accepted:
        log.error("No images passed quality selection.")
        return

    # Camera Calibration
    camera_matrix, dist_coeffs = camera_calibration()

    # Image Enhancement
    enhanced = image_enhancement(accepted, camera_matrix, dist_coeffs)
    if not enhanced:
        log.error("No images were enhanced.")
        return

    # Preprocessing Summary
    log.info("")
    log.info("=" * 65)
    log.info("PREPROCESSING SUMMARY")
    log.info("=" * 65)
    log.info(f"  Session processed    : {SESSION_FILTER if SESSION_FILTER else 'ALL'}")
    log.info(f"  Raw images found     : {len(all_raw)}")
    log.info(f"  Passed quality check : {len(accepted)}")
    log.info(f"  Enhanced             : {len(enhanced)}")
    log.info(f"  Calibration file     : {CALIBRATION_FILE}")
    log.info(f"  Full log             : {LOG_FILE}")
    log.info("")
    log.info("=" * 65)


if __name__ == "__main__":
    main()