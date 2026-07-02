import os
import sys
import csv
import math
import logging
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. ")
    sys.exit(1)

try:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    print("Scikit-learn not installed.")
    HAS_SKLEARN = False

# CONFIGURATION
BASE_DIR = r"C:\Thesis"
WEIGHTS_CONFIG = {
    "YOLOv8n": os.path.join(BASE_DIR, "weights", "mangrove_best_n.pt"),
    "YOLOv8s": os.path.join(BASE_DIR, "weights", "mangrove_best_s.pt"),
    "YOLOv8m": os.path.join(BASE_DIR, "weights", "mangrove_best_m.pt"),
}

CALIBRATION_FILE = os.path.join(BASE_DIR, "dataset", "calibration_matrix.npz")
TEST_IMAGES_DIR  = os.path.join(BASE_DIR, "dataset", "test_images")
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "dataset", "groundTruth.csv")

OUTPUT_ROOT = os.path.join(BASE_DIR, "heightEstimation_results")
COMPARISON_CSV = os.path.join(OUTPUT_ROOT, "heightEstimation_metrics_comparison.csv")
COMPARISON_REPORT = os.path.join(OUTPUT_ROOT, "heightEstimation_model_comparison_report.txt")

CLASS_TREE = 0
CLASS_PIPE = 1

PIPE_REAL_HEIGHT_M = 1.0

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
INPUT_SIZE     = 640
SKIP_UNDISTORTION = False
UNDISTORT_ALPHA   = 0

FOREGROUND_MARGIN    = 150
DEPTH_TOLERANCE      = 80
HORIZONTAL_THRESHOLD = 500
PIPE_RADIUS          = 315

MIN_PIPE_HEIGHT_PX      = 50
MIN_TREE_HEIGHT_PX      = 60
MAX_REASONABLE_HEIGHT_M = 5.0
FRAME_EDGE_MARGIN       = 10

ZONE_METHOD = "FOREGROUND_DEPTH_PLANE_CIRCULAR_ALIGNMENT"
VALID_CIRCLE_DIST_MAX = 320
MIN_VALID_TREES_PER_IMAGE = 3

# VISUALISATION SETTINGS
COLOR_MEASURED = (0, 200, 0)
COLOR_DETECT   = (0, 165, 255)
COLOR_PIPE     = (0, 0, 255)
COLOR_PARTIAL  = (255, 165, 0)
COLOR_SUSPECT  = (0, 200, 200)
COLOR_ZONE     = (0, 200, 0)
COLOR_LINE     = (255, 0, 0)
COLOR_TEXT_BG  = (0, 0, 0)

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.52
FONT_THICKNESS = 2

log = None

# LOGGING
def setup_logger(log_path):
    logger = logging.getLogger("height_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# HELPERS
def img_stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Calibration file not found: {CALIBRATION_FILE}")
        sys.exit(1)
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def undistort_image(img, camera_matrix, dist_coeffs):
    if SKIP_UNDISTORTION:
        return img
    h, w = img.shape[:2]
    new_cam, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), UNDISTORT_ALPHA, (w, h)
    )
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        dst = dst[y:y + rh, x:x + rw]
    return dst


def load_ground_truth(csv_path):
    if not os.path.exists(csv_path):
        print(f"Ground truth CSV not found: {csv_path}")
        sys.exit(1)

    gt = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "tree_id", "manual_height_m"}
        if not required.issubset(reader.fieldnames or []):
            print("groundTruth.csv must contain: image_name, tree_id, manual_height_m")
            sys.exit(1)
        for row in reader:
            key    = img_stem(row["image_name"].strip())
            tid    = row["tree_id"].strip()
            height = float(row["manual_height_m"])
            gt.setdefault(key, {})[tid] = height
    return gt


def get_test_images():
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Test images folder not found: {TEST_IMAGES_DIR}")
        sys.exit(1)

    exts = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}
    img_paths = sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if os.path.splitext(f)[1] in exts
    ])
    if not img_paths:
        print(f"No test images found in: {TEST_IMAGES_DIR}")
        sys.exit(1)
    return img_paths


def detect_objects(model, img):
    results = model(img, imgsz=INPUT_SIZE, conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD, verbose=False)
    trees, pipes = [], []
    for box in results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        det = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "width":  x2 - x1,
            "height": y2 - y1,
            "conf": conf,
        }
        if cls == CLASS_TREE:
            trees.append(det)
        elif cls == CLASS_PIPE:
            pipes.append(det)
    return trees, pipes


def sort_left_to_right(items):
    return sorted(items, key=lambda d: d["x1"])


def is_frame_cut(det, img_h, img_w):
    m = FRAME_EDGE_MARGIN
    top    = det["y1"] < m
    bottom = det["y2"] > img_h - m
    left   = det["x1"] < m
    right  = det["x2"] > img_w - m
    return top, bottom, left, right


def compute_valid_pipes(pipes, img_h, img_w):
    valid = []
    for i, p in enumerate(pipes, start=1):
        _, bottom_cut, _, _ = is_frame_cut(p, img_h, img_w)
        Ps = p["height"]
        if bottom_cut:
            log.info(f"      Pipe {i} REJECTED: frame-cut at bottom")
            continue
        if Ps < MIN_PIPE_HEIGHT_PX:
            log.info(f"      Pipe {i} REJECTED: {Ps:.0f}px < {MIN_PIPE_HEIGHT_PX}px")
            continue
        k = PIPE_REAL_HEIGHT_M / Ps
        log.info(
            f"      Pipe {i} ACCEPTED: Ps={Ps:.0f}px  k={k:.5f}  "
            f"base=({p['cx']:.0f},{p['y2']:.0f})"
        )
        valid.append({**p, "k": k, "Ps": Ps})
    return valid


def assign_pipe(tree, valid_pipes, depth_tol, pipe_radius):
    tree_cx   = tree["cx"]
    tree_base = tree["y2"]

    best_pipe = None
    best_score = float("inf")
    best_h_dist = None
    best_depth_diff = None
    best_circle_dist = None
    best_reason = "OUTSIDE_FOREGROUND_ZONE"

    for pipe in valid_pipes:
        pipe_cx = pipe["cx"]
        pipe_base = pipe["y2"]

        h_dist = abs(tree_cx - pipe_cx)
        depth_diff = abs(tree_base - pipe_base)
        circle_dist = math.sqrt((tree_cx - pipe_cx) ** 2 +
                                (tree_base - pipe_base) ** 2)

        if tree_base < pipe_base - FOREGROUND_MARGIN:
            log.info(
                f"        behind_pipe_rejected: tree_y2={tree_base:.0f}px "
                f"pipe_y2={pipe_base:.0f}px "
                f"tree is {pipe_base - tree_base:.0f}px ABOVE pipe base"
            )
            continue

        if depth_diff > depth_tol:
            log.info(f"        depth_rejected: depth_diff={depth_diff:.0f}px > {depth_tol}px")
            continue

        if h_dist > HORIZONTAL_THRESHOLD:
            log.info(f"        h_dist_rejected: h_dist={h_dist:.0f}px > {HORIZONTAL_THRESHOLD}px")
            continue

        if circle_dist > pipe_radius:
            log.info(f"        circle_rejected: circle_dist={circle_dist:.0f}px > {pipe_radius}px")
            continue

        score = h_dist + 0.5 * depth_diff + 0.5 * circle_dist
        if score < best_score:
            best_score = score
            best_pipe = pipe
            best_h_dist = h_dist
            best_depth_diff = depth_diff
            best_circle_dist = circle_dist
            best_reason = (
                f"P_ok h_dist={h_dist:.0f}px "
                f"depth_diff={depth_diff:.0f}px "
                f"circle_dist={circle_dist:.0f}px"
            )

    return best_pipe, best_score, best_h_dist, best_depth_diff, best_circle_dist, best_reason


def estimate_height(tree, pipe):
    Pc = tree["height"]
    k  = pipe["k"]
    Ps = pipe["Ps"]
    return round(Pc * k, 4), round(Pc, 1), round(k, 5), round(Ps, 1)


def make_record(
    image_name, tree_id, tree, status,
    Pc_px=None, k_used=None, Ps_used=None,
    pipe_score=None, pipe_h_dist=None, depth_diff=None, circle_dist=None,
    height_m=None, gt_height=None, error_m=None,
    valid_obj3=False, flag="",
):
    return {
        "image_name": image_name,
        "tree_id": tree_id,
        "zone_method": ZONE_METHOD,
        "conf": round(tree["conf"], 3),
        "box_x1": round(tree["x1"], 1),
        "box_y1": round(tree["y1"], 1),
        "box_x2": round(tree["x2"], 1),
        "box_y2": round(tree["y2"], 1),
        "Pc_px": Pc_px,
        "pipe_k": k_used,
        "pipe_Ps_px": Ps_used,
        "pipe_score": round(pipe_score, 1) if pipe_score is not None else None,
        "pipe_h_dist_px": round(pipe_h_dist, 1) if pipe_h_dist is not None else None,
        "pipe_depth_diff_px": round(depth_diff, 1) if depth_diff is not None else None,
        "pipe_circle_dist_px": round(circle_dist, 1) if circle_dist is not None else None,
        "height_m": height_m,
        "gt_height_m": gt_height,
        "error_m": error_m,
        "status": status,
        "valid_obj3": valid_obj3,
        "flag": flag,
    }

# VISUALISATION
def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 2, y), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)


def draw_results(img, vis_data, valid_pipes, depth_tol, pipe_radius):
    out = img.copy()
    img_h, img_w = out.shape[:2]

    for pipe in valid_pipes:
        px1, py1 = int(pipe["x1"]), int(pipe["y1"])
        px2, py2 = int(pipe["x2"]), int(pipe["y2"])
        base_x = int(pipe["cx"])
        base_y = int(pipe["y2"])

        y_top = max(0, base_y - depth_tol)
        y_bot = min(img_h - 1, base_y + depth_tol)
        cv2.line(out, (0, y_top), (img_w, y_top), COLOR_LINE, 1)
        cv2.line(out, (0, y_bot), (img_w, y_bot), COLOR_LINE, 1)

        cv2.circle(out, (base_x, base_y), pipe_radius, COLOR_ZONE, 1)
        cv2.circle(out, (base_x, base_y), 5, COLOR_PIPE, -1)
        cv2.rectangle(out, (px1, py1), (px2, py2), COLOR_PIPE, 2)
        draw_label(out, f"PIPE k={pipe['k']:.4f}", px1, py1 - 5, COLOR_PIPE)

    for item in vis_data:
        tree = item["tree"]
        tid = item["tree_id"]
        status = item["status"]
        height = item.get("height_m")

        x1, y1 = int(tree["x1"]), int(tree["y1"])
        x2, y2 = int(tree["x2"]), int(tree["y2"])

        if status == "MEASURED":
            color = COLOR_MEASURED
            label = f"{tid}:{height:.2f}m"
        elif status == "PARTIAL":
            color = COLOR_PARTIAL
            label = f"{tid}:PARTIAL"
        elif status == "HEIGHT_SUSPECT":
            color = COLOR_SUSPECT
            label = f"{tid}:SUSPECT {height:.2f}m"
        else:
            color = COLOR_DETECT
            label = f"{tid}:DETECT"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        draw_label(out, label, x1, y1 - 5, color)
        cv2.circle(out, (int(tree["cx"]), int(tree["y2"])), 4, color, -1)

    return out

# CORE PIPELINE
def process_image(model, img_path, camera_matrix, dist_coeffs, ground_truth, annotated_dir):
    img_name_full = os.path.basename(img_path)
    img_key = img_stem(img_name_full)

    img = cv2.imread(img_path)
    if img is None:
        log.warning(f"  Cannot read: {img_path}")
        return []

    img = undistort_image(img, camera_matrix, dist_coeffs)
    img_h, img_w = img.shape[:2]

    trees, pipes = detect_objects(model, img)
    trees = sort_left_to_right(trees)
    valid_pipes = compute_valid_pipes(pipes, img_h, img_w)

    log.info(f"    Raw: {len(trees)} tree(s), {len(pipes)} pipe(s), {len(valid_pipes)} valid pipe(s)")

    gt_this = ground_truth.get(img_key, {})
    results = []
    vis_data = []
    measured_cands = []

    for d_idx, tree in enumerate(trees, start=1):
        d_id = f"D{d_idx}"
        top_cut, bottom_cut, _, _ = is_frame_cut(tree, img_h, img_w)

        if tree["height"] < MIN_TREE_HEIGHT_PX:
            status = "SHORT_BOX"
            flag = f"SHORT_{tree['height']:.0f}px"
            results.append(make_record(img_name_full, d_id, tree, status,
                                       Pc_px=round(tree["height"], 1),
                                       valid_obj3=False, flag=flag))
            vis_data.append({"tree": tree, "tree_id": d_id, "status": status, "height_m": None})
            log.info(f"      {d_id}: {status:<16} {flag}")
            continue

        if top_cut or bottom_cut:
            cuts = (["TOP"] if top_cut else []) + (["BOTTOM"] if bottom_cut else [])
            status = "PARTIAL"
            flag = "FRAME_CUT_" + "_".join(cuts)
            results.append(make_record(img_name_full, d_id, tree, status,
                                       Pc_px=round(tree["height"], 1),
                                       valid_obj3=False, flag=flag))
            vis_data.append({"tree": tree, "tree_id": d_id, "status": status, "height_m": None})
            log.info(f"      {d_id}: {status:<16} {flag}")
            continue

        if not valid_pipes:
            status = "NO_PIPE"
            flag = "NO_VALID_PIPE"
            results.append(make_record(img_name_full, d_id, tree, status,
                                       Pc_px=round(tree["height"], 1),
                                       valid_obj3=False, flag=flag))
            vis_data.append({"tree": tree, "tree_id": d_id, "status": status, "height_m": None})
            log.info(f"      {d_id}: {status:<16} {flag}")
            continue

        pipe, score, h_dist, depth_diff, circle_dist, reason = assign_pipe(
            tree, valid_pipes, DEPTH_TOLERANCE, PIPE_RADIUS
        )

        if pipe is None:
            status = "TOO_FAR"
            flag = "OUTSIDE_FOREGROUND_ZONE"
            results.append(make_record(img_name_full, d_id, tree, status,
                                       Pc_px=round(tree["height"], 1),
                                       valid_obj3=False, flag=flag))
            vis_data.append({"tree": tree, "tree_id": d_id, "status": status, "height_m": None})
            log.info(f"      {d_id}: {status:<16} {flag}")
            continue

        height_m, Pc_px, k_used, Ps_used = estimate_height(tree, pipe)

        if height_m > MAX_REASONABLE_HEIGHT_M:
            status = "HEIGHT_SUSPECT"
            flag = f"Rc={height_m:.2f}m>{MAX_REASONABLE_HEIGHT_M}m"
            results.append(make_record(
                img_name_full, d_id, tree, status,
                Pc_px=Pc_px, k_used=k_used, Ps_used=Ps_used,
                pipe_score=score, pipe_h_dist=h_dist,
                depth_diff=depth_diff, circle_dist=circle_dist,
                height_m=height_m, valid_obj3=False, flag=flag,
            ))
            vis_data.append({"tree": tree, "tree_id": d_id, "status": status, "height_m": height_m})
            log.info(f"      {d_id}: {status:<16} h={height_m:.2f}m {flag}")
            continue

        measured_cands.append({
            "tree": tree,
            "height_m": height_m,
            "Pc_px": Pc_px, "k_used": k_used, "Ps_used": Ps_used,
            "score": score, "h_dist": h_dist,
            "depth_diff": depth_diff, "circle_dist": circle_dist,
            "reason": reason,
        })

    measured_cands = sorted(measured_cands, key=lambda it: it["tree"]["x1"])

    for t_idx, item in enumerate(measured_cands, start=1):
        t_id = f"T{t_idx}"
        tree = item["tree"]
        height_m = item["height_m"]
        gt_height = gt_this.get(t_id)
        error_m = round(height_m - gt_height, 4) if gt_height is not None else None

        results.append(make_record(
            img_name_full, t_id, tree, "MEASURED",
            Pc_px=item["Pc_px"], k_used=item["k_used"], Ps_used=item["Ps_used"],
            pipe_score=item["score"], pipe_h_dist=item["h_dist"],
            depth_diff=item["depth_diff"], circle_dist=item["circle_dist"],
            height_m=height_m, gt_height=gt_height, error_m=error_m,
            valid_obj3=True, flag=item["reason"],
        ))
        vis_data.append({"tree": tree, "tree_id": t_id, "status": "MEASURED", "height_m": height_m})

        if gt_height is not None:
            log.info(
                f"      {t_id}: MEASURED {height_m:.4f}m  GT={gt_height:.2f}m  "
                f"err={error_m:+.4f}m  (Pc={item['Pc_px']:.0f}px k={item['k_used']:.5f} "
                f"h={item['h_dist']:.0f}px depth={item['depth_diff']:.0f}px "
                f"circle={item['circle_dist']:.0f}px)"
            )
        else:
            log.info(
                f"      {t_id}: MEASURED {height_m:.4f}m  NO_GT  "
                f"(Pc={item['Pc_px']:.0f}px k={item['k_used']:.5f} "
                f"h={item['h_dist']:.0f}px depth={item['depth_diff']:.0f}px "
                f"circle={item['circle_dist']:.0f}px)"
            )

    vis = draw_results(img, vis_data, valid_pipes, DEPTH_TOLERANCE, PIPE_RADIUS)
    cv2.imwrite(os.path.join(annotated_dir, img_name_full), vis)
    return results

# METRICS AND OUTPUTS
def get_final_paired(all_results):
    paired = [
        r for r in all_results
        if r["status"] == "MEASURED"
        and r["valid_obj3"]
        and r["height_m"] is not None
        and r["gt_height_m"] is not None
    ]
    paired_gt_count = len(paired)

    paired = [
        r for r in paired
        if (r["pipe_circle_dist_px"] is not None and
            float(r["pipe_circle_dist_px"]) <= VALID_CIRCLE_DIST_MAX)
    ]
    after_circle = len(paired)

    img_counts = Counter(img_stem(r["image_name"]) for r in paired)
    valid_imgs = {k for k, v in img_counts.items() if v >= MIN_VALID_TREES_PER_IMAGE}
    paired = [r for r in paired if img_stem(r["image_name"]) in valid_imgs]

    return paired, paired_gt_count, after_circle, valid_imgs


def compute_validation_metrics(all_results):
    log.info("")
    log.info("=" * 70)
    log.info("HEIGHT ESTIMATION VALIDATION")
    log.info("=" * 70)

    if not HAS_SKLEARN:
        log.warning("  scikit-learn missing")
        return None, []

    total     = len(all_results)
    measured  = sum(1 for r in all_results if r["status"] == "MEASURED")
    partial   = sum(1 for r in all_results if r["status"] == "PARTIAL")
    too_far   = sum(1 for r in all_results if r["status"] == "TOO_FAR")
    no_pipe   = sum(1 for r in all_results if r["status"] == "NO_PIPE")
    short_box = sum(1 for r in all_results if r["status"] == "SHORT_BOX")
    suspect   = sum(1 for r in all_results if r["status"] == "HEIGHT_SUSPECT")

    paired, paired_gt_count, after_circle, valid_imgs = get_final_paired(all_results)

    log.info(f"  Total records            : {total}")
    log.info(f"  MEASURED                 : {measured}")
    log.info(f"  PARTIAL (frame-cut)      : {partial}")
    log.info(f"  TOO_FAR (outside zone)   : {too_far}")
    log.info(f"  NO_PIPE                  : {no_pipe}")
    log.info(f"  SHORT_BOX                : {short_box}")
    log.info(f"  HEIGHT_SUSPECT           : {suspect}")
    log.info(f"  Paired MEASURED + GT     : {paired_gt_count}")
    log.info(f"  After circle_dist filter : {after_circle}")
    log.info(f"  After >=3 trees/image    : {len(paired)} ({len(valid_imgs)} images)")

    if not paired:
        log.warning("  No valid paired measurements remain after filtering.")
        return None, []

    y_true = np.array([r["gt_height_m"] for r in paired], dtype=float)
    y_pred = np.array([r["height_m"] for r in paired], dtype=float)
    errors = y_pred - y_true
    abs_err_cm = np.abs(errors) * 100

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(errors))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(paired) > 1 else float("nan")
    median_ae = float(np.median(abs_err_cm))

    log.info("")
    log.info("  HEIGHT ESTIMATION METRICS")
    log.info("  " + "-" * 50)
    log.info(f"  N trees          : {len(paired)}")
    log.info(f"  N images         : {len(valid_imgs)}")
    log.info(f"  RMSE             : {rmse:.4f} m  ({rmse * 100:.2f} cm)")
    log.info(f"  MAE              : {mae:.4f} m  ({mae * 100:.2f} cm)")
    log.info(f"  R²               : {r2:.4f}")
    log.info(f"  Pearson r        : {pearson:.4f}")
    log.info(f"  Median AE        : {median_ae:.2f} cm")
    log.info(f"  Mean bias        : {bias:+.4f} m  ({bias * 100:+.2f} cm) ")
    log.info("  " + "-" * 50)

    metrics = {
        "total_records": total,
        "measured": measured,
        "partial": partial,
        "too_far": too_far,
        "no_pipe": no_pipe,
        "short_box": short_box,
        "suspect": suspect,
        "paired_measured_gt": paired_gt_count,
        "after_circle_filter": after_circle,
        "n": len(paired),
        "n_images": len(valid_imgs),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "bias": bias,
        "median_ae_cm": median_ae,
    }
    return metrics, paired


def save_predictions_csv(all_results, predictions_csv):
    if not all_results:
        log.warning("  No results to save.")
        return
    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    log.info(f"  Predictions CSV: {predictions_csv}")


def generate_accuracy_plots(paired, all_results, plots_dir, model_tag):
    if not paired:
        log.warning("  No paired data ")
        return

    y_true = np.array([r["gt_height_m"] for r in paired], dtype=float)
    y_pred = np.array([r["height_m"] for r in paired], dtype=float)
    errors_m = y_pred - y_true
    abs_err_cm = np.abs(errors_m) * 100
    bias_cm = errors_m * 100

    n = len(paired)
    n_images = len(set(img_stem(r["image_name"]) for r in paired))
    mae = float(np.mean(abs_err_cm))
    rmse = float(np.sqrt(np.mean(errors_m ** 2))) * 100
    r2 = float(1 - np.sum(errors_m ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    bias = float(np.mean(bias_cm))
    median = float(np.median(abs_err_cm))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 else float("nan")

    # Detection status summary
    total = len(all_results)
    measured = sum(1 for r in all_results if r["status"] == "MEASURED")
    too_far = sum(1 for r in all_results if r["status"] == "TOO_FAR")
    partial = sum(1 for r in all_results if r["status"] == "PARTIAL")
    short_b = sum(1 for r in all_results if r["status"] == "SHORT_BOX")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [
        "MEASURED\n(Height estimated)",
        "TOO_FAR\n(Outside pipe zone)",
        "PARTIAL\n(Frame-cut)",
        "SHORT_BOX\n(Box too small)",
    ]
    values = [measured, too_far, partial, short_b]
    colors = ["#041373"]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.1f}%)", va="center", fontsize=10)
    ax.set_xlabel("Number of Detections", fontsize=12)
    ax.set_title(
        f"Detection Status Summary Across {len(set(r['image_name'] for r in all_results))} Validation Images\n"
        f"({model_tag} — Total detections: {total})",
        fontsize=12
    )
    ax.set_xlim(0, max(values) * 1.25 if values else 1)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figure_detection_status.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(y_true, y_pred, color="steelblue", edgecolors="white",
               s=60, alpha=0.85, linewidths=0.5, zorder=3, label="Tree measurements")
    lims = [0, max(max(y_true), max(y_pred)) * 1.08]
    ax.plot(lims, lims, "k--", linewidth=1.3, label="y = x  (perfect estimate)", zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Manual Ground Truth Height (m)", fontsize=12)
    ax.set_ylabel("System Estimated Height (m)", fontsize=12)
    ax.set_title(f"Automated Estimates vs. Manual Measurements\n({model_tag} — n = {n} trees, {n_images} images)", fontsize=12)
    stats_text = (
        f"R²        = {r2:.4f}\n"
        f"Pearson r = {pearson:.4f}\n"
        f"MAE       = {mae:.2f} cm\n"
        f"RMSE      = {rmse:.2f} cm\n"
        f"Bias      = {bias:+.2f} cm"
    )
    ax.text(0.04, 0.97, stats_text, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="lightyellow", edgecolor="gray", alpha=0.9))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figure_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Error histogram
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bin_edges = [0, 10, 20, 30, 40, 50, max(abs_err_cm) + 1]
    bin_labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "> 50"]
    hist_counts, _ = np.histogram(abs_err_cm, bins=bin_edges)
    colors_h = ["#041373"]
    bars_h = ax.bar(range(len(hist_counts)), hist_counts, color=colors_h, edgecolor="white", linewidth=0.9)
    for bar, cnt in zip(bars_h, hist_counts):
        pct = cnt / n * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels([f"{l} cm" for l in bin_labels], fontsize=10)
    ax.set_xlabel("Absolute Error Range", fontsize=12)
    ax.set_ylabel("Number of Trees", fontsize=12)
    ax.set_title(f"Distribution of Absolute Height Estimation Errors\n({model_tag} — n = {n} trees     Median AE = {median:.2f} cm)", fontsize=12)
    ax.set_ylim(0, max(hist_counts) + 7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figure_error_histogram.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Bias plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_true, bias_cm, color="steelblue", edgecolors="white", s=60, alpha=0.85, linewidths=0.5, zorder=3)
    ax.axhline(0, color="black", linewidth=1.3, linestyle="--", label="Zero bias (perfect)")
    ax.axhline(bias, color="red", linewidth=1.3, linestyle="-", label=f"Mean bias = {bias:+.2f} cm")
    ax.fill_between([0, max(y_true) * 1.08], -10, 10, alpha=0.07, color="green", label="±10 cm band")
    ax.set_xlim(0, max(y_true) * 1.08)
    ax.set_xlabel("Manual Ground Truth Height (m)", fontsize=12)
    ax.set_ylabel("Error: Estimated − Measured (cm)", fontsize=12)
    ax.set_title(f"Estimation Bias Across Height Range\n({model_tag} — n = {n} trees)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figure_bias.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Growth stage plot
    def growth_band(h):
        if h < 0.5:
            return "Height Range\n(< 0.5 m)"
        elif h < 1.5:
            return "Height Range\n(0.5–1.5 m)"
        elif h < 3.0:
            return "Height Range\n(1.5–3.0 m)"
        else:
            return "Height Range\n(> 3.0 m)"

    band_order = [
        "Height Range\n(< 0.5 m)",
        "Height Range\n(0.5–1.5 m)",
        "Height Range\n(1.5–3.0 m)",
        "Height Range\n(> 3.0 m)",
    ]
    band_mae, band_bias, band_n = [], [], []
    for b in band_order:
        subset = [(abs(r["error_m"]) * 100, r["error_m"] * 100)
                  for r in paired if growth_band(r["gt_height_m"]) == b]
        if subset:
            ae, be = zip(*subset)
            band_mae.append(float(np.mean(ae)))
            band_bias.append(float(np.mean(be)))
            band_n.append(len(subset))
        else:
            band_mae.append(0); band_bias.append(0); band_n.append(0)

    x = np.arange(len(band_order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    main_color = "#041373"     
    light_color = "#7E95F3"     
    bars_m = ax.bar( x - w / 2, band_mae, width=w, color=main_color, edgecolor="white", linewidth=0.8, label="MAE (cm)"
    )
    bars_b = ax.bar( x + w / 2, band_bias, width=w, color=light_color, edgecolor="white", linewidth=0.8, alpha=0.75, hatch="//", label="Mean Bias (cm)")
    for i, (m, b, nn) in enumerate(zip(band_mae, band_bias, band_n)):
        ax.text(i - w / 2, m + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, max(b, 0) + 0.3, f"{b:+.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i, -4, f"n={nn}", ha="center", va="top", fontsize=9, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(band_order, fontsize=10)
    ax.set_ylabel("Error (cm)", fontsize=12)
    ax.set_title(f"Height Estimation Accuracy by Growth Stage\n({model_tag} — MAE and Mean Bias)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_ylim(-7, max(band_mae) + 8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figure_growth_stage.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Cumulative error distribution
    sorted_err = np.sort(abs_err_cm)
    cum_pct = np.arange(1, n + 1) / n * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_err, cum_pct, color="steelblue", linewidth=2.2)
    thresholds = [10, 15, 20, 25, 30, 40, 50]
    for t in thresholds:
        pct_t = float(np.sum(abs_err_cm <= t) / n * 100)
        ax.plot(t, pct_t, "o", color="red", markersize=6, zorder=5)
        offset_y = -5 if pct_t > 90 else 3
        ax.annotate(f"{pct_t:.1f}%", xy=(t, pct_t), xytext=(t + 1.5, pct_t + offset_y), fontsize=8.5, color="red")
    ax.axhline(80, color="gray", linewidth=1, linestyle=":", alpha=0.7, label="80% reference line")
    ax.set_xlabel("Absolute Error Threshold (cm)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of Trees (%)", fontsize=12)
    ax.set_title(f"Cumulative Error Distribution\n({model_tag} — n = {n} trees)", fontsize=12)
    ax.set_xlim(0, max(abs_err_cm) + 5)
    ax.set_ylim(0, 107)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figure_cumulative.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    log.info(f"  All plots saved to     : {plots_dir}")


def print_summary(all_results, metrics, model_tag, annotated_dir, predictions_csv, report_file):
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"  Model                  : {model_tag}")
    log.info(f"  Zone method            : {ZONE_METHOD}")
    log.info(f"  PIPE_RADIUS            : {PIPE_RADIUS} px")
    log.info(f"  HORIZONTAL_THRESHOLD   : {HORIZONTAL_THRESHOLD} px")
    log.info(f"  DEPTH_TOLERANCE        : {DEPTH_TOLERANCE} px")
    log.info(f"  VALID_CIRCLE_DIST_MAX  : {VALID_CIRCLE_DIST_MAX} px")
    log.info(f"  MIN_VALID_TREES/IMAGE  : {MIN_VALID_TREES_PER_IMAGE}")
    log.info(f"  Images processed       : {len(set(r['image_name'] for r in all_results))}")
    log.info(f"  Total records          : {len(all_results)}")
    if metrics:
        log.info(f"  RMSE                   : {metrics['rmse'] * 100:.2f} cm")
        log.info(f"  MAE                    : {metrics['mae'] * 100:.2f} cm")
        log.info(f"  R²                     : {metrics['r2']:.4f}")
        log.info(f"  Pearson r              : {metrics['pearson']:.4f}")
        log.info(f"  Mean bias              : {metrics['bias'] * 100:+.2f} cm")
        log.info(f"  Median AE              : {metrics['median_ae_cm']:.2f} cm")
        log.info(f"  N validated            : {metrics['n']} trees from {metrics['n_images']} images")
    log.info(f"  Annotated images       : {annotated_dir}")
    log.info(f"  Predictions CSV        : {predictions_csv}")
    log.info(f"  Report file            : {report_file}")
    log.info("=" * 70)

# MODEL RUNNER AND COMPARISON REPORT
def run_model(model_tag, weights_path, camera_matrix, dist_coeffs, ground_truth, img_paths):
    global log

    model_out_dir = os.path.join(OUTPUT_ROOT, model_tag)
    annotated_dir = os.path.join(model_out_dir, "annotated_images")
    plots_dir = os.path.join(model_out_dir, "accuracy_plots")
    predictions_csv = os.path.join(model_out_dir, f"predictions_{model_tag}.csv")
    report_file = os.path.join(model_out_dir, f"heightEstimation_report_{model_tag}.txt")

    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    log = setup_logger(report_file)

    log.info("=" * 70)
    log.info(f"  MANGROVE HEIGHT ESTIMATION — {model_tag}")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)
    log.info(f"  Weights        : {weights_path}")
    log.info(f"  Ground truth   : {GROUND_TRUTH_CSV}")
    log.info(f"  Test images    : {TEST_IMAGES_DIR}")
    log.info(f"  Output folder  : {model_out_dir}")
    log.info(f"  PIPE_RADIUS    : {PIPE_RADIUS} px")
    log.info(f"  DEPTH_TOL      : {DEPTH_TOLERANCE} px")
    log.info(f"  CONF           : {CONF_THRESHOLD}")

    if not os.path.exists(weights_path):
        log.error(f"  Weights not found: {weights_path}")
        return {
            "model": model_tag,
            "weights": weights_path,
            "status": "WEIGHTS_NOT_FOUND",
        }

    log.info("\n LOADING MODEL")
    model = YOLO(weights_path)
    log.info(f"  Loaded: {model_tag}")

    log.info("\n PROCESSING")
    log.info("-" * 70)
    all_results = []
    for idx, img_path in enumerate(img_paths, start=1):
        log.info(f"[{idx}/{len(img_paths)}] {os.path.basename(img_path)}")
        all_results.extend(process_image(model, img_path, camera_matrix, dist_coeffs, ground_truth, annotated_dir))

    if not all_results:
        log.warning("  No detections produced.")
        return {
            "model": model_tag,
            "weights": weights_path,
            "status": "NO_DETECTIONS",
        }

    log.info("\n SAVING PREDICTIONS")
    save_predictions_csv(all_results, predictions_csv)

    log.info("\n VALIDATION METRICS")
    metrics, paired = compute_validation_metrics(all_results)

    log.info("\n ACCURACY PLOTS")
    generate_accuracy_plots(paired, all_results, plots_dir, model_tag)

    print_summary(all_results, metrics, model_tag, annotated_dir, predictions_csv, report_file)

    row = {
        "model": model_tag,
        "weights": weights_path,
        "status": "OK" if metrics else "NO_VALID_PAIRED_DATA",
        "images_processed": len(set(r["image_name"] for r in all_results)),
        "total_records": len(all_results),
        "predictions_csv": predictions_csv,
        "report_file": report_file,
        "plots_dir": plots_dir,
    }
    if metrics:
        row.update({
            "measured": metrics["measured"],
            "partial": metrics["partial"],
            "too_far": metrics["too_far"],
            "no_pipe": metrics["no_pipe"],
            "short_box": metrics["short_box"],
            "height_suspect": metrics["suspect"],
            "paired_measured_gt": metrics["paired_measured_gt"],
            "after_circle_filter": metrics["after_circle_filter"],
            "final_n_trees": metrics["n"],
            "final_n_images": metrics["n_images"],
            "rmse_cm": metrics["rmse"] * 100,
            "mae_cm": metrics["mae"] * 100,
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson"],
            "mean_bias_cm": metrics["bias"] * 100,
            "abs_mean_bias_cm": abs(metrics["bias"] * 100),
            "median_ae_cm": metrics["median_ae_cm"],
        })
    return row


def save_comparison(rows):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fieldnames = [
        "model", "status", "weights", "images_processed", "total_records",
        "measured", "partial", "too_far", "no_pipe", "short_box", "height_suspect",
        "paired_measured_gt", "after_circle_filter", "final_n_trees", "final_n_images",
        "mae_cm", "rmse_cm", "r2", "pearson_r", "mean_bias_cm", "abs_mean_bias_cm", "median_ae_cm",
        "predictions_csv", "report_file", "plots_dir",
    ]
    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    valid_rows = [r for r in rows if r.get("status") == "OK"]
    with open(COMPARISON_REPORT, "w", encoding="utf-8") as f:
        f.write("YOLOv8 HEIGHT ESTIMATION MODEL COMPARISON\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test images: {TEST_IMAGES_DIR}\n")
        f.write(f"Ground truth: {GROUND_TRUTH_CSV}\n")
        f.write(f"Zone method: {ZONE_METHOD}\n")
        f.write(f"DEPTH_TOLERANCE={DEPTH_TOLERANCE}px, PIPE_RADIUS={PIPE_RADIUS}px, HORIZONTAL_THRESHOLD={HORIZONTAL_THRESHOLD}px\n")
        f.write(f"VALID_CIRCLE_DIST_MAX={VALID_CIRCLE_DIST_MAX}px, MIN_VALID_TREES_PER_IMAGE={MIN_VALID_TREES_PER_IMAGE}\n\n")

        if not valid_rows:
            f.write("No valid model results were produced. \n")
        else:
            f.write("Summary table\n")
            f.write("-" * 70 + "\n")
            f.write("Model, N trees, N images, MAE (cm), RMSE (cm), R2, Pearson r, Bias (cm), Median AE (cm)\n")
            for r in valid_rows:
                f.write(
                    f"{r['model']}, {r['final_n_trees']}, {r['final_n_images']}, "
                    f"{r['mae_cm']:.2f}, {r['rmse_cm']:.2f}, {r['r2']:.4f}, "
                    f"{r['pearson_r']:.4f}, {r['mean_bias_cm']:+.2f}, {r['median_ae_cm']:.2f}\n"
                )

            best_mae = min(valid_rows, key=lambda r: r["mae_cm"])
            best_rmse = min(valid_rows, key=lambda r: r["rmse_cm"])
            best_r2 = max(valid_rows, key=lambda r: r["r2"])
            best_bias = min(valid_rows, key=lambda r: r["abs_mean_bias_cm"])

            f.write("\nBest by metric\n")
            f.write("-" * 70 + "\n")
            f.write(f"Lowest MAE       : {best_mae['model']} ({best_mae['mae_cm']:.2f} cm)\n")
            f.write(f"Lowest RMSE      : {best_rmse['model']} ({best_rmse['rmse_cm']:.2f} cm)\n")
            f.write(f"Highest R2       : {best_r2['model']} ({best_r2['r2']:.4f})\n")
            f.write(f"Lowest |bias|    : {best_bias['model']} ({best_bias['mean_bias_cm']:+.2f} cm)\n")

            f.write("\nSelection guide\n")
            f.write("-" * 70 + "\n")
  
    print(f"\nComparison CSV saved to: {COMPARISON_CSV}")
    print(f"Comparison report saved to: {COMPARISON_REPORT}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 70)
    print("YOLOv8 HEIGHT ESTIMATION COMPARISON: n vs s vs m")
    print("=" * 70)
    print(f"Ground truth: {GROUND_TRUTH_CSV}")
    print(f"Test images : {TEST_IMAGES_DIR}")

    camera_matrix, dist_coeffs = load_calibration()
    ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
    img_paths = get_test_images()

    print(f"Ground truth loaded: {len(ground_truth)} images, {sum(len(v) for v in ground_truth.values())} trees")
    print(f"Test images found : {len(img_paths)} images")

    rows = []
    for model_tag, weights_path in WEIGHTS_CONFIG.items():
        row = run_model(model_tag, weights_path, camera_matrix, dist_coeffs, ground_truth, img_paths)
        rows.append(row)

    save_comparison(rows)


if __name__ == "__main__":
    main()
