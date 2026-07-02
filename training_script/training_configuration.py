import os
import sys
import yaml
import shutil
import time
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed")
    sys.exit(1)

try:
    from roboflow import Roboflow
except ImportError:
    print("Roboflow not installed")
    sys.exit(1)

# PATHS
BASE_DIR    = r"C:\Thesis"
DATASET_DIR = os.path.join(BASE_DIR, "dataset_yolov8")
RUNS_DIR    = os.path.join(BASE_DIR, "runs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# ROBOFLOW
ROBOFLOW_API_KEY   = os.environ.get("ROBOFLOW_API_KEY", "CGrUAFIXc0kISvh18sDT")
ROBOFLOW_WORKSPACE = "ians-workspace-rjj3k"
ROBOFLOW_PROJECT   = "thesis_mangrovetree_and_referencepipe_detection"
ROBOFLOW_VERSION   = 1

# DATASET 
EXPECTED_CLASSES = ["mangrove_tree", "reference_object"]
EXPECTED_NC      = 2

# SHARED HYPERPARAMETERS
INPUT_SIZE       = 640
OPTIMIZER        = "AdamW"
LEARNING_RATE    = 0.01
LRF              = 0.01
MOMENTUM         = 0.937
WEIGHT_DECAY     = 0.0005
WARMUP_EPOCHS    = 3.0
WARMUP_MOMENTUM  = 0.8
DEVICE           = 0
WORKERS          = 4

# Loss weights
BOX_LOSS = 7.5
CLS_LOSS = 3.0
DFL_LOSS = 1.5

# GEOMETRY LOCKS
DEGREES     = 0.0
FLIPUD      = 0.0
SHEAR       = 0.0
PERSPECTIVE = 0.0
TRANSLATE   = 0.0
MIXUP       = 0.0

# AUGMENTATIONS
FLIPLR     = 0.5
SCALE      = 0.3
COPY_PASTE = 0.3
ERASING    = 0.3
MOSAIC     = 1.0

# Color augmentation
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4

# EVALUATION
CONF_THRESHOLD    = 0.25
IOU_THRESHOLD     = 0.5
SUCCESS_THRESHOLD = 0.80


# UTILITY FUNCTIONS
def check_api_key():
    if not ROBOFLOW_API_KEY:
        print("\nROBOFLOW_API_KEY is not set.")
        print("  Windows CMD:   set ROBOFLOW_API_KEY=your_key_here")
        sys.exit(1)
    print(f"  API key      : {ROBOFLOW_API_KEY[:6]}...{'*'*10} (loaded)")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU          : {name} ({vram:.1f} GB VRAM)")
            return vram
        else:
            print("No CUDA GPU detected ")
            return 0.0
    except ImportError:
        return 0.0


def _fix_yaml_paths(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    yaml_dir = os.path.dirname(yaml_path)
    changed  = False
    for key in ["train", "val", "test"]:
        val = data.get(key, "")
        if not isinstance(val, str):
            continue
        cleaned = val.replace("\\", "/")
        while cleaned.startswith("../"):
            cleaned = cleaned[3:]
        cleaned  = cleaned.lstrip("./")
        abs_path = os.path.join(yaml_dir, cleaned)
        if os.path.exists(abs_path) and val != abs_path:
            data[key] = abs_path
            changed   = True
    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print("  Fixed dataset paths in data.yaml")


def download_dataset():
    print("\nDATASET")
    print("-" * 50)

    local_yaml = os.path.join(BASE_DIR, "data.yaml")
    if os.path.exists(local_yaml):
        print(f"  Found local data.yaml: {local_yaml}")
        _fix_yaml_paths(local_yaml)
        return local_yaml

    version_dir = os.path.join(DATASET_DIR, f"v{ROBOFLOW_VERSION}")
    yaml_path   = os.path.join(version_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"  Already downloaded: {yaml_path}")
        _fix_yaml_paths(yaml_path)
        return yaml_path

    print(f"  Downloading Roboflow {ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    rf.workspace(ROBOFLOW_WORKSPACE) \
      .project(ROBOFLOW_PROJECT) \
      .version(ROBOFLOW_VERSION) \
      .download("yolov8", location=version_dir)

    yaml_path = None
    for root, _, files in os.walk(version_dir):
        for fn in files:
            if fn == "data.yaml":
                yaml_path = os.path.join(root, fn)
                break

    if not yaml_path or not os.path.exists(yaml_path):
        print("data.yaml not found after download.")
        sys.exit(1)

    _fix_yaml_paths(yaml_path)
    print(f"  Saved: {yaml_path}")
    return yaml_path


def verify_dataset(yaml_path):
    print("\nDATASET VERIFICATION")
    print("-" * 50)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    nc    = data.get("nc", 0)
    names = data.get("names", [])
    print(f"  Classes (nc) : {nc}  ->  {names}")

    if nc != EXPECTED_NC:
        print(f"  Expected nc={EXPECTED_NC}, got nc={nc}.")
        sys.exit(1)

    yaml_dir = os.path.dirname(yaml_path)
    total    = 0
    for split in ["train", "valid", "test"]:
        for pattern in [f"{split}/images", split]:
            img_dir = os.path.join(yaml_dir, pattern)
            if os.path.exists(img_dir):
                n = len([f for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                print(f"  {split:<8} : {n:>5} images")
                total += n
                break

    print(f"  {'Total':<8} : {total:>5} images")
    if total == 0:
        print("No images found.")
        sys.exit(1)

    class_counts = {i: 0 for i in range(EXPECTED_NC)}
    for pattern in ["train/labels", "train"]:
        lbl_dir = os.path.join(yaml_dir, pattern)
        if os.path.exists(lbl_dir):
            for fn in os.listdir(lbl_dir):
                if fn.endswith(".txt"):
                    with open(os.path.join(lbl_dir, fn)) as fp:
                        for line in fp:
                            if line.strip():
                                cls_id = int(line.split()[0])
                                if cls_id in class_counts:
                                    class_counts[cls_id] += 1
            break

    total_ann = sum(class_counts.values())
    if total_ann > 0:
        print(f"\n  Class balance (training labels):")
        for cls_id, count in sorted(class_counts.items()):
            name = EXPECTED_CLASSES[cls_id] if cls_id < len(EXPECTED_CLASSES) else f"class_{cls_id}"
            pct  = count / total_ann * 100
            bar  = "X" * int(pct / 3)
            print(f"    [{cls_id}] {name:<25}: {count:>5}  ({pct:5.1f}%)  {bar}")
        pipe_pct = class_counts.get(1, 0) / total_ann * 100
        if pipe_pct < 8:
            print(f"\n  Pipes are only {pipe_pct:.1f}% of annotations.")
            print(f"     CLS_LOSS={CLS_LOSS} compensates for this imbalance.")
        else:
            print(f"\n  Class balance acceptable ({pipe_pct:.1f}% pipes).")

    return data, yaml_path


def evaluate_model(weights_path, run_name, yaml_path, model_tag):
    print(f"\nEVALUATION")
    print("-" * 50)

    if not os.path.exists(weights_path):
        alt = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")
        if os.path.exists(alt):
            weights_path = alt
        else:
            print(f"Weights not found: {weights_path}")
            return False

    print(f"  Loading: {weights_path}")
    model   = YOLO(weights_path)
    results = model.val(
        data     = yaml_path,
        split    = "test",
        imgsz    = INPUT_SIZE,
        device   = DEVICE,
        conf     = CONF_THRESHOLD,
        iou      = IOU_THRESHOLD,
        verbose  = True,
        plots    = True,
        project  = RUNS_DIR,
        name     = f"{run_name}_eval",
        exist_ok = True,
    )

    box   = results.box
    p     = float(box.mp)
    r     = float(box.mr)
    map50 = float(box.map50)
    map95 = float(box.map)
    f1    = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    # Per-class mAP@0.5 
    per_class = {}
    try:
        class_names = results.names
        ap50_vals   = box.ap50
        if hasattr(ap50_vals, "tolist"):
            ap50_vals = ap50_vals.tolist()
        for cls_idx, ap_val in enumerate(ap50_vals):
            cls_name = class_names.get(cls_idx, f"class_{cls_idx}")
            per_class[cls_name] = float(ap_val)
    except Exception as e:
        print(f"  Per-class extraction skipped ({e})")

    # Print results
    sep = "=" * 62

    print(f"\n  {sep}")
    print(f"  {model_tag} - TEST SET RESULTS")
    print(f"  {sep}")
    print(f"  {'Metric':<25} {'Value':>10}  {'Target':>10}  {'Assessment'}")
    print(f"  {'-' * 62}")

    primary = [
        ("Precision", p),
        ("Recall", r),
        ("F1 Score", f1),
        ("mAP@0.5", map50)
    ]

    all_passed = True

    for name, val in primary:
        status = "Target Met" if val >= SUCCESS_THRESHOLD else "Below Target"

        print(
            f"  {name:<25} "
            f"{val*100:>9.2f}%  "
            f"{SUCCESS_THRESHOLD*100:>9.0f}%  "
            f"{status}"
        )

        if val < SUCCESS_THRESHOLD:
            all_passed = False

    print(
        f"  {'mAP@0.5:0.95 (ref)':<25} "
        f"{map95*100:>9.2f}%  "
        f"{'---':>10}  "
        f"Reference"
    )

    if per_class:
        print(f"  {'-' * 62}")
        print("  PER-CLASS mAP@0.5\n")

        for cls_name, ap_val in per_class.items():
            class_status = (
                "Target Met"
                if ap_val >= SUCCESS_THRESHOLD
                else "Below Target"
            )

            print(
                f"  {cls_name:<25}: "
                f"{ap_val:.4f} "
                f"({ap_val*100:.2f}%) "
                f"{class_status}"
            )

        print("\n  Note:")
        print("  Height estimation requires successful detection")
        print("  of both mangrove trees and reference objects.")

    print(f"\n  {'-' * 62}")
    print(
        f"  Detection Requirement : "
        f"{'MET' if all_passed else 'NOT MET'}"
    )

    print("\n  Target:")
    print(
        f"  {SUCCESS_THRESHOLD*100:.0f}% for Precision, Recall, "
        f"F1 Score, and mAP@0.5"
    )
    print(f"\n  {sep}")

    metrics_file = os.path.join(RUNS_DIR, run_name, "thesis_metrics.txt")
    os.makedirs(os.path.join(RUNS_DIR, run_name), exist_ok=True)

    with open(metrics_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"THESIS METRICS - {model_tag}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Weights        : {weights_path}\n")
        f.write(f"Dataset        : Roboflow v{ROBOFLOW_VERSION}\n")
        f.write(f"Classes        : {EXPECTED_CLASSES}\n")
        f.write(f"Conf threshold : {CONF_THRESHOLD}\n")
        f.write(f"IOU threshold  : {IOU_THRESHOLD}\n\n")

        f.write("-" * 60 + "\n")
        f.write("OVERALL DETECTION METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Precision    : {p:.4f}  ({p*100:.2f}%)\n")
        f.write(f"  Recall       : {r:.4f}  ({r*100:.2f}%)\n")
        f.write(f"  F1 Score     : {f1:.4f}  ({f1*100:.2f}%)\n")
        f.write(f"  mAP@0.5      : {map50:.4f}  ({map50*100:.2f}%)\n")
        f.write(f"  mAP@0.5:0.95 : {map95:.4f}  ({map95*100:.2f}%)\n\n")

        if per_class:
            f.write("-" * 60 + "\n")
            f.write("PER-CLASS mAP@0.5\n")
            f.write("-" * 60 + "\n")

            for cls_name, ap_val in per_class.items():
                class_status = (
                    "Target Met"
                    if ap_val >= SUCCESS_THRESHOLD
                    else "Below Target"
                )

                f.write(
                    f"{cls_name:<25}: "
                    f"{ap_val:.4f} "
                    f"({ap_val*100:.2f}%) "
                    f"{class_status}\n"
                )

            f.write("\n")

        f.write("-" * 60 + "\n\n")

        f.write(
            f"Detection Requirement : "
            f"{'MET' if all_passed else 'NOT MET'}\n\n"
        )

        f.write("Target:\n")
        f.write(
            f"{SUCCESS_THRESHOLD*100:.0f}% for Precision, Recall, "
            f"F1 Score, and mAP@0.5\n\n"
        )

        f.write("Note:\n")
        f.write("Height estimation requires successful detection of both\n")
        f.write("mangrove trees and reference objects.\n")

    print(f"\n  Metrics saved: {metrics_file}")
    return all_passed