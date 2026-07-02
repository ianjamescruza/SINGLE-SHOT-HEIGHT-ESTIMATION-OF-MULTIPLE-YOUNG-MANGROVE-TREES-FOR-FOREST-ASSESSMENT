import os
import sys
import time
import shutil
from datetime import datetime
import training_configuration as cfg

MODEL_WEIGHTS = "yolov8m.pt"
MODEL_TAG     = "YOLOv8m"
EPOCHS        = 150
BATCH_SIZE    = 4       
PATIENCE      = 50
WEIGHTS_OUT   = os.path.join(cfg.WEIGHTS_DIR, "mangrove_best_m.pt")


def train(yaml_path):
    print(f"\nTRAINING — {MODEL_TAG}")
    print("-" * 60)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mangrove_m_{ts}"
    os.makedirs(cfg.RUNS_DIR,    exist_ok=True)
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)

    print(f"  Run            : {run_name}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Patience       : {PATIENCE}")
    print(f"  Batch          : {BATCH_SIZE}")
    print(f"  imgsz          : {cfg.INPUT_SIZE}")
    print(f"  Optimizer      : {cfg.OPTIMIZER}")
    print(f"  lr0            : {cfg.LEARNING_RATE}")
    print(f"  Device         : {cfg.DEVICE}")
    print(f"  Workers        : {cfg.WORKERS}")
    print(f"  CLS_LOSS       : {cfg.CLS_LOSS}")
    print(f"  Mosaic         : {cfg.MOSAIC}")
    print("  Geometry locks  : degrees=0      flipud=0     shear=0")
    print("                    perspective=0  translate=0")
    print(f"  Safe augs      : fliplr={cfg.FLIPLR}          scale={cfg.SCALE}")
    print(f"                   copy_paste={cfg.COPY_PASTE}  erasing={cfg.ERASING}")
    print("-" * 60)

    model = cfg.YOLO(MODEL_WEIGHTS)
    t0    = time.time()

    model.train(
        data            = yaml_path,
        epochs          = EPOCHS,
        batch           = BATCH_SIZE,
        imgsz           = cfg.INPUT_SIZE,
        device          = cfg.DEVICE,
        workers         = cfg.WORKERS,
        optimizer       = cfg.OPTIMIZER,
        lr0             = cfg.LEARNING_RATE,
        lrf             = cfg.LRF,
        momentum        = cfg.MOMENTUM,
        weight_decay    = cfg.WEIGHT_DECAY,
        warmup_epochs   = cfg.WARMUP_EPOCHS,
        warmup_momentum = cfg.WARMUP_MOMENTUM,
        box             = cfg.BOX_LOSS,
        cls             = cfg.CLS_LOSS,
        dfl             = cfg.DFL_LOSS,
        degrees         = cfg.DEGREES,
        flipud          = cfg.FLIPUD,
        shear           = cfg.SHEAR,
        perspective     = cfg.PERSPECTIVE,
        translate       = cfg.TRANSLATE,
        mixup           = cfg.MIXUP,
        fliplr          = cfg.FLIPLR,
        scale           = cfg.SCALE,
        copy_paste      = cfg.COPY_PASTE,
        erasing         = cfg.ERASING,
        mosaic          = cfg.MOSAIC,
        hsv_h           = cfg.HSV_H,
        hsv_s           = cfg.HSV_S,
        hsv_v           = cfg.HSV_V,
        project         = cfg.RUNS_DIR,
        name            = run_name,
        exist_ok        = True,
        pretrained      = True,
        patience        = PATIENCE,
        save            = True,
        save_period     = 25,
        val             = True,
        plots           = True,
        verbose         = True,
        seed            = 42,
    )

    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed / 60:.1f} min")

    src = os.path.join(cfg.RUNS_DIR, run_name, "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy2(src, WEIGHTS_OUT)
        print(f"  Best weights saved : {WEIGHTS_OUT}")
    else:
        print("  best.pt not found")

    return run_name


def main():
    print("\n" + "=" * 60)
    print(f"  {MODEL_TAG}")
    print("  SINGLE-SHOT HEIGHT ESTIMATION OF MULTIPLE YOUNG")
    print("  MANGROVE TREES FOR FOREST ASSESSMENT")
    print(f"  Hyperparameters")
    print(f"  Dataset         : Roboflow v{cfg.ROBOFLOW_VERSION}")
    print("=" * 60)

    cfg.check_api_key()
    vram = cfg.check_gpu()
    if vram > 0 and vram < 5.5:
        print(f"  WARNING Only {vram:.1f} GB VRAM detected.")

    yaml_path = cfg.download_dataset()
    cfg.verify_dataset(yaml_path)

    if input("\n  Start training? (yes/no): ").strip().lower() not in ("yes", "y"):
        print("  Cancelled.")
        return

    run_name = train(yaml_path)
    passed   = cfg.evaluate_model(WEIGHTS_OUT, run_name, yaml_path, MODEL_TAG)

    print("\n" + "=" * 60)
    if passed:
        print("  PASSED ")
    else:
        print("  FAILED ")
    print(f"  Weights : {WEIGHTS_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
