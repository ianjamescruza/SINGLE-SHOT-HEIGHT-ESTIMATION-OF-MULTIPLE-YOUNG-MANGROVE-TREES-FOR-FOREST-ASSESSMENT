import os
import sys
import time
import shutil
from datetime import datetime

# Configuration 
import training_configuration as cfg

# Model settings 
MODEL_WEIGHTS = "yolov8n.pt"
MODEL_TAG     = "YOLOv8n"
WEIGHTS_OUT   = os.path.join(cfg.WEIGHTS_DIR, "mangrove_best_n.pt")

# Hyperparameters 
EPOCHS     = 100   
BATCH_SIZE = 16    
PATIENCE   = 50    


# TRAINING
def train(yaml_path):
    print(f"\n[STEP 3] TRAINING — {MODEL_TAG}")
    print("-" * 60)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mangrove_n_{ts}"  

    os.makedirs(cfg.RUNS_DIR,    exist_ok=True)
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)

    print(f"  {'Run':<16}: {run_name}")
    print(f"  {'Epochs':<16}: {EPOCHS}")
    print(f"  {'Patience':<16}: {PATIENCE}")
    print(f"  {'Batch':<16}: {BATCH_SIZE}")
    print(f"  {'imgsz':<16}: {cfg.INPUT_SIZE}")
    print(f"  {'Optimizer':<16}: {cfg.OPTIMIZER}")
    print(f"  {'lr0':<16}: {cfg.LEARNING_RATE}")
    print(f"  {'Device':<16}: {cfg.DEVICE}")
    print(f"  {'Workers':<16}: {cfg.WORKERS}")
    print(f"  {'CLS_LOSS':<16}: {cfg.CLS_LOSS}")
    print(f"  {'Mosaic':<16}: {cfg.MOSAIC}")
    print(f"  {'Geometry locks':<16}: degrees=0  flipud=0  shear=0")
    print(f"  {'':<16}  perspective=0  translate=0")
    print(f"  {'Safe augs':<16}: fliplr={cfg.FLIPLR}  scale={cfg.SCALE}")
    print(f"  {'':<16}  copy_paste={cfg.COPY_PASTE}  erasing={cfg.ERASING}")
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

        # Optimizer 
        lrf             = cfg.LRF,
        momentum        = cfg.MOMENTUM,
        weight_decay    = cfg.WEIGHT_DECAY,
        warmup_epochs   = cfg.WARMUP_EPOCHS,
        warmup_momentum = cfg.WARMUP_MOMENTUM,

        # Loss weights
        box             = cfg.BOX_LOSS,
        cls             = cfg.CLS_LOSS,
        dfl             = cfg.DFL_LOSS,

        # Geometry augmentations
        degrees         = cfg.DEGREES,      
        flipud          = cfg.FLIPUD,       
        shear           = cfg.SHEAR,        
        perspective     = cfg.PERSPECTIVE,  
        translate       = cfg.TRANSLATE,    
        mixup           = cfg.MIXUP,        

        # Additional augmentations 
        fliplr          = cfg.FLIPLR,       
        scale           = cfg.SCALE,        
        copy_paste      = cfg.COPY_PASTE,   
        erasing         = cfg.ERASING,      
        mosaic          = cfg.MOSAIC,      

        # Color augmentation 
        hsv_h           = cfg.HSV_H,
        hsv_s           = cfg.HSV_S,
        hsv_v           = cfg.HSV_V,

        # Run settings
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


# MAIN
def main():
    print("\n" + "=" * 60)
    print(f"  {MODEL_TAG}")
    print("  SINGLE-SHOT HEIGHT ESTIMATION OF MULTIPLE YOUNG")
    print("  MANGROVE TREES FOR FOREST ASSESSMENT")
    print(f"  Hyperparameters ")
    print(f"  Dataset         : Roboflow v{cfg.ROBOFLOW_VERSION}")
    print("=" * 60)

    cfg.check_api_key()
    cfg.check_gpu()

    yaml_path = cfg.download_dataset()
    cfg.verify_dataset(yaml_path)

    print(f"\n  Hyperparameter Summary")
    print(f"    Model      : {MODEL_WEIGHTS}")
    print(f"    Epochs     : {EPOCHS}")
    print(f"    Batch Size : {BATCH_SIZE}")
    print(f"    Workers    : {cfg.WORKERS}")
    print(f"    Resolution : {cfg.INPUT_SIZE}x{cfg.INPUT_SIZE}")
    print(f"    Optimizer  : {cfg.OPTIMIZER}")
    print(f"    Lr0        : {cfg.LEARNING_RATE}")
    print(f"    Device     : {cfg.DEVICE}")
    print(f"    Mosaic     : {cfg.MOSAIC} (enabled)")

    if input("\n  Start training? (yes/no): ").strip().lower() not in ("yes", "y"):
        print("  Cancelled.")
        return

    run_name = train(yaml_path)
    passed   = cfg.evaluate_model(WEIGHTS_OUT, run_name, yaml_path, MODEL_TAG)

    print("\n" + "=" * 60)
    if passed:
        print("  PASSED ")
    else:
        print("  Failed ")
    print(f"  Weights : {WEIGHTS_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()