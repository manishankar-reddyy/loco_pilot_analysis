"""
Train YOLOv8 on the custom Brake & Horn detection dataset.

Classes:
  0: horn_active
  1: horn_inactive
  2: brake_applied
  3: brake_released

Usage:
  python train_yolo.py
"""

from ultralytics import YOLO
import os

# ── Configuration ──────────────────────────────────────────────
DATA_YAML = os.path.join("yolo_dataset", "data.yaml")
BASE_MODEL = "yolov8s.pt"  # Using yolov8s (small) for better accuracy on small objects
EPOCHS = 100
IMG_SIZE = 480  # Match dataset resolution
BATCH_SIZE = 16
PROJECT = "runs"
EXPERIMENT = "brake_horn_detection"


def main():
    # Load pretrained model
    model = YOLO(BASE_MODEL)

    # Train with augmentation to handle class imbalance and small dataset
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=EXPERIMENT,
        # Augmentation (helps with small dataset & class imbalance)
        augment=True,
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation
        hsv_v=0.4,       # Value/brightness augmentation
        degrees=5.0,     # Small rotation (loco cab is mostly fixed angle)
        translate=0.1,   # Translation
        scale=0.3,       # Scale augmentation
        fliplr=0.5,      # Horizontal flip
        flipud=0.0,      # No vertical flip (not meaningful for cab view)
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.1,       # Mixup augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        # Training parameters
        patience=20,     # Early stopping patience
        lr0=0.01,        # Initial learning rate
        lrf=0.01,        # Final learning rate factor
        weight_decay=0.0005,
        warmup_epochs=3,
        # Save settings
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,      # Generate training plots
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best model: {os.path.join(PROJECT, EXPERIMENT, 'weights', 'best.pt')}")
    print(f"Last model: {os.path.join(PROJECT, EXPERIMENT, 'weights', 'last.pt')}")
    print(f"Results:    {os.path.join(PROJECT, EXPERIMENT)}")

    # Run validation on best model
    print("\nRunning validation on best model...")
    best_model = YOLO(os.path.join(PROJECT, EXPERIMENT, "weights", "best.pt"))
    metrics = best_model.val(data=DATA_YAML)

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Per-class results
    class_names = ["horn_active", "horn_inactive", "brake_applied", "brake_released"]
    print(f"\nPer-class AP50:")
    for i, name in enumerate(class_names):
        if i < len(metrics.box.ap50):
            print(f"  {name}: {metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
