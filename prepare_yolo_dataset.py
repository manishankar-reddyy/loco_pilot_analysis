"""
Prepare YOLO dataset from LabelMe JSON annotations.

Converts LabelMe polygon annotations to YOLO bounding box format,
renames classes (Option C), and splits into train/val sets.

Classes:
  0: horn_active
  1: horn_inactive
  2: brake_applied
  3: brake_released
"""

import json
import os
import glob
import shutil
import random
import numpy as np

# ── Configuration ──────────────────────────────────────────────
DATASET_DIR = "dataset"
OUTPUT_DIR = "yolo_dataset"
VAL_SPLIT = 0.2  # 20% for validation
RANDOM_SEED = 42

# Class mapping: LabelMe label → (YOLO class_id, new_name)
CLASS_MAP = {
    "horn": (0, "horn_active"),
    "no_horn": (1, "horn_inactive"),
    "brake": (2, "brake_applied"),
    "no_brake": (3, "brake_released"),
}

CLASS_NAMES = ["horn_active", "horn_inactive", "brake_applied", "brake_released"]


def polygon_to_bbox(points, img_w, img_h):
    """Convert polygon points to YOLO normalized bbox (x_center, y_center, w, h)."""
    pts = np.array(points)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    # Convert to YOLO format (normalized x_center, y_center, width, height)
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height


def convert_labelme_to_yolo(json_path):
    """Convert a single LabelMe JSON file to YOLO format lines."""
    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            print(f"  WARNING: Unknown label '{label}' in {json_path}, skipping")
            continue

        class_id = CLASS_MAP[label][0]
        points = shape["points"]
        x_c, y_c, w, h = polygon_to_bbox(points, img_w, img_h)
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return lines


def main():
    random.seed(RANDOM_SEED)

    # ── Collect all annotation files ──
    json_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    print(f"Found {len(json_files)} annotation files")

    # ── Create output directory structure ──
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    # ── Shuffle and split ──
    indices = list(range(len(json_files)))
    random.shuffle(indices)
    val_count = int(len(indices) * VAL_SPLIT)
    val_indices = set(indices[:val_count])
    train_indices = set(indices[val_count:])

    print(f"Train: {len(train_indices)} images, Val: {len(val_indices)} images")

    # ── Convert and copy ──
    stats = {"train": {c: 0 for c in CLASS_NAMES}, "val": {c: 0 for c in CLASS_NAMES}}
    skipped = 0

    for i, json_path in enumerate(json_files):
        split = "val" if i in val_indices else "train"
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = json_path.replace(".json", ".jpg")

        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found for {json_path}, skipping")
            skipped += 1
            continue

        # Convert annotations
        yolo_lines = convert_labelme_to_yolo(json_path)

        if not yolo_lines:
            print(f"  WARNING: No valid annotations in {json_path}, skipping")
            skipped += 1
            continue

        # Copy image
        dst_img = os.path.join(OUTPUT_DIR, split, "images", f"{base_name}.jpg")
        shutil.copy2(img_path, dst_img)

        # Write YOLO label file
        dst_label = os.path.join(OUTPUT_DIR, split, "labels", f"{base_name}.txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # Count stats
        for line in yolo_lines:
            cls_id = int(line.split()[0])
            stats[split][CLASS_NAMES[cls_id]] += 1

    # ── Print summary ──
    print(f"\n{'='*50}")
    print("CONVERSION COMPLETE")
    print(f"{'='*50}")
    print(f"Skipped: {skipped}")
    print(f"\nClass distribution:")
    print(f"{'Class':<20} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 48)
    for cls in CLASS_NAMES:
        t = stats["train"][cls]
        v = stats["val"][cls]
        print(f"{cls:<20} {t:>8} {v:>8} {t+v:>8}")

    total_t = sum(stats["train"].values())
    total_v = sum(stats["val"].values())
    print("-" * 48)
    print(f"{'Total':<20} {total_t:>8} {total_v:>8} {total_t+total_v:>8}")

    # ── Create data.yaml ──
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    abs_output = os.path.abspath(OUTPUT_DIR)
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Brake & Horn Detection for Loco Pilot Analysis

path: {abs_output}
train: train/images
val: val/images

# Classes (Option C - renamed)
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nCreated: {yaml_path}")
    print(f"Output directory: {abs_output}")


if __name__ == "__main__":
    main()
