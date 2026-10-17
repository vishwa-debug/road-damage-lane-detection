import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

# ============================================================
# RDD2022 XML → YOLO conversion
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RDD2022"
    / "RDD2022_released_through_CRDDC2022"
    / "RDD2022"
    / "India"
    / "India"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "road_damage"
)

IMAGE_OUTPUT = OUTPUT_ROOT / "images"
LABEL_OUTPUT = OUTPUT_ROOT / "labels"

CLASSES = [
    "D00",
    "D01",
    "D0w0",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASSES)
}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def convert_bbox_to_yolo(
    xmin,
    ymin,
    xmax,
    ymax,
    image_width,
    image_height,
):
    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return x_center, y_center, width, height


def find_image(image_name):
    candidates = list(DATASET_ROOT.rglob(image_name))

    if not candidates:
        return None

    return candidates[0]


def process_xml(xml_file, split):
    root = ET.parse(xml_file).getroot()

    filename = root.findtext("filename")

    if not filename:
        return False

    size = root.find("size")

    if size is None:
        return False

    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    image_path = find_image(filename)

    if image_path is None:
        print(f"WARNING: Image not found: {filename}")
        return False

    objects = root.findall("object")

    if not objects:
        return False

    label_lines = []

    for obj in objects:
        class_name = obj.findtext("name")

        if class_name not in CLASS_TO_ID:
            print(f"WARNING: Unknown class: {class_name}")
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        # Clamp coordinates
        xmin = max(0, min(xmin, image_width))
        ymin = max(0, min(ymin, image_height))
        xmax = max(0, min(xmax, image_width))
        ymax = max(0, min(ymax, image_height))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center, y_center, width, height = convert_bbox_to_yolo(
            xmin,
            ymin,
            xmax,
            ymax,
            image_width,
            image_height,
        )

        class_id = CLASS_TO_ID[class_name]

        label_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    if not label_lines:
        return False

    output_image_dir = IMAGE_OUTPUT / split
    output_label_dir = LABEL_OUTPUT / split

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    output_image = output_image_dir / image_path.name
    output_label = output_label_dir / f"{image_path.stem}.txt"

    shutil.copy2(image_path, output_image)

    output_label.write_text(
        "\n".join(label_lines),
        encoding="utf-8",
    )

    return True

def main():

    print("=" * 60)
    print("RDD2022 XML → YOLO CONVERSION")
    print("=" * 60)

    # Find all XML annotation files
    xml_files = list(DATASET_ROOT.rglob("*.xml"))

    print(f"XML files found: {len(xml_files)}")
    print("Scanning XML annotations...")

    # Find XML files that actually contain objects
    annotated_xml = []

    for index, xml_file in enumerate(xml_files, start=1):

        root = ET.parse(xml_file).getroot()
        objects = root.findall("object")

        if objects:
            annotated_xml.append(xml_file)

        if index % 500 == 0:
            print(
                f"Scanned {index}/{len(xml_files)} XML files..."
            )

    print(
        f"XML files containing annotations: "
        f"{len(annotated_xml)}"
    )

    # Shuffle for train/validation split
    random.seed(RANDOM_SEED)
    random.shuffle(annotated_xml)

    split_index = int(
        len(annotated_xml) * TRAIN_RATIO
    )

    train_files = annotated_xml[:split_index]
    val_files = annotated_xml[split_index:]

    print()
    print(f"Training XML files: {len(train_files)}")
    print(f"Validation XML files: {len(val_files)}")

    # Convert training data
    print()
    print("Converting training data...")

    train_count = 0

    for index, xml_file in enumerate(train_files, start=1):

        if process_xml(xml_file, "train"):
            train_count += 1

        if index % 500 == 0:
            print(
                f"Processed training: "
                f"{index}/{len(train_files)}"
            )

    # Convert validation data
    print()
    print("Converting validation data...")

    val_count = 0

    for index, xml_file in enumerate(val_files, start=1):

        if process_xml(xml_file, "val"):
            val_count += 1

        if index % 500 == 0:
            print(
                f"Processed validation: "
                f"{index}/{len(val_files)}"
            )

    print()
    print("=" * 60)
    print("CONVERSION COMPLETED")
    print("=" * 60)

    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

    print()
    print("Output:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
