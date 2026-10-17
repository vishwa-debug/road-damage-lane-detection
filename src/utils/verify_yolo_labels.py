from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "road_damage"
    / "images"
    / "train"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "road_damage"
    / "labels"
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "images"
    / "yolo_verification"
)

CLASS_NAMES = [
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


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list(IMAGE_DIR.glob("*.jpg"))

    print(f"Training images found: {len(images)}")

    # Check first 10 images
    for image_path in images[:10]:

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Missing label: {image_path.name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        height, width = image.shape[:2]

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"Invalid label in {label_path.name}: {line}"
                )
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates → pixels
            xmin = int(
                (x_center - box_width / 2) * width
            )

            ymin = int(
                (y_center - box_height / 2) * height
            )

            xmax = int(
                (x_center + box_width / 2) * width
            )

            ymax = int(
                (y_center + box_height / 2) * height
            )

            # Keep coordinates inside image
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width - 1, xmax)
            ymax = min(height - 1, ymax)

            class_name = (
                CLASS_NAMES[class_id]
                if 0 <= class_id < len(CLASS_NAMES)
                else f"UNKNOWN_{class_id}"
            )

            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                class_name,
                (xmin, max(20, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        output_path = (
            OUTPUT_DIR / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            f"Created: {output_path}"
        )

    print()
    print("YOLO label verification completed.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()